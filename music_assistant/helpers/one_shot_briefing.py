"""
Development-only MA-side, session-scoped one-shot briefing mixer.

Not imported by HA-AlarmClock. This module renders through registered MA TTS
engines and mixes trusted resolved clips. Public commands live in the reviewed
controller patch; no credentials or provider audio locations are returned.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import AsyncGenerator, Callable
from contextlib import suppress
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from music_assistant_models.media_items import AudioFormat

    from music_assistant.mass import MusicAssistant


@dataclass
class PreparedBriefing:
    """Private rendered clip, passed directly into one locked playback call."""

    run_id: str
    source: str = field(repr=False)
    levels: MixLevels = field(default_factory=lambda: MixLevels())


async def prepare_briefing(
    mass: MusicAssistant,
    run_id: str,
    message: str,
    tts_engine: str,
    language: str | None = None,
    voice: str | None = None,
    *,
    levels: MixLevels | None = None,
) -> PreparedBriefing:
    """Render using a registered engine, never a caller-provided audio location."""
    from music_assistant.helpers.plugin_engines import resolve_tts_engine  # noqa: PLC0415
    from music_assistant.helpers.tts import (  # noqa: PLC0415
        query_tts_engine,
        resolve_tts_stream_path,
    )

    _identifier(run_id)
    _identifier(tts_engine)
    if not isinstance(message, str) or not message.strip() or len(message) > 4096:
        raise ValueError("Briefing text must contain between one and 4096 characters")
    for value, limit in ((language, 32), (voice, 128)):
        if value is not None and (
            not isinstance(value, str)
            or not value
            or len(value) > limit
            or any(ord(c) < 32 for c in value)
        ):
            raise ValueError("Invalid language or voice identifier")
    levels = levels or MixLevels()
    levels.validate()
    try:
        async with asyncio.timeout(30):
            engine = await resolve_tts_engine(mass, tts_engine)
            if engine is None:
                raise ValueError("Selected TTS engine is unavailable")
            # No fallback to another language/engine: preserve the requested assistant.
            details = await query_tts_engine(
                engine, message, language, timeout=25, options={"voice": voice} if voice else None
            )
            duration = details.duration
            if duration is not None and (not math.isfinite(duration) or not 0 < duration <= 240):
                raise ValueError("Briefing exceeds the supported duration")
            source, _ = await resolve_tts_stream_path(engine, details)
            _resolved_source(source)
            return PreparedBriefing(run_id, source, levels)
    except Exception:
        # Engine errors may contain private text, URLs or file paths.
        raise ValueError("Unable to prepare briefing with the selected TTS engine") from None


@dataclass(frozen=True)
class MixLevels:
    """Bounded linear gains for the two PCM inputs."""

    music: float = 0.25
    speech: float = 0.7
    delay_seconds: int = 0

    def validate(self) -> None:
        """Reject invalid gains and preserve mix headroom."""
        for value in (self.music, self.speech):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Mix gains must be finite values between zero and one")
        if self.music + self.speech > 1:
            raise ValueError("Leave headroom: combined gains must not exceed one")
        if type(self.delay_seconds) is not int or not 0 <= self.delay_seconds <= 120:
            raise ValueError("Speech delay must be between zero and 120 seconds")


DEFAULT_LEVELS = MixLevels()


@dataclass
class Briefing:
    """A private finite clip tied to one queue/session/run."""

    queue_id: str
    session_id: str
    run_id: str
    source: str = field(repr=False)
    levels: MixLevels
    expires: float
    state: str = "pending"
    task: asyncio.Task[Any] | None = field(default=None, repr=False)


def _identifier(value: str) -> None:
    if not isinstance(value, str) or not 1 <= len(value) <= 128 or any(ord(c) < 32 for c in value):
        raise ValueError("Expected a nonempty bounded queue/session/run identifier")


def _resolved_source(value: str) -> None:
    if not isinstance(value, str) or not 1 <= len(value) <= 4096 or any(ord(c) < 32 for c in value):
        raise ValueError("Expected a bounded, MA-resolved audio source")
    parts = urlsplit(value)
    if parts.scheme in ("http", "https") and parts.hostname:
        if parts.username is not None or parts.password is not None:
            raise ValueError("Embedded credentials are not allowed")
    elif not Path(value).is_absolute():
        raise ValueError(
            "Only trusted MA-resolved HTTP(S) audio or absolute local paths are supported"
        )


def _check_format(pcm: AudioFormat) -> None:
    kind = getattr(pcm.content_type, "value", pcm.content_type)
    depths = {"s16le": 16, "s24le": 24, "s32le": 32, "f32le": 32}
    if (
        depths.get(kind) != pcm.bit_depth
        or pcm.channels not in (1, 2)
        or not 8000 <= pcm.sample_rate <= 192000
    ):
        raise ValueError("Prototype requires mono/stereo little-endian PCM with a matching format")


def native_mix(
    song_input: AsyncGenerator[bytes], source: str, pcm_format: AudioFormat, levels: MixLevels
) -> AsyncGenerator[bytes]:
    """Use MA 2.10.12's real ComplexFilter/get_ffmpeg_stream machinery lazily."""
    from music_assistant.helpers.dsp import ComplexFilter, ComplexFilterInput  # noqa: PLC0415
    from music_assistant.helpers.ffmpeg import get_ffmpeg_stream  # noqa: PLC0415

    # Same two-input amix design as MA's ambient overlay, but no looping or
    # reconnection of the voice. Reconnecting a voice URL could repeat its words.
    args = ["-stream_loop", "0", "-t", "240"]
    if urlsplit(source).scheme in ("http", "https"):
        args += ["-rw_timeout", "5000000"]
    layout = "mono" if pcm_format.channels == 1 else "stereo"
    clip = ComplexFilterInput(
        path=source,
        input_args=args,
        filters=(
            f"atrim=duration=240,volume={levels.speech},"
            f"aresample={pcm_format.sample_rate},aformat=channel_layouts={layout},"
            f"adelay=delays={levels.delay_seconds * 1000}:all=1"
        ),
    )
    graph = ComplexFilter(
        body="amix=inputs=2:duration=first:dropout_transition=0:normalize=0",
        inputs=[clip],
    )
    # The song is always input zero. No -shortest: ending the voice must not end
    # the song. Its ordinary MA provider, queue and repeat behaviour stay intact.
    return get_ffmpeg_stream(
        audio_input=song_input,
        input_format=copy(pcm_format),
        output_format=pcm_format,
        filter_params=[f"volume={levels.music}", graph],
        chunk_size=pcm_format.sample_rate * pcm_format.channels * (pcm_format.bit_depth // 8),
    )


async def _close(stream: AsyncGenerator[bytes]) -> None:
    if (close := getattr(stream, "aclose", None)) is not None:
        with suppress(Exception):
            await close()


class OneShotMixer:
    """
    Runtime-only, single-consumer proof; never serialize sources or speech.

    Arm only after MA has created the intended session and before dispatching its
    stream. A run is consumed before opening FFmpeg, including failed attempts.
    Duplicate GETs, reconnections and song repeats therefore do not replay voice.
    """

    def __init__(
        self,
        *,
        engine: Callable[
            [AsyncGenerator[bytes], str, AudioFormat, MixLevels], AsyncGenerator[bytes]
        ] = native_mix,
        clock: Callable[[], float] = monotonic,
        max_runs: int = 65536,
    ) -> None:
        """Initialize bounded runtime state with an injectable audio engine."""
        self.engine = engine
        self.clock = clock
        self.max_runs = max_runs
        self._pending: dict[str, Briefing] = {}
        self._active: dict[str, Briefing] = {}
        self._seen: set[tuple[str, str]] = set()
        self._owners: dict[str, dict[str, Any]] = {}

    def bind_owner(self, queue_id: str, session_id: str | None, owner: dict[str, Any]) -> None:
        """Retain private test ownership after the finite speech stream finishes."""
        self._owners[queue_id] = {**owner, "session_id": session_id}

    def owner(self, queue_id: str, run_id: str, session_id: str | None) -> dict[str, Any] | None:
        """Return ownership only for the exact active or snoozed test session."""
        owner = self._owners.get(queue_id)
        if owner and owner["run_id"] == run_id and owner["session_id"] == session_id:
            return dict(owner)
        return None

    def arm(
        self,
        queue_id: str,
        session_id: str,
        run_id: str,
        source: str,
        *,
        levels: MixLevels = DEFAULT_LEVELS,
        ttl: float = 30,
    ) -> bool:
        """Arm one trusted clip before dispatching this queue/session."""
        for identifier in (queue_id, session_id, run_id):
            _identifier(identifier)
        _resolved_source(source)
        levels.validate()
        if type(ttl) not in (int, float) or not math.isfinite(ttl) or not 0 < ttl <= 120:
            raise ValueError("Arming TTL must be finite and between zero and 120 seconds")
        if (queue_id, run_id) in self._seen:
            return False
        if len(self._seen) >= self.max_runs:
            raise ValueError("Prototype run limit reached; do not evict replay protection")
        self._expire()
        if queue_id in self._pending or queue_id in self._active:
            raise ValueError("This queue already has an armed or active briefing")
        self._pending[queue_id] = Briefing(
            queue_id, session_id, run_id, source, levels, self.clock() + ttl
        )
        self._seen.add((queue_id, run_id))
        return True

    def has_run(self, queue_id: str, run_id: str) -> bool:
        """Check replay protection before doing any expensive TTS work."""
        _identifier(queue_id)
        _identifier(run_id)
        return (queue_id, run_id) in self._seen

    def reject_run(self, queue_id: str, run_id: str) -> None:
        """Tombstone a stopped request even if its start has not acquired the lock."""
        _identifier(queue_id)
        _identifier(run_id)
        if len(self._seen) < self.max_runs:
            self._seen.add((queue_id, run_id))

    def claim_music_run(self, queue_id: str, run_id: str) -> None:
        """Consume a music-only start id with the same fail-closed replay bound."""
        _identifier(queue_id)
        _identifier(run_id)
        if (queue_id, run_id) in self._seen:
            raise ValueError("This alarm start was already attempted")
        if len(self._seen) >= self.max_runs:
            raise ValueError("Alarm run limit reached; replay protection was retained")
        self._seen.add((queue_id, run_id))

    def cancel(self, queue_id: str, run_id: str) -> None:
        """Cancel only this run; do not cancel another alarm that owns the queue."""
        for collection in (self._pending, self._active):
            plan = collection.get(queue_id)
            if plan is None or plan.run_id != run_id:
                continue
            del collection[queue_id]
            plan.state = "cancelled"
            plan.source = ""
            if plan.task is not None:
                plan.task.cancel()

    def cancel_queue(self, queue_id: str) -> None:
        """For MA's permission-checked queue stop/pause handlers only."""
        for collection in (self._pending, self._active):
            if (plan := collection.get(queue_id)) is not None:
                self.cancel(queue_id, plan.run_id)
        self._owners.pop(queue_id, None)

    async def wrap(  # noqa: PLR0915
        self,
        queue_id: str,
        session_id: str | None,
        song_input: AsyncGenerator[bytes],
        pcm_format: AudioFormat,
    ) -> AsyncGenerator[bytes]:
        """Mix a claimed clip once and close owned generators on every exit path."""
        self._expire()
        plan = self._pending.get(queue_id)
        if plan is None or plan.session_id != session_id:
            owner = self._owners.get(queue_id)
            if owner and owner["session_id"] == session_id:
                # Music-only restarts keep the separately chosen background gain.
                from music_assistant.helpers.ffmpeg import get_ffmpeg_stream  # noqa: PLC0415

                song_input = get_ffmpeg_stream(
                    audio_input=song_input,
                    input_format=copy(pcm_format),
                    output_format=pcm_format,
                    filter_params=[f"volume={owner['music_gain']}"],
                )
            try:
                async for chunk in song_input:
                    yield chunk
            finally:
                await _close(song_input)
            return

        try:
            _check_format(pcm_format)
        except ValueError:
            self.cancel(queue_id, plan.run_id)
            await _close(song_input)
            raise
        # No await between checking/removing the plan and claiming ownership.
        del self._pending[queue_id]
        plan.state = "claimed"
        plan.task = asyncio.current_task()
        self._active[queue_id] = plan
        forwarded = 0
        mixed = None

        async def tracked_song() -> AsyncGenerator[bytes]:
            nonlocal forwarded
            async for chunk in song_input:
                forwarded += len(chunk)
                yield chunk

        tracked = tracked_song()
        try:
            mixed = self.engine(tracked, plan.source, pcm_format, plan.levels)
            async for chunk in mixed:
                yield chunk
            plan.state = "finished"
        except asyncio.CancelledError:
            plan.state = "cancelled"
            raise
        except Exception as err:
            plan.state = "failed"
            # Never log a clip URL, token, path, or speech. If nothing has been
            # consumed yet it is safe to continue the original song unchanged.
            LOGGER.warning("Briefing mix failed (%s)", type(err).__name__)
            if forwarded:
                raise RuntimeError(
                    "Briefing mix failed after consuming song audio; restart music-only"
                ) from None
            async for chunk in song_input:
                yield chunk
        finally:
            if plan.state == "claimed":
                plan.state = "cancelled"
            if mixed is not None:
                await _close(mixed)
            await _close(tracked)
            await _close(song_input)
            plan.task = None
            plan.source = ""
            if self._active.get(queue_id) is plan:
                del self._active[queue_id]

    def _expire(self) -> None:
        for key, plan in list(self._pending.items()):
            if self.clock() >= plan.expires:
                plan.state = "expired"
                plan.source = ""
                del self._pending[key]
