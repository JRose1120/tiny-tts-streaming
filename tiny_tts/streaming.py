import queue
import re
import threading
from typing import Iterator, Optional, TYPE_CHECKING

import numpy as np

from tiny_tts.utils.config import SAMPLING_RATE

if TYPE_CHECKING:
    from tiny_tts import TinyTTS


class StreamingTinyTTS:
    """
    Chunked streaming wrapper around TinyTTS.

    This class accepts text incrementally, chunks it at punctuation/whitespace,
    synthesizes each chunk in a background worker, and returns stitched audio
    blocks with overlap-add blending to reduce seam artifacts.
    """

    _STOP = object()
    _END = object()
    _PUNCT_RE = re.compile(r"[.!?;:]$")

    def __init__(
        self,
        tts: "TinyTTS",
        speaker: str = "MALE",
        speed: float = 1.0,
        min_chunk_chars: int = 40,
        target_chunk_chars: int = 100,
        max_chunk_chars: int = 180,
        crossfade_ms: float = 40.0,
        punctuation_pause_ms: float = 60.0,
        continuation_pause_ms: float = 8.0,
        trim_chunk_silence: bool = True,
        silence_threshold_db: float = -42.0,
        trim_edge_pad_ms: float = 8.0,
        min_chunk_keep_ms: float = 60.0,
    ):
        if min_chunk_chars <= 0 or target_chunk_chars <= 0 or max_chunk_chars <= 0:
            raise ValueError("Chunk sizes must be positive")
        if not (min_chunk_chars <= target_chunk_chars <= max_chunk_chars):
            raise ValueError("Expected min_chunk_chars <= target_chunk_chars <= max_chunk_chars")

        self.tts = tts
        self.speaker = speaker
        self.speed = speed

        self.min_chunk_chars = min_chunk_chars
        self.target_chunk_chars = target_chunk_chars
        self.max_chunk_chars = max_chunk_chars

        self.crossfade_samples = max(0, int(SAMPLING_RATE * (crossfade_ms / 1000.0)))
        self.punctuation_pause_samples = max(0, int(SAMPLING_RATE * (punctuation_pause_ms / 1000.0)))
        self.continuation_pause_samples = max(0, int(SAMPLING_RATE * (continuation_pause_ms / 1000.0)))
        self.trim_chunk_silence = trim_chunk_silence
        self.silence_threshold_db = float(silence_threshold_db)
        self.trim_edge_pad_samples = max(0, int(SAMPLING_RATE * (trim_edge_pad_ms / 1000.0)))
        self.min_chunk_keep_samples = max(1, int(SAMPLING_RATE * (min_chunk_keep_ms / 1000.0)))

        self._text_buffer = ""
        self._audio_tail: Optional[np.ndarray] = None
        self._closed = False

        self._text_queue: queue.Queue = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def push_text(self, text: str) -> None:
        """Push newly arrived text into the stream."""
        if self._closed:
            raise RuntimeError("Cannot push text after end_input()")
        if not text:
            return

        self._text_buffer += text
        self._emit_ready_chunks(force=False)

    def flush_text(self) -> None:
        """Force chunk emission for all currently buffered text."""
        self._emit_ready_chunks(force=True)

    def end_input(self) -> None:
        """Signal no more text will arrive and finalize worker output."""
        if self._closed:
            return
        self.flush_text()
        self._closed = True
        self._text_queue.put(self._STOP)

    def read_audio(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Read one stitched audio block.

        Returns:
            np.ndarray for an audio chunk, or None when the stream has ended.
        """
        item = self._audio_queue.get(timeout=timeout)
        if item is self._END:
            return None
        return item

    def iter_audio(self, timeout: Optional[float] = None) -> Iterator[np.ndarray]:
        """Iterate over output audio blocks until stream completion."""
        while True:
            block = self.read_audio(timeout=timeout)
            if block is None:
                break
            yield block

    def close(self) -> None:
        """Gracefully stop the stream and worker thread."""
        self.end_input()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)

    def _worker_loop(self) -> None:
        while True:
            item = self._text_queue.get()
            if item is self._STOP:
                final = self._finalize_tail()
                if final.size > 0:
                    self._audio_queue.put(final)
                self._audio_queue.put(self._END)
                break

            chunk = item
            audio = self.tts.synthesize_to_array(chunk, speaker=self.speaker, speed=self.speed)
            audio = np.asarray(audio, dtype=np.float32)
            audio = self._trim_edge_silence(audio)
            audio = self._append_pause(audio, chunk)

            stitched = self._stitch_chunk(audio)
            if stitched.size > 0:
                self._audio_queue.put(stitched)

    def _emit_ready_chunks(self, force: bool) -> None:
        while True:
            split_idx = self._choose_split_index(force=force)
            if split_idx is None:
                if force and self._text_buffer.strip():
                    self._text_queue.put(self._text_buffer.strip())
                    self._text_buffer = ""
                break

            chunk = self._text_buffer[:split_idx].strip()
            self._text_buffer = self._text_buffer[split_idx:].lstrip()
            if chunk:
                self._text_queue.put(chunk)

    def _choose_split_index(self, force: bool) -> Optional[int]:
        text = self._text_buffer
        n = len(text)

        if n < self.min_chunk_chars and not force:
            return None

        hard_limit = min(n, self.max_chunk_chars)
        preferred_limit = min(n, self.target_chunk_chars)

        # Prefer punctuation up to preferred target, then up to hard cap.
        punct_split = self._last_split_after_pattern(text, r"[.!?;:](?:\s+|$)", preferred_limit)
        if punct_split is None:
            punct_split = self._last_split_after_pattern(text, r"[.!?;:](?:\s+|$)", hard_limit)
        if punct_split is not None and punct_split >= self.min_chunk_chars:
            return punct_split

        # Otherwise split on whitespace.
        ws_split = self._last_split_after_pattern(text, r"\s+", preferred_limit)
        if ws_split is None:
            ws_split = self._last_split_after_pattern(text, r"\s+", hard_limit)
        if ws_split is not None and ws_split >= self.min_chunk_chars:
            return ws_split

        if n >= self.max_chunk_chars:
            return self.max_chunk_chars

        if force and n > 0:
            return n

        return None

    @staticmethod
    def _last_split_after_pattern(text: str, pattern: str, limit: int) -> Optional[int]:
        last = None
        for match in re.finditer(pattern, text[:limit]):
            last = match.end()
        return last

    def _stitch_chunk(self, audio: np.ndarray) -> np.ndarray:
        overlap = self.crossfade_samples
        if overlap <= 0:
            if self._audio_tail is None:
                return audio
            out = np.concatenate([self._audio_tail, audio]).astype(np.float32)
            self._audio_tail = None
            return out

        if self._audio_tail is None:
            if len(audio) <= overlap:
                self._audio_tail = audio
                return np.zeros(0, dtype=np.float32)
            self._audio_tail = audio[-overlap:]
            return audio[:-overlap]

        prev = self._audio_tail
        mix = min(len(prev), len(audio), overlap)

        prev_overlap = prev[-mix:]
        next_overlap = audio[:mix]

        # Fixed equal-power fade for a simple low-overhead stitch path.
        phase = np.linspace(0.0, np.pi / 2.0, mix, dtype=np.float32)
        fade_in = np.sin(phase)
        fade_out = np.cos(phase)

        prefix = prev[:-mix] if len(prev) > mix else np.zeros(0, dtype=np.float32)
        cross = prev_overlap * fade_out + next_overlap * fade_in

        if len(audio) <= overlap:
            # Keep accumulating very short audio into tail until we have enough
            # samples to confidently emit a body region.
            self._audio_tail = np.concatenate([prefix, cross, audio[mix:]]).astype(np.float32)
            return np.zeros(0, dtype=np.float32)

        body = audio[mix:-overlap]
        self._audio_tail = audio[-overlap:]
        return np.concatenate([prefix, cross, body]).astype(np.float32)

    def _finalize_tail(self) -> np.ndarray:
        if self._audio_tail is None:
            return np.zeros(0, dtype=np.float32)
        out = self._audio_tail.astype(np.float32)
        self._audio_tail = None
        return out

    def _append_pause(self, audio: np.ndarray, chunk_text: str) -> np.ndarray:
        text = chunk_text.rstrip()
        if not text:
            return audio

        if self._PUNCT_RE.search(text):
            n_pause = self.punctuation_pause_samples
        else:
            n_pause = self.continuation_pause_samples

        if n_pause <= 0:
            return audio

        silence = np.zeros(n_pause, dtype=np.float32)
        return np.concatenate([audio, silence]).astype(np.float32)

    def _trim_edge_silence(self, audio: np.ndarray) -> np.ndarray:
        if not self.trim_chunk_silence or audio.size == 0:
            return audio

        threshold = 10.0 ** (self.silence_threshold_db / 20.0)
        mask = np.abs(audio) >= threshold
        if not np.any(mask):
            return audio

        first_idx = int(np.argmax(mask))
        last_idx = int(len(mask) - 1 - np.argmax(mask[::-1]))

        start = max(0, first_idx - self.trim_edge_pad_samples)
        end = min(audio.size, last_idx + 1 + self.trim_edge_pad_samples)
        trimmed = audio[start:end]

        if trimmed.size < self.min_chunk_keep_samples:
            center = (start + end) // 2
            half = self.min_chunk_keep_samples // 2
            safe_start = max(0, center - half)
            safe_end = min(audio.size, safe_start + self.min_chunk_keep_samples)
            safe_start = max(0, safe_end - self.min_chunk_keep_samples)
            return audio[safe_start:safe_end].astype(np.float32)

        return trimmed.astype(np.float32)
