import time

import numpy as np
import soundfile as sf

from tiny_tts import TinyTTS, StreamingTinyTTS
from tiny_tts.utils.config import SAMPLING_RATE


def main():
    tts = TinyTTS()

    streamer = StreamingTinyTTS(
        tts,
        speaker="MALE",
        speed=1.0,
        min_chunk_chars=30,
        target_chunk_chars=90, #90
        max_chunk_chars=160, #160
        crossfade_ms=100,
        punctuation_pause_ms=90,
        continuation_pause_ms=0,
    )

    # Simulate text arriving over time, e.g., from ASR/LLM token stream.
    partial_text_messages = [
        "Here ", "is ", "the ", "plan." ,"If ", "We ", "Can ", "Tokenize ", "a ", "stream ", "of ", "text ", "we ", "can ", "make ", "audio ", "that ", "sounds ", "like ", "this."
    ]

    for msg in partial_text_messages:
        streamer.push_text(msg)
        time.sleep(0.08)

    streamer.end_input()

    blocks = list(streamer.iter_audio(timeout=10.0))
    if not blocks:
        raise RuntimeError("No audio blocks were produced")

    audio = np.concatenate(blocks)
    sf.write("stream_output.wav", audio, SAMPLING_RATE)
    print(f"Saved stream_output.wav ({len(audio) / SAMPLING_RATE:.2f}s)")


if __name__ == "__main__":
    main()
