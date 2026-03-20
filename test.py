from tiny_tts import TinyTTS

# Initialize the TTS model (auto-detects device and downloads default checkpoint if missing)
tts = TinyTTS()
# OR specify a custom checkpoint: tts = TinyTTS(checkpoint_path="...")

# Synthesize a single sentence
tts.speak("Hello, this is a test of the Python API.", output_path="hello.wav")

# Adjust speech speed (1.0=normal, 1.5=faster, 0.7=slower)
tts.speak("This is faster speech.", output_path="fast.wav", speed=1.5)
tts.speak("This is slower speech.", output_path="slow.wav", speed=0.7)

# Synthesize a long paragraph (5 sentences)
paragraph = (
    "TinyTTS is an ultra-lightweight text-to-speech model. "
    "It has only one point six million parameters, which makes it extremely fast. "
    "You can run it easily on your local CPU without a dedicated graphics card. "
    "The audio quality remains surprisingly clear despite the small model size. "
    "I hope you enjoy building exciting applications with it!"
)
tts.speak(paragraph, output_path="paragraph.wav")