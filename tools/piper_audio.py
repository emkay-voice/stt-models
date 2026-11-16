"""Speak tests/sentences_LANG.txt with a Piper voice into audio/LANG/NN.wav.

    python tools/piper_audio.py da da_DK-talesyntese-medium

The voice is downloaded from huggingface.co/rhasspy/piper-voices into voices/
the first time. Use the voice Emkay answers with in that language: the test
then hears what Emkay itself would say.
"""
import subprocess
import sys
import urllib.request
from pathlib import Path

VOICES = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def voice_path(voice: str) -> Path:
    """Download VOICE (like da_DK-talesyntese-medium) and return its .onnx."""
    locale, name, quality = voice.split("-")
    folder = f"{VOICES}/{locale.split('_')[0]}/{locale}/{name}/{quality}"
    out = Path("voices")
    out.mkdir(exist_ok=True)
    for f in (f"{voice}.onnx", f"{voice}.onnx.json"):
        if not (out / f).exists():
            urllib.request.urlretrieve(f"{folder}/{f}", out / f)
    return out / f"{voice}.onnx"


def main(lang: str, voice: str):
    model = voice_path(voice)
    out = Path(f"audio/{lang}")
    out.mkdir(parents=True, exist_ok=True)
    piper = Path(sys.executable).with_name("piper")
    sentences = Path(f"tests/sentences_{lang}.txt").read_text(encoding="utf-8").split("\n")
    for i, text in enumerate(filter(None, sentences), 1):
        subprocess.run([piper, "-m", model, "-f", out / f"{i:02d}.wav"], input=text.encode(),
                       check=True, capture_output=True)
    print(len(list(out.glob("*.wav"))), "sentences in", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
