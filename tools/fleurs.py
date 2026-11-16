"""Pick test sentences from Google's FLEURS (CC-BY 4.0) for compare.mjs.

    python fleurs.py da_dk 50

Downloads data/<code>/dev.tsv and dev.tar.gz from huggingface.co/datasets/google/fleurs
(if not already in data/fleurs-<lang>/), and writes the first N distinct
recordings to audio/<lang>-fleurs/NNN.wav with their transcripts in
audio/<lang>-fleurs/sentences.txt. FLEURS is read speech by real people, and none of
the models compared here were trained on its dev split.
"""
import csv
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"


def main(code: str, n: str = "50"):
    lang = code.split("_")[0]
    data = Path(f"data/fleurs-{lang}")
    data.mkdir(parents=True, exist_ok=True)
    for name, url in (("dev.tsv", f"{BASE}/{code}/dev.tsv"),
                      ("dev.tar.gz", f"{BASE}/{code}/audio/dev.tar.gz")):
        if not (data / name).exists():
            urllib.request.urlretrieve(url, data / name)
    rows, seen = [], set()
    with open(data / "dev.tsv", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if row[0] not in seen:  # several speakers read the same sentence
                seen.add(row[0])
                rows.append((row[1], row[2]))
            if len(rows) == int(n):
                break
    out = Path(f"audio/{lang}-fleurs")
    out.mkdir(parents=True, exist_ok=True)
    wanted = {name: i for i, (name, _) in enumerate(rows)}
    with tarfile.open(data / "dev.tar.gz") as tar:
        for member in tar:
            i = wanted.get(Path(member.name).name)
            if i is not None:
                (out / f"{i:03d}.wav").write_bytes(tar.extractfile(member).read())
    (out / "sentences.txt").write_text("\n".join(t for _, t in rows) + "\n", encoding="utf-8")
    print(len(list(out.glob("*.wav"))), "recordings in", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
