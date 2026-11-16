"""Write the model card (README.md) and license for a converted model.

    python tools/model_card.py ORG/MODEL out/NAME [RESULTS_URL]

The card keeps the original model's license, languages and datasets, says
where it came from and how it was converted, and copies the original's
license files (LICENSE*, *.md licenses) next to it. A converted model is
still the original model, so its license terms apply unchanged.
"""
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, ModelCard, hf_hub_download

REPO = "https://github.com/emkay-voice/stt-models"


def main(source: str, out: str, results: str = ""):
    out = Path(out)
    meta = ModelCard.load(source).data.to_dict()
    whisper = (out / "onnx" / "decoder_model_merged.onnx").exists()
    data = {
        "base_model": source,
        "library_name": "transformers.js",
        "pipeline_tag": "automatic-speech-recognition",
        "tags": ["onnx", "emkay", "openvoiceos"],
    }
    for key in ("license", "license_name", "license_link", "language", "datasets"):
        if meta.get(key):
            data[key] = meta[key]
    # license files of the original go along unchanged
    copied = []
    for f in HfApi().list_repo_files(source):
        if Path(f).name.upper().startswith(("LICENSE", "LICENCE")):
            hf_hub_download(source, f, local_dir=out)
            copied.append(f)
    if not copied:
        # Some cards link to a license kept in another repository (Røst links
        # to Alvenir/coral-1-whisper-large/LICENSE); OpenRAIL licenses require
        # their use restrictions to travel with the model, so fetch it.
        card = Path(hf_hub_download(source, "README.md")).read_text(encoding="utf-8")
        m = re.search(r"huggingface\.co/([\w.-]+/[\w.-]+)/blob/main/(LICEN[SC]E[\w.-]*)", card)
        if m:
            Path(out / "LICENSE").write_text(
                Path(hf_hub_download(m[1], m[2])).read_text(encoding="utf-8"), encoding="utf-8")
            copied.append(f"LICENSE, from {m[1]}")
    files = ("onnx/encoder_model_quantized.onnx + onnx/decoder_model_merged_quantized.onnx"
             if whisper else "onnx/model_quantized.onnx")
    license_note = (f"the license files ({', '.join(copied)}) are copied from the original"
                    if copied else f"see [{source}](https://huggingface.co/{source}) for the full terms")
    text = f"""
# {source.split('/')[-1]} in ONNX

[{source}](https://huggingface.co/{source}) converted to ONNX for
[transformers.js](https://github.com/huggingface/transformers.js), so it runs
in a browser. It is the speech recognition of
[Emkay](https://github.com/emkay-voice/emkay), OpenVoiceOS in the browser.

Nothing was retrained: the weights are the original's, exported with
[optimum](https://github.com/huggingface/optimum) and also quantized to
8 bit ({files}, the files Emkay uses). The conversion script and the tests are
in [emkay-voice/stt-models]({REPO}).
{f"{chr(10)}Test results and the reasons this model was chosen: {results}{chr(10)}" if results else ""}
## License

This is the original model in another file format, so **the original
license applies unchanged**, including any use restrictions in it:
{license_note}.

## Use

```js
import {{ pipeline }} from "@huggingface/transformers";
const asr = await pipeline("automatic-speech-recognition", "emkay-voice/{out.name}",
  {{ dtype: "q8" }});
const {{ text }} = await asr(audio16kHz);
```
"""
    (out / "README.md").write_text(
        "---\n" + "".join(f"{k}: {v!r}\n" if not isinstance(v, list) else
                          f"{k}:\n" + "".join(f"- {x}\n" for x in v) for k, v in data.items())
        + "---\n" + text, encoding="utf-8")
    print("wrote", out / "README.md", "license files:", copied or "none")


if __name__ == "__main__":
    main(*sys.argv[1:4])
