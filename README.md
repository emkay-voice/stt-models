# stt-models

Speech recognition models for [Emkay](https://github.com/emkay-voice/emkay),
OpenVoiceOS in the browser: the tools that convert a model to ONNX for
[transformers.js](https://github.com/huggingface/transformers.js), test it
the way Emkay runs it, and publish it on Hugging Face, and the reasons
behind each model Emkay uses.

Emkay hears on the phone itself, with
[ovos-stt-plugin-whisper-browser](https://github.com/emkay-voice/ovos-stt-plugin-whisper-browser).
The plugin picks the model by language:

| language | model | why |
|---|---|---|
| English | `onnx-community/whisper-base.en` | OpenAI's English-only Whisper: better and faster than the multilingual one |
| Danish | `emkay-voice/roest-wav2vec2-315m-onnx` | [docs/danish.md](docs/danish.md) |
| others | `onnx-community/whisper-base` | multilingual; good enough for the big languages |

Whisper learned the big languages from tens of thousands of hours each, a
small language from a few hundred, and a model small enough for a phone does
not learn it well from that. Many countries have trained models of their own
on national data. [docs/adding-a-language.md](docs/adding-a-language.md)
shows how to bring one into Emkay.

## Contents

| | |
|---|---|
| `tools/convert.py` | Hugging Face model → ONNX in transformers.js' layout, plus 8-bit versions. Whisper and wav2vec2 (CTC). |
| `tools/compare.mjs` | Runs models on test recordings with transformers.js, prints transcripts, word error rate and time. |
| `tools/fleurs.py` | Test recordings from Google's FLEURS (real people reading, 102 languages). |
| `tools/piper_audio.py` | Test recordings of `tests/sentences_LANG.txt`, spoken by Piper. |
| `tools/model_card.py` | Model card for Hugging Face, with the original's license carried over. |
| `tools/smoke.mjs` | Checks that a converted model runs before it is published. |
| `.github/workflows/publish-model.yml` | Converts and publishes a model on Hugging Face (run by hand). |
| `docs/` | Why each model was chosen, how to add a language, results. |

## Quick start

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
npm install

.venv/bin/python tools/convert.py CoRal-project/roest-v3-wav2vec2-315m out/roest-wav2vec2-315m
.venv/bin/python tools/fleurs.py da_dk 50
node tools/compare.mjs da-fleurs ./roest-wav2vec2-315m onnx-community/whisper-small:q8:q8
```

A model is `ORG/NAME` on Hugging Face or `./NAME` under `out/`, optionally
with the precision of the encoder and decoder (`:fp32:q8`); a wav2vec2 model
has one file and uses the last one.

## Publishing

Actions → **Publish model** → Run workflow, with the source model and the
name under emkay-voice. It needs the repository secret `HF_TOKEN` (a Hugging
Face token with write access to the emkay-voice organisation), which an
owner adds by hand.

## License

The tools here are Apache-2.0. Each converted model keeps the license of
the model it was converted from; its model card says which.
