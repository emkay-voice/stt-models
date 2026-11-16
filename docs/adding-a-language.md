# Adding a language

Emkay's speech recognition runs on the phone, with
[transformers.js](https://github.com/huggingface/transformers.js), and picks
the model by language. Every language starts with the multilingual
`onnx-community/whisper-base`. That is good enough for the big languages
(English, Spanish, German, French …), because Whisper heard tens of thousands
of hours of each while it was trained. For a small language it heard a few
hundred hours, and a model small enough for a phone does not learn a
language well from that. Danish was the first case: whisper-base got about
two words in three wrong (see [danish.md](danish.md)).

Many countries have trained a model of their own on national data:
broadcasts, parliament, read speech collected by a national library or a
university. Those are usually far better at their language than any
multilingual model of the same size. This repository turns such a model into
files the browser can run, measures it against what Emkay already has, and
publishes it on Hugging Face. The language then gets its own line in the
plugin's model table.

## 1. Find candidates

Look on [Hugging Face](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition)
for the language, and for its national projects. Some starting points:

| language | where to look |
|---|---|
| Danish | [CoRal](https://huggingface.co/CoRal-project) (Alexandra Institute and partners) |
| Norwegian | [NbAiLab](https://huggingface.co/NbAiLab) (National Library of Norway) |
| Swedish | [KBLab](https://huggingface.co/KBLab) (National Library of Sweden) |
| Finnish | [Finnish-NLP](https://huggingface.co/Finnish-NLP) |

A candidate must:

- be a **Whisper** model (`small` is the largest a phone takes) or a
  **wav2vec2 CTC** model. Those are the two kinds transformers.js and Emkay
  are tested with.
- be at most about **350 MB in 8 bit**: it is downloaded to the phone once
  and kept in the browser's cache.
- have a **license that allows** downloading it to people's phones and
  redistributing it converted. Read it, and carry its terms over to the
  model card.

## 2. Convert

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
.venv/bin/python tools/convert.py ORG/MODEL out/NAME
```

`convert.py` exports to ONNX and writes the layout transformers.js reads:

```
out/NAME/
  config.json, preprocessor_config.json, tokenizer.json, …
  onnx/encoder_model.onnx                     Whisper
  onnx/encoder_model_quantized.onnx
  onnx/decoder_model_merged.onnx
  onnx/decoder_model_merged_quantized.onnx
  onnx/model.onnx                             wav2vec2
  onnx/model_quantized.onnx
```

The `_quantized` files are 8 bit, about a quarter of the size. Emkay uses
those. For wav2vec2, `convert.py` also writes the `tokenizer.json` that
transformers.js needs (see [danish.md](danish.md) for a pitfall it avoids).

## 3. Measure

Two test sets, both made by scripts here:

- **FLEURS** (`tools/fleurs.py`): Google's read speech, real people in 102
  languages ([CC-BY 4.0](https://huggingface.co/datasets/google/fleurs)),
  long sentences from Wikipedia. None of the models here were trained on its
  dev split, so it is a fair judge.
- **Commands** (`tests/sentences_LANG.txt`, spoken by Piper with
  `tools/piper_audio.py`): short things people say to a voice assistant, in
  the voice Emkay answers with. Piper's pronunciation is not perfect, so this
  set is strict. It is for spotting a model that fails on commands, not for
  ranking.

```sh
npm install
.venv/bin/python tools/fleurs.py nb_no 50             # FLEURS language code
.venv/bin/python tools/piper_audio.py nb nb_NO-talesyntese-medium
node tools/compare.mjs nb-fleurs onnx-community/whisper-base:fp32:q8 onnx-community/whisper-small:q8:q8 ./NAME:q8:q8
node tools/compare.mjs nb ./NAME:q8:q8
```

`compare.mjs` prints every transcript, the word error rate (WER) and the time
per sentence. Write the results down in `docs/LANGUAGE.md`, as
[danish.md](danish.md) does, with the reasons for the choice, so we later
know why.

A model is worth a line in the table when it is clearly better than
whisper-base and not much slower. On a phone without WebGPU the time per
sentence matters as much as the WER: three seconds is already a long wait
for an answer.

## 4. Publish

The **Publish model** workflow (Actions → Publish model → Run workflow)
converts the model on GitHub, writes a model card with the original license
and a link to the results, and uploads it to the emkay-voice organisation on
Hugging Face as `NAME-onnx`. It needs the repository secret `HF_TOKEN`, a
Hugging Face token with write access to the organisation, which an owner adds
by hand.

## 5. Use it

Add the language to `MODELS` in
[ovos-stt-plugin-whisper-browser](https://github.com/emkay-voice/ovos-stt-plugin-whisper-browser)
and to the table in its README, then try it on a phone: download time, the
first answer, and a few commands in a normal voice.
