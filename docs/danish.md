# Danish: why Røst

**Choice:** Danish speech recognition in Emkay uses
[Røst](https://huggingface.co/CoRal-project/roest-v3-wav2vec2-315m)
(`CoRal-project/roest-v3-wav2vec2-315m`), converted to
`emkay-voice/roest-wav2vec2-315m-onnx`, at 8 bit (~320 MB). Every other
language except English keeps the multilingual Whisper base.

Decided November 2026, from the measurements below.

## The problem

Emkay first used `onnx-community/whisper-base` for every language. In
English it was right; in Danish it got about two words in three wrong
("Hva er klodene?" for "Hvad er klokken?").

Whisper is trained on about 680,000 hours of audio. Roughly two thirds is
English; the big languages have tens of thousands of hours each, Danish a
few hundred. A model small enough to run on a phone (base, small) does not
learn a language well from so little. Danish is also hard in itself: soft
consonants and stød make words like "klokken" and "kloden" sound alike.

Denmark has a national speech dataset, [CoRal](https://huggingface.co/CoRal-project)
(Alexandra Institute and partners): Danish speech from many dialects, ages
and accents. Røst is a wav2vec2 model trained on it.

## Candidates

| model | kind | trained on | 8 bit |
|---|---|---|---|
| `onnx-community/whisper-base` | Whisper, multilingual | Whisper's data | ~80 MB |
| `onnx-community/whisper-small` | Whisper, multilingual | Whisper's data | ~250 MB |
| `WasuratS/whisper-small-da` | Whisper small, fine-tuned | Common Voice Danish | 287 MB |
| `CoRal-project/roest-v3-wav2vec2-315m` | wav2vec2 (CTC) | CoRal v3 | 317 MB |

Not tried: CoRal's Whisper large models (several GB, far too big for a
phone) and Whisper medium (too slow without a GPU).

## Results

Word error rate (WER, lower is better; digits count as the spoken number)
and average time per sentence, with transformers.js 4.3.2 in Node on an
AMD Ryzen 9 5950X CPU. A phone is slower; the order between the models is
what counts.

**Real speech**: [FLEURS](https://huggingface.co/datasets/google/fleurs)
Danish, the first 50 sentences of the dev split (read Wikipedia sentences,
real people; none of the models were trained on it).
All transcripts: [results/da-fleurs.txt](results/da-fleurs.txt).

| model | WER | time per sentence |
|---|---|---|
| **Røst, 8 bit** | **21.8%** | **3.5 s** |
| whisper-small-da, 8 bit | 37.4% | 3.6 s |
| whisper-small, 8 bit | 38.6% | 5.1 s |
| whisper-base (fp32 encoder, 8-bit decoder, as Emkay ran it) | 62.4% | 2.3 s |

**Commands**: 12 short things one says to an assistant ("Tænd lyset i
stuen."), spoken by Piper's Danish voice (`da_DK-talesyntese-medium`, the one
Emkay answers with). Piper mispronounces several words itself, so every
model is punished for that. The set is strict; it shows how the models
cope with commands, not their real accuracy.
All transcripts: [results/da-piper.txt](results/da-piper.txt).

| model | WER | time per sentence |
|---|---|---|
| **Røst, 8 bit** | **29.3%** | **0.5 s** |
| whisper-small-da, 8 bit | 37.9% | 2.4 s |
| whisper-small, 8 bit | 37.9% | 2.8 s |
| whisper-base | 62.1% | 1.5 s |

## Why Røst

- **Most accurate**: on real speech it gets about one word in five wrong,
  Whisper small two in five, Whisper base three in five.
- **Fastest**: on short commands about five times faster than Whisper small.
  On a phone without WebGPU that is the difference between an answer and a
  wait.
- **It writes what it hears**. Its mistakes are spelling ("kloppene",
  "væjret"), not invented sentences, as Whisper can do. The intent parsers
  often still get the meaning.
- **Fine-tuning Whisper on Common Voice did not help**: whisper-small-da was
  no better than the plain whisper-small, so there is no reason to keep a
  conversion of it.

What it costs:

- It knows only Danish. That is fine, since the plugin picks the model by
  language.
- No punctuation or capitals ("hvad er klokken"). OVOS normalises the text
  anyway.
- Its own language model (a KenLM 3-gram that fixes some spelling) cannot run
  in a browser, so Emkay decodes without it.

## License

Røst's license is adapted from OpenRAIL-M (Alvenir ApS; the full text is
copied as `LICENSE` with the converted model). It allows commercial use but
forbids using the model for speech synthesis and for biometric
identification. Emkay uses it only to turn speech into text on the user's
own device, which is allowed. The model card on Hugging Face carries the
license on.

## Pitfalls found on the way

- **Double letters disappeared** ("sliper" for "slipper", 28.4% instead of
  21.8%). transformers.js 4.3.2 decodes wav2vec2 with
  `skip_special_tokens`, which removes the CTC blank before repeated
  letters are merged. `convert.py` works around it by marking the blank as
  not special in `tokenizer.json`. It also affects transformers.js' own
  English wav2vec2 models ("FELOW" for "fellow"), so it belongs fixed
  upstream: [huggingface/transformers.js#1947](https://github.com/huggingface/transformers.js/issues/1947).
- **The merged Whisper decoder was not quantized**: `quantize_dynamic`
  leaves the weights in the decoder's `If` subgraphs alone unless asked
  (`EnableSubgraph`).
- **Whisper sometimes hears nothing**, and transformers.js then throws
  instead of returning empty text. Emkay treats it as an empty transcript.
- **Conversion tools**: optimum's ONNX exporter does not work with torch
  2.7 or newer; `requirements.txt` pins versions that work together.

## Next

Measure Røst on a phone, with and without WebGPU: download time, the first
answer, and commands in a normal voice.
