# Danish: why Røst

**Choice:** Danish speech recognition in Emkay uses
[Røst](https://huggingface.co/CoRal-project/roest-v3-wav2vec2-315m)
(`CoRal-project/roest-v3-wav2vec2-315m`), converted to
`emkay-voice/roest-wav2vec2-315m-onnx`, at 8 bit (~320 MB). Every other
language except English keeps the multilingual Whisper base.

On devices with little memory (a tablet with 4 GB, an iPhone) Røst is
too big to open, and Danish uses a light model instead:
[Alvenir's wav2vec2 base](https://huggingface.co/Alvenir/wav2vec2-base-da-ft-nst),
converted to `emkay-voice/alvenir-wav2vec2-base-da-onnx` (~95 MB). See
[A light model](#a-light-model).

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

## Memory

On a Lenovo tablet Emkay in Danish closed and started over again and again,
while it was fine on a Samsung S24. Røst is the biggest thing Emkay opens,
and the way it was opened cost far more memory than its size.

onnxruntime-web, in the browser, copies a one-file model into WebAssembly
memory and parses it there, so while the model opens its weights are in
that memory two to three times. WebAssembly memory never shrinks: what was
taken while opening stays taken for as long as the page lives. With the
weights in a file of their own (`model_quantized.onnx_data`), each one is
read straight into its tensor.

`tools/memory.mjs`, Røst at 8 bit (303 MB), onnxruntime-web as
transformers.js runs it:

| | WebAssembly memory, opened | after one run (3 s) |
|---|---|---|
| one file | 635 MB | 635 MB |
| weights in their own file | 358 MB | 429 MB |

The transcripts are the same, word for word. `convert.py` now saves every
model this way and writes `"transformers.js_config":
{"use_external_data_format": …}` into `config.json`, so Emkay needs no
change: transformers.js reads that and fetches both files.

## A light model

Split, Røst still did not fit a Lenovo Tab M10 Plus (3rd gen, 4 GB): the
tab closed while Røst opened. Android gives one Chrome tab far less than
the device's memory, and Emkay is near that before its ears open; while
Røst opens, transformers.js holds its 303 MB in JavaScript as well as the
358 MB in WebAssembly. A Samsung S24 (8 GB and more) has no trouble.

So Danish has a second, light model for small devices:
[Alvenir/wav2vec2-base-da-ft-nst](https://huggingface.co/Alvenir/wav2vec2-base-da-ft-nst),
a wav2vec2 base (95 M parameters) fine-tuned on the Danish NST corpus by
Alvenir, Apache-2.0. Commands, as above
([results/da-piper-alvenir.txt](results/da-piper-alvenir.txt)):

| model | WER | time per sentence | 8 bit | WebAssembly memory, opened / after a run |
|---|---|---|---|---|
| Røst | 29.3% | 1.0 s | 303 MB | 358 / 429 MB |
| **Alvenir base** | **34.5%** | **0.4 s** | **95 MB** | **120 / 144 MB** |
| whisper-base | 62.1% | 1.5 s | 136 MB | 255 / 368 MB |

Close to Røst on commands, three times as fast, a third of the memory.
Its mistakes are spelling too ("hvær er klokken", "sin lyset i stuen").
NST is read speech from 2000, so dialects and casual speech will suffer
more than with Røst: Røst stays the model wherever it fits.

Which one a device gets is Emkay's choice (ovos-stt-plugin-whisper-browser's
`light` setting): the light one where the browser says the device has
less than 8 GB (`navigator.deviceMemory`, which Chrome rounds down: a
6 GB phone says 4) or says nothing (Safari, Firefox), and after a start
that ended with the page closed.

## Next

Measure Røst on a phone, with and without WebGPU: download time, the first
answer, and commands in a normal voice.
