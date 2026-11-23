"""Convert a speech recognition model on Hugging Face to ONNX for transformers.js.

    python convert.py MODEL_ID OUT_DIR

Whisper (encoder-decoder): onnx/encoder_model.onnx (full precision; the
encoder loses too much at 8 bit) and onnx/decoder_model_merged.onnx plus
onnx/decoder_model_merged_quantized.onnx (8 bit).
wav2vec2 (CTC): onnx/model.onnx and onnx/model_quantized.onnx.
The configs, tokenizer and feature extractor files go next to onnx/, as
in onnx-community's models.

Every .onnx is saved with its weights in a file of their own
(NAME.onnx_data), and config.json tells transformers.js so. A browser then
opens the model with far less memory: see externalize().

    python convert.py --externalize OUT_DIR    (only that, on a converted model)
"""
import json
import shutil
import sys
from pathlib import Path

import onnx as onnxlib


def quantize(onnx: Path, name: str):
    """8-bit dynamic quantization of onnx/NAME.onnx -> onnx/NAME_quantized.onnx"""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(onnx / f"{name}.onnx"), str(onnx / f"{name}_quantized.onnx"),
                     weight_type=QuantType.QInt8,
                     # the merged decoder keeps its weights in If-subgraphs
                     extra_options={"EnableSubgraph": True})


def ctc_tokenizer_json(out: Path):
    """Write tokenizer.json for a wav2vec2 CTC tokenizer.

    transformers.js only reads tokenizer.json, and Wav2Vec2CTCTokenizer has no
    fast version to save one. Same layout as transformers.js' own conversion:
    one token per character, "|" between words, CTC decoding."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(out)
    vocab = tok.get_vocab()
    delim = tok.word_delimiter_token or "|"
    data = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [
            {"id": i, "content": t, "single_word": False, "lstrip": True, "rstrip": True,
             "normalized": False,
             # The CTC blank must not be "special": transformers.js decodes with
             # skip_special_tokens, which would drop the blanks before the CTC
             # decoder merges repeats, so "p <pad> p" became "p" ("sliper").
             # The CTC decoder removes the blanks itself.
             "special": t != tok.pad_token}
            for t, i in sorted(vocab.items(), key=lambda x: x[1]) if t.startswith("<") and t.endswith(">")],
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": delim},
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": ""}, "behavior": "Isolated", "invert": False},
        "post_processor": None,
        "decoder": {"type": "CTC", "pad_token": tok.pad_token, "word_delimiter_token": delim, "cleanup": True},
        "model": {"vocab": vocab},
    }
    (out / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))


def externalize(out: Path):
    """Save every onnx/*.onnx with its weights in NAME.onnx_data, and tell
    transformers.js in config.json ("transformers.js_config").

    In a browser, onnxruntime-web copies a one-file model into WebAssembly
    memory and parses it there, so the weights are in it two to three times
    while the model opens; WebAssembly memory never shrinks, and the space
    stays taken. With the weights in a file of their own, each one is read
    straight into its tensor. Røst at 8 bit (303 MB): 635 MB of WebAssembly
    memory for one file, 358 MB like this (docs/danish.md)."""
    onnx = out / "onnx"
    names = {}
    for f in sorted(onnx.glob("*.onnx")):
        data = f.with_name(f.name + "_data")
        model = onnxlib.load(f)     # reads an existing NAME.onnx_data too
        data.unlink(missing_ok=True)
        onnxlib.save_model(model, f, save_as_external_data=True, all_tensors_to_one_file=True,
                           location=data.name, size_threshold=1024)
        names[f.name] = 1           # one data file each (transformers.js counts chunks)
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.setdefault("transformers.js_config", {})["use_external_data_format"] = names
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")


def main(model_id: str, out: str):
    from optimum.exporters.onnx import main_export
    from transformers import AutoConfig, AutoFeatureExtractor, AutoProcessor, AutoTokenizer

    out = Path(out)
    tmp = out / "_export"
    shutil.rmtree(out, ignore_errors=True)
    cfg = AutoConfig.from_pretrained(model_id)
    whisper = cfg.model_type == "whisper"
    task = "automatic-speech-recognition-with-past" if whisper else "automatic-speech-recognition"
    main_export(model_id, output=tmp, task=task, opset=17)
    onnx = out / "onnx"
    onnx.mkdir(parents=True)
    names = ["encoder_model", "decoder_model_merged"] if whisper else ["model"]
    for name in names:
        shutil.move(tmp / f"{name}.onnx", onnx / f"{name}.onnx")
        data = tmp / f"{name}.onnx_data"
        if data.exists():
            shutil.move(data, onnx / data.name)
    names_q = ["decoder_model_merged"] if whisper else ["model"]
    for name in names_q:
        quantize(onnx, name)
    # configs, tokenizer, feature extractor
    for f in tmp.iterdir():
        if f.suffix == ".json" or f.name.endswith(".txt"):
            shutil.move(f, out / f.name)
    if whisper:
        AutoProcessor.from_pretrained(model_id).save_pretrained(out)
    else:
        # Some CTC models (Røst) ship a KenLM processor; the browser decodes
        # greedily, so only the tokenizer and feature extractor are kept.
        AutoTokenizer.from_pretrained(model_id).save_pretrained(out)
        AutoFeatureExtractor.from_pretrained(model_id).save_pretrained(out)
        ctc_tokenizer_json(out)
    shutil.rmtree(tmp)
    externalize(out)
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"{f.relative_to(out)}  {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    if sys.argv[1] == "--ctc-tokenizer":  # only write tokenizer.json: --ctc-tokenizer OUT_DIR
        ctc_tokenizer_json(Path(sys.argv[2]))
    elif sys.argv[1] == "--quantize":  # redo only the quantization: --quantize OUT_DIR NAME
        quantize(Path(sys.argv[2]) / "onnx", sys.argv[3])
    elif sys.argv[1] == "--externalize":  # only move the weights out: --externalize OUT_DIR
        externalize(Path(sys.argv[2]))
    else:
        main(*sys.argv[1:3])
