// Compare speech recognition models on test recordings, the way Emkay runs them.
//
//   node tools/compare.mjs LANG MODEL[:ENCODER_DTYPE:DECODER_DTYPE] ...
//
// A model id starting with "./" is a local folder made by convert.py (under out/);
// anything else comes from the Hugging Face hub. Prints each transcript, the
// time it took and the word error rate (WER) against the text that was spoken.
import { pipeline, env } from "@huggingface/transformers";
import { existsSync, readFileSync, readdirSync } from "fs";

// run from the repository root: models in out/, audio in audio/
env.cacheDir = `${process.cwd()}/.cache/`;
env.localModelPath = `${process.cwd()}/out/`;
env.backends.onnx.wasm.numThreads = 1;

const [lang = "da", ...models] = process.argv.slice(2);  // "da", or a test set like "da-fleurs"
const base = lang.split("-")[0];
// the text of each recording: tests/sentences_LANG.txt (Piper), or the
// sentences.txt fleurs.py writes next to the recordings
const textFile = existsSync(`audio/${lang}/sentences.txt`) ? `audio/${lang}/sentences.txt` : `tests/sentences_${lang}.txt`;
const sentences = readFileSync(textFile, "utf8").trim().split("\n");
const files = readdirSync(`audio/${lang}`).filter((f) => f.endsWith(".wav")).sort();

// mono WAV (16-bit or float) -> Float32 at 16 kHz (linear resampling, as the browser does)
function load(path) {
  const wav = readFileSync(path), view = new DataView(wav.buffer, wav.byteOffset);
  let off = 12, rate = 16000, format = 1, data;
  while (off < wav.length) {
    const tag = wav.toString("ascii", off, off + 4), size = view.getUint32(off + 4, true);
    if (tag === "fmt ") { format = view.getUint16(off + 8, true); rate = view.getUint32(off + 12, true); }
    if (tag === "data") {
      const bytes = wav.buffer.slice(wav.byteOffset + off + 8, wav.byteOffset + off + 8 + size);
      // format 3 is 32-bit float (FLEURS), 1 is 16-bit PCM (Piper)
      data = format === 3 ? new Float32Array(bytes) : Float32Array.from(new Int16Array(bytes), (v) => v / 32768);
      break;
    }
    off += 8 + size + (size % 2);
  }
  const n = Math.floor(data.length * 16000 / rate), out = new Float32Array(n), step = rate / 16000;
  for (let i = 0; i < n; i++) { const x = i * step, j = Math.floor(x), f = x - j; const a = data[j] ?? 0, b = data[j + 1] ?? a; out[i] = a + (b - a) * f; }
  return out;
}

// digits count as the spoken number: OVOS understands "5" and "fem" alike
const NUMBERS = { da: "nul en to tre fire fem seks syv otte ni ti elleve tolv".split(" "),
                  en: "zero one two three four five six seven eight nine ten eleven twelve".split(" ") };
const words = (s) => s.toLowerCase().replace(/[^\p{L}\p{N} ]/gu, " ").split(/\s+/).filter(Boolean)
  .map((w) => (/^\d+$/.test(w) && NUMBERS[base]?.[+w]) || w);
function edits(a, b) {
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++)
    for (let j = 1; j <= b.length; j++)
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  return d[a.length][b.length];
}

const audio = files.map((f) => load(`audio/${lang}/${f}`));
const summary = [];
for (const spec of models) {
  const [id, enc = "fp32", dec = "q8"] = spec.split(":");
  const local = id.startsWith("./");
  env.allowLocalModels = local; env.allowRemoteModels = !local;
  const name = local ? id.slice(2) : id;
  let asr;
  const t0 = Date.now();
  try {
    asr = await pipeline("automatic-speech-recognition", name, {
      dtype: { encoder_model: enc, decoder_model_merged: dec, model: dec },
    });
  } catch (e) { console.log(spec, "LOAD ERROR", e.message.slice(0, 300)); continue; }
  const whisper = asr.model.config.model_type === "whisper";
  const english = /\.en$/.test(name);
  const opts = whisper && !english ? { language: base, task: "transcribe" } : {};
  console.log(`\n== ${spec} (loaded in ${Date.now() - t0} ms)`);
  let err = 0, total = 0, ms = 0;
  for (let i = 0; i < audio.length; i++) {
    const t = Date.now();
    let text;
    try { text = (await asr(audio[i], opts)).text.trim(); }
    catch (e) { text = ""; console.log("  (no text: " + e.message.slice(0, 80) + ")"); }  // Whisper heard nothing
    const dt = Date.now() - t; ms += dt;
    const ref = words(sentences[i]), e = edits(ref, words(text));
    err += e; total += ref.length;
    console.log(`${e ? "✗" : "✓"} ${dt} ms  ${JSON.stringify(text)}  (${sentences[i]})`);
  }
  const wer = (100 * err / total).toFixed(1), avg = Math.round(ms / audio.length);
  console.log(`WER ${wer}%  avg ${avg} ms`);
  summary.push(`${spec.padEnd(50)} WER ${wer.padStart(5)}%  ${String(avg).padStart(6)} ms/sentence`);
  await asr.dispose?.();
}
console.log("\n" + summary.join("\n"));
