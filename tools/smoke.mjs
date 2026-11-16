// Check that a converted model loads and transcribes in transformers.js,
// with the 8-bit files Emkay uses, before it is published.
//
//   node tools/smoke.mjs ./NAME        (a folder under out/)
//
// tests/smoke_da.wav is Piper saying "Hvad er klokken?". The check is only
// that some text comes out; how good it is, is compare.mjs's job.
import { pipeline, env } from "@huggingface/transformers";
import { readFileSync } from "fs";

env.localModelPath = `${process.cwd()}/out/`;
env.allowRemoteModels = false;

const wav = readFileSync("tests/smoke_da.wav");
const view = new DataView(wav.buffer, wav.byteOffset);
let off = 12, rate = 16000, pcm;
while (off < wav.length) {
  const tag = wav.toString("ascii", off, off + 4), size = view.getUint32(off + 4, true);
  if (tag === "fmt ") rate = view.getUint32(off + 12, true);
  if (tag === "data") { pcm = new Int16Array(wav.buffer.slice(wav.byteOffset + off + 8, wav.byteOffset + off + 8 + size)); break; }
  off += 8 + size + (size % 2);
}
const n = Math.floor(pcm.length * 16000 / rate), audio = new Float32Array(n);
for (let i = 0; i < n; i++) audio[i] = pcm[Math.floor(i * rate / 16000)] / 32768;

const asr = await pipeline("automatic-speech-recognition", process.argv[2].replace(/^\.\//, ""), { dtype: "q8" });
const { text } = await asr(audio);
console.log(JSON.stringify(text));
if (!text.trim()) { console.error("no text"); process.exit(1); }
