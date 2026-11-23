// How much memory a model takes in a browser: its WebAssembly memory in
// onnxruntime-web (the one transformers.js runs), opened the way Emkay opens
// it (one thread, no memory arena, no memory pattern). WebAssembly memory
// never shrinks, so what a model takes while opening stays taken for as
// long as the page lives; on a phone or tablet that decides whether Emkay
// starts at all.
//
//   node tools/memory.mjs out/NAME/onnx/FILE.onnx
//
// FILE.onnx_data next to it (convert.py's external data) is used when it is
// there. A CTC model (input_values) also does one run of 3 s of silence.

// the WebAssembly memory onnxruntime-web makes: its size is its high-water mark
let heap = null;
const Memory = WebAssembly.Memory;
WebAssembly.Memory = new Proxy(Memory, { construct(T, args) { return (heap = new T(...args)); } });
const ort = await import("onnxruntime-web");
const { existsSync, readFileSync } = await import("node:fs");

const MB = (n) => `${(n / 1048576).toFixed(0)} MB`.padStart(8);
const show = (what) => {
  console.log(`${what.padEnd(16)} wasm ${MB(heap?.buffer.byteLength ?? 0)}`);
};
const read = (p) => { const b = readFileSync(p); return new Uint8Array(b.buffer, b.byteOffset, b.byteLength); };

const file = process.argv[2];
if (!file) { console.error("usage: node tools/memory.mjs out/NAME/onnx/FILE.onnx"); process.exit(2); }
const data = `${file}_data`;
const options = { executionProviders: ["wasm"], enableCpuMemArena: false, enableMemPattern: false };
if (existsSync(data)) options.externalData = [{ path: data.split("/").pop(), data: read(data) }];
console.log(`${file}${options.externalData ? " + " + data.split("/").pop() : " (one file)"}`);

ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(read(file), options);
show("opened");
if (session.inputNames[0] === "input_values") {
  const n = 16000 * 3;
  await session.run({ input_values: new ort.Tensor("float32", new Float32Array(n), [1, n]) });
  show("after one run");
}
