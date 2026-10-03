# NInfer-HIP

> From-scratch C++/HIP inference engine. AMD Instinct MI50 (gfx906) only.

NInfer-HIP is a C++ rewrite of [NInfer](https://github.com/Neroued/ninfer)'s
architecture, re-targeted from a single RTX 5090 (CUDA `sm_120a`) to a single
AMD Instinct MI50 via HIP / ROCm 6.3.3. It keeps the original's philosophy —
one GPU, one resident model, a startup-fixed capacity, no distributed serving —
and re-implements every layer natively for gfx906:

- **FP16 storage, FP32 accumulation.** gfx906 has no BF16 units, so weights are
  converted to FP16 at load time and all GEMMs accumulate in FP32
  (`rocblas_gemm_ex`).
- **Hand-written HIP kernels** for RMSNorm, per-head QK-Norm, RoPE, SwiGLU,
  residual add, embedding gather, causal prefill softmax, and a fused GQA
  flash-decode attention kernel with streaming (online) softmax.
- **rocBLAS** for the GEMM-bound paths: fused QKV, o_proj, FFN, MoE experts,
  and the language-model head.
- **Zero third-party dependencies** — JSON, BPE tokenizer (with Qwen's
  pretokenizer rules), ChatML rendering, safetensors reading, and the HTTP
  server are all implemented in-tree.

Supported models: `qwen2` / `qwen3` dense and `qwen3_moe` (MoE expert routing
with grouped expert execution), loaded directly from HuggingFace checkpoints
(`config.json` + `model.safetensors` / sharded index).

## Requirements

- 64-bit Linux with an AMD Instinct MI50 (gfx906) GPU
- ROCm 6.3.3 installed at `/opt/rocm` (or `ROCM_PATH` set)
- CMake >= 3.24 with HIP language support, Ninja, a C++17 host compiler

## Build

```bash
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DCMAKE_HIP_COMPILER=hipcc
cmake --build build -j$(nproc)
```

The configure step rejects any architecture other than `gfx906`. The build
produces two binaries:

| Binary | Purpose |
|---|---|
| `build/ninfer` | One-shot or interactive CLI chat |
| `build/ninfer-serve` | OpenAI-compatible HTTP server with SSE streaming |

## Usage

One-shot generation (answer on stdout, diagnostics on stderr):

```bash
./build/ninfer /path/to/Qwen3-8B \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 --max-new 1024 --temperature 0.7
```

Interactive multi-turn chat (omit `--prompt`); `--raw` bypasses the chat
template, `--no-thinking` disables the Qwen3 thinking marker:

```bash
./build/ninfer /path/to/Qwen3-8B --system "You are a helpful assistant."
```

OpenAI-compatible serving (`/v1/chat/completions`, streaming and non-streaming;
`GET /health` for liveness):

```bash
./build/ninfer-serve /path/to/Qwen3-8B --port 8080 --max-context 32768

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"stream":true}'
```

Key options (both binaries): `--max-context` (KV capacity in tokens,
startup-fixed), `--prefill-chunk` (chunked prefill size), `--max-new`,
`--temperature`, `--top-k`, `--top-p`, `--repeat-penalty`, `--seed`,
`--gpus 0,1` (dual-GPU layer-split pipeline), `--verify` (dual vs. single-GPU
golden comparison), `--log-level debug|info|warn|error`.

### Dual-GPU (layer-split pipeline)

With `--gpus 0,1` the transformer layers are split into two contiguous shards
(one per MI50): shard 0 holds the embedding and the first half of the layers,
shard 1 holds the second half plus the final norm and LM head. Activations
cross the shard boundary once per forward through `hipMemcpyPeerAsync`
(PCIe P2P when the platform enables it, host-staged otherwise); each shard
keeps its own KV cache, so per-device VRAM is roughly halved — a 14B model
fits comfortably across 2×32 GiB. Correctness can be checked against a
single-GPU golden run:

```bash
./build/ninfer /path/to/Qwen3-14B --gpus 0,1 --verify \
  --prompt "..." --max-new 256
```

## vLLM integration (vLLM + NInfer-HIP running side by side)

Following the [gfx906-vllm](https://github.com/ttdxq/gfx906-vllm) approach, the
gfx906-supported vLLM branch and NInfer-HIP run **at the same time**, each on
its own GPU, behind one gateway port:

```
client -> :9000 gateway -> :8000 vLLM    (GPU0: continuous batching, PagedAttention)
                        -> :8080 ninfer  (GPU1: low-latency HIP engine)
```

```bash
# one GPU each, unified entry point on :9000
MODEL=/data/models/Qwen3-14B VLLM_DIR=$HOME/gfx906-vllm ./tools/run_dual.sh

# route by model prefix
curl http://127.0.0.1:9000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"ninfer/Qwen3-14B","messages":[{"role":"user","content":"hi"}]}'
curl http://127.0.0.1:9000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"vllm/Qwen3-14B","messages":[{"role":"user","content":"hi"}]}'
```

### C API (`libninfer.so`)

`src/api/ninfer_c_api.h` exposes an `extern "C"` surface (model load,
tokenizer, per-sequence prefill/decode, **batched decode step**, KV-cache
introspection, sampling) so any scheduler — including a vLLM worker — can drive
the HIP engine without touching C++:

```c
ninfer_options_t opt = {4096, 512, 4, 1, gpus};   // ctx, chunk, seq slots, gpus
ninfer_engine_t* eng; ninfer_load(dir, &opt, &eng);
ninfer_session_t* s = ninfer_session_create(eng);
float* logits = malloc(vocab * sizeof(float));
ninfer_prefill(s, ids, n, logits, vocab);
long long tok = ninfer_sample(eng, logits, 0.7f, 50, 0.95f, 1.0f, seed);
ninfer_decode(s, tok, logits, vocab);
// continuous batching: one GPU step for N requests
ninfer_decode_batch(eng, sessions, tokens, n, all_logits, n * vocab);
```

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 -DNINFER_BUILD_SHARED=ON
cmake --build build -j$(nproc)          # -> build/libninfer.so

export NINFER_LIB=$PWD/build/libninfer.so
PYTHONPATH=tools python3 tools/demo_ninfer_capi.py \
  --model /data/models/Qwen3-14B --gpus 1 --sequences 4 --max-new 64
```

`ninfer_decode_batch` maps N in-flight requests onto the reserved KV slabs and
executes them in a single forward pass — the continuous-batching primitive a
PagedAttention scheduler needs.

## Architecture

```
src/
├── common/    JSON parser, logging, file utilities (replaces nlohmann)
├── core/      HIP device management, RAII device buffers, safetensors reader
├── frontend/  Unicode tables, byte-level BPE tokenizer, ChatML renderer
├── ops/       HIP kernels + rocBLAS wrappers (linear, attention, MoE)
├── models/    HF config parsing; Qwen2/Qwen3 dense + MoE forward pass
├── runtime/   FP16 KV cache; engine (chunked prefill → decode → CPU sampling)
└── apps/      ninfer CLI; ninfer-serve (self-contained HTTP server)
```

Layer correspondence with upstream NInfer: `artifact/` → direct safetensors
loading; `core/` → `core/` + `runtime/kv_cache`; `ops/` → `ops/`;
`models/qwen3_5/` → `models/` (standard attention subset, no GDN/MTP);
`frontend/` → `frontend/`; `runtime/engine/` → `runtime/engine`;
`apps/` → `apps/`; vendored `third_party/` (nlohmann, cpp-httplib, utf8proc,
jinja, spdlog) → in-tree minimal implementations.

### gfx906 notes

- Decode attention is a fused per-head kernel (one block per head, wave64,
  online softmax over the KV cache, no score materialization).
- Prefill attention is chunked: per-head `QK^T` and `PV` GEMMs through
  rocBLAS with a causal-softmax pass in between; the score scratch is
  allocated once at startup (6 bytes/token-pair).
- MoE routing (softmax + top-k) runs on the host; expert execution is grouped
  by expert so each expert runs one batched set of GEMMs.
- Only `head_dim` 64 and 128 are supported (template-instantiated decode
  kernel).
- VRAM: FP16 weights are ~2 bytes/parameter; the 32 GiB MI50 fits models up to
  ~14B parameters.

## License

The upstream NInfer project is Apache-2.0; this rewrite follows the same
license spirit. Model weights remain subject to their own licenses
(Qwen is Apache-2.0).
