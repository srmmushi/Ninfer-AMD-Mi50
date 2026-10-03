# NInfer-HIP

> From-scratch C++/HIP inference engine for AMD gfx906 (MI50 / MI60 / Radeon VII / Radeon Pro VII).

NInfer-HIP is a C++ rewrite of [NInfer](https://github.com/Neroued/ninfer)'s
architecture, re-targeted from a single RTX 5090 (CUDA `sm_120a`) to gfx906 via
HIP / ROCm 6.3.3. It keeps the original's philosophy — a resident model, a
startup-fixed capacity, no distributed serving — and re-implements every layer
natively for CDNA1, with one or two MI50-class GPUs:

- **FP16 / INT4 storage, FP32 accumulation.** gfx906 has no BF16, FP8 or FP4
  units: weights are FP16 by default or groupwise INT4 with `--quant q4`, and
  all GEMMs accumulate in FP32.
- **Hand-written HIP kernels**: RMSNorm, per-head QK-Norm, RoPE (row-wise
  positions), SwiGLU, residual add (fused with RMSNorm), embedding gather, and
  fused GQA attention — flash-decode with online softmax, and an all-heads
  prefill kernel with LDS double-buffered K/V tiles (gfx906 has no TMA /
  cp.async, so the pipeline is software-managed).
- **Decode fast path**: at M=1 the projections run as custom wave64 GEMVs; the
  post-QKV stage (qk-norm + RoPE + KV append) collapses into one kernel.
- **rocBLAS** for the GEMM-bound prefill paths.
- **Multi-GPU**: `--parallel pp` (layer-split pipeline) or `--parallel tp`
  (Megatron tensor parallel), both ordered purely by HIP events.
- **Zero third-party dependencies**: JSON, BPE tokenizer (Qwen pretokenizer
  rules), ChatML rendering, safetensors reader and the HTTP server are all
  in-tree.

Supported models: `qwen2` / `qwen3` dense and `qwen3_moe` / `qwen2_moe`
(MoE with grouped expert execution), loaded directly from HuggingFace
checkpoints. `head_dim` must be 64 or 128.

## Requirements

- 64-bit Linux, one or two gfx906 GPUs (MI50 / MI60 / Radeon VII / Radeon Pro VII)
- ROCm 6.3.3 at `/opt/rocm` (or `ROCM_PATH` set)
- CMake >= 3.24 with HIP language support, Ninja, a C++17 host compiler

## Build

```bash
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DCMAKE_HIP_COMPILER=hipcc \
  -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_SHARED=ON
cmake --build build -j$(nproc)
```

Configure rejects any architecture other than `gfx906` (and every GPU listed in
`--gpus` is checked for gfx906 at load time).

| Artifact | Purpose |
|---|---|
| `build/ninfer` | One-shot or interactive CLI chat |
| `build/ninfer-serve` | OpenAI-compatible HTTP server with SSE streaming |
| `build/libninfer.so` | C API shared library (vLLM / external schedulers) |

## Model format (HuggingFace safetensors)

```
/path/to/Qwen3-14B/
├── config.json                 # architecture
├── tokenizer.json              # BPE vocab + merges
├── model.safetensors           # weights (or sharded model-*.safetensors +
│                               # model.safetensors.index.json)
```

Accepted checkpoint dtypes: `F16`, `BF16`, `F32`, `F64` (converted to FP16 at
load). Quantised checkpoints (GGUF / GPTQ / AWQ) are **not** read directly —
use the original FP16/BF16 weights and let `--quant q4` quantize at load time.

```bash
hf download Qwen/Qwen3-14B --local-dir /data/models/Qwen3-14B \
  --include "config.json" "tokenizer.json" "model*.safetensors"
```

## Usage

```bash
# one-shot
./build/ninfer /data/models/Qwen3-14B \
  --prompt "Explain prefill and decode, then conclude." \
  --max-context 32768 --max-new 1024 --temperature 0.7

# interactive multi-turn chat (omit --prompt)
./build/ninfer /data/models/Qwen3-14B --system "You are a helpful assistant."

# OpenAI-compatible server
./build/ninfer-serve /data/models/Qwen3-14B --port 8080 --max-context 32768
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"stream":true}'
```

### Options (both binaries)

| Option | Meaning |
|---|---|
| `--max-context N` | KV capacity per sequence (startup-fixed) |
| `--prefill-chunk N` | rows per prefill call (default 512) |
| `--max-new N` | new tokens per generation |
| `--temperature/--top-k/--top-p/--repeat-penalty/--seed` | sampling |
| `--gpus 0,1` | devices used by the engine |
| `--quant fp16\|q4` | weight format (q4 = groupwise INT4) |
| `--parallel pp\|tp` | layer-split pipeline (default) or tensor parallel |
| `--verify` | dual-GPU output vs single-GPU golden (CLI, needs `--gpus` with 2 ids) |
| `--raw`, `--no-thinking` | skip chat template / disable Qwen3 thinking marker |
| `--log-level debug\|info\|warn\|error` | diagnostics |

## Weight format: Q4 groupwise quantization

`--quant q4` quantizes weights at load time (symmetric, group of 32, FP32
accumulation, dequantized inside the GEMM/GEMV) — ~4x less weight traffic, the
dominant cost in decode:

- 27B: FP16 ≈ 54 GB → Q4 ≈ 14 GB (fits 2×16 GB Radeon VII)
- dual MI50 27B decode ceiling: ~19 tok/s (FP16) → ~68 tok/s theoretical (Q4);
  expect ~30–45 tok/s in practice

```bash
./build/ninfer /data/models/Qwen3-27B --quant q4 --gpus 0,1 \
  --prompt "..." --max-new 256
```

## Parallelism modes

| `--parallel` | Layout | Notes |
|---|---|---|
| `pp` (default) | layers split into contiguous shards; activation crosses once per forward over PCIe (P2P when available) | halves per-card VRAM; decode bandwidth **not** doubled |
| `tp` | Megatron tensor parallel: q/k/v and gate/up row-split, o_proj/down column-split, 2 all-reduces per layer | halves per-token weight traffic → up to ~2x decode; **dense models only** |

Both modes are ordered by HIP events only (no host device drains). MoE runs in
`pp` only. Each shard keeps its own KV cache, so VRAM scales down with GPU count.

## Performance notes (gfx906 reality check)

- Decode is weight-bandwidth bound: FP16 ≈ 19 tok/s ceiling for 27B on dual
  MI50 (1 TB/s each, sequential reads); Q4 removes ~75% of that traffic.
- Prefill is MFMA-bound; the fused attention kernel avoids score materialization
  but is not yet MFMA-tuned.
- MoE routing (softmax + top-k) still runs on the host — one sync per layer on
  the MoE decode path.
- Measure with `rocprof`: `rocprof --stats ./build/ninfer ...`, then compare
  kernel time vs. the roofline (weight bytes / HBM bandwidth).

## vLLM integration

Two complementary paths, both usable today.

### 1. Side-by-side (gateway)

vLLM (gfx906 branch) and NInfer-HIP run at the same time, one GPU each, behind a
single port:

```
client -> :9000 gateway -> :8000 vLLM    (GPU0: continuous batching, PagedAttention)
                        -> :8080 ninfer  (GPU1: low-latency HIP engine)
```

```bash
MODEL=/data/models/Qwen3-14B VLLM_DIR=$HOME/gfx906-vllm ./tools/run_dual.sh
curl http://127.0.0.1:9000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"ninfer/Qwen3-14B","messages":[{"role":"user","content":"hi"}]}'
```

### 2. Worker patch (real backend replacement)

`tools/vllm_plugin/` installs a backend plugin into a
[gfx906-vllm](https://github.com/ttdxq/gfx906-vllm) checkout: vLLM keeps
scheduling, sampling and the HTTP surface, while every scheduled step runs on
the HIP engine through the C API.

```bash
./tools/vllm_plugin/install.sh ~/gfx906-vllm
export NINFER_LIB=$PWD/build/libninfer.so
cd ~/gfx906-vllm
HIP_VISIBLE_DEVICES=0,1 \
NINFER_MODEL=/data/models/Qwen3-14B NINFER_GPUS=0,1 \
NINFER_QUANT=q4 NINFER_PARALLEL=pp NINFER_SEQUENCES=8 \
vllm serve Qwen3-14B --port 8000 --max-model-len 32768
```

Patched points: `Worker.determine_available_memory` / `Worker.initialize_cache`
(report the engine's KV footprint, no torch KV cache), `ModelRunner.load_model`
(no torch module), `ModelRunner.execute_model` (`ninfer_prefill` for new
requests, `ninfer_decode_batch` for the running batch). Unset `NINFER_MODEL` to
fall back to stock vLLM.

### C API (`libninfer.so`)

`src/api/ninfer_c_api.h` exposes an `extern "C"` surface — model load,
tokenizer, per-sequence prefill/decode, **batched decode step**, KV introspection,
sampling:

```c
ninfer_options_t opt = {4096, 512, 4, 1, gpus, "q4", "pp"};
ninfer_engine_t* eng; ninfer_load(dir, &opt, &eng);
ninfer_session_t* s = ninfer_session_create(eng);
ninfer_prefill(s, ids, n, logits, vocab);
long long tok = ninfer_sample(eng, logits, 0.7f, 50, 0.95f, 1.0f, seed);
ninfer_decode(s, tok, logits, vocab);
// continuous batching: one GPU step for N requests
ninfer_decode_batch(eng, sessions, tokens, n, all_logits, n * vocab);
```

```bash
export NINFER_LIB=$PWD/build/libninfer.so
PYTHONPATH=tools python3 tools/demo_ninfer_capi.py \
  --model /data/models/Qwen3-14B --gpus 1 --sequences 4 --max-new 64
```

`tools/ninfer_ctypes.py` wraps the API; `tools/gateway.py` and
`tools/run_dual.sh` implement the side-by-side deployment.

Known limits: no true PagedAttention (one contiguous slab per sequence → no
prefix sharing / block reuse yet), dense-only for `tp`, chunked prefill and
speculative decoding are not wired through the vLLM patch.

## Architecture

```
src/
├── common/    JSON parser, logging, file utilities (replaces nlohmann)
├── core/      HIP device management (DeviceGuard), RAII buffers, safetensors
├── frontend/  Unicode tables, byte-level BPE tokenizer, ChatML renderer
├── ops/       HIP kernels + rocBLAS wrappers
│              linear.hip  GEMM/GEMM-view wrappers (FP16 in, FP32 accumulate)
│              gemv.hip    M=1 GEMV + fused post-QKV (qk-norm+RoPE+KV append)
│              quant.hip   groupwise INT4 quantize / GEMV / GEMM
│              attention.hip fused GQA decode + all-heads prefill (LDS pipeline)
│              kernels.hip norm / RoPE / SwiGLU / residual / slice / add
│              moe.hip     token gather-scatter + host routing
├── models/    HF config parsing; Qwen2/Qwen3 dense + MoE, sharding + tp
├── runtime/   multi-sequence FP16 KV cache; engine (prefill → decode → sampler)
├── api/       extern "C" surface (libninfer.so)
└── apps/      ninfer CLI; ninfer-serve (self-contained HTTP server)
tools/         ctypes binding, C-API demo, gateway, run_dual.sh, vllm_plugin/
```

Layer correspondence with upstream NInfer: `artifact/` → direct safetensors
loading; `core/` → `core/` + `runtime/kv_cache`; `ops/` → `ops/`;
`models/qwen3_5/` → `models/` (standard attention subset, no GDN/MTP);
`frontend/` → `frontend/`; `runtime/engine/` → `runtime/engine`;
`apps/` → `apps/`; vendored `third_party/` (nlohmann, cpp-httplib, utf8proc,
jinja, spdlog) → in-tree minimal implementations.

### gfx906 kernel notes

- Decode attention: one block per head, wave64, streaming (online) softmax over
  the KV cache — no score materialization.
- Prefill attention: one launch for all heads; BQ=16 × BK=32 tiles, K/V staged
  through double-buffered LDS, next tile prefetched into registers while the
  current tile is consumed.
- Fusions: Q/K/V into one weight (1 GEMM), gate/up into one weight, residual add
  with the following RMSNorm, and the decode post-QKV stage.
- RoPE frequencies are a precomputed device table; positions are per-row, so
  batched decode rows with different lengths are supported.
- Only `head_dim` 64 and 128 are supported (template-instantiated kernels).
- VRAM: FP16 ≈ 2 bytes/parameter, Q4 ≈ 0.5 + scale overhead; 32 GiB MI50 fits
  ~14B in FP16 or ~50B in Q4 (minus KV/workspace).

## License

The upstream NInfer project is Apache-2.0; this rewrite follows the same
license spirit. Model weights remain subject to their own licenses
(Qwen is Apache-2.0).
