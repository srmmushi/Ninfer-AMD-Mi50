#!/usr/bin/env bash
# Run vLLM (gfx906 branch) and NInfer-HIP side by side, one GPU each, behind a
# single gateway port.
#
#   GPU0 -> vLLM    (throughput / continuous batching / PagedAttention)
#   GPU1 -> ninfer  (low-latency single/dual-GPU pipeline engine)
#   :9000 -> unified OpenAI-compatible entry point
#
# Requirements:
#   * gfx906-vllm installed in $VLLM_DIR (https://github.com/ttdxq/gfx906-vllm)
#   * NInfer-HIP built: cmake --build build -j$(nproc)
#   * Model in HuggingFace safetensors format (see README)
#
# Usage:
#   MODEL=/data/models/Qwen3-14B ./tools/run_dual.sh
set -euo pipefail

MODEL="${MODEL:?set MODEL=/path/to/hf/model}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VLLM_DIR="${VLLM_DIR:-$HOME/gfx906-vllm}"
VLLM_PORT="${VLLM_PORT:-8000}"
NINFER_PORT="${NINFER_PORT:-8080}"
GATEWAY_PORT="${NINFER_GATEWAY_PORT:-9000}"

VLLM_GPU="${VLLM_GPU:-0}"
NINFER_GPUS="${NINFER_GPUS:-1}"
NINFER_MAX_CTX="${NINFER_MAX_CTX:-32768}"
NINFER_PREFILL_CHUNK="${NINFER_PREFILL_CHUNK:-512}"

LOGDIR="$ROOT/.run"; mkdir -p "$LOGDIR"

echo "==> vLLM  (GPU$VLLM_GPU)  :$VLLM_PORT   log: $LOGDIR/vllm.log"
HIP_VISIBLE_DEVICES="$VLLM_GPU" \
  nohup "$VLLM_DIR/.venv/bin/vllm" serve "$MODEL" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size 1 \
    --max-model-len "$NINFER_MAX_CTX" \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    > "$LOGDIR/vllm.log" 2>&1 &
echo $! > "$LOGDIR/vllm.pid"

echo "==> ninfer (GPU$NINFER_GPUS) :$NINFER_PORT  log: $LOGDIR/ninfer.log"
nohup "$ROOT/build/ninfer-serve" "$MODEL" \
    --port "$NINFER_PORT" \
    --gpus "$NINFER_GPUS" \
    --max-context "$NINFER_MAX_CTX" \
    --prefill-chunk "$NINFER_PREFILL_CHUNK" \
    > "$LOGDIR/ninfer.log" 2>&1 &
echo $! > "$LOGDIR/ninfer.pid"

echo "==> gateway :$GATEWAY_PORT  log: $LOGDIR/gateway.log"
VLLM_PORT="$VLLM_PORT" NINFER_PORT="$NINFER_PORT" \
  NINFER_GATEWAY_PORT="$GATEWAY_PORT" \
  nohup python3 "$ROOT/tools/gateway.py" > "$LOGDIR/gateway.log" 2>&1 &
echo $! > "$LOGDIR/gateway.pid"

echo
echo "Client endpoint: http://127.0.0.1:$GATEWAY_PORT/v1/chat/completions"
echo "  model=\"ninfer/<name>\" -> NInfer-HIP ; model=\"vllm/<name>\" -> vLLM"
echo "Stop everything:  kill \$(cat $LOGDIR/*.pid)"
