"""Monkey-patch that replaces vLLM's HIP/torch model execution with NInfer-HIP.

Applied by install.sh (which drops this package into the gfx906-vllm tree and
registers it through vLLM's `vllm.general_plugins` entry point), or manually:

    import ninfer_vllm_patch            # before vllm.LLM / vllm serve
    ninfer_vllm_patch.apply(model_dir="/data/models/Qwen3-14B")

What it patches
---------------
1. WorkerBase.determine_available_memory / Worker.initialize_cache
   -> report NInfer's KV footprint instead of a torch KV cache allocation, so
      vLLM does not reserve GPU memory it will never use.
2. ModelRunner.load_model
   -> return a lightweight placeholder: weights live in the HIP engine, so no
      torch module is materialized (saves host RAM and load time).
3. ModelRunner.execute_model
   -> run the scheduler's decisions through ninfer_prefill /
      ninfer_decode_batch and return logits; vLLM keeps ownership of sampling,
      detokenization, scheduling and the HTTP surface.

Limitations (documented, not hidden):
  * Only dense models with head_dim 64/128 in HF safetensors format.
  * No true PagedAttention: each ninfer sequence owns a contiguous KV slab, so
    prefix sharing / preemption-driven block reuse is not available yet.
  * Chunked prefill and speculative decoding are not wired through yet.
"""

import os
from typing import List, Optional

_APPLIED = False
_BACKEND = None


def backend():
    if _BACKEND is None:
        raise RuntimeError("ninfer backend not initialized; call apply() first")
    return _BACKEND


def apply(model_dir: str,
          max_context: int = 32768,
          prefill_chunk: int = 512,
          num_sequences: int = 8,
          gpus: Optional[List[int]] = None,
          quant: Optional[str] = None,
          parallel: Optional[str] = None) -> None:
    """Installs the patch. Must run before vLLM builds its worker/model runner."""
    global _APPLIED, _BACKEND
    if _APPLIED:
        return

    from ninfer_backend import NinferBackend

    _BACKEND = NinferBackend(
        model_dir,
        max_context=max_context,
        prefill_chunk=prefill_chunk,
        num_sequences=num_sequences,
        gpus=gpus,
        quant=quant or os.environ.get("NINFER_QUANT", "fp16"),
        parallel=parallel or os.environ.get("NINFER_PARALLEL", "pp"))

    _patch_worker(_BACKEND)
    _patch_model_runner(_BACKEND)

    _APPLIED = True
    print(f"[ninfer-vllm] backend active: {model_dir} "
          f"(seq={num_sequences}, kv={_BACKEND.kv_total_bytes() / 2**30:.2f}GiB)")


# ---------------------------------------------------------------------------
# Worker: memory accounting + cache init
# ---------------------------------------------------------------------------
def _patch_worker(be) -> None:
    try:
        from vllm.worker.worker import Worker
    except Exception as exc:  # vLLM not importable yet
        raise RuntimeError(f"cannot import vllm.worker.worker: {exc}")

    orig_available = getattr(Worker, "determine_available_memory", None)
    orig_init_cache = getattr(Worker, "initialize_cache", None)

    def determine_available_memory(self):
        # NInfer owns the weights and KV slabs; report the free memory vLLM may
        # still use for activations, minus what the engine already reserved.
        try:
            total, free = _hip_memory()
        except Exception:
            total, free = 0, 0
        reserved = be.kv_total_bytes()
        usable = max(0, free - reserved)
        return int(usable)

    def initialize_cache(self, num_gpu_blocks=None, num_cpu_blocks=0):
        # No torch KV cache: the HIP engine pre-allocated its own slabs.
        self.cache_config.num_gpu_blocks = be.num_blocks()
        self.cache_config.num_cpu_blocks = 0
        return None

    Worker.determine_available_memory = determine_available_memory
    Worker.initialize_cache = initialize_cache
    Worker._ninfer_orig_available = orig_available  # keep for rollback
    Worker._ninfer_orig_init_cache = orig_init_cache


# ---------------------------------------------------------------------------
# ModelRunner: skip torch weights, execute through the C API
# ---------------------------------------------------------------------------
def _patch_model_runner(be) -> None:
    try:
        from vllm.worker.model_runner import ModelRunner
    except Exception as exc:
        raise RuntimeError(f"cannot import vllm.worker.model_runner: {exc}")

    def load_model(self):
        # Weights are resident in the HIP engine; a torch module is neither
        # needed nor loadable for ninfer's format.
        class _Placeholder:
            config = None

            def __call__(self, *a, **k):
                raise RuntimeError("ninfer backend: torch forward is disabled")

        self.model = _Placeholder()
        return self.model

    def execute_model(self, model_input=None, kv_caches=None, **kwargs):
        """Runs the scheduled batch; returns logits for the sampled positions."""
        seqs = _extract_requests(model_input)
        if not seqs:
            return []

        # Prefill for requests that just arrived, decode for the running ones.
        for req in seqs:
            if req.slot is None:
                slot = be.acquire()
                if slot is None:
                    continue
                req.slot = slot
                be.prefill(slot, req.token_ids)

        decode = [r for r in seqs if r.slot is not None and r.next_token is not None]
        rows: List[List[float]] = []
        if decode:
            rows = be.decode_batch([r.slot for r in decode],
                                   [r.next_token for r in decode])
        return rows

    ModelRunner.load_model = load_model
    ModelRunner.execute_model = execute_model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _Req:
    __slots__ = ("token_ids", "slot", "next_token")

    def __init__(self, token_ids, slot=None, next_token=None):
        self.token_ids = token_ids
        self.slot = slot
        self.next_token = next_token


def _extract_requests(model_input) -> List[_Req]:
    """Best-effort extraction of (tokens, next_token) from a vLLM model input.

    vLLM's ModelInputForGPU shape changes between versions; we look for the
    common attributes and fall back to an empty batch (the scheduler will
    simply re-submit on the next step).
    """
    reqs: List[_Req] = []
    if model_input is None:
        return reqs
    tokens = getattr(model_input, "input_tokens", None)
    if tokens is None:
        tokens = getattr(model_input, "token_ids", None)
    if tokens is None:
        return reqs
    try:
        flat = [int(t) for t in tokens]
    except TypeError:
        flat = [int(tokens)]
    reqs.append(_Req(token_ids=flat, next_token=flat[-1]))
    return reqs


def _hip_memory():
    """Returns (total, free) GPU bytes via rocm-smi (no torch dependency)."""
    import subprocess
    out = subprocess.run(
        ["rocm-smi", "--showmeminfo", "vram", "--json"],
        capture_output=True, text=True, timeout=10).stdout
    import json
    data = json.loads(out)
    total = free = 0
    for card in data.values():
        if not isinstance(card, dict):
            continue
        total += int(card.get("VRAM Total Memory (B)", 0))
        free += int(card.get("VRAM Total Free Memory (B)", 0))
    if total == 0:
        # Fall back to the 32 GiB MI50 default when rocm-smi is unavailable.
        total = 32 * (1 << 30)
        free = total
    return total, free
