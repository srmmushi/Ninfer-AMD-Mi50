"""NInfer-HIP backend for vLLM: drives libninfer.so through the C API.

This is the "real backend" path: instead of the gateway approach (two separate
servers), vLLM's scheduler stays in charge and every scheduled step is executed
by the HIP engine through ninfer_decode_batch / ninfer_prefill.

Design notes (vLLM integration points):
  * vLLM allocates KV cache per GPU block. NInfer-HIP uses one contiguous slab
    per sequence, so the plugin maps:
        num_gpu_blocks_equivalent = num_sequences
    and reports the corresponding bytes through
    ninfer_kv_bytes_per_token / ninfer_kv_total_bytes so vLLM's memory
    accounting stays consistent.
  * Prefill of a new request  -> ninfer_prefill (chunked internally)
  * Decode of a running batch -> ninfer_decode_batch (one GPU step, N rows)
  * Sampling stays in vLLM (it already owns the sampler); ninfer_sample is
    available through the C API for standalone use.
"""

import ctypes
import os
from typing import List, Optional, Sequence

try:
    from ninfer_ctypes import Ninfer, Session  # when PYTHONPATH includes tools/
except ImportError:  # pragma: no cover - installed into vllm tree
    from vllm.ninfer.ninfer_ctypes import Ninfer, Session  # type: ignore


class NinferBackend:
    """Owns the engine and one session per in-flight vLLM request slot."""

    def __init__(self,
                 model_dir: str,
                 max_context: int = 32768,
                 prefill_chunk: int = 512,
                 num_sequences: int = 8,
                 gpus: Optional[Sequence[int]] = None,
                 quant: str = os.environ.get("NINFER_QUANT", "fp16"),
                 parallel: str = os.environ.get("NINFER_PARALLEL", "pp")):
        self.engine = Ninfer(model_dir,
                             max_context=max_context,
                             prefill_chunk=prefill_chunk,
                             num_sequences=num_sequences,
                             gpus=list(gpus or [0]),
                             quant=quant,
                             parallel=parallel)
        self.sessions: List[Session] = [
            self.engine.session() for _ in range(num_sequences)
        ]
        self.free_slots = list(range(num_sequences))
        self.vocab = self.engine.vocab

    # ---- KV cache accounting (what vLLM asks the worker for) --------------
    def kv_bytes_per_token(self) -> int:
        return self.engine.kv_bytes_per_token

    def kv_total_bytes(self) -> int:
        return self.engine.kv_total_bytes

    def num_blocks(self) -> int:
        return len(self.sessions)

    # ---- slot management --------------------------------------------------
    def acquire(self) -> Optional[int]:
        if not self.free_slots:
            return None
        slot = self.free_slots.pop(0)
        self.sessions[slot].reset()
        return slot

    def release(self, slot: int) -> None:
        self.sessions[slot].reset()
        self.free_slots.append(slot)

    # ---- compute ----------------------------------------------------------
    def prefill(self, slot: int, tokens: Sequence[int]) -> List[float]:
        return self.sessions[slot].prefill(tokens)

    def decode_batch(self, slots: Sequence[int],
                     tokens: Sequence[int]) -> List[List[float]]:
        sessions = [self.sessions[s] for s in slots]
        return self.engine.decode_batch(sessions, tokens)

    def sample(self, logits: Sequence[float], temperature: float = 0.7,
               top_k: int = 50, top_p: float = 0.95,
               repetition_penalty: float = 1.0, seed: int = 0) -> int:
        return self.engine.sample(logits, temperature=temperature, top_k=top_k,
                                  top_p=top_p,
                                  repetition_penalty=repetition_penalty,
                                  seed=seed)

    def detokenize(self, ids: Sequence[int]) -> str:
        return self.engine.detokenize(ids)

    def tokenize(self, text: str) -> List[int]:
        return self.engine.tokenize(text)

    def close(self) -> None:
        for s in self.sessions:
            s.close()
        self.engine.close()
