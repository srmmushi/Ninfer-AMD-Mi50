"""ctypes binding for libninfer.so (NInfer-HIP C API).

Usage:
    from ninfer_ctypes import Ninfer
    eng = Ninfer("/path/to/Qwen3-14B", gpus=[0, 1], num_sequences=4)
    ids = eng.tokenize("Hello")
    with eng.session() as s:
        logits = s.prefill(ids)
        tok = eng.sample(logits, temperature=0.7)
        logits = s.decode(tok)

Layout mirrors src/api/ninfer_c_api.h.
"""

import ctypes
import os
from typing import List, Optional, Sequence


def _find_lib() -> str:
    env = os.environ.get("NINFER_LIB")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, "..", "build", "libninfer.so"),
        os.path.join(here, "..", "build-debug", "libninfer.so"),
        "/usr/local/lib/libninfer.so",
    ):
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError(
        "libninfer.so not found; set NINFER_LIB=/path/to/libninfer.so")


class _Options(ctypes.Structure):
    _fields_ = [
        ("max_context", ctypes.c_int),
        ("prefill_chunk", ctypes.c_int),
        ("num_sequences", ctypes.c_int),
        ("gpu_count", ctypes.c_int),
        ("gpus", ctypes.POINTER(ctypes.c_int)),
    ]


class Session:
    def __init__(self, engine: "Ninfer", handle):
        self._engine = engine
        self._handle = handle

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def slot(self) -> int:
        return self._engine.lib.ninfer_session_slot(self._handle)

    @property
    def length(self) -> int:
        return self._engine.lib.ninfer_session_length(self._handle)

    def reset(self) -> None:
        self._engine.lib.ninfer_session_reset(self._handle)

    def prefill(self, tokens: Sequence[int]) -> List[float]:
        arr = (ctypes.c_longlong * len(tokens))(*tokens)
        out = (ctypes.c_float * self._engine.vocab)()
        rc = self._engine.lib.ninfer_prefill(
            self._handle, arr, len(tokens), out, self._engine.vocab)
        if rc != 0:
            raise RuntimeError(f"ninfer_prefill failed: {rc}")
        return list(out)

    def decode(self, token: int) -> List[float]:
        out = (ctypes.c_float * self._engine.vocab)()
        rc = self._engine.lib.ninfer_decode(
            self._handle, ctypes.c_longlong(token), out, self._engine.vocab)
        if rc != 0:
            raise RuntimeError(f"ninfer_decode failed: {rc}")
        return list(out)

    def close(self) -> None:
        if self._handle is not None:
            self._engine.lib.ninfer_session_free(self._handle)
            self._handle = None


class Ninfer:
    def __init__(self,
                 model_dir: str,
                 max_context: int = 32768,
                 prefill_chunk: int = 512,
                 num_sequences: int = 1,
                 gpus: Optional[Sequence[int]] = None):
        self.lib = ctypes.CDLL(_find_lib(), mode=ctypes.RTLD_GLOBAL)
        # signatures
        self.lib.ninfer_load.restype = ctypes.c_int
        self.lib.ninfer_load.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(_Options),
            ctypes.POINTER(ctypes.c_void_p)
        ]
        for name in ("ninfer_vocab_size", "ninfer_max_context",
                     "ninfer_num_sequences", "ninfer_device_count"):
            getattr(self.lib, name).restype = ctypes.c_int
            getattr(self.lib, name).argtypes = [ctypes.c_void_p]
        for name in ("ninfer_kv_bytes_per_token", "ninfer_kv_total_bytes",
                     "ninfer_kv_seq_stride_elems"):
            getattr(self.lib, name).restype = ctypes.c_longlong
            getattr(self.lib, name).argtypes = [ctypes.c_void_p]
        self.lib.ninfer_tokenize.restype = ctypes.c_int
        self.lib.ninfer_tokenize.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_longlong), ctypes.c_int
        ]
        self.lib.ninfer_detokenize.restype = ctypes.c_int
        self.lib.ninfer_detokenize.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong), ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int
        ]
        self.lib.ninfer_session_create.restype = ctypes.c_void_p
        self.lib.ninfer_session_create.argtypes = [ctypes.c_void_p]
        self.lib.ninfer_session_free.argtypes = [ctypes.c_void_p]
        self.lib.ninfer_session_reset.argtypes = [ctypes.c_void_p]
        self.lib.ninfer_session_slot.argtypes = [ctypes.c_void_p]
        self.lib.ninfer_session_slot.restype = ctypes.c_int
        self.lib.ninfer_session_length.argtypes = [ctypes.c_void_p]
        self.lib.ninfer_session_length.restype = ctypes.c_int
        self.lib.ninfer_prefill.restype = ctypes.c_int
        self.lib.ninfer_prefill.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int
        ]
        self.lib.ninfer_decode.restype = ctypes.c_int
        self.lib.ninfer_decode.argtypes = [
            ctypes.c_void_p, ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int
        ]
        self.lib.ninfer_decode_batch.restype = ctypes.c_int
        self.lib.ninfer_decode_batch.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_longlong), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int
        ]
        self.lib.ninfer_sample.restype = ctypes.c_longlong
        self.lib.ninfer_sample.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_float,
            ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_ulonglong
        ]
        self.lib.ninfer_engine_free.argtypes = [ctypes.c_void_p]

        gpus = list(gpus or [0])
        gpu_arr = (ctypes.c_int * len(gpus))(*gpus)
        opt = _Options(max_context=max_context,
                       prefill_chunk=prefill_chunk,
                       num_sequences=num_sequences,
                       gpu_count=len(gpus),
                       gpus=gpu_arr)
        handle = ctypes.c_void_p()
        rc = self.lib.ninfer_load(model_dir.encode(), ctypes.byref(opt),
                                  ctypes.byref(handle))
        if rc != 0:
            raise RuntimeError(f"ninfer_load failed: {rc}")
        self._handle = handle
        self.vocab = self.lib.ninfer_vocab_size(self._handle)

    # ---- queries ----
    @property
    def devices(self) -> int:
        return self.lib.ninfer_device_count(self._handle)

    @property
    def kv_bytes_per_token(self) -> int:
        return self.lib.ninfer_kv_bytes_per_token(self._handle)

    @property
    def kv_total_bytes(self) -> int:
        return self.lib.ninfer_kv_total_bytes(self._handle)

    # ---- tokenizer ----
    def tokenize(self, text: str) -> List[int]:
        buf = (ctypes.c_longlong * (len(text) + 8))()
        n = self.lib.ninfer_tokenize(self._handle, text.encode(), buf,
                                     len(text) + 8)
        if n < 0:
            raise RuntimeError("ninfer_tokenize failed")
        return [int(buf[i]) for i in range(n)]

    def detokenize(self, ids: Sequence[int]) -> str:
        arr = (ctypes.c_longlong * len(ids))(*ids)
        out = ctypes.create_string_buffer(len(ids) * 8 + 16)
        n = self.lib.ninfer_detokenize(self._handle, arr, len(ids), out,
                                       len(out))
        return out.raw[:n].decode("utf-8", errors="replace")

    # ---- sessions ----
    def session(self) -> Session:
        h = self.lib.ninfer_session_create(self._handle)
        if not h:
            raise RuntimeError("ninfer_session_create failed")
        return Session(self, h)

    def decode_batch(self, sessions: Sequence[Session],
                     tokens: Sequence[int]) -> List[List[float]]:
        """One GPU step for N active requests (continuous batching)."""
        n = len(sessions)
        sarr = (ctypes.c_void_p * n)(*[s._handle for s in sessions])
        tarr = (ctypes.c_longlong * n)(*tokens)
        out = (ctypes.c_float * (n * self.vocab))()
        rc = self.lib.ninfer_decode_batch(self._handle, sarr, tarr, n, out,
                                          n * self.vocab)
        if rc != 0:
            raise RuntimeError(f"ninfer_decode_batch failed: {rc}")
        return [list(out[i * self.vocab:(i + 1) * self.vocab])
                for i in range(n)]

    # ---- sampling ----
    def sample(self,
               logits: Sequence[float],
               temperature: float = 0.7,
               top_k: int = 50,
               top_p: float = 0.95,
               repetition_penalty: float = 1.0,
               seed: int = 0) -> int:
        arr = (ctypes.c_float * len(logits))(*logits)
        return int(self.lib.ninfer_sample(self._handle, arr, temperature,
                                          top_k, top_p, repetition_penalty,
                                          seed))

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.lib.ninfer_engine_free(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
