#!/usr/bin/env python3
"""Demo: drive NInfer-HIP through libninfer.so, including batched decode.

  export NINFER_LIB=~/Ninfer-AMD-Mi50/build/libninfer.so
  python3 tools/demo_ninfer_capi.py --model /path/to/Qwen3-14B --gpus 0,1 \
      --sequences 4 --prompt "Explain prefill vs decode." --max-new 64
"""

import argparse
import sys
import time

from ninfer_ctypes import Ninfer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--sequences", type=int, default=1)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--prompt", default="Hello, who are you?")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chat", action="store_true",
                    help="wrap the prompt in the ChatML template")
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x != ""]
    t0 = time.time()
    eng = Ninfer(args.model, max_context=args.max_context,
                 num_sequences=args.sequences, gpus=gpus)
    print(f"[load] {time.time() - t0:.1f}s  vocab={eng.vocab} "
          f"devices={eng.devices} kv/token={eng.kv_bytes_per_token}B "
          f"kv_total={eng.kv_total_bytes / 2**30:.2f}GiB", file=sys.stderr)

    prompt = args.prompt
    if args.chat:
        prompt = ("<|im_start|>user\n" + args.prompt +
                  "<|im_end|>\n<|im_start|>assistant\n")

    # ---- multi-request continuous batching across the reserved slots ----
    n = args.sequences
    sessions, next_tokens, generated = [], [], []
    for i in range(n):
        ids = eng.tokenize(prompt if i == 0 else f"{prompt} (request {i})")
        s = eng.session()
        logits = s.prefill(ids)
        tok = eng.sample(logits, temperature=args.temperature,
                         seed=args.seed + i)
        sessions.append(s)
        next_tokens.append(tok)
        generated.append([tok])

    t1 = time.time()
    steps = 0
    for _ in range(args.max_new):
        # ONE forward pass drives all n requests (continuous batching).
        rows = eng.decode_batch(sessions, next_tokens)
        next_tokens = []
        for i, logits in enumerate(rows):
            tok = eng.sample(logits, temperature=args.temperature,
                             seed=args.seed + i + steps + 1)
            generated[i].append(tok)
            next_tokens.append(tok)
        steps += 1
        if all(eng.detokenize(g).find("<|im_end|>") >= 0 for g in generated):
            break
    dt = time.time() - t1
    per_tok = (steps * n) / dt if dt > 0 else 0.0
    print(f"[decode] {steps} steps x {n} seq in {dt:.2f}s -> "
          f"{per_tok:.1f} tok/s aggregate", file=sys.stderr)

    for i in range(n):
        text = eng.detokenize(generated[i]).replace("<|im_end|>", "")
        print(f"\n--- request {i} (slot {sessions[i].slot}) ---\n{text}")

    for s in sessions:
        s.close()
    eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
