#!/usr/bin/env bash
# Install the NInfer-HIP backend into a gfx906-vllm checkout.
#
#   ./tools/vllm_plugin/install.sh ~/gfx906-vllm
#
# It copies the plugin into vllm/ninfer/, registers it through vLLM's
# `vllm.general_plugins` entry point (so `vllm serve` imports and activates it
# before building workers), and prints the run command.
set -euo pipefail

VLLM_DIR="${1:-${VLLM_DIR:-$HOME/gfx906-vllm}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$VLLM_DIR/vllm" ]; then
  echo "error: $VLLM_DIR does not look like a vLLM checkout" >&2
  exit 1
fi

echo "==> installing ninfer backend into $VLLM_DIR/vllm/ninfer"
mkdir -p "$VLLM_DIR/vllm/ninfer"
cp "$HERE/ninfer_vllm_patch.py" "$VLLM_DIR/vllm/ninfer/"
cp "$HERE/ninfer_backend.py"    "$VLLM_DIR/vllm/ninfer/"
cp "$HERE/../ninfer_ctypes.py"  "$VLLM_DIR/vllm/ninfer/"
touch "$VLLM_DIR/vllm/ninfer/__init__.py"

cat > "$VLLM_DIR/vllm/ninfer/plugin.py" <<'PY'
"""vLLM general plugin: activate the NInfer-HIP backend when configured."""
import os


def register():
    model = os.environ.get("NINFER_MODEL")
    if not model:
        return  # not requested -> vanilla vLLM
    from . import ninfer_vllm_patch
    ninfer_vllm_patch.apply(
        model_dir=model,
        max_context=int(os.environ.get("NINFER_MAX_CONTEXT", "32768")),
        prefill_chunk=int(os.environ.get("NINFER_PREFILL_CHUNK", "512")),
        num_sequences=int(os.environ.get("NINFER_SEQUENCES", "8")),
        gpus=[int(x) for x in os.environ.get("NINFER_GPUS", "0").split(",") if x],
        quant=os.environ.get("NINFER_QUANT", "fp16"),
        parallel=os.environ.get("NINFER_PARALLEL", "pp"))
PY

# Register the entry point in pyproject.toml (idempotent).
PYPROJECT="$VLLM_DIR/pyproject.toml"
if ! grep -q "ninfer-plugin" "$PYPROJECT"; then
  cp "$PYPROJECT" "$PYPROJECT.ninfer.bak"
  python3 - "$PYPROJECT" <<'PY'
import sys
path = sys.argv[1]
with open(path) as f:
    text = f.read()
marker = '[project.entry-points."vllm.general_plugins"]'
if marker not in text:
    text += '\n[project.entry-points."vllm.general_plugins"]\nninfer-plugin = "vllm.ninfer.plugin:register"\n'
else:
    text = text.replace(
        marker,
        marker + '\nninfer-plugin = "vllm.ninfer.plugin:register"')
with open(path, "w") as f:
    f.write(text)
PY
  echo "==> registered entry point in $PYPROJECT (backup: $PYPROJECT.ninfer.bak)"
else
  echo "==> entry point already registered"
fi

cat <<TXT

Installed. Run vLLM with the NInfer-HIP backend:

  export NINFER_LIB=$PWD/build/libninfer.so
  cd $VLLM_DIR
  HIP_VISIBLE_DEVICES=0,1 \\
  NINFER_MODEL=/data/models/Qwen3-14B \\
  NINFER_GPUS=0,1 NINFER_PARALLEL=pp NINFER_QUANT=q4 \\
  NINFER_SEQUENCES=8 NINFER_MAX_CONTEXT=32768 \\
  vllm serve Qwen3-14B --port 8000 --max-model-len 32768

Unset NINFER_MODEL to fall back to the stock vLLM path.
Uninstall: rm -rf $VLLM_DIR/vllm/ninfer && mv $PYPROJECT.ninfer.bak $PYPROJECT
TXT
