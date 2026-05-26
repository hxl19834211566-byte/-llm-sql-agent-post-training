#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-/root/venv-route-a}"
PROJECT_DIR="${PROJECT_DIR:-/root/project}"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3-pip \
  python3-venv \
  git \
  git-lfs \
  build-essential

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${PROJECT_DIR}/requirements.txt"

python - <<'PY'
import torch
import transformers
import trl
import peft
import datasets
import accelerate
import bitsandbytes

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("mem_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("peft", peft.__version__)
print("datasets", datasets.__version__)
print("accelerate", accelerate.__version__)
print("bitsandbytes", bitsandbytes.__version__)
PY
