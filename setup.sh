#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="ca22"
PYTHON_VERSION="3.10"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda was not found. Install Miniconda or Anaconda first." >&2
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create --yes --name "${ENV_NAME}" "python=${PYTHON_VERSION}" pip=24.0
fi

# Versions below are a mutually compatible CUDA 11.8/OpenMMLab stack.
conda run --name "${ENV_NAME}" python -m pip install --upgrade pip==24.0 openmim==0.3.9
conda run --name "${ENV_NAME}" python -m pip install \
  torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
conda run --name "${ENV_NAME}" mim install "mmcv==2.1.0"
conda run --name "${ENV_NAME}" python -m pip install \
  mmengine==0.10.4 mmsegmentation==1.2.2 mmdet==3.3.0
conda run --name "${ENV_NAME}" python -m pip install -r "${PROJECT_DIR}/requirements.txt"

conda run --name "${ENV_NAME}" python - <<'PY'
import mmcv
import mmdet
import mmengine
import mmseg
import torch

print("Environment ca22 is ready")
print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("mmcv:", mmcv.__version__, "mmengine:", mmengine.__version__)
print("mmseg:", mmseg.__version__, "mmdet:", mmdet.__version__)
PY

echo "Activate with: conda activate ${ENV_NAME}"
echo "Then verify with: pytest -q tests"
