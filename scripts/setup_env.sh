#!/bin/bash
set -euo pipefail

ENV_NAME="ctrgcn_ava"

# 1. Validate NVIDIA driver (Ada Lovelace requires >= 525.xx)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
MAJOR="${DRIVER%%.*}"
if (( MAJOR < 525 )); then
    echo "[ERROR] Driver $DRIVER too old. RTX 4500 Ada needs >= 525.xx"; exit 1
fi

# 2. Create conda environment with Python 3.10
conda create -y -n "$ENV_NAME" python=3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 3. PyTorch 2.2 + CUDA 12.1 (Ada Lovelace / SM 8.9)
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 4. OpenMMLab v2 stack — MUST use cu121 wheel, not mmcv-full
pip install mmengine==0.10.3
pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html
pip install mmdet==3.3.0
pip install mmpose==1.3.1

# 5. Core science + data
pip install numpy==1.26.4 pandas==2.2.0 scipy==1.12.0 \
    scikit-learn==1.4.0 einops==0.7.0

# 6. Video / download tools
pip install yt-dlp==2024.3.10 ffmpeg-python==0.2.0 \
    boto3==1.34.0 gdown==5.1.0 requests==2.31.0 \
    opencv-python-headless==4.9.0.80

# 7. Config + validation
pip install pyyaml==6.0.1 jsonschema==4.21.0

# 8. Logging + experiment tracking
pip install wandb==0.16.3 tensorboard==2.16.0 tqdm==4.66.2 rich==13.7.0

# 9. Testing
pip install pytest==8.0.0 pytest-cov==4.1.0 pytest-timeout==2.2.0

# 10. Freeze exact versions for reproducibility
pip freeze > requirements.txt

# 11. Validate GPU capabilities
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
p = torch.cuda.get_device_properties(0)
print(f"GPU : {p.name}")
print(f"VRAM: {p.total_memory/1e9:.1f} GB")
print(f"SM  : {p.major}.{p.minor}")
assert p.major >= 8, "Need SM 8.x+ for BF16 on Ada Lovelace"
assert torch.cuda.is_bf16_supported(), "BF16 not supported"
# Enable globally for all matrix ops
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # safe — input shapes are fixed
print("All GPU checks PASSED")
EOF

# 12. Create directory tree + __init__.py files
python - <<'EOF'
import os
pkg_dirs = ["models", "models/ctrgcn", "annotation", "data_pipeline",
            "training", "evaluation", "utils", "scripts", "tests"]
data_dirs = ["data/raw/ava_kinetics/videos", "data/raw/ava/videos",
             "data/processed/ava_kinetics/skeletons",
             "data/processed/ava/skeletons",
             "data/annotations/ava_kinetics", "data/annotations/ava",
             "checkpoints/pretrained", "checkpoints/runs",
             "configs", "logs", "outputs"]
for d in pkg_dirs + data_dirs:
    os.makedirs(d, exist_ok=True)
for d in pkg_dirs:
    open(f"{d}/__init__.py", "a").close()
print("Directory tree created")
EOF
