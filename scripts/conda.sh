#!/usr/bin/env bash
set -e

ENV_NAME="samsap"
PYTHON_VER="3.12"

# 1. 아나콘다 약관 수락
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "=== [1/5] Conda 가상환경 생성 (${ENV_NAME}, Python ${PYTHON_VER}) ==="
conda create -y -n "${ENV_NAME}" python="${PYTHON_VER}"

# Conda 환경 활성화
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "=== [2/5] 기본 빌드 툴 및 컴파일러 설치 ==="
conda install -y -c conda-forge \
    ninja \
    cmake \
    git \
    ffmpeg

echo "=== [3/5] PyTorch 및 CUDA 설치 ==="
pip install --upgrade pip
pip install "setuptools<70.0.0" wheel Cython
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "=== [4/5] NumPy 1.26.4 고정 및 필수 라이브러리 일괄 설치 ==="
pip install "numpy==1.26.4"
pip install pycocotools --no-build-isolation

# 💡 SAM3, Sapiens2 내부적으로 요구하는 패키지 추가 (ftfy, prettytable 등)
pip install \
    "scipy<1.14.0" \
    "opencv-python<4.10.0" \
    "matplotlib<3.9.0" \
    "contourpy<1.3.0" \
    safetensors \
    pandas \
    timm \
    einops \
    huggingface_hub \
    hydra-core \
    omegaconf \
    tqdm \
    pillow \
    python-dotenv \
    accelerate \
    transformers \
    iopath \
    rich \
    portalocker \
    ftfy \
    prettytable \
    ultralytics

echo "=== [4.5/5] OpenMMLab (mmengine, mmcv-lite, mmpose) 설치 ==="
pip install mmengine
pip install mmcv-lite
pip install mmpose --no-deps

echo "=== [5/5] SAM3 및 Sapiens2 소스코드 다운로드 ==="
SAM3_PATH="/workspace/sam3"
if [ ! -d "${SAM3_PATH}" ]; then
    echo "⬇️ SAM3 소스코드 다운로드 중..."
    git clone https://github.com/facebookresearch/sam3.git "${SAM3_PATH}"
else
    echo "✅ SAM3 폴더 존재: ${SAM3_PATH}"
fi

SAPIENS_PATH="/workspace/sapiens2"
if [ ! -d "${SAPIENS_PATH}" ]; then
    echo "⬇️ Sapiens2 소스코드 다운로드 중..."
    git clone https://github.com/facebookresearch/sapiens2.git "${SAPIENS_PATH}"
else
    echo "✅ Sapiens2 폴더 존재: ${SAPIENS_PATH}"
fi

echo "=== [검증] 패키지 및 모듈 로드 확인 ==="
python -c "
import sys
import os

sam3_p = '${SAM3_PATH}'
sapiens_p = '${SAPIENS_PATH}'

if os.path.exists(sam3_p):
    sys.path.insert(0, sam3_p)
if os.path.exists(sapiens_p):
    sys.path.insert(0, sapiens_p)

import torch, numpy, safetensors, mmpose
import pycocotools.mask as mask_util
print(f'✅ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'✅ NumPy: {numpy.__version__}')
print(f'✅ mmpose: {mmpose.__version__}')

import traceback

try:
    import sam3
    print('✅ SAM3: sys.path 연동 성공!')
except ImportError as e:
    print('⚠️ SAM3: 모듈 임포트 실패')
    print('   -> 상세 에러:', e)
    traceback.print_exc()

try:
    import sapiens
    print('✅ Sapiens2: sys.path 연동 성공!')
except ImportError as e:
    print('⚠️ Sapiens2: 모듈 임포트 실패')
    print('   -> 상세 에러:', e)
    traceback.print_exc()
"

echo "=== 🎉 환경 구축 완료: conda activate ${ENV_NAME} ==="