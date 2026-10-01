#!/bin/bash
#SBATCH -J tojihoo_d02
#SBATCH -t 7-00:00:00
#SBATCH -o /home/tojihoo/logs/%A.out
#SBATCH --mail-type END,TIME_LIMIT_90,REQUEUE,INVALID_DEPEND
#SBATCH --mail-user jihu6033@gmail.com
#SBATCH -p V100
#SBATCH --gpus 1

# ------------------------------------------------------------
# 환경 설정
# ------------------------------------------------------------
export HTTP_PROXY=http://192.168.45.108:3128
export HTTPS_PROXY=http://192.168.45.108:3128
export http_proxy=http://192.168.45.108:3128
export https_proxy=http://192.168.45.108:3128

DOCKER_IMAGE_NAME="tojihoo/base"
DOCKER_CONTAINER_NAME="tojihoo_sapiens2"
DOCKERFILE_PATH="/mnt/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/jupyter/base/Dockerfile"
WORKSPACE_PATH="/mnt/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/jupyter/base/"
RANDOM_PORT=$(( (RANDOM % 101) + 8000 ))  # 8000~8100 사이 포트

# ------------------------------------------------------------
# Docker 이미지 빌드
# ------------------------------------------------------------
echo "[INFO] Building Docker image: ${DOCKER_IMAGE_NAME}"
docker build -t ${DOCKER_IMAGE_NAME} -f ${DOCKERFILE_PATH} ${WORKSPACE_PATH}
if [ $? -ne 0 ]; then
    echo "[❌ ERROR] Docker build failed."
    exit 1
fi

# ------------------------------------------------------------
# Docker 컨테이너 실행
# ------------------------------------------------------------
echo "[INFO] Running container: ${DOCKER_CONTAINER_NAME}"
docker run --rm --device=nvidia.com/gpu=all --shm-size 1TB \
    --name "${DOCKER_CONTAINER_NAME}" \
    -p ${RANDOM_PORT}:${RANDOM_PORT} \
    -v /mnt/nas203/ds_RehabilitationMedicineData/IDs/tojihoo:/workspace/ \
    -v /tmp:/data \
    -e HTTP_PROXY=${HTTP_PROXY} \
    -e HTTPS_PROXY=${HTTPS_PROXY} \
    -e http_proxy=${http_proxy} \
    -e https_proxy=${https_proxy} \
    ${DOCKER_IMAGE_NAME} bash -c "
        cd /workspace/SAM3Sapiens2 && \
        source scripts/conda.sh && \
        conda activate samsap && \
        python scripts/total_pipeline_with_csv_d02.py
    "
    
echo "[✅ DONE] total_pipeline_with_csv_d02.py finished."
