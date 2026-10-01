from pathlib import Path
import pandas as pd

PRJ_PATH = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/workspace/data_new")
CSV_PATH = DATA_DIR / "metadata_v3.0.csv"

WEIGHTS_DIR = PRJ_PATH / "data" / "checkpoints"

# 1) YOLOv8n (Start Frame 자동 탐색용)
YOLO_CKPT_PATH = WEIGHTS_DIR / "yolov8n" / "yolov8n.pt"

# 2) SAM3 (검출 & 양방향 트래킹)
SAM_CKPT_PATH = WEIGHTS_DIR / "SAM3" / "sam3.pt"
BPE_PATH = WEIGHTS_DIR / "SAM3" / "bpe_simple_vocab_16e6.txt.gz"

# 3) Sapiens2 (308 키포인트 포즈 추론)
SAPIENS_CONFIG = (
    PRJ_PATH
    / "configs"
    / "sapiens"
    / "sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py"
)
SAPIENS_CHECKPOINT = WEIGHTS_DIR / "sapiens2" / "sapiens2_0.4b_pose.safetensors"

df = pd.read_csv(CSV_PATH)
target=300
common_path = df.iloc[target]['common_path']

# 원본 영상 및 프레임 폴더
# vid_path = Path(f"/video_data/{common_path}")
frm_dir = Path(f"/workspace/data/01_FRAME/{common_path}")

# Phase 1: SAM3 출력 경로
# run_bidirectional_tracking에 bbox_out_base를 넘기면 내부에서 .json과 _masks.npz를 생성
bbox_out_base = DATA_DIR / "02_SAM" / common_path
sam_json_path = DATA_DIR / "02_SAM" / f"{common_path}.json"        # BBox 메타데이터 JSON
sam_mask_path = DATA_DIR / "02_SAM" / f"{common_path}_masks.npz"   # 3D 라벨 마스크 NPZ

# Phase 2: Sapiens2 입력 및 출력 경로
# 입력: Phase 1에서 생성된 sam_json_path
# 출력: 308 키포인트 및 BBox 시계열 압축 NPZ
skt_npz_path = DATA_DIR / "03_KPT308" / f"{common_path}.npz"

# Phase 3: 최종 오버레이 렌더링 출력 경로
# 입력: frm_dir, skt_npz_path, sam_mask_path
# 출력: 합성 비디오 MP4
out_mp4_path = DATA_DIR / "04_MP4" / f"{common_path}.mp4"
