import os
from pathlib import Path
import sys

# =====================================================================
# Sapiens 및 SAM3 로컬 소스코드 모듈 경로 주입
# =====================================================================
SAM3_DIR = "/workspace/sam3"
SAPIENS_DIR = "/workspace/sapiens2"

if os.path.exists(SAM3_DIR) and SAM3_DIR not in sys.path:
    sys.path.insert(0, SAM3_DIR)
if os.path.exists(SAPIENS_DIR) and SAPIENS_DIR not in sys.path:
    sys.path.insert(0, SAPIENS_DIR)

# 1. 경로 설정
PRJ_PATH = Path(__file__).resolve().parent.parent
VID_PATH = PRJ_PATH / "data" / "00_video" / "frontal__alternating_shoulder_press__incline1.MP4"
FRM_PATH = PRJ_PATH / "data" / "01_frame" / "frontal__alternating_shoulder_press__incline1"

# BBox 출력 폴더 및 생성될 JSON 경로 분리
BBOX_OUT_DIR = PRJ_PATH / "data" / "02_bboxes"
BBOX_JSON_PATH = BBOX_OUT_DIR / "frontal__alternating_shoulder_press__incline1.json"

SKT_PATH = PRJ_PATH / "data" / "03_skeleton" / "frontal__alternating_shoulder_press__incline1.json"
OUT_PATH = PRJ_PATH / "data" / "04_overlay" / "frontal__alternating_shoulder_press__incline1.mp4"

# Sapiens2 0.4B Pose Config 및 Checkpoint 매칭
SAPIENS_CONFIG = (
    PRJ_PATH
    / "configs"
    / "sapiens"
    / "sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py"
)

WEIGHTS_DIR = PRJ_PATH / "data" / "checkpoints"
SAPIENS_CHECKPOINT = WEIGHTS_DIR / "sapiens2" / "sapiens2_0.4b_pose.safetensors"

# SAM3 체크포인트 및 환경 변수
SAM_CKPT_PATH = WEIGHTS_DIR / "SAM3" / "sam3.pt"
BPE_PATH = WEIGHTS_DIR / "SAM3" / "bpe_simple_vocab_16e6.txt.gz"
ENV_PATH = "/workspace/.env"

# 2. sys.path 등록 및 모듈 임포트
sys.path.insert(0, str(PRJ_PATH))

from core.extract_frame import extract_frames
from core.huggingface_login import login_to_huggingface
from core.sapiens2_skeleton import extract_sapiens2_skeletons
from core.tracking_sam3 import detect_objects, run_bidirectional_tracking
from utils.visualize.sapiens_skeleton_overlay import overlay_skeleton_to_video


def main():
    print(f"Project Path:       {PRJ_PATH}")
    print(f"Weights Path:       {SAPIENS_CHECKPOINT}")
    print(f"Video Path:         {VID_PATH}")
    print(f"Frame Path:         {FRM_PATH}")
    print(f"BBox Out Dir:       {BBOX_OUT_DIR}")
    print(f"Skeleton Path:      {SKT_PATH}")
    print(f"Sapiens Config:     {SAPIENS_CONFIG}")
    print(f"Sapiens Checkpoint: {SAPIENS_CHECKPOINT}\n")

    # -------------------------------------------------------------
    # Step 0: 사전 검증 및 환경 초기화
    # -------------------------------------------------------------
    if not VID_PATH.exists():
        print(f"[오류] 원본 비디오 파일을 찾을 수 없습니다: {VID_PATH}")
        return
    if not SAPIENS_CHECKPOINT.exists():
        print(f"[오류] Sapiens 가중치를 찾을 수 없습니다: {SAPIENS_CHECKPOINT}")
        return
    if not SAM_CKPT_PATH.exists():
        print(f"[오류] SAM3 가중치를 찾을 수 없습니다: {SAM_CKPT_PATH}")
        return

    try:
        login_to_huggingface(ENV_PATH)
    except Exception as e:
        print(f"[경고] HuggingFace 로그인 실패 (로컬 가중치 사용 시 무시 가능): {e}")

    # -------------------------------------------------------------
    # Step 1: 비디오 프레임 추출
    # -------------------------------------------------------------
    print("\n>>> [Frame] 1단계: 비디오 프레임 추출 시작...")
    extract_frames(video_path=str(VID_PATH), frame_dir=str(FRM_PATH))

    # -------------------------------------------------------------
    # Step 2: SAM3 객체 검출 및 전 프레임 양방향 트래킹
    # -------------------------------------------------------------
    START_FRAME_IDX = 0  # 객체들이 잘 보이는 기준 프레임 번호

    print(f"\n>>> [SAM3] 2단계-1: {START_FRAME_IDX}번 프레임에서 객체 검출 시작...")
    detection_results = detect_objects(
        frame_dir=str(FRM_PATH),
        text_prompt="person",
        target_frame_idx=START_FRAME_IDX,
        checkpoint_path=str(SAM_CKPT_PATH),
        bpe_path=str(BPE_PATH),
    )

    if detection_results is None or len(detection_results.get("boxes", [])) == 0:
        print("[오류] 기준 프레임에서 인물을 검출하지 못했습니다. target_frame_idx를 조정하세요.")
        return

    print(f"[완료] 인물 검출 완료: {len(detection_results['boxes'])}명 식별")

    print("\n>>> [SAM3] 2단계-2: 양방향 트래킹 및 BBox 생성 시작...")
    bbox_json_path = run_bidirectional_tracking(
        frame_dir=str(FRM_PATH),
        detection_results=detection_results,
        output_dir=str(BBOX_OUT_DIR),
        start_frame_idx=START_FRAME_IDX,
        checkpoint_path=str(SAM_CKPT_PATH),
        bpe_path=str(BPE_PATH),
    )
    print(f"[완료] BBox JSON 저장 완료 -> {bbox_json_path}")

    # -------------------------------------------------------------
    # Step 3: Sapiens2 포즈 추론 (308개 키포인트 배치 추론)
    # -------------------------------------------------------------
    print("\n>>> [Sapiens2] 3단계: 스켈레톤 추출 시작 (0.4B Pose Model / Batch Size=4)...")
    skeleton_json_path = extract_sapiens2_skeletons(
        frame_dir=FRM_PATH,
        bbox_json_path=bbox_json_path,
        output_json_path=SKT_PATH,
        config_path=SAPIENS_CONFIG,
        checkpoint_path=SAPIENS_CHECKPOINT,
        device="cuda:0",
        batch_size=4,   # VRAM 여유에 따라 8로 상향 가능
        use_fp16=True,  # FP16 가속 활성화
    )
    print(f"[완료] 스켈레톤 생성 완료 -> {skeleton_json_path}")
    
    # -------------------------------------------------------------
    # Step 4: BBox, ID, 308 스켈레톤 비디오 오버레이 합성
    # -------------------------------------------------------------
    print("\n>>> [Overlay] 4단계: 비디오 오버레이 렌더링 시작...")
    overlay_video_path = overlay_skeleton_to_video(
        frame_dir=FRM_PATH,
        skeleton_json_path=SKT_PATH,
        output_video_path=OUT_PATH,
        fps=30.0,
        score_thresh=0.35,      # 신뢰도 0.35 이상인 관절만 표시
        draw_bones=True,        # 주요 신체 뼈대 라인 활성화
        draw_dense_points=True, # 얼굴/손가락 세부 308 키포인트 점 활성화
    )
    print(f"[완료] 최종 오버레이 영상 생성 -> {overlay_video_path}")

if __name__ == "__main__":
    main()