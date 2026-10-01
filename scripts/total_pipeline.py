import gc
import os
from pathlib import Path
import sys
import time

# GPU 메모리 단편화 방지 플래그 선언
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import torch

# =====================================================================
# ⚙️ [CONFIG] 모든 경로 및 실행 환경 설정 (여기서만 수정하세요)
# =====================================================================
# 1) 기본 루트 및 작업 경로
PRJ_PATH = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = Path("/workspace/data_new")
FRAME_ROOT_DIR = Path("/workspace/data/01_FRAME")
CSV_PATH = BASE_DATA_DIR / "metadata_v3.0.csv"
ENV_PATH = "/workspace/.env"

# 2) 처리 대상 선택 (단일 테스트용 타겟 인덱스)
TARGET_ROW_IDX = 300

# 3) 외부 소스코드 모듈 경로
SAM3_SOURCE_DIR = "/workspace/sam3"
SAPIENS_SOURCE_DIR = "/workspace/sapiens2"

# 4) 모델 체크포인트 및 설정 파일
WEIGHTS_DIR = PRJ_PATH / "data" / "checkpoints"
YOLO_CKPT_PATH = WEIGHTS_DIR / "yolov8n" / "yolov8n.pt"
SAM_CKPT_PATH = WEIGHTS_DIR / "SAM3" / "sam3.pt"
BPE_PATH = WEIGHTS_DIR / "SAM3" / "bpe_simple_vocab_16e6.txt.gz"
SAPIENS_CONFIG = (
    PRJ_PATH
    / "configs"
    / "sapiens"
    / "sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py"
)
SAPIENS_CHECKPOINT = WEIGHTS_DIR / "sapiens2" / "sapiens2_0.4b_pose.safetensors"

# 5) 타겟 기반 파이프라인 입출력 경로 빌더 함수
def get_pipeline_paths(common_path: str):
    """common_path를 입력받아 각 단계별 일관된 경로 객체들을 반환합니다."""
    return {
        "frame_dir": FRAME_ROOT_DIR / common_path,
        "bbox_base": BASE_DATA_DIR / "02_SAM" / common_path,
        "sam_json": BASE_DATA_DIR / "02_SAM" / f"{common_path}.json",
        "sam_mask": BASE_DATA_DIR / "02_SAM" / f"{common_path}_masks.npz",
        "skt_npz": BASE_DATA_DIR / "03_KPT308" / f"{common_path}.npz",
        "out_mp4": BASE_DATA_DIR / "04_MP4" / f"{common_path}.mp4",
    }


# =====================================================================
# 📦 [SYS PATH] 모듈 임포트 주입
# =====================================================================
for src_dir in [SAM3_SOURCE_DIR, SAPIENS_SOURCE_DIR, str(PRJ_PATH)]:
    if os.path.exists(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)

from core.extract_frame import extract_frames
from core.huggingface_login import login_to_huggingface
from core.sapiens2_skeleton import extract_sapiens2_skeletons
from core.tracking_sam3 import detect_objects, run_bidirectional_tracking
from core.detect_start_frame import LightweightPersonDetector, find_optimal_start_frame
from utils.visualize.sapiens_skeleton_overlay import overlay_skeleton_to_video
from utils.filter_keypoints import extract_keypoints_summary


# =====================================================================
# ⏱️ [PROFILER & UTILS] 성능 측정 및 메모리 관리
# =====================================================================
class Profiler:
    def __init__(self):
        self.records = {}
        self._cur_tag = None
        self._cur_start = 0

    def start(self, tag: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._cur_tag = tag
        self._cur_start = time.perf_counter()

    def stop(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._cur_start
        self.records[self._cur_tag] = elapsed
        print(f"  ⏱️ [{self._cur_tag}] 완료 소요: {elapsed:.2f}초")

    def print_report(self):
        total_time = sum(self.records.values())
        print("\n" + "=" * 65)
        print("📊 [병목 진단 프로파일링 리포트]")
        print("=" * 65)
        print(f"{'단계별 태스크':<30} | {'소요 시간':>9} | {'점유율':>7}")
        print("-" * 65)
        for tag, sec in self.records.items():
            pct = (sec / total_time * 100) if total_time > 0 else 0
            bar = "■" * int(pct / 4)
            print(f"{tag:<30} | {sec:>8.2f}s | {pct:>6.1f}%  {bar}")
        print("-" * 65)
        print(f"{'총 실행 시간 (Total)':<30} | {total_time:>8.2f}s | 100.0%")
        print("=" * 65 + "\n")


def clear_gpu_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# =====================================================================
# 🚀 [MAIN PIPELINE]
# =====================================================================
def main():
    if not CSV_PATH.exists():
        print(f"[오류] CSV 파일을 찾을 수 없습니다: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    common_path = str(df.iloc[TARGET_ROW_IDX]["common_path"]).strip()
    paths = get_pipeline_paths(common_path)

    print("=" * 70)
    print(f"🎯 Target Row [{TARGET_ROW_IDX}] : {common_path}")
    print(f"• Frame Dir     : {paths['frame_dir']}")
    print(f"• SAM Output    : {paths['sam_json']}")
    print(f"• Skeleton Path : {paths['skt_npz']}")
    print(f"• Output Video  : {paths['out_mp4']}")
    print("=" * 70 + "\n")

    # 출력 폴더 사전 생성
    paths["bbox_base"].parent.mkdir(parents=True, exist_ok=True)
    paths["skt_npz"].parent.mkdir(parents=True, exist_ok=True)
    paths["out_mp4"].parent.mkdir(parents=True, exist_ok=True)

    if not paths["frame_dir"].exists():
        print(f"[오류] 프레임 경로가 존재하지 않습니다: {paths['frame_dir']}")
        return

    try:
        login_to_huggingface(ENV_PATH)
    except Exception as e:
        print(f"[경고] HuggingFace 로그인 예외: {e}")

    prof = Profiler()

    # -------------------------------------------------------------
    # Step 2: SAM3 BBox 및 트래킹 (결과가 없을 때만 실행)
    # -------------------------------------------------------------
    if not paths["sam_json"].exists():
        print(">>> [Detector] 2단계-0: 최적의 Start Frame 탐색 중...")
        prof.start("Step 2-0: Detect Optimal Start Frame")
        detector = LightweightPersonDetector(
            weights_path=str(YOLO_CKPT_PATH) if YOLO_CKPT_PATH.exists() else None,
            device="cuda:0",
        )
        START_FRAME_IDX = find_optimal_start_frame(
            frame_dir=paths["frame_dir"],
            detector=detector,
            sample_stride=60,
            min_confidence=0.6,
            device="cuda:0",
        )
        prof.stop()
        print(f"[선정] 자동 선정된 시작 프레임: {START_FRAME_IDX}번")

        print(f"\n>>> [SAM3] 2단계-1: {START_FRAME_IDX}번 프레임에서 객체 검출...")
        prof.start("Step 2-1: SAM3 Model Load & Detect")
        detection_results = detect_objects(
            frame_dir=str(paths["frame_dir"]),
            text_prompt="person",
            target_frame_idx=START_FRAME_IDX,
            checkpoint_path=str(SAM_CKPT_PATH),
            bpe_path=str(BPE_PATH),
        )
        prof.stop()

        if not detection_results or len(detection_results.get("boxes", [])) == 0:
            print(f"[오류] 프레임({START_FRAME_IDX})에서 인물을 감지하지 못했습니다.")
            return

        print("\n>>> [SAM3] 2단계-2: 양방향 트래킹 실행...")
        prof.start("Step 2-2: SAM3 Video Tracking")
        run_bidirectional_tracking(
            frame_dir=str(paths["frame_dir"]),
            detection_results=detection_results,
            output_dir=str(paths["bbox_base"]),
            start_frame_idx=START_FRAME_IDX,
            checkpoint_path=str(SAM_CKPT_PATH),
            bpe_path=str(BPE_PATH),
        )
        prof.stop()
    else:
        print(f"[INFO] 기존 SAM3 결과 재사용: {paths['sam_json']}")

    # -------------------------------------------------------------
    # Step 3: Sapiens2 포즈 추론
    # -------------------------------------------------------------
    print("\n>>> [Sapiens2] 3단계: 스켈레톤 추출 시작 (0.4B Pose Model)...")
    clear_gpu_cache()  # 직전 모델 VRAM 반환

    prof.start("Step 3: Sapiens2 Inference (Batch=8)")
    extract_sapiens2_skeletons(
        frame_dir=paths["frame_dir"],
        bbox_json_path=paths["sam_json"],
        output_json_path=paths["skt_npz"],
        config_path=SAPIENS_CONFIG,
        checkpoint_path=SAPIENS_CHECKPOINT,
        device="cuda:0",
        batch_size=32,        # 프리징 방지 및 안정적인 VRAM 운용
        use_fp16=True,
    )
    prof.stop()
    print(f"[완료] 스켈레톤 저장: {paths['skt_npz']}")

    # -------------------------------------------------------------
    # Step 4: 오버레이 비디오 렌더링
    # -------------------------------------------------------------
    print("\n>>> [Overlay] 4단계: 비디오 오버레이 렌더링 시작...")
    prof.start("Step 4: Overlay Render & Encode")
    overlay_skeleton_to_video(
        frame_dir=paths["frame_dir"],
        skeleton_npz_path=paths["skt_npz"],
        output_video_path=paths["out_mp4"],
        sam_npz_path=paths["sam_mask"] if paths["sam_mask"].exists() else None,
        fps=30.0,
        score_thresh=0.35,
        draw_bones=True,
        draw_dense_points=True,
    )
    prof.stop()
    print(f"[완료] 렌더링 완료: {paths['out_mp4']}")

    prof.print_report()


if __name__ == "__main__":
    main()