import sys
import time
from pathlib import Path

print("\n" + "=" * 70, flush=True)
print("[DEBUG PIPELINE] Initialization started", flush=True)
print("=" * 70, flush=True)

_t_global_start = time.perf_counter()

import gc
import os
import psutil

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["YOLO_OFFLINE"] = "true"
os.environ["ULTRALYTICS_ANALYTICS"] = "false"

import pandas as pd
import torch

def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    print(f"  [MEM DEBUG] [{tag}] Current RAM usage: {ram_mb:.2f} MB", flush=True)
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        vram_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"  [GPU DEBUG] [{tag}] VRAM Allocated: {vram_allocated:.2f} MB / Reserved: {vram_reserved:.2f} MB", flush=True)

print(f"Standard libraries and psutil loaded ({time.perf_counter() - _t_global_start:.2f}s)", flush=True)
print_memory_usage("After Initial Import")

PRJ_PATH = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = Path("/workspace/data_new")
FRAME_ROOT_DIR = Path("/workspace/data/01_FRAME")
CSV_PATH = BASE_DATA_DIR / "metadata_v3.0.csv"
ENV_PATH = "/workspace/.env"

SAM3_SOURCE_DIR = "/workspace/sam3"
SAPIENS_SOURCE_DIR = "/workspace/sapiens2"

WEIGHTS_DIR = PRJ_PATH / "data" / "checkpoints"
YOLO_CKPT_PATH = WEIGHTS_DIR / "yolov8n" / "yolov8n.pt"
SAM_CKPT_PATH = PRJ_PATH / "data/checkpoints" / "SAM3" / "sam3.pt"
BPE_PATH = PRJ_PATH / "data/checkpoints" / "SAM3" / "bpe_simple_vocab_16e6.txt.gz"
SAPIENS_CONFIG = (
    PRJ_PATH
    / "configs"
    / "sapiens"
    / "sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py"
)
SAPIENS_CHECKPOINT = PRJ_PATH / "data/checkpoints" / "sapiens2" / "sapiens2_0.4b_pose.safetensors"

SAPIENS_BATCH_SIZE = 100
VIDEO_FPS = 30.0
KPT_SCORE_THRESH = 0.35
START_FRAME_STRIDE = 60
START_FRAME_CONF = 0.6

def get_pipeline_paths(common_path: str):
    return {
        "frame_dir": FRAME_ROOT_DIR / common_path,
        "bbox_base": BASE_DATA_DIR / "02_SAM" / common_path,
        "sam_json": BASE_DATA_DIR / "02_SAM" / f"{common_path}.json",
        "sam_mask": BASE_DATA_DIR / "02_SAM" / f"{common_path}_masks.npz",
        "skt_npz": BASE_DATA_DIR / "03_KPT308" / f"{common_path}.npz",
        "out_mp4": BASE_DATA_DIR / "04_MP4" / f"{common_path}.mp4",
    }

for src_dir in [SAM3_SOURCE_DIR, SAPIENS_SOURCE_DIR, str(PRJ_PATH)]:
    if os.path.exists(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)

def clear_gpu_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def main():
    print(f"\n[MAIN] Entering main function (Elapsed: {time.perf_counter() - _t_global_start:.2f}s)", flush=True)

    if not CSV_PATH.exists():
        print(f"[Error] Metadata CSV not found: {CSV_PATH}")
        return

    print("Reading metadata CSV...", flush=True)
    _t_csv = time.perf_counter()
    df = pd.read_csv(CSV_PATH)
    print(f"CSV load complete ({time.perf_counter() - _t_csv:.2f}s)", flush=True)

    total_videos = len(df)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Total target videos: {total_videos} (Device: {device})\n", flush=True)
    print_memory_usage("After CSV Load")

    if "sam_done" not in df.columns:
        df["sam_done"] = False
    if "sapiens_done" not in df.columns:
        df["sapiens_done"] = False
    if "overlay_done" not in df.columns:
        df["overlay_done"] = False

    # -------------------------------------------------------------
    # Diagnostic Inspection of Metadata Statuses
    # -------------------------------------------------------------
    false_sam_rows = df[df["sam_done"] != True]
    false_sapiens_rows = df[df["sapiens_done"] != True]
    false_overlay_rows = df[df["overlay_done"] != True]

    print(f"[DIAGNOSTIC] Status summary across {total_videos} records:")
    print(f"  - sam_done pending count: {len(false_sam_rows)}")
    print(f"  - sapiens_done pending count: {len(false_sapiens_rows)}")
    print(f"  - overlay_done pending count: {len(false_overlay_rows)}")

    # Inspect why records are pending for Phase 1 (SAM)
    for idx, row in false_sam_rows.iterrows():
        common_path = str(row.get("common_path", "")).strip()
        if not common_path or common_path == "nan":
            continue
        paths = get_pipeline_paths(common_path)
        issues = []
        if not paths["frame_dir"].exists():
            issues.append(f"Frame directory missing: {paths['frame_dir']}")
        if paths["sam_json"].exists() and paths["sam_mask"].exists():
            issues.append("Output files exist on disk but CSV sam_done is False")
        if issues:
            print(f"  [DIAGNOSTIC WARNING] Phase 1 pending target [{common_path}] at index {idx}: {', '.join(issues)}")

    # # -------------------------------------------------------------
    # # Phase 1 Execution Block
    # # -------------------------------------------------------------
    # need_sam = not bool(df["sam_done"].all())
    # if need_sam:
    #     print("\n" + "=" * 70, flush=True)
    #     print("[PHASE 1] Initializing YOLO & SAM3 for BBox Tracking", flush=True)
    #     print("=" * 70, flush=True)

    #     print("Importing HuggingFace login module...", flush=True)
    #     from core.huggingface_login import login_to_huggingface
    #     try:
    #         login_to_huggingface(ENV_PATH)
    #     except Exception as e:
    #         print(f"[Warning] HF login bypassed: {e}")

    #     print("Importing detector and SAM3 core modules...", flush=True)
    #     _t_import = time.perf_counter()
    #     from core.detect_start_frame import LightweightPersonDetector, find_optimal_start_frame
    #     from core.tracking_sam3 import (
    #         build_sam3_image_model,
    #         build_sam3_video_model,
    #         detect_objects,
    #         run_bidirectional_tracking,
    #     )
    #     print(f"Core modules import complete ({time.perf_counter() - _t_import:.2f}s)", flush=True)
    #     print_memory_usage("Before Phase 1 Model Build")

    #     print(f"Building LightweightPersonDetector (Path: {YOLO_CKPT_PATH})...", flush=True)
    #     _t_yolo = time.perf_counter()
    #     start_frame_detector = LightweightPersonDetector(
    #         checkpoint_path=str(YOLO_CKPT_PATH) if YOLO_CKPT_PATH.exists() else None,
    #         device=device,
    #     )
    #     print(f"YOLO detector built ({time.perf_counter() - _t_yolo:.2f}s)", flush=True)

    #     print(f"Building SAM3 Image Model (Path: {SAM_CKPT_PATH})...", flush=True)
    #     _t_sam_img = time.perf_counter()
    #     sam_detector = build_sam3_image_model(
    #         checkpoint_path=str(SAM_CKPT_PATH), bpe_path=str(BPE_PATH)
    #     ).to(device)
    #     print(f"SAM3 Image Model built ({time.perf_counter() - _t_sam_img:.2f}s)", flush=True)

    #     print("Building SAM3 Video Model...", flush=True)
    #     _t_sam_vid = time.perf_counter()
    #     sam_tracker = build_sam3_video_model(
    #         checkpoint_path=str(SAM_CKPT_PATH),
    #         bpe_path=str(BPE_PATH),
    #         apply_temporal_disambiguation=True,
    #         device=device,
    #     )
    #     print(f"SAM3 Video Model built ({time.perf_counter() - _t_sam_vid:.2f}s)", flush=True)
    #     print_memory_usage("After Phase 1 Model Build Complete")

    #     for idx, row in df.iterrows():
    #         common_path = str(row.get("common_path", "")).strip()
    #         if not common_path or common_path == "nan":
    #             continue

    #         sam_done = bool(row.get("sam_done", False))
    #         if sam_done:
    #             continue

    #         paths = get_pipeline_paths(common_path)
    #         if not paths["frame_dir"].exists():
    #             print(f"[Phase 1 Error] Target skipped due to missing frame directory. Target: {common_path}, Path: {paths['frame_dir']}", flush=True)
    #             continue

    #         print(f"\n[Phase 1 - {idx+1}/{total_videos}] Target: {common_path}", flush=True)
    #         paths["bbox_base"].parent.mkdir(parents=True, exist_ok=True)

    #         try:
    #             print("  - Searching for start frame...", flush=True)
    #             _t_start = time.perf_counter()
    #             start_idx = find_optimal_start_frame(
    #                 frame_dir=paths["frame_dir"],
    #                 detector=start_frame_detector,
    #                 sample_stride=START_FRAME_STRIDE,
    #                 min_confidence=START_FRAME_CONF,
    #                 device=device,
    #             )
    #             print(f"  - Start frame found (Index: {start_idx}, {time.perf_counter() - _t_start:.2f}s)", flush=True)
    #         except Exception as e:
    #             print(f"  [Phase 1 Warning] Start frame search failed ({e}), fallback to index 0.")
    #             start_idx = 0

    #         try:
    #             print("  - Detecting objects...", flush=True)
    #             _t_det = time.perf_counter()
    #             detection_results = detect_objects(
    #                 frame_dir=str(paths["frame_dir"]),
    #                 text_prompt="person",
    #                 target_frame_idx=start_idx,
    #                 model=sam_detector,
    #             )
    #             print(f"  - Object detection complete ({time.perf_counter() - _t_det:.2f}s)", flush=True)

    #             if detection_results and len(detection_results.get("boxes", [])) > 0:
    #                 print("  - Running bidirectional tracking...", flush=True)
    #                 _t_trk = time.perf_counter()
    #                 run_bidirectional_tracking(
    #                     frame_dir=str(paths["frame_dir"]),
    #                     detection_results=detection_results,
    #                     output_dir=str(paths["bbox_base"]),
    #                     start_frame_idx=start_idx,
    #                     model=sam_tracker,
    #                 )
                    
    #                 if paths["sam_json"].exists() and paths["sam_mask"].exists():
    #                     df.at[idx, "sam_done"] = True
    #                     df.to_csv(CSV_PATH, index=False)
    #                     print(f"  [Phase 1 SUCCESS] Tracking complete ({time.perf_counter() - _t_trk:.2f}s)", flush=True)
    #                 else:
    #                     print(f"  [Phase 1 Error] Tracking executed but expected output files were not generated for target: {common_path}", flush=True)
    #             else:
    #                 print(f"  [Phase 1 Error] Object detection failed to find any persons for target: {common_path}", flush=True)

    #         except Exception as e:
    #             print(f"  [Phase 1 Error] Tracking execution failed for target {common_path}. Exception: {e}", flush=True)
    #         finally:
    #             if "detection_results" in locals():
    #                 del detection_results
    #             clear_gpu_cache()

    #     print("\n[Phase 1 Cleanup] Releasing YOLO and SAM3 memory...", flush=True)
    #     del start_frame_detector
    #     del sam_detector
    #     del sam_tracker
    #     clear_gpu_cache()
    #     print("Phase 1 memory cleanup complete\n", flush=True)
    #     print_memory_usage("After Phase 1 Cleanup")
    # else:
    #     print("[Phase 1 Skipped] All videos have completed SAM processing.", flush=True)

    # -------------------------------------------------------------
    # Phase 2 Execution Block
    # -------------------------------------------------------------
    need_sapiens = not bool(df["sapiens_done"].all())
    if need_sapiens:
        print("\n" + "=" * 70, flush=True)
        print("[PHASE 2] Loading Sapiens2 and Starting 308 Keypoint Pose Estimation", flush=True)
        print("=" * 70, flush=True)

        clear_gpu_cache()
        print_memory_usage("Before Phase 2 Entry Cache Clear")

        print("Importing Sapiens2 core modules...", flush=True)
        _t_sapiens_imp = time.perf_counter()
        from core.sapiens2_skeleton import SapiensPoseEstimator, extract_sapiens2_skeletons
        print(f"Sapiens modules import complete ({time.perf_counter() - _t_sapiens_imp:.2f}s)", flush=True)

        print(f"Building SapiensPoseEstimator and loading weights (Path: {SAPIENS_CHECKPOINT})...", flush=True)
        _t_sapiens_build = time.perf_counter()
        estimator = SapiensPoseEstimator(
            config_path=SAPIENS_CONFIG,
            checkpoint_path=SAPIENS_CHECKPOINT,
            device="cuda:0",
            use_fp16=True,
        )
        print(f"Sapiens model built ({time.perf_counter() - _t_sapiens_build:.2f}s)", flush=True)
        print_memory_usage("After Phase 2 Sapiens Model Build")

        for idx, row in df.iterrows():
            common_path = str(row.get("common_path", "")).strip()
            if not common_path or common_path == "nan":
                continue

            sapiens_done = bool(row.get("sapiens_done", False))
            if sapiens_done:
                continue

            paths = get_pipeline_paths(common_path)
            if not paths["sam_json"].exists():
                print(f"[Phase 2 Error] Prerequisite SAM JSON missing. Target skipped: {common_path}, Path: {paths['sam_json']}", flush=True)
                continue

            print(f"\n[Phase 2 - {idx+1}/{total_videos}] Target: {common_path}", flush=True)
            paths["skt_npz"].parent.mkdir(parents=True, exist_ok=True)

            try:
                print("  - Running Sapiens pose inference...", flush=True)
                _t_sk = time.perf_counter()
                extract_sapiens2_skeletons(
                    frame_dir=paths["frame_dir"],
                    bbox_json_path=paths["sam_json"],
                    output_json_path=paths["skt_npz"],
                    estimator=estimator,
                    batch_size=SAPIENS_BATCH_SIZE,
                    use_fp16=True,
                )
                
                if paths["skt_npz"].exists():
                    df.at[idx, "sapiens_done"] = True
                    df.to_csv(CSV_PATH, index=False)
                    print(f"  [Phase 2 SUCCESS] Pose estimation complete ({time.perf_counter() - _t_sk:.2f}s)", flush=True)
                else:
                    print(f"  [Phase 2 Error] Pose estimation executed but output NPZ was not generated for target: {common_path}", flush=True)

            except Exception as e:
                print(f"  [Phase 2 Error] Sapiens pose estimation failed for target {common_path}. Exception: {e}", flush=True)
            finally:
                clear_gpu_cache()

        print("\n[Phase 2 Cleanup] Releasing Sapiens2 model memory...", flush=True)
        del estimator
        clear_gpu_cache()
        print("Phase 2 memory cleanup complete\n", flush=True)
    else:
        print("[Phase 2 Skipped] All videos have completed Sapiens processing.", flush=True)

    # -------------------------------------------------------------
    # Phase 3 Execution Block
    # -------------------------------------------------------------
    need_overlay = not bool(df["overlay_done"].all())
    if need_overlay:
        print("\n" + "=" * 70, flush=True)
        print("[PHASE 3] Starting Visualization Overlay Video Synthesis", flush=True)
        print("=" * 70, flush=True)

        print("Importing overlay rendering module...", flush=True)
        from utils.visualize.sapiens_skeleton_overlay import overlay_skeleton_to_video
        print("Overlay rendering module import complete", flush=True)

        for idx, row in df.iterrows():
            common_path = str(row.get("common_path", "")).strip()
            if not common_path or common_path == "nan":
                continue

            overlay_done = bool(row.get("overlay_done", False))
            if overlay_done:
                continue

            paths = get_pipeline_paths(common_path)
            if not paths["skt_npz"].exists():
                print(f"[Phase 3 Error] Prerequisite skeleton NPZ missing. Target skipped: {common_path}, Path: {paths['skt_npz']}", flush=True)
                continue

            print(f"\n[Phase 3 - {idx+1}/{total_videos}] Target: {common_path}", flush=True)
            paths["out_mp4"].parent.mkdir(parents=True, exist_ok=True)

            try:
                print("  - Rendering overlay MP4...", flush=True)
                _t_rnd = time.perf_counter()
                overlay_skeleton_to_video(
                    frame_dir=paths["frame_dir"],
                    skeleton_npz_path=paths["skt_npz"],
                    output_video_path=paths["out_mp4"],
                    sam_npz_path=paths["sam_mask"] if paths["sam_mask"].exists() else None,
                    fps=VIDEO_FPS,
                    score_thresh=KPT_SCORE_THRESH,
                    draw_bones=True,
                    draw_dense_points=True,
                )
            except Exception as e:
                print(f"  [Phase 3 Error] Overlay video rendering failed for target {common_path}. Exception: {e}", flush=True)
                continue

            if paths["out_mp4"].exists():
                df.at[idx, "overlay_done"] = True
                df.to_csv(CSV_PATH, index=False)
                print(f"  [Phase 3 SUCCESS] MP4 rendering complete ({time.perf_counter() - _t_rnd:.2f}s)", flush=True)
            else:
                print(f"  [Phase 3 Error] Output MP4 was not created for target: {common_path}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"[ALL COMPLETED] Pipeline execution finished successfully. (Total time: {time.perf_counter() - _t_global_start:.2f}s)", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()