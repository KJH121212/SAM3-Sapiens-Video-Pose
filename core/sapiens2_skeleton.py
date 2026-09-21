import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from tqdm import tqdm

# =====================================================================
# Sapiens 모듈 경로 주입
# =====================================================================
SAPIENS_DIR = "/workspace/sapiens2"
if os.path.exists(SAPIENS_DIR):
    if SAPIENS_DIR not in sys.path:
        sys.path.insert(0, SAPIENS_DIR)
        print(f"[INFO] sys.path 최상단에 Sapiens 모듈 경로 주입 완료: {SAPIENS_DIR}")
else:
    print(f"[WARN] Sapiens 기본 디렉터리가 존재하지 않습니다: {SAPIENS_DIR}")

try:
    from sapiens.engine.config import Config
    from sapiens.pose.src.models.init_model import init_model
    from sapiens.pose.src.datasets.codecs.udp_heatmap import UDPHeatmap
except ImportError as e:
    print(f"[ERROR] Sapiens 엔진 임포트 실패: {e}")
    raise

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def box_xyxy_to_cs(
    bbox: List[float], aspect_ratio: float = 0.75, padding: float = 1.25
) -> Tuple[np.ndarray, np.ndarray]:
    """Sapiens 2 입력 비율(W:H = 768:1024, AR=0.75)에 맞춰 Center와 Scale 계산"""
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    center = np.array([x1 + w * 0.5, y1 + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    scale = np.array([w * padding, h * padding], dtype=np.float32)
    return center, scale


def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    output_size: Tuple[int, int],  # (W, H)
) -> np.ndarray:
    """Center/Scale 기반 2x3 아핀 변환 행렬 생성"""
    src_w, src_h = scale[0], scale[1]
    dst_w, dst_h = output_size[0], output_size[1]

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center
    src[1, :] = center + [0, -src_h * 0.5]
    src[2, :] = center + [src_w * 0.5, 0]

    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = [dst_w * 0.5, 0]
    dst[2, :] = [dst_w, dst_h * 0.5]

    return cv2.getAffineTransform(src, dst)


def transform_points_vectorized(pts: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """(N, 2) 크롭 좌표계를 2x3 역 아핀 변환 행렬로 원본 좌표계로 일괄 변환"""
    pts_homo = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    return pts_homo @ trans.T


class SapiensPoseEstimator:
    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda:0",
        use_fp16: bool = True,
    ):
        cfg_path = Path(config_path)
        ckpt_path = Path(checkpoint_path)

        if not cfg_path.is_file():
            raise FileNotFoundError(f"[ERROR] Config 파일이 없습니다: {cfg_path.resolve()}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"[ERROR] Checkpoint 가중치가 없습니다: {ckpt_path.resolve()}")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and (self.device.type == "cuda")

        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(self.device)
            vram_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            print(f"[INFO] GPU: {gpu_name} (총 VRAM: {vram_gb:.2f} GB) | FP16={self.use_fp16}")

        print(f"[INFO] Config 로드: {cfg_path.name}")
        self.cfg = Config.fromfile(str(cfg_path))
        self.input_size = (768, 1024)  # (Width, Height)

        print(f"[INFO] 모델 빌드: {ckpt_path.name}")
        self.model = init_model(str(cfg_path), checkpoint=None, device=str(self.device))

        if ckpt_path.suffix == ".safetensors":
            if not HAS_SAFETENSORS:
                raise ImportError("pip install safetensors 가 필요합니다.")
            state_dict = load_safetensors(str(ckpt_path), device=str(self.device))
            clean_dict = {
                (k[6:] if k.startswith("model.") else k): v
                for k, v in state_dict.items()
            }
            self.model.load_state_dict(clean_dict, strict=False)
        else:
            sd = torch.load(str(ckpt_path), map_location=self.device)
            if "state_dict" in sd:
                sd = sd["state_dict"]
            self.model.load_state_dict(sd, strict=False)

        self.model.eval()

        self.codec = UDPHeatmap(
            input_size=self.input_size,
            heatmap_size=(int(self.input_size[0] / 4), int(self.input_size[1] / 4)),
            sigma=6,
        )

        # ImageNet 정규화 상수 캐싱
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def preprocess_crop(
        self, img_rgb: np.ndarray, bbox: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """BBox 크롭 및 아핀 변환 수행 후 (C, H, W) 정규화 텐서용 ndarray와 역변환 행렬 반환"""
        center, scale = box_xyxy_to_cs(bbox)
        trans = get_affine_transform(center, scale, self.input_size)
        inv_trans = cv2.invertAffineTransform(trans)

        crop_img = cv2.warpAffine(
            img_rgb,
            trans,
            self.input_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114),
        )

        inp = crop_img.astype(np.float32)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return inp, inv_trans

    @torch.inference_mode()
    def predict_batch(
        self,
        batch_tensors: List[np.ndarray],
        inv_trans_list: List[np.ndarray],
    ) -> List[Tuple[List[List[float]], List[float]]]:
        """(B, C, H, W) 텐서 단위 일괄 순전파 후 308개 키포인트 역변환 리스트 반환"""
        if not batch_tensors:
            return []

        # 1. 배치 텐서 빌드: (B, 3, 1024, 768)
        inp_batch = torch.from_numpy(np.stack(batch_tensors, axis=0)).to(self.device)

        # 2. 순전파 (FP16 적용)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_fp16):
            feats = self.model.backbone(inp_batch)
            if hasattr(self.model, "neck") and self.model.neck is not None:
                feats = self.model.neck(feats)

            while isinstance(feats, (list, tuple)):
                feats = feats[-1]

            if hasattr(self.model, "head"):
                pred_heatmaps = self.model.head(feats)
            else:
                pred_heatmaps = self.model.decode_head(feats)

            if isinstance(pred_heatmaps, tuple):
                pred_heatmaps = pred_heatmaps[0]

        heatmaps_np = pred_heatmaps.float().detach().cpu().numpy()  # (B, 308, H/4, W/4)

        # 3. 인스턴스별 UDP 디코딩 및 좌표 복원
        batch_results = []
        for b_idx in range(len(batch_tensors)):
            hm = heatmaps_np[b_idx]
            pred_kpts_crop, scores = self.codec.decode(hm)

            if pred_kpts_crop.ndim == 3:
                pred_kpts_crop = pred_kpts_crop[0]
                scores = scores[0]

            orig_kpts = transform_points_vectorized(pred_kpts_crop, inv_trans_list[b_idx])
            
            kpts_rounded = np.round(orig_kpts, 3).tolist()
            scores_rounded = np.round(scores, 3).tolist()
            batch_results.append((kpts_rounded, scores_rounded))

        return batch_results


def flush_batch_queue(
    queue: List[Dict[str, Any]],
    estimator: SapiensPoseEstimator,
):
    """대기 중인 BBox 큐를 실행하고 원본 instance_dict에 결과를 직접 주입"""
    if not queue:
        return

    batch_tensors = [item["crop_tensor"] for item in queue]
    inv_trans_list = [item["inv_trans"] for item in queue]

    results = estimator.predict_batch(batch_tensors, inv_trans_list)

    for item, (kpts, scores) in zip(queue, results):
        target_dict = item["target_dict"]
        target_dict["keypoints"] = kpts
        target_dict["keypoint_scores"] = scores


def extract_sapiens2_skeletons(
    frame_dir: str | Path,
    bbox_json_path: str | Path,
    output_json_path: str | Path,
    config_path: Optional[str | Path] = None,
    checkpoint_path: Optional[str | Path] = None,
    estimator: Optional[SapiensPoseEstimator] = None,
    device: str = "cuda:0",
    batch_size: int = 4,
    use_fp16: bool = True,
) -> Path:
    start_time = time.time()
    frame_dir = Path(frame_dir)
    bbox_json_path = Path(bbox_json_path)
    output_json_path = Path(output_json_path)

    print("\n" + "=" * 70)
    print(f"[INFO] Sapiens 2 배치 스켈레톤 추출 시작 (Batch Size: {batch_size}, FP16: {use_fp16})")
    print("=" * 70)

    if not bbox_json_path.is_file():
        raise FileNotFoundError(f"[ERROR] BBox JSON 파일이 없습니다: {bbox_json_path.resolve()}")
    if not frame_dir.is_dir():
        raise NotADirectoryError(f"[ERROR] 프레임 디렉터리가 없습니다: {frame_dir.resolve()}")

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(bbox_json_path, "r", encoding="utf-8") as f:
        bbox_meta = json.load(f)

    raw_frames = bbox_meta.get("frames", [])
    if not raw_frames:
        raise ValueError("[ERROR] BBox JSON 내 'frames'가 비어 있습니다.")

    video_name = bbox_meta.get("video_name", frame_dir.name)
    bbox_by_frame_idx: Dict[int, List[Dict[str, Any]]] = {}
    bbox_by_filename: Dict[str, List[Dict[str, Any]]] = {}

    for f_entry in raw_frames:
        fidx = f_entry.get("frame_index")
        fname = f_entry.get("file_name")
        objs = f_entry.get("objects", [])
        if fidx is not None:
            bbox_by_frame_idx[int(fidx)] = objs
        if fname is not None:
            bbox_by_filename[fname] = objs

    all_frame_paths = sorted(
        [p for p in frame_dir.rglob("*.jpg")]
        + [p for p in frame_dir.rglob("*.png")]
        + [p for p in frame_dir.rglob("*.jpeg")]
    )
    try:
        all_frame_paths.sort(key=lambda p: int(p.stem))
    except Exception:
        all_frame_paths.sort()

    total_frames = len(all_frame_paths)
    if total_frames == 0:
        raise FileNotFoundError(f"[ERROR] 프레임 이미지가 없습니다: {frame_dir.resolve()}")

    if estimator is None:
        if config_path is None or checkpoint_path is None:
            raise ValueError("[ERROR] estimator가 없는 경우 config_path, checkpoint_path가 필수입니다.")
        estimator = SapiensPoseEstimator(
            config_path, checkpoint_path, device=device, use_fp16=use_fp16
        )

    skeleton_frames_output = []
    batch_queue: List[Dict[str, Any]] = []
    total_instances = 0

    for fidx, img_path in enumerate(tqdm(all_frame_paths, desc="Sapiens2 Batch Pose")):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]

        target_objs = bbox_by_filename.get(img_path.name, bbox_by_frame_idx.get(fidx, []))

        # 결측 프레임 Fallback 탐색
        if not target_objs:
            off = 1
            while True:
                left_idx = fidx - off
                right_idx = fidx + off
                if left_idx in bbox_by_frame_idx and len(bbox_by_frame_idx[left_idx]) > 0:
                    target_objs = bbox_by_frame_idx[left_idx]
                    break
                if right_idx in bbox_by_frame_idx and len(bbox_by_frame_idx[right_idx]) > 0:
                    target_objs = bbox_by_frame_idx[right_idx]
                    break
                if left_idx < 0 and right_idx >= total_frames:
                    break
                off += 1

        frame_instances = []
        for obj in target_objs:
            bbox = obj.get("bbox")
            obj_id = obj.get("id")

            if obj_id is None or not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1 or x2 < 0 or y2 < 0 or x1 >= img_w or y1 >= img_h:
                continue

            # BBox 전처리 및 아핀 변환 행렬 생성
            crop_tensor, inv_trans = estimator.preprocess_crop(img_rgb, bbox)

            instance_dict: Dict[str, Any] = {
                "id": int(obj_id),
                "bbox": [round(float(c), 3) for c in bbox],
                "keypoints": None,         # 배치 추론 후 주입
                "keypoint_scores": None,    # 배치 추론 후 주입
            }
            if "score" in obj:
                instance_dict["bbox_score"] = round(float(obj["score"]), 3)

            frame_instances.append(instance_dict)

            # 배치 버퍼에 추가
            batch_queue.append({
                "crop_tensor": crop_tensor,
                "inv_trans": inv_trans,
                "target_dict": instance_dict,
            })
            total_instances += 1

            # 배치 크기 충족 시 GPU 추론 실행
            if len(batch_queue) >= batch_size:
                flush_batch_queue(batch_queue, estimator)
                batch_queue.clear()

        skeleton_frames_output.append({
            "frame_index": fidx,
            "file_name": img_path.name,
            "instances": frame_instances,
        })

    # 루프 종료 후 잔여 큐 비우기
    if batch_queue:
        flush_batch_queue(batch_queue, estimator)
        batch_queue.clear()

    final_output = {
        "video_name": video_name,
        "total_frames": len(skeleton_frames_output),
        "num_keypoints": 308,
        "keypoint_format": "GOLIATH_308",
        "frames": skeleton_frames_output,
    }

    print(f"\n[INFO] 파일 저장 중: {output_json_path} ...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, separators=(",", ":"))

    file_size_mb = output_json_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("✨ [SUCCESS] 배치 처리 완료")
    print(f"  • 총 처리 인스턴스   : {total_instances}개")
    print(f"  • 소요 시간 / 처리속도: {elapsed:.1f}초 ({total_instances / max(elapsed, 0.001):.1f} instances/sec)")
    print(f"  • 파일 크기          : {file_size_mb:.2f} MB")
    print("=" * 70 + "\n")

    return output_json_path