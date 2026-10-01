from pathlib import Path
import re
from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch

# ==============================================================================
# 기본 경로 설정
# ==============================================================================
# detect_start_frame.py(core/) 기준 상위 폴더(PRJ_PATH) 산출
PRJ_PATH = Path(__file__).resolve().parent.parent
DEFAULT_YOLO_DIR = PRJ_PATH / "data" / "checkpoints" / "yolov8n"
DEFAULT_YOLO_PATH = DEFAULT_YOLO_DIR / "yolov8n.pt"


# ==============================================================================
# 0. 유틸리티 함수 (전역 정의)
# ==============================================================================
def natural_sort_key(path_obj: Path) -> List:
    """파일명의 숫자를 자연 순서(0, 1, 2, ... 10)로 정렬하기 위한 키 함수"""
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", path_obj.name)
    ]


# ==============================================================================
# 1. 중복/포함 BBox 정제 필터 (Containment + NMS)
# ==============================================================================
def cleanup_duplicate_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.6,
    containment_thresh: float = 0.70,
) -> Tuple[np.ndarray, np.ndarray]:
    """한 사람에게 중복 생성되거나 상/하체로 쪼개진 박스를 제거."""
    if len(boxes) <= 1:
        return boxes, scores

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        cur_box = boxes[i]
        other_boxes = boxes[order[1:]]

        # 교집합(Intersection)
        xx1 = np.maximum(cur_box[0], other_boxes[:, 0])
        yy1 = np.maximum(cur_box[1], other_boxes[:, 1])
        xx2 = np.minimum(cur_box[2], other_boxes[:, 2])
        yy2 = np.minimum(cur_box[3], other_boxes[:, 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        area_cur = (cur_box[2] - cur_box[0]) * (cur_box[3] - cur_box[1])
        area_others = (other_boxes[:, 2] - other_boxes[:, 0]) * (
            other_boxes[:, 3] - other_boxes[:, 1]
        )

        # 1) IoU 계산 (동일 크기 중복)
        union = area_cur + area_others - inter
        iou = inter / np.maximum(union, 1.0)

        # 2) IoA 계산 (작은 박스가 큰 박스 안에 포함되었는지)
        min_area = np.minimum(area_cur, area_others)
        ioa = inter / np.maximum(min_area, 1.0)

        # IoU가 높거나 70% 이상 포함되면 중복으로 판정
        duplicate_mask = (iou >= iou_thresh) | (ioa >= containment_thresh)
        order = order[1:][~duplicate_mask]

    keep = np.array(keep, dtype=int)
    return boxes[keep], scores[keep]


# ==============================================================================
# 2. 경량 Detector 클래스 (YOLOv8n 기본, Torchvision Fallback)
# ==============================================================================
class LightweightPersonDetector:

    def __init__(
        self,
        device: str = "cuda:0",
        model_type: str = "yolo",
        checkpoint_path: Optional[str | Path] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_type = model_type.lower()

        if self.model_type == "yolo":
            try:
                from ultralytics import YOLO

                # 체크포인트 경로 설정 및 저장 디렉토리 생성
                ckpt = Path(checkpoint_path) if checkpoint_path else DEFAULT_YOLO_PATH
                ckpt.parent.mkdir(parents=True, exist_ok=True)

                print(f"[Detector] Ultralytics YOLOv8n 로드 중... (위치: {ckpt})")
                self.model = YOLO(str(ckpt))
            except ImportError:
                print("[WARN] ultralytics 미설치 -> Torchvision FasterRCNN 경량 모드로 대체")
                self.model_type = "torchvision"

        if self.model_type == "torchvision":
            import torchvision
            from torchvision.models.detection import (
                FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                fasterrcnn_mobilenet_v3_large_fpn,
            )

            print("[Detector] Torchvision MobileNetV3-FasterRCNN 로드 중...")
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).to(
                self.device
            )
            self.model.eval()

    def detect_person(
        self, img_bgr: np.ndarray, conf_thresh: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """단일 프레임에서 사람(person)의 BBox와 Confidence를 추출하여 반환."""
        if self.model_type == "yolo":
            results = self.model(
                img_bgr,
                classes=[0],  # class 0: person
                conf=conf_thresh,
                verbose=False,
                device=str(self.device),
            )[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            return boxes, scores

        elif self.model_type == "torchvision":
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            ).unsqueeze(0)
            tensor = tensor.to(self.device)

            with torch.inference_mode():
                outputs = self.model(tensor)[0]

            labels = outputs["labels"].cpu().numpy()
            scores = outputs["scores"].cpu().numpy()
            boxes = outputs["boxes"].cpu().numpy()

            person_idx = (labels == 1) & (scores >= conf_thresh)
            return boxes[person_idx], scores[person_idx]

        return np.empty((0, 4)), np.empty((0,))


# ==============================================================================
# 3. 최적 Start Frame 탐색 메인 함수
# ==============================================================================
def find_optimal_start_frame(
    frame_dir: str | Path,
    detector: Optional[LightweightPersonDetector] = None,
    checkpoint_path: Optional[str | Path] = None,
    sample_stride: int = 60,  # 30fps 기준 약 2초마다 1프레임 샘플링
    min_confidence: float = 0.45,
    device: str = "cuda:0",
) -> int:
    """
    전체 프레임을 스트라이드 스캔하여
    1) 가장 많은 인원이 포착되고
    2) 상/하체 중복 박스가 없으며
    3) 사람 간 겹침이 최소화된 최적의 start_frame_idx를 반환합니다.
    """
    frame_dir = Path(frame_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_frames = sorted(
        [p for p in frame_dir.rglob("*") if p.suffix.lower() in valid_exts],
        key=natural_sort_key
    )
    total_frames = len(all_frames)

    if total_frames == 0:
        raise FileNotFoundError(f"[ERROR] 프레임이 없습니다: {frame_dir}")

    if detector is None:
        detector = LightweightPersonDetector(device=device, checkpoint_path=checkpoint_path)

    candidate_indices = list(range(0, total_frames, sample_stride))
    if total_frames - 1 not in candidate_indices:
        candidate_indices.append(total_frames - 1)

    best_idx = candidate_indices[len(candidate_indices) // 4]  # 기본 Fallback (25% 지점)
    max_score = -float("inf")
    best_person_count = 0

    print(
        f"\n🔍 [Start Frame 자동 탐색] 총 {total_frames}개 프레임 중"
        f" {len(candidate_indices)}개 후보 평가 (Stride={sample_stride})..."
    )

    for idx in candidate_indices:
        img = cv2.imread(str(all_frames[idx]))
        if img is None:
            continue

        h_img, w_img = img.shape[:2]
        raw_boxes, raw_scores = detector.detect_person(
            img, conf_thresh=min_confidence
        )

        # 1. 중복/상하체 분할 박스 병합 필터링
        boxes, scores = cleanup_duplicate_boxes(
            raw_boxes, raw_scores, iou_thresh=0.4, containment_thresh=0.65
        )
        n_person = len(boxes)

        if n_person == 0:
            continue

        # 2. 객체 간 겹침(가림) 패널티 계산
        overlap_penalty = 0.0
        for i in range(n_person):
            for j in range(i + 1, n_person):
                b1, b2 = boxes[i], boxes[j]
                xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
                if inter > 0:
                    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    iou = inter / max(1.0, area1 + area2 - inter)
                    overlap_penalty += iou * 80.0  # 겹칠수록 큰 감점

        # 3. 화면 테두리에 잘린 인물 패널티
        boundary_penalty = 0.0
        for b in boxes:
            if b[0] < 5 or b[1] < 5 or b[2] > (w_img - 5) or b[3] > (h_img - 5):
                boundary_penalty += 5.0

        # 4. 종합 스코어 계산
        score = (
            (n_person * 1000.0)
            - overlap_penalty
            - boundary_penalty
            + float(np.sum(scores) * 10.0)
        )

        if score > max_score:
            max_score = score
            best_idx = idx
            best_person_count = n_person

    print(
        f"🎯 [선택 완료] Start Frame Index: {best_idx} (최다 인원:"
        f" {best_person_count}명, 평가 점수: {max_score:.1f})"
    )
    return best_idx