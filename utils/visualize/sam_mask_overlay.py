import json
import os
from pathlib import Path
import sys
import cv2
import numpy as np
from tqdm import tqdm

# 초고속 디코딩용 pycocotools mask 임포트
try:
    from pycocotools import mask as mask_util

    HAS_COCOTOOLS = True
except ImportError:
    HAS_COCOTOOLS = False


def decode_rle(rle_dict: dict) -> np.ndarray:
    """RLE 포맷 딕셔너리를 2D uint8 이진 마스크로 복원"""
    if HAS_COCOTOOLS:
        rle = {
            "size": rle_dict["size"],
            "counts": (
                rle_dict["counts"].encode("utf-8")
                if isinstance(rle_dict["counts"], str)
                else rle_dict["counts"]
            ),
        }
        return mask_util.decode(rle)

    # pycocotools가 없을 경우 순수 파이썬 폴백
    h, w = rle_dict["size"]
    counts = rle_dict["counts"]
    pixels = np.zeros(h * w, dtype=np.uint8)
    cur = 0
    val = 0
    for count in counts:
        pixels[cur : cur + count] = val
        cur += count
        val = 1 - val
    return pixels.reshape((h, w))


def get_color_by_id(obj_id: int):
    """객체 ID마다 일관된 고유 BGR 색상 반환"""
    np.random.seed(int(obj_id) * 31)
    color = np.random.randint(60, 255, size=3).tolist()
    return tuple(color)


def overlay_masks_and_bboxes_to_video(
    frame_dir: str | Path,
    json_path: str | Path,
    output_video_path: str | Path,
    fps: int = 30,
    mask_alpha: float = 0.45,
    bbox_thickness: int = 3,
):
    """프레임 이미지와 JSON을 읽어 Mask, 두꺼운 BBox, ID, Score를 합성한 비디오를 생성합니다."""
    frame_dir = Path(frame_dir)
    json_path = Path(json_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {json_path}")

    print(f"[Mask Overlay] JSON 로드 중: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames_meta = data.get("frames", [])
    if not frames_meta:
        print("[Mask Overlay] JSON에 프레임 데이터가 없습니다.")
        return

    frames_dict = {f["frame_index"]: f for f in frames_meta}

    # 하위 디렉토리(00/, 01/ 등)의 모든 이미지 경로 수집 및 정렬
    all_frame_paths = sorted(
        [p for p in frame_dir.rglob("*.jpg")]
        + [p for p in frame_dir.rglob("*.png")]
    )
    try:
        all_frame_paths.sort(key=lambda p: int(p.stem))
    except Exception:
        all_frame_paths.sort()

    if not all_frame_paths:
        raise FileNotFoundError(
            f"프레임 디렉터리에 이미지가 없습니다: {frame_dir}"
        )

    sample_img = cv2.imread(str(all_frame_paths[0]))
    if sample_img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {all_frame_paths[0]}")
    h, w, _ = sample_img.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

    print(
        f"[Mask Overlay] 렌더링 시작 (총 {len(all_frame_paths)} 프레임, {w}x{h} @ {fps}fps)"
    )

    for idx, img_path in enumerate(
        tqdm(all_frame_paths, desc="Rendering Mask & BBox Video")
    ):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        frame_info = frames_dict.get(idx)
        if frame_info:
            # 1단계: 마스크 색상 합성 캔버스 준비
            mask_layer = img.copy()
            objects = frame_info.get("objects", [])

            # 마스크 영역 채우기 및 윤곽선 추출
            for obj in objects:
                obj_id = obj.get("id", 0)
                color = get_color_by_id(obj_id)
                rle = obj.get("segmentation")

                if rle and "counts" in rle:
                    bin_mask = decode_rle(rle)

                    # 반투명 채색 레이어
                    mask_layer[bin_mask > 0] = color

                    # 마스크 외곽 윤곽선 강조선 (두께 2)
                    contours, _ = cv2.findContours(
                        bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(img, contours, -1, color, thickness=2)

            # 원본 이미지와 마스크 레이어를 알파 블렌딩
            img = cv2.addWeighted(
                mask_layer, mask_alpha, img, 1.0 - mask_alpha, 0
            )

            # 2단계: 마스크 위에 두꺼운 BBox 및 ID/Score 라벨 그리기
            for obj in objects:
                obj_id = obj.get("id", 0)
                score = obj.get("score", None)
                bbox = obj.get("bbox")
                color = get_color_by_id(obj_id)

                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]

                    # BBox 그리기 (요청하신 두꺼운 선)
                    cv2.rectangle(
                        img, (x1, y1), (x2, y2), color, thickness=bbox_thickness
                    )

                    # ID와 Score 표시
                    if score is not None:
                        label = f"ID:{obj_id} ({float(score):.2f})"
                    else:
                        label = f"ID:{obj_id}"

                    font_scale = 0.55
                    font_thickness = 1
                    (tw, th), baseline = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        font_thickness,
                    )

                    # 라벨 텍스트 배경 박스 (가독성 확보)
                    label_y1 = max(0, y1 - th - baseline - 6)
                    label_y2 = y1
                    cv2.rectangle(
                        img,
                        (x1, label_y1),
                        (x1 + tw + 8, label_y2),
                        color,
                        cv2.FILLED,
                    )
                    cv2.putText(
                        img,
                        label,
                        (x1 + 4, label_y2 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        font_thickness,
                        cv2.LINE_AA,
                    )

        # 좌측 상단 프레임 카운트 표시
        cv2.putText(
            img,
            f"Frame: {idx:06d}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        writer.write(img)

    writer.release()
    print(f"\n[성공] 마스크+BBox 오버레이 비디오 생성 완료: {output_video_path}")