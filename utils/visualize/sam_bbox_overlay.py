import json
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def get_color_by_id(obj_id: int):
    """객체 ID마다 고유한 BGR 색상 반환"""
    np.random.seed(int(obj_id) * 31)
    color = np.random.randint(50, 255, size=3).tolist()
    return tuple(color)


def overlay_bboxes_to_video(
    frame_dir: str | Path,
    bbox_json_path: str | Path,
    output_video_path: str | Path,
    fps: int = 30,
):
    """프레임 이미지 폴더와 SAM3 BBox JSON을 읽어 BBox가 오버레이된 비디오를 렌더링합니다."""
    frame_dir = Path(frame_dir)
    bbox_json_path = Path(bbox_json_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if not bbox_json_path.exists():
        raise FileNotFoundError(
            f"BBox JSON 파일을 찾을 수 없습니다: {bbox_json_path}"
        )

    # 1. JSON 데이터 로드
    print(f"[Overlay] BBox JSON 로드 중: {bbox_json_path}")
    with open(bbox_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames_meta = data.get("frames", [])
    if not frames_meta:
        print("[Overlay] JSON에 프레임 데이터가 없습니다.")
        return

    # 빠른 조회를 위해 frame_index를 키로 매핑
    frames_dict = {f["frame_index"]: f for f in frames_meta}

    # 2. 프레임 경로 매핑 수집 (00/000000.jpg, 01/000001.jpg 등 하위 폴더 전체 대응)
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

    # 3. 비디오 해상도 확인
    sample_img = cv2.imread(str(all_frame_paths[0]))
    if sample_img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {all_frame_paths[0]}")
    h, w, _ = sample_img.shape

    # 4. VideoWriter 세팅 (mp4v 코덱)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

    print(
        f"[Overlay] 비디오 렌더링 시작 (총 {len(all_frame_paths)} 프레임, {w}x{h} @ {fps}fps)"
    )

    for idx, img_path in enumerate(
        tqdm(all_frame_paths, desc="Rendering Overlay Video")
    ):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 해당 프레임의 객체 목록 조회
        frame_info = frames_dict.get(idx)
        if frame_info:
            for obj in frame_info.get("objects", []):
                bbox = obj.get("bbox")
                obj_id = obj.get("id", 0)

                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    color = get_color_by_id(obj_id)

                    # 바운딩 박스 그리기
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

                    # ID 텍스트 레이블 태그
                    label = f"ID: {obj_id}"
                    font_scale = 0.6
                    thickness = 1
                    (tw, th), baseline = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        thickness,
                    )

                    # 텍스트 배경 박스 (가독성 확보)
                    label_y1 = max(0, y1 - th - baseline - 4)
                    label_y2 = y1
                    cv2.rectangle(
                        img,
                        (x1, label_y1),
                        (x1 + tw + 6, label_y2),
                        color,
                        cv2.FILLED,
                    )
                    cv2.putText(
                        img,
                        label,
                        (x1 + 3, label_y2 - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )

        # 좌측 상단 현재 프레임 번호 표기
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
    print(f"\n[성공] 오버레이 비디오 생성 완료: {output_video_path}")