import os
from pathlib import Path
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

# =====================================================================
# Goliath 308 키포인트 체계 기반 주요 신체 연결선 (Bone Links)
# =====================================================================
GOLIATH_BODY_LINKS = [
    # 몸통 (Torso)
    (5, 6),    # left_shoulder - right_shoulder
    (5, 9),    # left_shoulder - left_hip
    (6, 10),   # right_shoulder - right_hip
    (9, 10),   # left_hip - right_hip
    # 팔 (Arms)
    (5, 7),    # left_shoulder - left_elbow
    (7, 62),   # left_elbow - left_wrist (Goliath 62번: left_wrist)
    (6, 8),    # right_shoulder - right_elbow
    (8, 41),   # right_elbow - right_wrist (Goliath 41번: right_wrist)
    # 다리 (Legs)
    (9, 11),   # left_hip - left_knee
    (11, 13),  # left_knee - left_ankle
    (10, 12),  # right_hip - right_knee
    (12, 14),  # right_knee - right_ankle
    # 발 (Feet)
    (13, 15),  # left_ankle - left_big_toe
    (13, 16),  # left_ankle - left_small_toe
    (13, 17),  # left_ankle - left_heel
    (14, 18),  # right_ankle - right_big_toe
    (14, 19),  # right_ankle - right_small_toe
    (14, 20),  # right_ankle - right_heel
    # 머리 및 얼굴 외곽 (Head)
    (0, 1),    # nose - left_eye
    (0, 2),    # nose - right_eye
    (1, 3),    # left_eye - left_ear
    (2, 4),    # right_eye - right_ear
    (3, 5),    # left_ear - left_shoulder
    (4, 6),    # right_ear - right_shoulder
]

# 인스턴스 ID별 구분을 위한 고대비 BGR 색상 팔레트 (SAM 마스크 및 BBox 라벨용)
DISTINCT_COLORS = [
    (0, 255, 255),    # 노랑
    (0, 255, 128),    # 연두
    (255, 105, 65),   # 청록
    (0, 165, 255),    # 주황
    (255, 0, 255),    # 마젠타
    (255, 191, 0),    # 딥스카이블루
    (50, 205, 50),    # 라임그린
    (180, 105, 255),  # 핑크
    (0, 215, 255),    # 골드
    (255, 144, 30),   # 블루
]


def get_instance_color(instance_id: int) -> Tuple[int, int, int]:
    """ID에 따라 고유 BGR 색상 반환"""
    if instance_id <= 0:
        return (200, 200, 200)
    return DISTINCT_COLORS[(abs(int(instance_id)) - 1) % len(DISTINCT_COLORS)]


def draw_label(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = (255, 255, 255),
):
    """가독성을 위해 배경 사각형 및 외곽 테두리가 포함된 라벨 출력"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    y = max(y, th + 8)

    # 텍스트 배경 박스
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + baseline - 2), bg_color, -1)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + baseline - 2), (0, 0, 0), 1, cv2.LINE_AA)

    # 텍스트 그림자 + 글씨
    cv2.putText(img, text, (x + 4, y - 3), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x + 3, y - 4), font, scale, text_color, thickness, cv2.LINE_AA)


def overlay_skeleton_to_video(
    frame_dir: str | Path,
    skeleton_npz_path: str | Path,
    output_video_path: str | Path,
    sam_npz_path: Optional[str | Path] = None,
    fps: float = 30.0,
    score_thresh: float = 0.35,
    draw_bones: bool = True,
    draw_dense_points: bool = True,
) -> Path:
    """프레임 이미지, Sapiens 스켈레톤 NPZ, SAM 마스크 NPZ를 합성하여 비디오 생성"""
    start_time = time.time()
    frame_dir = Path(frame_dir)
    skt_npz_path = Path(skeleton_npz_path)
    out_vid_path = Path(output_video_path)

    print("\n" + "=" * 70)
    print("[INFO] SAM 마스크 + 스켈레톤 복합 오버레이 비디오 렌더링 시작")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. 파일 검증
    # -------------------------------------------------------------
    if not skt_npz_path.exists():
        raise FileNotFoundError(f"[ERROR] 스켈레톤 NPZ 파일이 없습니다: {skt_npz_path.resolve()}")
    if not frame_dir.is_dir():
        raise NotADirectoryError(f"[ERROR] 프레임 경로가 폴더가 아닙니다: {frame_dir.resolve()}")

    out_vid_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 2. NPZ 데이터 로드 (스켈레톤 및 SAM 마스크)
    # -------------------------------------------------------------
    print(f"[INFO] 스켈레톤 NPZ 로드: {skt_npz_path.name}")
    data = np.load(skt_npz_path)
    kpts_all = data["keypoints"]  # (T, M, 308, 3) -> [x, y, score]
    track_ids = data["track_ids"]  # (M,)
    total_npz_frames, num_slots, num_kpts, _ = kpts_all.shape

    # SAM 마스크 로드 (지정되지 않았거나 존재하지 않을 경우 자동 탐색 fallback)
    sam_masks_3d = None
    if sam_npz_path is None:
        # data_new/03_KPT308/path.npz -> data_new/02_SAM/path_masks.npz 유추
        guess_sam_path = skt_npz_path.parent.parent / "02_SAM" / f"{skt_npz_path.stem}_masks.npz"
        if guess_sam_path.exists():
            sam_npz_path = guess_sam_path

    if sam_npz_path and Path(sam_npz_path).exists():
        print(f"[INFO] SAM 마스크 NPZ 로드: {Path(sam_npz_path).name}")
        with np.load(sam_npz_path) as sdata:
            sam_masks_3d = sdata["masks"]  # (T, H, W)
    else:
        print("[WARN] SAM 마스크 파일이 지정되지 않았거나 없습니다. 마스크 렌더링을 생략합니다.")

    # -------------------------------------------------------------
    # 3. 프레임 이미지 목록 정렬
    # -------------------------------------------------------------
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_frame_paths = sorted(
        [p for p in frame_dir.rglob("*") if p.suffix.lower() in valid_exts]
    )
    try:
        all_frame_paths.sort(key=lambda p: int(p.stem))
    except Exception:
        all_frame_paths.sort()

    total_frames = len(all_frame_paths)
    if total_frames == 0:
        raise FileNotFoundError(f"[ERROR] 디스크에 렌더링할 이미지가 없습니다: {frame_dir.resolve()}")

    first_img = cv2.imread(str(all_frame_paths[0]))
    if first_img is None:
        raise ValueError(f"[ERROR] 첫 번째 이미지를 읽을 수 없습니다: {all_frame_paths[0]}")
    height, width = first_img.shape[:2]
    print(f"[INFO] 비디오 규격: {width}x{height} @ {fps}fps (총 {total_frames} 프레임)")

    # -------------------------------------------------------------
    # 4. VideoWriter 초기화
    # -------------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"[ERROR] 비디오 라이터 생성 실패: {out_vid_path}")

    # -------------------------------------------------------------
    # 5. 프레임별 오버레이 렌더링 루프
    # -------------------------------------------------------------
    render_len = min(total_frames, total_npz_frames)
    if sam_masks_3d is not None:
        render_len = min(render_len, len(sam_masks_3d))

    for fidx in tqdm(range(render_len), desc="Overlay Rendering"):
        frame = cv2.imread(str(all_frame_paths[fidx]))
        if frame is None:
            continue

        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        # [1] SAM 마스크 반투명 오버레이 및 SAM 기반 BBox/라벨 렌더링
        if sam_masks_3d is not None:
            frame_mask = sam_masks_3d[fidx]
            unique_uids = [int(u) for u in np.unique(frame_mask) if u > 0]

            if len(unique_uids) > 0:
                # 1-1. 마스크 반투명 합성
                overlay = frame.copy()
                for uid in unique_uids:
                    color = get_instance_color(uid)
                    overlay[frame_mask == uid] = color
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

                # 1-2. SAM 마스크 픽셀에서 BBox 도출 및 ID 표시
                for uid in unique_uids:
                    y_indices, x_indices = np.where(frame_mask == uid)
                    if len(y_indices) > 50:  # 노이즈 방지
                        bx1, bx2 = int(np.min(x_indices)), int(np.max(x_indices))
                        by1, by2 = int(np.min(y_indices)), int(np.max(y_indices))
                        id_color = get_instance_color(uid)

                        # BBox 사각형 (외곽 검은선 + 내부 ID 색상)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), id_color, 2, cv2.LINE_AA)

                        # ID 라벨
                        draw_label(frame, f"ID: {uid}", (bx1, by1), bg_color=id_color)

        # [2] Sapiens 스켈레톤 렌더링 (모든 인물 동일하게 '검은색 외곽선 + 흰색 실선' 적용)
        for slot in range(num_slots):
            kpts_with_score = kpts_all[fidx, slot]

            if np.isnan(kpts_with_score).any():
                continue

            pts = kpts_with_score[:, :2]
            scores = kpts_with_score[:, 2]

            # 2-1. 뼈대 연결선 (Bones) - 검은색(두께 4) + 흰색(두께 2) 이중선
            if draw_bones:
                for idx1, idx2 in GOLIATH_BODY_LINKS:
                    if idx1 >= num_kpts or idx2 >= num_kpts:
                        continue
                    if scores[idx1] < score_thresh or scores[idx2] < score_thresh:
                        continue

                    pt1, pt2 = pts[idx1], pts[idx2]
                    if (
                        pt1[0] < -50 or pt1[0] > width + 50 or pt1[1] < -50 or pt1[1] > height + 50
                        or pt2[0] < -50 or pt2[0] > width + 50 or pt2[1] < -50 or pt2[1] > height + 50
                    ):
                        continue

                    p1 = (int(round(pt1[0])), int(round(pt1[1])))
                    p2 = (int(round(pt2[0])), int(round(pt2[1])))

                    # 1단계: 검은색 굵은 외곽선
                    cv2.line(frame, p1, p2, (0, 0, 0), 4, cv2.LINE_AA)
                    # 2단계: 흰색 내부 중심선
                    cv2.line(frame, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)

            # 2-2. 관절 포인트 그리기
            for kidx, (pt, score) in enumerate(zip(pts, scores)):
                if score < score_thresh:
                    continue

                x, y = pt[0], pt[1]
                if x < 0 or x >= width or y < 0 or y >= height:
                    continue

                center = (int(round(x)), int(round(y)))
                is_major_joint = (kidx <= 20) or (kidx in [41, 62])

                if is_major_joint:
                    # 주요 관절: 검은 테두리(반경 5) + 내부 흰색 원(반경 3)
                    cv2.circle(frame, center, 5, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(frame, center, 3, (255, 255, 255), -1, cv2.LINE_AA)
                elif draw_dense_points:
                    # 세부 관절: 시각 간섭을 줄이기 위한 작은 회색 점(반경 2)
                    cv2.circle(frame, center, 2, (160, 160, 160), -1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    elapsed = time.time() - start_time
    file_size_mb = out_vid_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 70)
    print("✨ [SUCCESS] SAM 마스크 & 스켈레톤 오버레이 비디오 저장 완료")
    print(f"  • 출력 비디오 파일 : {out_vid_path.resolve()}")
    print(f"  • 비디오 파일 크기 : {file_size_mb:.2f} MB")
    print(f"  • 렌더링 프레임    : {render_len} 장")
    print(f"  • 처리 속도        : {render_len / max(elapsed, 0.001):.1f} fps (소요시간: {elapsed:.1f}초)")
    print("=" * 70 + "\n")

    return out_vid_path