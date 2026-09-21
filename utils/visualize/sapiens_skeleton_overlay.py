import json
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
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

# 인스턴스 ID별 구분을 위한 고대비 BGR 색상 팔레트
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
    """ID에 따라 고정된 고유 BGR 색상 반환"""
    return DISTINCT_COLORS[(abs(int(instance_id)) - 1) % len(DISTINCT_COLORS)]


def draw_label(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = (255, 255, 255),
):
    """가독성을 위해 배경 사각형이 포함된 텍스트 라벨 출력"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    y = max(y, th + 6)

    # 텍스트 배경 박스
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + baseline - 2), bg_color, -1)
    # 텍스트 렌더링
    cv2.putText(img, text, (x + 3, y - 4), font, scale, text_color, thickness, cv2.LINE_AA)


def overlay_skeleton_to_video(
    frame_dir: str | Path,
    skeleton_json_path: str | Path,
    output_video_path: str | Path,
    fps: float = 30.0,
    score_thresh: float = 0.35,
    draw_bones: bool = True,
    draw_dense_points: bool = True,
) -> Path:
    """프레임 이미지와 Sapiens 308 스켈레톤 JSON을 읽어 BBox + ID + Skeleton 합성 비디오 생성"""
    start_time = time.time()
    frame_dir = Path(frame_dir)
    skt_json_path = Path(skeleton_json_path)
    out_vid_path = Path(output_video_path)

    print("\n" + "=" * 70)
    print("[INFO] 스켈레톤 오버레이 비디오 렌더링 시작")
    print("=" * 70)

    # -------------------------------------------------------------
    # 💡 1. 파일 검증 및 사전 체크 디버깅
    # -------------------------------------------------------------
    if not skt_json_path.exists():
        raise FileNotFoundError(f"[ERROR] 스켈레톤 JSON 파일이 없습니다: {skt_json_path.resolve()}")
    if not skt_json_path.is_file():
        raise ValueError(f"[ERROR] 지정한 스켈레톤 경로가 파일이 아닙니다: {skt_json_path.resolve()}")

    if not frame_dir.exists():
        raise FileNotFoundError(f"[ERROR] 프레임 디렉터리가 존재하지 않습니다: {frame_dir.resolve()}")
    if not frame_dir.is_dir():
        raise NotADirectoryError(f"[ERROR] 프레임 경로가 폴더가 아닙니다: {frame_dir.resolve()}")

    out_vid_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 💡 2. JSON 로드 및 인덱싱 매핑
    # -------------------------------------------------------------
    print(f"[INFO] 스켈레톤 데이터 파싱 중: {skt_json_path.name}")
    try:
        with open(skt_json_path, "r", encoding="utf-8") as f:
            skt_meta = json.load(f)
    except Exception as e:
        raise ValueError(f"[ERROR] 스켈레톤 JSON을 파싱할 수 없습니다: {e}")

    raw_frames = skt_meta.get("frames", [])
    if not raw_frames:
        raise ValueError("[ERROR] 스켈레톤 JSON 내 'frames' 목록이 비어 있습니다.")

    num_keypoints = skt_meta.get("num_keypoints", 308)
    print(f"[INFO] 키포인트 포맷: {skt_meta.get('keypoint_format', 'UNKNOWN')} ({num_keypoints} pts)")

    skt_by_frame_idx: Dict[int, List[Dict]] = {}
    skt_by_filename: Dict[str, List[Dict]] = {}
    for f_entry in raw_frames:
        fidx = f_entry.get("frame_index")
        fname = f_entry.get("file_name")
        insts = f_entry.get("instances", [])
        if fidx is not None:
            skt_by_frame_idx[int(fidx)] = insts
        if fname is not None:
            skt_by_filename[fname] = insts

    # -------------------------------------------------------------
    # 💡 3. 프레임 이미지 목록 정렬
    # -------------------------------------------------------------
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
        raise FileNotFoundError(f"[ERROR] 디스크에 렌더링할 이미지가 없습니다: {frame_dir.resolve()}")

    first_img = cv2.imread(str(all_frame_paths[0]))
    if first_img is None:
        raise ValueError(f"[ERROR] 첫 번째 이미지를 읽을 수 없습니다: {all_frame_paths[0]}")
    height, width = first_img.shape[:2]
    print(f"[INFO] 비디오 규격: {width}x{height} @ {fps}fps (총 {total_frames} 프레임)")

    # -------------------------------------------------------------
    # 💡 4. VideoWriter 초기화
    # -------------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"[ERROR] 비디오 라이터 생성 실패. 코덱 및 경로를 확인하세요: {out_vid_path}")

    # -------------------------------------------------------------
    # 💡 5. 프레임별 오버레이 렌더링 루프
    # -------------------------------------------------------------
    rendered_instances = 0

    for fidx, img_path in enumerate(tqdm(all_frame_paths, desc="Overlay Rendering")):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        instances = skt_by_filename.get(img_path.name, skt_by_frame_idx.get(fidx, []))

        for inst in instances:
            obj_id = inst.get("id", -1)
            bbox = inst.get("bbox")
            keypoints = inst.get("keypoints")
            scores = inst.get("keypoint_scores")

            color = get_instance_color(obj_id)

            # [1] BBox & ID 라벨 그리기
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(round(c)) for c in bbox]
                # BBox 경계선
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                # 라벨 텍스트
                label_txt = f"ID: {obj_id}"
                if "bbox_score" in inst and inst["bbox_score"] is not None:
                    label_txt += f" ({inst['bbox_score']:.2f})"
                draw_label(frame, label_txt, (x1, y1), bg_color=color)

            if not keypoints:
                continue

            rendered_instances += 1
            kpts_arr = np.array(keypoints, dtype=np.float32)
            scores_arr = (
                np.array(scores, dtype=np.float32)
                if scores is not None
                else np.ones(len(kpts_arr), dtype=np.float32)
            )

            # [2] 주요 뼈대 연결선 (Bones) 그리기
            if draw_bones:
                for idx1, idx2 in GOLIATH_BODY_LINKS:
                    if idx1 >= len(kpts_arr) or idx2 >= len(kpts_arr):
                        continue
                    sc1, sc2 = scores_arr[idx1], scores_arr[idx2]
                    if sc1 < score_thresh or sc2 < score_thresh:
                        continue

                    pt1 = kpts_arr[idx1]
                    pt2 = kpts_arr[idx2]

                    # 💡 비정상 발산 좌표(Out-of-bounds) 필터링
                    if (
                        pt1[0] < -50 or pt1[0] > width + 50 or pt1[1] < -50 or pt1[1] > height + 50
                        or pt2[0] < -50 or pt2[0] > width + 50 or pt2[1] < -50 or pt2[1] > height + 50
                    ):
                        continue

                    p1 = (int(round(pt1[0])), int(round(pt1[1])))
                    p2 = (int(round(pt2[0])), int(round(pt2[1])))
                    cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

            # [3] 308개 키포인트 점(Circles) 그리기
            for kidx, (pt, score) in enumerate(zip(kpts_arr, scores_arr)):
                if score < score_thresh:
                    continue

                x, y = pt[0], pt[1]
                # 화면 범위 이탈 필터링
                if x < 0 or x >= width or y < 0 or y >= height:
                    continue

                center = (int(round(x)), int(round(y)))

                # 주요 관절(0~20, 41, 62번)은 크고 굵게, 얼굴/손가락 등 밀집 포인트는 미세하게 표시
                is_major_joint = (kidx <= 20) or (kidx in [41, 62])
                if is_major_joint:
                    cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)
                    cv2.circle(frame, center, 5, (255, 255, 255), 1, cv2.LINE_AA)
                elif draw_dense_points:
                    # 손가락/얼굴 세부 점 (반경 2)
                    cv2.circle(frame, center, 2, color, -1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    elapsed = time.time() - start_time
    file_size_mb = out_vid_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 70)
    print("✨ [SUCCESS] 스켈레톤 오버레이 비디오 저장 완료")
    print(f"  • 출력 비디오 파일 : {out_vid_path.resolve()}")
    print(f"  • 비디오 파일 크기 : {file_size_mb:.2f} MB")
    print(f"  • 렌더링 프레임    : {total_frames} 장")
    print(f"  • 누적 인스턴스    : {rendered_instances} 개")
    print(f"  • 처리 속도        : {total_frames / max(elapsed, 0.001):.1f} fps (소요시간: {elapsed:.1f}초)")
    print("=" * 70 + "\n")

    return out_vid_path