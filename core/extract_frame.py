import shutil
from pathlib import Path
import cv2
from tqdm import tqdm

def extract_frames(
    video_path: str | Path,
    frame_dir: str | Path,
    target_short: int = 720,
    jpeg_quality: int = 80,
    max_per_folder: int = 10000,
) -> int:
    """영상에서 프레임 추출 (1만 개 단위 하위 폴더 분산 저장)"""
    video_path, frame_dir = Path(video_path), Path(frame_dir)

    if not video_path.exists():
        print(f"[오류] 영상 파일을 찾을 수 없습니다: {video_path}")
        return 0

    # 1. 기존 출력 폴더 초기화
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[오류] 영상 스트림을 열 수 없습니다: {video_path}")
        return 0

    # 영상 정보 획득
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if w == 0 or h == 0 or n_frames == 0:
        print(f"[오류] 유효하지 않은 영상 메타데이터입니다: {video_path}")
        cap.release()
        return 0

    # 리사이즈 스케일 계산 (짧은 축 기준)
    min_dim = min(w, h)
    scale = target_short / min_dim
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    # 확대(360p -> 720p 등) 시 계단 현상 방지를 위해 INTER_LINEAR, 축소 시 INTER_AREA 사용
    interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA

    count = 0

    for idx in tqdm(range(n_frames), total=n_frames, desc=f"Extracting {video_path.name}"):
        ret, frame = cap.read()
        if not ret:
            break

        # 1만 장 단위 서브 폴더 분산 ('00', '01', ...)
        subdir_num = f"{idx // max_per_folder:02d}"
        target_subdir = frame_dir / subdir_num
        target_subdir.mkdir(exist_ok=True)

        # 리사이즈 및 저장
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        out_path = target_subdir / f"{idx:06d}.jpg"

        cv2.imwrite(str(out_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        count += 1

    cap.release()
    return count