# SAM 3 & Sapiens 2 기반 고품질 비디오 스켈레톤 추출 파이프라인

단일/다중 인물 비디오를 입력받아 모션 분석 및 모델 학습 데이터셋(Training Data)으로 활용할 수 있는 **Goliath 308 키포인트 스켈레톤 데이터**를 일관된 객체 ID와 함께 추출하는 엔드투엔드(End-to-End) 파이프라인입니다.

---

## 📌 주요 특징 및 모델 선정 배경

### 1. 객체 검출 및 비디오 트래킹: SAM 3 (Segment Anything Model 3)

* **메모리 뱅크(Memory Bank) 기반 양방향 추적**
일반적인 프레임 단위 2D 객체 감지기(YOLO 등)는 인물 간 교차, 가림(Occlusion), 카메라 각도 변화 시 ID 스위칭이 빈번합니다. SAM 3 비디오 엔진은 과거 및 미래 프레임의 특징을 메모리 뱅크에 누적해 전 프레임에 걸쳐 안정적인 인스턴스 ID와 정밀한 BBox를 유지합니다.
* **텍스트 프롬프트 기반 Zero-Shot 검출**
기준 프레임에서 `"person"` 텍스트 쿼리를 통해 타깃 인물을 초기에 분할하고, 이를 비디오 전 구간(정방향/역방향)으로 전파(Propagate)합니다.

### 2. 고정밀 포즈 추정: Sapiens 2 (Foundation Model)

* **초대형 파운데이션 모델 기반의 고정밀도 추출**
단순 경량 회귀(Regression) 모델 대비 관절 떨림과 위치 왜곡이 적은 고품질 좌표를 확보하기 위해 대규모 데이터셋으로 사전 학습된 Sapiens 2 모델을 채택했습니다.
* **Goliath 308 전신 키포인트**
기본 COCO 17개 관절을 넘어 안면 디테일/시선, 양손 손가락(Hand 40개), 발가락 및 뒤꿈치(Foot 6개) 등 총 308개 키포인트를 추출해 정밀한 운동·동작 평가 데이터를 구축합니다.

---

## 🔄 파이프라인 아키텍처

```
[00_video: 비디오 입력]
         │
         ▼
[01_frame: 프레임 시퀀스 추출 (extract_frame.py)]
         │
         ▼
[SAM 3 기준 프레임 "person" 검출 (detect_objects)]
         │
         ▼
[SAM 3 메모리 뱅크 기반 정방향/역방향 양방향 트래킹 (run_bidirectional_tracking)]
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
[02_bboxes: ID & BBox JSON 저장]        [02_bboxes: (T, H, W) Label Map NPZ 저장]
         │
         ▼
[Sapiens 2 배치 추론 (extract_sapiens2_skeletons, Batch Size=4~8, FP16)]
         │
         ▼
[03_skeleton: Goliath 308 키포인트 JSON 생성]
         │
         ▼
[04_overlay: BBox + ID + 308 스켈레톤 합성 MP4 렌더링 (sapiens_skeleton_overlay.py)]

```

---

## 📁 디렉터리 구조

```bash
.
├── configs/
│   └── sapiens/                     # Sapiens2 파라미터별 모델 설정 파일
│       ├── sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py
│       ├── sapiens2_0.8b_keypoints308_shutterstock_goliath_3po-1024x768.py
│       ├── sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py
│       └── sapiens2_5b_keypoints308_shutterstock_goliath_3po-1024x768.py
├── core/
│   ├── extract_frame.py             # 영상 프레임 추출 모듈
│   ├── huggingface_login.py         # Hugging Face 토큰 인증 모듈
│   ├── tracking_sam3.py             # SAM 3 객체 검출 및 양방향 트래킹
│   └── sapiens2_skeleton.py         # Sapiens 2 배치 스켈레톤 추론
├── utils/
│   └── visualize/
│       ├── sam_bbox_overlay.py      # SAM 3 BBox 비디오 시각화
│       ├── sam_mask_overlay.py      # SAM 3 마스크/BBox 시각화
│       └── sapiens_skeleton_overlay.py # BBox + ID + 308 스켈레톤 통합 렌더러
├── scripts/
│   ├── conda.sh                     # 가상환경 설정 셸 스크립트
│   └── total_pipeline.py            # 전 과정 파이프라인 통합 실행 스크립트
└── data/
    ├── checkpoints/
    │   ├── SAM3/                    # SAM 3 체크포인트 (sam3.pt, BPE vocab 등)
    │   └── sapiens2/                # Sapiens 2 체크포인트 (.safetensors)
    ├── 00_video/                    # 원본 입력 비디오 파일
    ├── 01_frame/                    # 추출된 프레임 이미지 디렉터리
    ├── 02_bboxes/                   # SAM 3 트래킹 결과
    │   ├── <video_name>.json        # 프레임별 인물 ID, BBox, 해상도 메타데이터
    │   └── <video_name>_masks.npz   # 프레임별 ID 라벨 맵 3D 압축 마스크 (T, H, W)
    ├── 03_skeleton/                 # Sapiens 2 308 키포인트 JSON
    └── 04_overlay/                  # 최종 합성된 검증용 오버레이 영상

```

---

## ⚙️ 환경 구축 및 가중치 설정

### 1. 필수 체크포인트 경로

체크포인트 파일들이 `data/checkpoints/` 하위에 위치해 있는지 확인합니다.

* **SAM 3 가중치**

* `data/checkpoints/SAM3/sam3.pt`

* `data/checkpoints/SAM3/bpe_simple_vocab_16e6.txt.gz`



* **Sapiens 2 가중치**

* `data/checkpoints/sapiens2/sapiens2_0.4b_pose.safetensors` (또는 상위 모델 `sapiens2_5b_pose.safetensors`)





### 2. 가상환경 세팅 및 활성화

```bash
# 가상환경 생성
bash scripts/conda.sh

# 가상환경 활성화
conda activate samsap

```

---

## 🚀 실행 방법

### 1. 설정 경로 확인

`scripts/total_pipeline.py`를 열어 처리할 비디오 파일명 및 가중치 경로를 확인합니다.

```python
# 입력 비디오 지정 예시
VID_PATH = PRJ_PATH / "data" / "00_video" / "frontal__alternating_shoulder_press__incline1.MP4"

# 모델 가중치 및 Config 매칭
SAPIENS_CONFIG = PRJ_PATH / "configs" / "sapiens" / "sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py"
SAPIENS_CHECKPOINT = PRJ_PATH / "data" / "checkpoints" / "sapiens2" / "sapiens2_0.4b_pose.safetensors"
SAM_CKPT_PATH = PRJ_PATH / "data" / "checkpoints" / "SAM3" / "sam3.pt"
BPE_PATH = PRJ_PATH / "data" / "checkpoints" / "SAM3" / "bpe_simple_vocab_16e6.txt.gz"

```

### 2. 전체 파이프라인 실행

프레임 추출부터 SAM 3 트래킹, Sapiens 2 스켈레톤 추출, 시각화 오버레이 비디오 생성까지 원클릭으로 순차 수행됩니다.

```bash
python scripts/total_pipeline.py

```

---

## 📊 출력 데이터 포맷 (`data/03_skeleton/*.json`)

추출된 결과물은 저장 용량 최적화 및 I/O 속도를 위해 소수점 3자리로 경량화 직렬화되어 저장됩니다.

```json
{
  "video_name": "frontal__alternating_shoulder_press__incline1",
  "total_frames": 558,
  "num_keypoints": 308,
  "keypoint_format": "GOLIATH_308",
  "frames": [
    {
      "frame_index": 0,
      "file_name": "000000.jpg",
      "instances": [
        {
          "id": 1,
          "bbox": [257.0, 365.0, 743.0, 658.0],
          "keypoints": [
            [704.686, 374.658],
            [716.478, 377.035]
          ],
          "keypoint_scores": [0.994, 1.008]
        }
      ]
    }
  ]
}

```

## 🎭 마스크 데이터 포맷 (`data/02_bboxes/*_masks.npz`)

대규모 비디오 처리 시 I/O 병목 및 디스크 용량을 최적화하기 위해, SAM 3 인스턴스 분할 마스크는 **ID 맵(Label Map) 형태의 3D 압축 넘파이 아카이브(`.npz`)**로 저장됩니다.

### 1. 포맷 특징
* **Shape**: `(T, H, W)` (3D `numpy.ndarray`, 정수형 `uint8`)
  * `T`: 전체 비디오 프레임 수
  * `H, W`: 원본 비디오 해상도 (세로, 가로)
* **Pixel Encoding**: 픽셀 값 자체가 인물의 고유 식별자(ID)입니다.
  * `0`: 배경 (Background)
  * `1`: 타깃 1번 인물 (예: 영아)
  * `2, 3...`: 타깃 2번, 3번 인물 (예: 검사자, 보호자 등)
* **Metadata**: 원본 해상도 `[W, H]`가 `resolution` 키로 내장되어 있습니다.

---

### 2. 마스크 로드 및 활용 예제 (Python)

```python
import numpy as np

# 1. NPZ 마스크 파일 로드
mask_path = "data/02_bboxes/p01_gross_motor_4_masks.npz"
data = np.load(mask_path)

masks = data["masks"]              # Shape: (T, H, W), dtype: uint8
orig_w, orig_h = data["resolution"] # [가로, 세로]

print(f"총 프레임: {masks.shape[0]}, 해상도: {orig_w}x{orig_h}")

# 2. 특정 프레임(t)에서 특정 인물(ID=1)의 바이너리 마스크 추출
t = 120
target_id = 1
baby_mask = (masks[t] == target_id)  # (H, W) boolean 배열 (True/False)

# 3. 키포인트 (px, py)가 아기 마스크 내부에 있는지 즉시 검증 (In-Mask Check)
px, py = 450, 320
is_in_mask = (masks[t, py, px] == target_id)  # True/False

# 4. 검사자(ID=2)와의 신체 겹침/가림(Occlusion) 영역 계산
doctor_mask = (masks[t] == 2)
overlap_area = baby_mask & doctor_mask  # 인물 간 겹침 픽셀