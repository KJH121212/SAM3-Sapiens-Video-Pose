import glob
import json
import os
from collections import OrderedDict
from pathlib import Path
import sys
from typing import Any, Dict
import cv2
import numpy as np
from PIL import Image
import torch
import dataclasses

# --- SAM3 라이브러리 ---
import sam3
from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.model_builder import build_sam3_video_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
sys.path.append(f"{sam3_root}/examples")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
GLOBAL_COUNTER = 1


# ==============================================================================
# [Helper] 유틸리티
# ==============================================================================
def mask_to_rle(mask):
    """이진 마스크를 RLE 포맷으로 변환"""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return {"size": mask.shape, "counts": runs.tolist()}


def mask_to_bbox(mask):
    """이진 마스크로부터 Bounding Box 추출.

    Returns: [x_min, y_min, x_max, y_max]
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def create_empty_datapoint():
    return Datapoint(find_queries=[], images=[])


def set_image(datapoint, pil_image):
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def add_text_prompt(datapoint, text_query):
    global GLOBAL_COUNTER
    w, h = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER,
                original_image_id=GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    GLOBAL_COUNTER += 1


# ==============================================================================
# [Part 1] 객체 검출
# ==============================================================================
# ==============================================================================
# [Part 1] 객체 검출
# ==============================================================================
def detect_objects(
    frame_dir: str,
    text_prompt: str,
    target_frame_idx: int = 0,
    model=None,
    checkpoint_path: str = None,
    bpe_path: str = None,
) -> Dict[str, Any]:
    frame_path_obj = Path(frame_dir)
    candidates = sorted(
        [str(p) for p in frame_path_obj.rglob("*.jpg")]
        + [str(p) for p in frame_path_obj.rglob("*.png")]
    )

    try:
        candidates.sort(key=lambda p: int(Path(p).stem))
    except Exception:
        candidates.sort()

    if not candidates:
        print(f"[Error] 이미지 없음: {frame_dir}")
        return None

    img_path = candidates[target_frame_idx]

    # [디버그 1] 실제 읽어들이는 이미지 검증
    pil_img = Image.open(img_path).convert("RGB")
    print(f"\n========== [DEBUG LOG START] ==========")
    print(f"1. 검출 대상 이미지 파일: {img_path}")
    print(
        f"   - 이미지 크기: {pil_img.size} (W, H), 포맷: {pil_img.format}, 모드: {pil_img.mode}"
    )

    # 텍스트 프롬프트 뒤에 온점(.)이 없으면 붙여서 시도
    clean_prompt = text_prompt.strip()
    if not clean_prompt.endswith("."):
        clean_prompt += "."
    print(f"2. 적용 프롬프트: '{clean_prompt}' (원문: '{text_prompt}')")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        print("3. 모델 로드 시작...")
        ckpt_path = (
            checkpoint_path
            if checkpoint_path is not None
            else "/workspace/data/checkpoints/SAM3/sam3.pt"
        )
        vocab_path = (
            bpe_path
            if bpe_path is not None
            else "/workspace/data/checkpoints/SAM3/bpe_simple_vocab_16e6.txt.gz"
        )
        model = build_sam3_image_model(
            checkpoint_path=ckpt_path, bpe_path=vocab_path
        )
        model.to(device)
        print("   - 모델 로드 완료")

    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=1008, max_size=1008, square=True, consistent_transform=False
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # [디버그 2] 임계값을 극단적으로 낮춤 (0.05)
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=False,
    )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        datapoint = create_empty_datapoint()
        set_image(datapoint, pil_img)
        # 프롬프트는 'person' (마침표 없이)
        add_text_prompt(datapoint, "person")
        datapoint = transform(datapoint)

        batch = collate([datapoint], dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, device, non_blocking=True)

        raw_outputs = model(batch)

        # [디버그] raw_outputs 내부 확인
        print(f"4. raw_outputs 타입: {type(raw_outputs)}")
        if hasattr(raw_outputs, "loss_stages"):
            find_stages = raw_outputs
        elif hasattr(raw_outputs, "output") and hasattr(
            raw_outputs.output, "loss_stages"
        ):
            find_stages = raw_outputs.output
        elif isinstance(raw_outputs, (list, tuple)) and len(raw_outputs) > 0:
            find_stages = raw_outputs[0]
        else:
            find_stages = raw_outputs

        # 후처리 실행
        processed_results = postprocessor.process_results(
            find_stages, batch.find_metadatas
        )

    del batch

    print(f"5. Processed Results 개수: {len(processed_results)}")
    if len(processed_results) > 0:
        result = list(processed_results.values())[0]
        scores = result.get("scores", torch.tensor([]))
        boxes = result.get("boxes", torch.tensor([]))
        print(f"   - 검출된 BBox 개수: {len(boxes)}")
        if scores.numel() > 0:
            print(
                f"   - 검출 스코어 (전체 {len(scores)}개): {[round(s, 4) for s in scores.tolist()]}"
            )
        print(f"========== [DEBUG LOG END] ==========\n")
        return result

    print(f"   - 검출된 객체 없음.")
    print(f"========== [DEBUG LOG END] ==========\n")
    return None

    del batch

    print(f"5. Processed Results 개수: {len(processed_results)}")
    if len(processed_results) > 0:
        result = list(processed_results.values())[0]
        scores = result.get("scores", torch.tensor([]))
        boxes = result.get("boxes", torch.tensor([]))
        print(f"   - 통과된 BBox 개수: {len(boxes)}")
        print(f"   - 검출 스코어(Top 5): {scores[:5].tolist()}")
        print(f"========== [DEBUG LOG END] ==========\n")
        return result

    print(f"   - 0.05 임계값 기준 통과 객체 없음.")
    print(f"========== [DEBUG LOG END] ==========\n")
    return None

# ==============================================================================
# [Part 2] 양방향 트래킹 및 단일 JSON 저장
# ==============================================================================
class LazyVideoLoader:

    def __init__(self, video_path, image_size=1008):
        video_path_obj = Path(video_path)
        self.frame_paths = sorted(
            [str(p) for p in video_path_obj.rglob("*.jpg")]
            + [str(p) for p in video_path_obj.rglob("*.png")]
        )
        try:
            self.frame_paths.sort(key=lambda p: int(Path(p).stem))
        except Exception:
            self.frame_paths.sort()

        self.image_size = image_size

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.frame_paths[idx]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = (
            img.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])
        ) / np.array([0.229, 0.224, 0.225])
        return torch.from_numpy(img).permute(2, 0, 1)


def init_state_lazy(predictor, video_path):
    state = {
        "offload_video_to_cpu": True,
        "offload_state_to_cpu": True,
        "device": predictor.device,
        "storage_device": torch.device("cpu"),
        "images": LazyVideoLoader(video_path, predictor.image_size),
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": {},
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict": {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        },
        "tracking_has_started": False,
        "frames_already_tracked": {},
        "first_ann_frame_idx": None,
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "consolidated_frame_inds": {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        },
    }
    state["num_frames"] = len(state["images"])
    first = cv2.imread(state["images"].frame_paths[0])
    state["video_height"], state["video_width"] = first.shape[:2]
    predictor.clear_all_points_in_video(state)
    return state


def collect_frame_data(
    frame_idx, obj_ids, video_res_masks, state, total_results: Dict[int, Any]
):
    """메모리 딕셔너리에 프레임별 BBox 및 객체 정보 누적"""
    frame_entry = {
        "frame_index": int(frame_idx),
        "file_name": os.path.basename(state["images"].frame_paths[frame_idx]),
        "objects": [],
    }

    if video_res_masks is not None and len(video_res_masks) > 0:
        for k, obj_id in enumerate(obj_ids):
            if isinstance(obj_id, torch.Tensor):
                obj_id = obj_id.item()
            mask_tensor = video_res_masks[k]
            if mask_tensor.dim() == 3:
                mask_tensor = mask_tensor.squeeze(0)
            mask_np = (mask_tensor.cpu().numpy() > 0.0).astype(np.uint8)

            if np.any(mask_np):
                bbox = mask_to_bbox(mask_np)
                obj_info = {
                    "id": int(obj_id),
                    "bbox": bbox,  # [x_min, y_min, x_max, y_max]
                    # "segmentation": mask_to_rle(mask_np)  # 필요 시 주석 해제
                }
                frame_entry["objects"].append(obj_info)

    total_results[int(frame_idx)] = frame_entry


def run_bidirectional_tracking(
    frame_dir: str,
    detection_results: Dict,
    output_dir: str,
    start_frame_idx: int,
    model=None,
    checkpoint_path: str = None,  # 💡 추가
    bpe_path: str = None,  # 💡 추가
) -> Path:
    frame_dir_path = Path(frame_dir)
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    json_file_path = out_dir_path / f"{frame_dir_path.name}.json"
    mask_key = "masks" if "masks" in detection_results else "segmentation"
    num_objs = detection_results["scores"].numel()

    if model is None:
        print("[SAM 3] Video Model 로드 중 (로컬 체크포인트 사용)...")
        ckpt_path = (
            checkpoint_path
            if checkpoint_path is not None
            else "/workspace/data/checkpoints/SAM3/sam3.pt"
        )
        vocab_path = (
            bpe_path
            if bpe_path is not None
            else "/workspace/data/checkpoints/SAM3/bpe_simple_vocab_16e6.txt.gz"
        )

        # 💡 checkpoint_path와 bpe_path를 명시적으로 전달
        sam3_model = build_sam3_video_model(
            checkpoint_path=ckpt_path,
            bpe_path=vocab_path,
            apply_temporal_disambiguation=True,
            device="cuda",
        )
    else:
        sam3_model = model

    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone

    state = init_state_lazy(predictor, frame_dir)

    # 1. 초기 마스크 등록
    for i in range(num_objs):
        mask = detection_results[mask_key][i].cuda().float()
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=start_frame_idx,
            obj_id=i + 1,
            mask=mask,
        )

    # 전체 결과를 수집할 딕셔너리
    total_results = {}

    # 2. 정방향 추적 (start_frame_idx -> End)
    for (
        frame_idx,
        obj_ids,
        _,
        video_res_masks,
        _,
    ) in predictor.propagate_in_video(
        state,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=None,
        reverse=False,
        propagate_preflight=True,
    ):
        collect_frame_data(
            frame_idx, obj_ids, video_res_masks, state, total_results
        )

        # VRAM 메모리 정리
        if frame_idx > start_frame_idx and frame_idx % 100 == 0:
            cutoff = frame_idx - 7
            if cutoff > start_frame_idx:
                outputs = state["output_dict"]["non_cond_frame_outputs"]
                keys_to_remove = [k for k in outputs.keys() if k < cutoff]
                for k in keys_to_remove:
                    del outputs[k]
                for obj_dict in state["output_dict_per_obj"].values():
                    outputs_obj = obj_dict["non_cond_frame_outputs"]
                    keys_to_remove_obj = [
                        k for k in outputs_obj.keys() if k < cutoff
                    ]
                    for k in keys_to_remove_obj:
                        del outputs_obj[k]

    # 중간 캐시 정리
    outputs = state["output_dict"]["non_cond_frame_outputs"]
    keys_to_remove = [k for k in outputs.keys() if k > start_frame_idx]
    for k in keys_to_remove:
        del outputs[k]

    # 3. 역방향 추적 (start_frame_idx -> 0)
    for (
        frame_idx,
        obj_ids,
        _,
        video_res_masks,
        _,
    ) in predictor.propagate_in_video(
        state,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=None,
        reverse=True,
        propagate_preflight=True,
    ):
        collect_frame_data(
            frame_idx, obj_ids, video_res_masks, state, total_results
        )

        if frame_idx < start_frame_idx and frame_idx % 100 == 0:
            cutoff = frame_idx + 7
            if cutoff < start_frame_idx:
                outputs = state["output_dict"]["non_cond_frame_outputs"]
                keys_to_remove = [
                    k
                    for k in outputs.keys()
                    if k > cutoff and k < start_frame_idx
                ]
                for k in keys_to_remove:
                    del outputs[k]
                for obj_dict in state["output_dict_per_obj"].values():
                    outputs_obj = obj_dict["non_cond_frame_outputs"]
                    keys_to_remove_obj = [
                        k
                        for k in outputs_obj.keys()
                        if k > cutoff and k < start_frame_idx
                    ]
                    for k in keys_to_remove_obj:
                        del outputs_obj[k]

    del state

    # 4. 프레임 번호 기준 오름차순 정렬 후 단일 파일로 덤프
    sorted_frames = [
        total_results[idx] for idx in sorted(total_results.keys())
    ]
    final_output = {
        "video_name": frame_dir_path.name,
        "total_frames": len(sorted_frames),
        "frames": sorted_frames,
    }

    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"✅ 통합 BBox JSON 저장 완료: {json_file_path}")
    return json_file_path