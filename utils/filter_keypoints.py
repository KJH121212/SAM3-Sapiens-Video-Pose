import json
from typing import List, Optional

DEFAULT_SELECTED_INDICES = [
    0, 1, 2, 3, 4,         # 머리[cite: 2]
    5, 6, 7, 8, 41, 62,    # 상체[cite: 2]
    9, 10, 11, 12, 13, 14, # 하체[cite: 2]
    15, 16, 17, 18, 19, 20 # 발[cite: 2]
]

def extract_keypoints_summary(
    ori_path: str,
    out_path: str,
    selected_indices: Optional[List[int]] = None,
    indent: int = 2  # 줄바꿈 및 들여쓰기 공백 수 (기본 2칸, 보통 2 또는 4 사용)
) -> None:
    if selected_indices is None:
        selected_indices = DEFAULT_SELECTED_INDICES

    with open(ori_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_kps = len(selected_indices)
    data["num_keypoints"] = num_kps
    data["keypoint_format"] = f"GOLIATH_SUBSET_{num_kps}"

    for frame in data.get("frames", []):
        for instance in frame.get("instances", []):
            raw_kps = instance.get("keypoints", [])
            raw_scores = instance.get("keypoint_scores", [])

            instance["keypoints"] = [raw_kps[i] for i in selected_indices if i < len(raw_kps)]
            if raw_scores:
                instance["keypoint_scores"] = [
                    raw_scores[i] for i in selected_indices if i < len(raw_scores)
                ]

    # indent와 ensure_ascii 옵션을 적용해 줄바꿈 저장
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    print(f"변환 완료 (들여쓰기 적용): {out_path}")


# 사용 예시
if __name__ == "__main__":
    ori_file = "/workspace/data_d03_new/03_KPT308/1_bayley/p01/p01_gross_motor_4.json"
    out_file = "/workspace/data_d03_new/03_KPT308/1_bayley/p01/p01_gross_motor_4_summary.json"
    
    # 기본 21개 관절로 변환[cite: 2]
    extract_keypoints_summary(ori_file, out_file)