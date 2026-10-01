import shutil
from pathlib import Path
import pandas as pd

# =====================================================================
# 경로 설정
# =====================================================================
DATA_DIR = Path("/workspace/data_new")
CSV_PATH = DATA_DIR / "metadata_v3.0.csv"
SAM_DIR = DATA_DIR / "02_SAM"

if not CSV_PATH.exists():
    print(f"[오류] CSV 파일을 찾을 수 없습니다: {CSV_PATH}")
    exit(1)

# =====================================================================
# 1. 안전을 위한 백업 생성
# =====================================================================
backup_path = CSV_PATH.with_suffix(".csv.bak")
shutil.copy2(CSV_PATH, backup_path)
print(f"📦 원본 CSV 백업 완료: {backup_path}")

# =====================================================================
# 2. 메타데이터 로드 및 검사
# =====================================================================
df = pd.read_csv(CSV_PATH)
total_count = len(df)

true_count = 0
false_count = 0

status_list = []

for idx, row in df.iterrows():
    common_path = str(row.get("common_path", "")).strip()

    if not common_path or common_path == "nan":
        status_list.append(False)
        false_count += 1
        continue

    # 검사 대상 파일 경로
    json_path = SAM_DIR / f"{common_path}.json"
    mask_path = SAM_DIR / f"{common_path}_masks.npz"

    # 둘 다 디스크에 실제로 존재하는 경우만 True
    is_valid = json_path.exists() and mask_path.exists()
    status_list.append(is_valid)

    if is_valid:
        true_count += 1
    else:
        false_count += 1

# =====================================================================
# 3. CSV 갱신 및 저장
# =====================================================================
df["sam_done"] = status_list
df.to_csv(CSV_PATH, index=False)

print("\n" + "=" * 50)
print("📊 [sam_done 컬럼 갱신 완료]")
print("=" * 50)
print(f"• 총 검사 항목 : {total_count:,}건")
print(f"• True (완료)  : {true_count:,}건 (JSON 및 Mask NPZ 모두 존재)")
print(f"• False (미완) : {false_count:,}건 (1개 이상 누락 또는 미생성)")
print(f"• 저장 경로    : {CSV_PATH}")
print("=" * 50)