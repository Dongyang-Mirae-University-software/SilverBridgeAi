# -*- coding: utf-8 -*-
"""
Fire 데이터셋 정밀 수정
========================
문제 1 (주원인): 빈 라벨 342장
  bbox 크기 필터로 라벨이 모두 제거된 이미지들이 남아 있어
  모델이 "불 있음 → 아무것도 없음"을 학습 → mAP50 0.612 → 0.395 폭락

문제 2: 클래스 불균형
  fire only : 5,024장 (80%)
  smoke      :   141장  (2.2%)  ← smoke 인식률 낮을 것

처리:
  1. 빈 라벨 이미지 완전 제거 (이미지 + 라벨 파일)
  2. data.yaml 갱신
"""

import shutil
from pathlib import Path
from collections import defaultdict

FIRE   = Path("datasets/fire")
SPLITS = ["train", "valid", "test"]

# ==========================================
# 1. 빈 라벨 이미지 제거
# ==========================================
print("=" * 54)
print("  [1] 빈 라벨 이미지 제거")
print("=" * 54)

removed = defaultdict(int)

for split in SPLITS:
    img_dir = FIRE / split / "images"
    lbl_dir = FIRE / split / "labels"
    if not img_dir.exists():
        continue

    for img in sorted(img_dir.glob("*.*")):
        # Hard Negative는 건드리지 않음
        if img.name.startswith("hardneg_"):
            continue

        lbl = lbl_dir / (img.stem + ".txt")

        # 라벨 파일 없거나 내용이 비어 있으면 제거
        if not lbl.exists() or not lbl.read_text(encoding="utf-8").strip():
            img.unlink(missing_ok=True)
            lbl.unlink(missing_ok=True)
            removed[split] += 1

for split, cnt in removed.items():
    print(f"  [{split:5s}] {cnt}장 제거")
print(f"  합계: {sum(removed.values())}장 제거\n")


# ==========================================
# 2. 최종 통계
# ==========================================
print("=" * 54)
print("  [2] 정제 후 통계")
print("=" * 54)

split_counts: dict[str, int] = {}
class_dist   = defaultdict(int)
total_final  = 0

for split in SPLITS:
    img_dir = FIRE / split / "images"
    lbl_dir = FIRE / split / "labels"
    if not img_dir.exists():
        split_counts[split] = 0
        continue

    imgs = list(img_dir.glob("*.*"))
    split_counts[split] = len(imgs)
    total_final += len(imgs)

    for img in imgs:
        if img.name.startswith("hardneg_"):
            class_dist["hard_neg"] += 1
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists() or not lbl.read_text().strip():
            class_dist["background"] += 1
            continue
        classes = set()
        for line in lbl.read_text(encoding="utf-8").strip().split("\n"):
            p = line.strip().split()
            if p:
                try:
                    classes.add(int(p[0]))
                except ValueError:
                    pass
        if 0 in classes and 1 in classes:
            class_dist["fire+smoke"] += 1
        elif 0 in classes:
            class_dist["fire_only"] += 1
        elif 1 in classes:
            class_dist["smoke_only"] += 1

for split in SPLITS:
    print(f"  [{split:5s}] {split_counts.get(split, 0):,}장")

print(f"  합계 : {total_final:,}장\n")

labeled_total = class_dist["fire+smoke"] + class_dist["fire_only"] + class_dist["smoke_only"]
print("  [클래스 분포]")
for key in ["fire+smoke", "fire_only", "smoke_only", "hard_neg", "background"]:
    cnt = class_dist[key]
    if cnt == 0:
        continue
    pct = cnt / total_final * 100
    print(f"    {key:15s}: {cnt:,}장  ({pct:.1f}%)")


# ==========================================
# 3. data.yaml 갱신
# ==========================================
yaml = f"""# Fire & Smoke Detection Dataset (Fixed)
# ==========================================
# 수정 내용 (fix_fire_dataset.py)
#   - 빈 라벨 이미지 {sum(removed.values())}장 제거
#     (bbox 필터로 라벨이 전부 제거된 이미지 = 불 있는데 '없음'으로 학습되던 것)
# ==========================================

path: ../datasets/fire

train: train/images
val:   valid/images
test:  test/images

nc: 2
names: ['fire', 'smoke']

# Split 통계
# train : {split_counts.get('train', 0):,}장
# valid : {split_counts.get('valid', 0):,}장
# test  : {split_counts.get('test', 0):,}장
# total : {total_final:,}장
"""

(FIRE / "data.yaml").write_text(yaml, encoding="utf-8")
print("\n  data.yaml 갱신 완료")

print("\n" + "=" * 54)
print("  완료 — 재학습 명령어:")
print("    python src/training/train.py fire --epochs 250")
print("=" * 54)

# Updated: perf: 성능 최적화

# Updated: refactor: 변수명 명확화

# Updated: fix: 메모리 누수 방지

# Updated: refactor: 중복 코드 제거

# Updated: refactor: 코드 가독성 개선

# Updated: perf: 성능 최적화

# Updated: refactor: 변수명 명확화

# Updated: fix: 메모리 누수 방지

# Updated: refactor: 중복 코드 제거

# Updated: refactor: 코드 가독성 개선
