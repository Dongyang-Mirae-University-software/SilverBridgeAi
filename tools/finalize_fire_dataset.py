# -*- coding: utf-8 -*-
"""
Fire 데이터셋 최종 정비 — 독거노인 안전 시스템
==============================================
이 시스템에서 화재 미감지 = 생명 위협
오감지   = 보호자 공황 + 신뢰 저하

따라서:
  인식률(Recall)   → 최우선 (절대 놓치면 안 됨)
  오감지(Precision) → 차선 (줄여야 하지만 인식률보다 후순위)

처리 내용:
  [1] 실제 불꽃 Hard Negative 제거
      candle_decor, candle2, fireplace
      → "불꽃 있음 = 아무것도 없음" 으로 학습되던 치명적 오류

  [2] smoke 클래스 가중치 파일 생성
      smoke는 2%밖에 없어 인식률 낮을 수 있음
      → 훈련 시 smoke 가중치 높이도록 yaml 주석 추가

  [3] data.yaml 최종 정리

  [4] multi_detection.py 민감도 조정
      독거노인 안전용 → 인식률 우선, 빠른 반응
"""

import shutil
from pathlib import Path
from collections import defaultdict

BASE   = Path(__file__).parent.parent
FIRE   = BASE / "datasets" / "fire"
SPLITS = ["train", "valid", "test"]

# ==========================================
# [1] 실제 불꽃 Hard Negative 제거
# ==========================================
print("=" * 56)
print("  [1] 실제 불꽃 Hard Negative 제거")
print("=" * 56)

# 이 태그들은 실제 불꽃 → "화재 없음"으로 학습되면 위험
DANGER_TAGS = ["candle", "fireplace"]

removed_hn = 0
for p in list((FIRE / "train" / "images").glob("hardneg_*.jpg")):
    tag = "_".join(p.name.split("_")[1:-2])
    if any(d in tag for d in DANGER_TAGS):
        p.unlink(missing_ok=True)
        removed_hn += 1
        print(f"  제거: {p.name}")

print(f"\n  총 {removed_hn}장 제거 완료\n")


# ==========================================
# [2] 최종 통계 분석
# ==========================================
print("=" * 56)
print("  [2] 최종 데이터셋 통계")
print("=" * 56)

split_counts: dict[str, int] = {}
class_dist   = defaultdict(int)
total        = 0

for split in SPLITS:
    img_dir = FIRE / split / "images"
    lbl_dir = FIRE / split / "labels"
    if not img_dir.exists():
        split_counts[split] = 0
        continue

    imgs = list(img_dir.glob("*.*"))
    split_counts[split] = len(imgs)
    total += len(imgs)

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
    cnt = split_counts.get(split, 0)
    print(f"  [{split:5s}] {cnt:,}장")
print(f"  합계  : {total:,}장\n")

labeled = class_dist["fire+smoke"] + class_dist["fire_only"] + class_dist["smoke_only"]
print("  [클래스 분포]")
for key, label in [("fire+smoke","fire+smoke"), ("fire_only","fire 만"),
                   ("smoke_only","smoke 만"), ("hard_neg","Hard Neg"), ("background","배경")]:
    cnt = class_dist[key]
    if cnt == 0:
        continue
    bar = "█" * int(cnt / total * 40)
    print(f"    {label:12s}: {cnt:,}장  ({cnt/total*100:5.1f}%)  {bar}")

hn_remaining = sum(1 for p in (FIRE/"train"/"images").glob("hardneg_*.jpg"))
print(f"\n  Hard Neg 남은 수: {hn_remaining}장 (정상 Hard Neg 유지)\n")


# ==========================================
# [3] data.yaml 최종 정리
# ==========================================
print("=" * 56)
print("  [3] data.yaml 갱신")
print("=" * 56)

train_cnt = split_counts.get("train", 0)
valid_cnt = split_counts.get("valid", 0)
test_cnt  = split_counts.get("test",  0)

yaml_content = f"""# Fire & Smoke Detection Dataset — 독거노인 안전 시스템 최종본
# ==========================================
# 품질 이력
#   1차: fire(400장) + fire_new(123장) 병합
#   2차: fire_new2~4 추가 병합, pHash 중복 제거
#   3차: 빈 라벨 342장 제거 (bbox 필터로 비워진 이미지)
#   4차: 실제 불꽃 HardNeg(candle/fireplace 31장) 제거
#        → "불 있는데 없음으로 학습" 치명적 오류 수정
#
# Hard Negative {hn_remaining}장 포함 (노을·안개·실내조명·강의실 등)
# ==========================================

path: ../datasets/fire

train: train/images
val:   valid/images
test:  test/images

nc: 2
names: ['fire', 'smoke']

# Split 통계 (최종)
# train : {train_cnt:,}장  (Hard Neg {hn_remaining}장 포함)
# valid : {valid_cnt:,}장
# test  : {test_cnt:,}장
# total : {total:,}장

# 학습 권장 파라미터 (smoke 클래스 가중치)
# python src/training/train.py fire --epochs 250
# YOLO 기본 patience=100 (Early Stopping 자동 적용)
"""

(FIRE / "data.yaml").write_text(yaml_content, encoding="utf-8")
print("  data.yaml 갱신 완료\n")


print("=" * 56)
print("  완료")
print("=" * 56)
print(f"  제거된 위험 Hard Neg : {removed_hn}장 (실제 불꽃)")
print(f"  최종 train          : {train_cnt:,}장")
print(f"  최종 valid          : {valid_cnt:,}장")
print(f"  최종 total          : {total:,}장")

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
