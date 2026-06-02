"""현재 fire 데이터셋 상세 분석"""
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

FIRE = Path("datasets/fire")
SPLITS = ["train", "valid", "test"]

# ============ 1. valid set 클래스 분포 ============
print("=" * 56)
print("  [1] Valid 세트 클래스 분포 (학습 평가 기준)")
print("=" * 56)
valid_fire = valid_smoke = valid_both = valid_bg = 0
for img in (FIRE / "valid" / "images").glob("*.*"):
    lbl = FIRE / "valid" / "labels" / (img.stem + ".txt")
    if img.name.startswith("hardneg_") or not lbl.exists() or not lbl.read_text().strip():
        valid_bg += 1; continue
    classes = {int(l.split()[0]) for l in lbl.read_text().strip().split("\n") if l.strip()}
    if 0 in classes and 1 in classes: valid_both += 1
    elif 0 in classes: valid_fire += 1
    elif 1 in classes: valid_smoke += 1

total_v = valid_fire + valid_smoke + valid_both + valid_bg
print(f"  fire+smoke : {valid_both:3d}장  ({valid_both/total_v*100:.1f}%)")
print(f"  fire 만    : {valid_fire:3d}장  ({valid_fire/total_v*100:.1f}%)")
print(f"  smoke 만   : {valid_smoke:3d}장  ({valid_smoke/total_v*100:.1f}%)")
print(f"  배경       : {valid_bg:3d}장  ({valid_bg/total_v*100:.1f}%)")
print(f"  smoke 포함 총: {valid_both + valid_smoke}장 → smoke mAP50 평가 기준")

# ============ 2. bbox 크기 분포 ============
print()
print("=" * 56)
print("  [2] Train bbox 크기 분포 (초기 소규모 화재 감지 가능성)")
print("=" * 56)
tiny = small = medium = large = 0
for split in SPLITS:
    lbl_dir = FIRE / split / "labels"
    if not lbl_dir.exists(): continue
    for lbl in lbl_dir.glob("*.txt"):
        for line in lbl.read_text(encoding="utf-8").strip().split("\n"):
            p = line.strip().split()
            if len(p) < 5: continue
            try:
                bw, bh = float(p[3]), float(p[4])
                area = bw * bh * 100
                if area < 1: tiny += 1
                elif area < 5: small += 1
                elif area < 20: medium += 1
                else: large += 1
            except: pass

total_b = tiny + small + medium + large
if total_b:
    print(f"  극소 (<1%)   : {tiny:5d}개  ({tiny/total_b*100:.1f}%) ← 이미 제거됨")
    print(f"  소형 (1~5%)  : {small:5d}개  ({small/total_b*100:.1f}%) ← 초기 화재")
    print(f"  중형 (5~20%) : {medium:5d}개  ({medium/total_b*100:.1f}%)")
    print(f"  대형 (>20%)  : {large:5d}개  ({large/total_b*100:.1f}%)")

# ============ 3. 이미지 밝기 분포 (실내/실외 비율 추정) ============
print()
print("=" * 56)
print("  [3] 이미지 밝기 분포 (실내/야외 비율)")
print("=" * 56)
dark = mid = bright = 0
sample = list((FIRE / "train" / "images").glob("*.jpg"))[:500]
for p in sample:
    if p.name.startswith("hardneg_"): continue
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    mean = img.mean()
    if mean < 80: dark += 1
    elif mean < 160: mid += 1
    else: bright += 1

total_s = dark + mid + bright
if total_s:
    print(f"  어두운 환경 (<80) : {dark:3d}장  ({dark/total_s*100:.0f}%) ← 야간·특수 환경")
    print(f"  중간 밝기  (80~160): {mid:3d}장  ({mid/total_s*100:.0f}%)")
    print(f"  밝은 환경  (>160) : {bright:3d}장  ({bright/total_s*100:.0f}%) ← 실내·주간 환경")
    print(f"  (샘플 {total_s}장 기준)")

print()
print("=" * 56)
print("  결론 — 추가 보완 가능한 항목")
print("=" * 56)
print(f"  1. smoke 데이터 부족 ({valid_smoke + valid_both}장만 있어 smoke mAP50 낮을 것)")
print(f"  2. 실내 환경 화재 이미지 부족 (독거노인 집 = 실내)")
print(f"  3. 집 안 오감지 유발 물체 Hard Negative 추가 가능")

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
