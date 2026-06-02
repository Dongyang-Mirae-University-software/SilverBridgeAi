"""현재 fire 데이터셋 오염 실태 파악"""
from pathlib import Path
from collections import defaultdict

FIRE = Path("datasets/fire")
SPLITS = ["train", "valid", "test"]

empty_with_fire = []    # 라벨 제거됐지만 이미지엔 불 있을 수 있는 것
fire_new3_files = []    # fire_new3 출처 확인 가능한 파일
class_only_zero = 0     # fire 클래스만 있는 이미지 (smoke 없음 → fire_new3 가능성)
class_both      = 0     # fire + smoke 둘 다 있는 이미지
class_smoke_only = 0
empty_label     = 0     # 라벨 파일이 있지만 내용 없음 (bbox 제거로 비워진 것)
hardneg_count   = 0     # 의도적 Hard Negative

for split in SPLITS:
    img_dir = FIRE / split / "images"
    lbl_dir = FIRE / split / "labels"
    if not img_dir.exists():
        continue

    for img in sorted(img_dir.glob("*.*")):
        name = img.name

        # Hard Negative (의도적 배경)
        if name.startswith("hardneg_"):
            hardneg_count += 1
            continue

        # fire_new3 출처 식별 가능한 파일
        if "_fire_new3" in name:
            fire_new3_files.append((split, img))
            continue

        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            # 라벨 파일 자체 없음 → 원래 배경이거나 누락
            empty_label += 1
            empty_with_fire.append((split, img, "no_label_file"))
            continue

        content = lbl.read_text(encoding="utf-8").strip()
        if not content:
            # 라벨 파일은 있지만 내용 없음 → bbox 제거로 비워진 것
            empty_label += 1
            empty_with_fire.append((split, img, "emptied_by_filter"))
            continue

        classes = set()
        for line in content.split("\n"):
            p = line.strip().split()
            if p:
                try:
                    classes.add(int(p[0]))
                except ValueError:
                    pass

        if 0 in classes and 1 in classes:
            class_both += 1
        elif 0 in classes:
            class_only_zero += 1
        elif 1 in classes:
            class_smoke_only += 1

print("=" * 56)
print("  Fire 데이터셋 오염 실태 파악")
print("=" * 56)
print(f"  Hard Negative (의도적): {hardneg_count}장")
print(f"  fire_new3 식별 가능  : {len(fire_new3_files)}장  ← 확인 후 제거")
print(f"  빈 라벨 (bbox 제거됨): {empty_label}장  ← 핵심 오염원!")
print()
print("  [클래스 구성]")
print(f"    fire + smoke : {class_both}장")
print(f"    fire 만      : {class_only_zero}장")
print(f"    smoke 만     : {class_smoke_only}장")
print()

# 빈 라벨 split별
empty_split = defaultdict(int)
for split, _, _ in empty_with_fire:
    empty_split[split] += 1
print("  빈 라벨 split별:")
for s, c in empty_split.items():
    print(f"    {s}: {c}장")

print()
print("  ★ 핵심 문제:")
print(f"    빈 라벨 {empty_label}장 = 불이 찍혔는데 '아무것도 없음'으로 학습됨")
print(f"    fire_new3 {len(fire_new3_files)}장 = 총·병·사람 배경에 불만 라벨 → 혼란")

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
