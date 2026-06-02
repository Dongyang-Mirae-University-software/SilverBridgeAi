# -*- coding: utf-8 -*-
"""
Fire 최종 데이터셋 빌드 — 독거노인 안전 시스템
================================================
소스:
  datasets/fire/     : 기존 정제 데이터 (5,958장, Hard Neg 730장 포함)
  datasets/fire_new/ : 22,751장 (class 0 = fire → 0)
  datasets/smoke/    : 10,042장 (Fire=0→0, Other=1→스킵, Smoke=2→1)

목표:
  - 화재·연기 모두 높은 인식률
  - Hard Negative로 오감지 최소화
  - MD5 + pHash 중복 제거로 다양성 확보

처리 순서:
  1. 소스 데이터셋 분석 및 클래스 매핑
  2. 기존 fire/ 를 base로 fire_new + smoke 병합
     - MD5 중복 스킵
     - 클래스 인덱스 변환
     - smoke의 Other(1) 클래스 → 스킵 (라벨 없는 배경 Hard Neg)
  3. pHash 유사 중복 제거
  4. 라벨 품질 정제
     - 좌표 클리핑
     - 빈 라벨 이미지 제거 (이번엔 엄격하게)
  5. data.yaml 갱신
"""

import cv2
import csv
import hashlib
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

# ==========================================
# 경로 설정
# ==========================================
BASE     = Path(__file__).parent.parent
DEST     = BASE / "datasets" / "fire"
SPLITS   = ["train", "valid", "test"]

# ==========================================
# 소스별 클래스 매핑
# src_names : 소스 data.yaml 의 names
# class_map : { src_index -> dst_index }  None = 스킵
# ==========================================
SOURCES = {
    "fire_new": {
        "src_names": ["FireDataSeason2 - v1 2024-06-03 11-25am"],
        "class_map": {0: 0},          # fire → fire
    },
    "smoke": {
        "src_names": ["Fire", "Other", "Smoke"],
        "class_map": {0: 0, 1: None, 2: 1},  # Fire→fire, Other→스킵, Smoke→smoke
    },
}

# ==========================================
# pHash 파라미터
# ==========================================
PHASH_THRESHOLD    = 8
BATCH_SIZE         = 128
MIN_BBOX_AREA      = 0.002   # 0.2% 미만 → 제거
MAX_BBOX_AREA      = 0.92    # 92% 초과 → 제거

# ==========================================
# 유틸리티
# ==========================================
def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def phash_bits(img) -> np.ndarray | None:
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct  = cv2.dct(np.float32(gray))
    low  = dct[:8, :8].flatten()
    return (low > low.mean()).astype(np.uint8)

def phash_from_file(p: Path) -> np.ndarray | None:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return phash_bits(img) if img is not None else None

def remap_labels(src_lbl: Path, class_map: dict) -> list[str] | None:
    if not src_lbl.exists(): return None
    raw = [l.strip() for l in src_lbl.read_text(encoding="utf-8").split("\n") if l.strip()]
    new_lines = []
    for line in raw:
        parts = line.split()
        if len(parts) < 5: continue
        try:
            src_idx = int(parts[0])
            vals    = list(map(float, parts[1:]))
        except ValueError:
            continue
        dst_idx = class_map.get(src_idx)
        if dst_idx is None: continue
        new_lines.append(f"{dst_idx} " + " ".join(f"{v:.6f}" for v in vals))
    return new_lines


# ==========================================
# Step 1: 기존 fire/ MD5 해시 수집
# ==========================================
print("=" * 60)
print("  Step 1 — 기존 fire/ 이미지 MD5 수집")
print("=" * 60)

existing_hashes: set[str] = set()
existing_names:  set[str] = set()

for split in SPLITS:
    img_dir = DEST / split / "images"
    if img_dir.exists():
        for p in img_dir.glob("*.*"):
            existing_hashes.add(md5(p))
            existing_names.add(p.name)

print(f"  기존 fire/ : {len(existing_hashes):,}장\n")


# ==========================================
# Step 2: 소스 데이터셋 병합
# ==========================================
print("=" * 60)
print("  Step 2 — 소스 데이터셋 병합")
print("=" * 60)

merge_stats: dict[str, dict] = {}

for src_name, cfg in SOURCES.items():
    src_dir   = BASE / "datasets" / src_name
    class_map = cfg["class_map"]

    if not src_dir.exists():
        print(f"  [{src_name}] 폴더 없음 — 건너뜀")
        continue

    s = {"added": 0, "skip_md5": 0, "skip_no_target": 0, "renamed": 0}

    for split in SPLITS:
        src_img = src_dir / split / "images"
        src_lbl = src_dir / split / "labels"
        dst_img = DEST    / split / "images"
        dst_lbl = DEST    / split / "labels"

        if not src_img.exists(): continue
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        for img in sorted(src_img.glob("*.*")):
            h = md5(img)
            if h in existing_hashes:
                s["skip_md5"] += 1; continue

            new_lines = remap_labels(src_lbl / (img.stem + ".txt"), class_map)

            # 유효한 라벨이 하나도 없고 라벨 파일이 존재할 때만 스킵
            # (라벨 파일 없음 → 배경 이미지 → 유지)
            if new_lines is not None and len(new_lines) == 0:
                # smoke의 Other-only 이미지 → 배경 Hard Neg 로 활용
                # 단 라벨 파일은 빈 파일로 생성
                pass  # new_lines = [] → 빈 라벨로 저장

            # 파일명 충돌 처리
            dst_name = img.name
            if dst_name in existing_names:
                dst_name = img.stem + f"_{src_name}" + img.suffix
                s["renamed"] += 1

            dst_img_path = dst_img / dst_name
            dst_lbl_path = dst_lbl / (Path(dst_name).stem + ".txt")

            shutil.copy2(img, dst_img_path)
            if new_lines is not None:
                dst_lbl_path.write_text(
                    "\n".join(new_lines) + ("\n" if new_lines else ""),
                    encoding="utf-8"
                )

            existing_hashes.add(h)
            existing_names.add(dst_name)
            s["added"] += 1

    merge_stats[src_name] = s
    print(f"  [{src_name}]  추가={s['added']:,}  MD5중복={s['skip_md5']:,}  이름변경={s['renamed']:,}")

total_after_merge = len(existing_hashes)
print(f"\n  병합 후 총 이미지: {total_after_merge:,}장\n")


# ==========================================
# Step 3: pHash 유사 중복 제거
# ==========================================
print("=" * 60)
print(f"  Step 3 — pHash 유사 중복 제거 (Hamming ≤ {PHASH_THRESHOLD})")
print("=" * 60)

all_imgs: list[tuple[str, Path]] = []
for split in SPLITS:
    d = DEST / split / "images"
    if d.exists():
        all_imgs += [(split, p) for p in sorted(d.glob("*.*"))]

n_before = len(all_imgs)
print(f"  pHash 계산 중... ({n_before:,}장)")

indexed:   list[tuple[str, Path]] = []
hash_list: list[np.ndarray]       = []

for i, (split, p) in enumerate(all_imgs):
    if i % 2000 == 0 and i > 0:
        print(f"    {i:,} / {n_before:,}")
    bits = phash_from_file(p)
    if bits is not None:
        indexed.append((split, p))
        hash_list.append(bits)

n = len(hash_list)
hash_matrix = np.stack(hash_list)
print(f"  해시 완료 ({n:,}장) — 배치 비교 중...")

to_remove: set[int] = set()
for b_start in range(0, n, BATCH_SIZE):
    b_end  = min(b_start + BATCH_SIZE, n)
    batch  = hash_matrix[b_start:b_end]
    dists  = np.sum(
        batch[:, np.newaxis, :] != hash_matrix[np.newaxis, :, :],
        axis=2, dtype=np.uint8
    )
    for k in range(b_end - b_start):
        i = b_start + k
        if i in to_remove: continue
        mask = dists[k] <= PHASH_THRESHOLD
        mask[:i + 1] = False
        for j in np.where(mask)[0]:
            if int(j) not in to_remove:
                to_remove.add(int(j))

print(f"  유사 중복 {len(to_remove):,}장 제거 중...")
for idx in to_remove:
    split, img_path = indexed[idx]
    lbl = DEST / split / "labels" / (img_path.stem + ".txt")
    img_path.unlink(missing_ok=True)
    lbl.unlink(missing_ok=True)

all_imgs = [(s, p) for s, p in all_imgs if p.exists()]
print(f"  pHash 제거 후 남음: {len(all_imgs):,}장\n")


# ==========================================
# Step 4: 라벨 품질 정제 + 빈 라벨 제거
# ==========================================
print("=" * 60)
print("  Step 4 — 라벨 품질 정제")
print("=" * 60)

qc = defaultdict(int)
class_count: dict[int, int] = defaultdict(int)

imgs_to_delete = []

for split, img_path in all_imgs:
    lbl_path = DEST / split / "labels" / (img_path.stem + ".txt")

    # Hard Negative — 라벨 파일 없음 → 유지
    if img_path.name.startswith("hardneg_"):
        qc["hardneg"] += 1
        continue

    # 라벨 파일 없음
    if not lbl_path.exists():
        # smoke의 Other-only → 배경 Hard Neg로 활용 (유지)
        qc["no_label_kept"] += 1
        continue

    content = lbl_path.read_text(encoding="utf-8").strip()

    # 빈 라벨 파일 → "불 있는데 없음" 오염 방지 → 이미지 삭제
    if not content:
        imgs_to_delete.append((split, img_path, lbl_path))
        qc["empty_deleted"] += 1
        continue

    new_lines: list[str] = []
    changed = False

    for line in content.split("\n"):
        parts = line.strip().split()
        if len(parts) < 5: continue
        try:
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
        except ValueError:
            changed = True; continue

        # 좌표 클리핑
        x1 = max(0.0, cx - bw / 2);  y1 = max(0.0, cy - bh / 2)
        x2 = min(1.0, cx + bw / 2);  y2 = min(1.0, cy + bh / 2)
        nw = x2 - x1;  nh = y2 - y1
        ncx = (x1 + x2) / 2;  ncy = (y1 + y2) / 2

        if nw <= 0 or nh <= 0:
            qc["clipped_out"] += 1; changed = True; continue

        area = nw * nh
        if area < MIN_BBOX_AREA:
            qc["tiny_removed"] += 1; changed = True; continue
        if area > MAX_BBOX_AREA:
            qc["huge_removed"] += 1; changed = True; continue

        class_count[cls] += 1
        new_lines.append(f"{cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")

    # 모든 bbox 제거됨 → 이미지 삭제 (빈 라벨 오염 방지)
    if not new_lines:
        imgs_to_delete.append((split, img_path, lbl_path))
        qc["all_bbox_removed"] += 1
        continue

    if changed:
        lbl_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        qc["files_fixed"] += 1

# 삭제 실행
for split, img_path, lbl_path in imgs_to_delete:
    img_path.unlink(missing_ok=True)
    lbl_path.unlink(missing_ok=True)

CLASS_NAMES = {0: "fire", 1: "smoke"}
print(f"  빈 라벨 삭제        : {qc['empty_deleted']:,}장")
print(f"  bbox 전부 제거 삭제 : {qc['all_bbox_removed']:,}장")
print(f"  Hard Negative 유지  : {qc['hardneg']:,}장")
print(f"  클리핑 수정         : {qc['clipped_out'] + qc['files_fixed']:,}개")
print(f"  극소 bbox 제거      : {qc['tiny_removed']:,}개")
print(f"  극대 bbox 제거      : {qc['huge_removed']:,}개")

total_cls = sum(class_count.values()) or 1
print(f"\n  [최종 클래스 분포]")
for idx, name in CLASS_NAMES.items():
    cnt   = class_count[idx]
    pct   = cnt / total_cls * 100
    bar   = "█" * int(pct / 2)
    print(f"    {idx} {name:6s}: {cnt:7,}개  ({pct:5.1f}%)  {bar}")


# ==========================================
# Step 5: 최종 통계 및 data.yaml 갱신
# ==========================================
print(f"\n{'='*60}")
print("  Step 5 — 최종 통계 및 data.yaml")
print("=" * 60)

split_counts: dict[str, int] = {}
for split in SPLITS:
    d = DEST / split / "images"
    split_counts[split] = len(list(d.glob("*.*"))) if d.exists() else 0

total_final = sum(split_counts.values())
hn_count    = len([p for p in (DEST / "train" / "images").glob("hardneg_*.jpg")])

for split in SPLITS:
    print(f"  [{split:5s}] {split_counts[split]:,}장")
print(f"  합계   : {total_final:,}장")

yaml_content = f"""# Fire & Smoke Detection Dataset — 독거노인 안전 시스템 최종본
# ==========================================
# 소스 데이터셋
#   fire/     : 기존 정제 데이터 + Hard Neg {hn_count}장
#   fire_new/ : 화재 전용 대규모 (22,751장, CC BY 4.0)
#   smoke/    : Fire+Smoke 혼합 (10,042장, CC BY 4.0)
#              Fire(0)→fire, Other(1)→스킵, Smoke(2)→smoke
#
# 품질 처리
#   MD5 완전 중복 제거
#   pHash 유사 중복 제거 (Hamming ≤ {PHASH_THRESHOLD})
#   빈 라벨 이미지 삭제 (fire 있는데 없음으로 학습 방지)
#   bbox 클리핑 / 극소({MIN_BBOX_AREA*100:.1f}%)·극대({MAX_BBOX_AREA*100:.0f}%) 제거
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

(DEST / "data.yaml").write_text(yaml_content, encoding="utf-8")
print("\n  data.yaml 갱신 완료")

# 소스 폴더 삭제
for src_name in SOURCES:
    src_dir = BASE / "datasets" / src_name
    if src_dir.exists():
        shutil.rmtree(src_dir)
        print(f"  삭제: datasets/{src_name}/")

print(f"\n{'='*60}")
print("  완료")
print("=" * 60)
print(f"  train : {split_counts.get('train', 0):,}장")
print(f"  valid : {split_counts.get('valid', 0):,}장")
print(f"  test  : {split_counts.get('test', 0):,}장")
print(f"  합계  : {total_final:,}장")
print(f"\n  재학습:")
print(f"    python src/training/train.py fire --epochs 250")

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
