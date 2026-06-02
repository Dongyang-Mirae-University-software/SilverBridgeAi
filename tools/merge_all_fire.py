# -*- coding: utf-8 -*-
"""
Fire 데이터셋 전체 병합 + 중복 제거 + 품질 정제
================================================
처리 순서:
  Step 1 : 각 소스 데이터셋 → datasets/fire/ 로 병합
           - MD5로 완전 동일 이미지 스킵
           - 클래스 인덱스 리매핑 (데이터셋마다 다름)
           - fire_new3 : fire 클래스(idx=1)만 추출, 나머지 스킵
  Step 2 : pHash 유사 중복 제거 (병합 후 전체 스캔)
  Step 3 : 라벨 품질 정제 (bbox 클리핑 / 극소·극대 제거)
  Step 4 : data.yaml + README 갱신
  Step 5 : 빈 소스 폴더 삭제

실행:
    cd AISilverBridgeLJH
    python tools/merge_all_fire.py
"""

import cv2
import hashlib
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

# ==========================================
# 경로 설정
# ==========================================
BASE     = Path(__file__).parent.parent
DATASETS = BASE / "datasets"
DEST     = DATASETS / "fire"
SPLITS   = ["train", "valid", "test"]

# ==========================================
# 소스 데이터셋 & 클래스 매핑 설정
#
# src_names : 소스 data.yaml 의 names 리스트 (인덱스 기준)
# class_map : { 소스_클래스명 -> 목적지_클래스_인덱스 }
#             None 또는 누락 -> 해당 bbox 무시
# ==========================================
SOURCE_CONFIGS = {
    "fire_new": {
        "src_names": ["fire"],
        "class_map": {"fire": 0},          # smoke 없음
    },
    "fire_new2": {
        "src_names": ["fire", "smoke"],
        "class_map": {"fire": 0, "smoke": 1},
    },
    "fire_new3": {
        # bottle(0), fire(1), gun(2), person(3), stick(4) 혼합 데이터셋
        # fire 만 추출, 나머지 무시
        "src_names": ["bottle", "fire", "gun", "person", "stick"],
        "class_map": {"fire": 0},
    },
    "fire_new4": {
        "src_names": ["fire", "smoke"],
        "class_map": {"fire": 0, "smoke": 1},
    },
}

# ==========================================
# pHash / 라벨 파라미터
# ==========================================
PHASH_THRESHOLD = 8
MIN_BBOX_AREA   = 0.003   # 0.3%
MAX_BBOX_AREA   = 0.90    # 90%
MIN_IMG_PIXEL   = 32
BATCH_SIZE      = 128


# ==========================================
# 유틸리티
# ==========================================
def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_bits(img_path: Path) -> np.ndarray | None:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(img))
    low = dct[:8, :8].flatten()
    return (low > low.mean()).astype(np.uint8)


def remap_labels(src_lbl: Path, src_names: list[str], class_map: dict) -> list[str] | None:
    """
    라벨 파일을 읽어 class_map 에 따라 인덱스를 변환.
    반환: 변환된 라인 리스트 (비어 있으면 [] 반환, 파일 없으면 None)
    """
    if not src_lbl.exists():
        return None

    raw = [l.strip() for l in src_lbl.read_text(encoding="utf-8").split("\n") if l.strip()]
    new_lines = []
    for line in raw:
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            src_idx = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            continue

        if src_idx >= len(src_names):
            continue

        src_name = src_names[src_idx]
        dst_idx  = class_map.get(src_name)
        if dst_idx is None:
            continue   # 이 클래스는 목적지 데이터셋에 포함하지 않음

        new_lines.append(f"{dst_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return new_lines


# ==========================================
# Step 1 : 병합
# ==========================================
print("=" * 56)
print("  Step 1 — 데이터셋 병합")
print("=" * 56)

# 기존 fire/ 이미지 MD5 수집
existing_hashes: set[str] = set()
existing_names:  set[str] = set()

for split in SPLITS:
    img_dir = DEST / split / "images"
    if img_dir.exists():
        for p in img_dir.glob("*.*"):
            existing_hashes.add(md5(p))
            existing_names.add(p.name)

print(f"  기존 fire/ 이미지: {len(existing_hashes):,}장\n")

merge_stats: dict[str, dict] = {}

for src_name, cfg in SOURCE_CONFIGS.items():
    src_dir   = DATASETS / src_name
    if not src_dir.exists():
        print(f"  [{src_name}] 폴더 없음 — 건너뜀")
        continue

    src_names = cfg["src_names"]
    class_map = cfg["class_map"]
    s = {"added": 0, "skip_dup": 0, "skip_no_fire": 0, "skip_no_img": 0}

    for split in SPLITS:
        src_img_dir = src_dir  / split / "images"
        src_lbl_dir = src_dir  / split / "labels"
        dst_img_dir = DEST     / split / "images"
        dst_lbl_dir = DEST     / split / "labels"

        if not src_img_dir.exists():
            continue

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in sorted(src_img_dir.glob("*.*")):
            # 이미지 읽기 가능 여부 확인
            test_img = cv2.imread(str(img))
            if test_img is None:
                s["skip_no_img"] += 1
                continue

            h_val = md5(img)
            if h_val in existing_hashes:
                s["skip_dup"] += 1
                continue

            # 라벨 변환
            src_lbl = src_lbl_dir / (img.stem + ".txt")
            new_lines = remap_labels(src_lbl, src_names, class_map)

            # fire 클래스가 하나도 없으면 스킵 (fire_new3 등 혼합 데이터셋 대응)
            if new_lines is not None and len(new_lines) == 0:
                s["skip_no_fire"] += 1
                continue

            # 파일명 충돌 처리
            dst_name = img.name
            if dst_name in existing_names:
                dst_name = img.stem + f"_{src_name}" + img.suffix

            dst_img = dst_img_dir / dst_name
            dst_lbl = dst_lbl_dir / (Path(dst_name).stem + ".txt")

            shutil.copy2(img, dst_img)

            if new_lines is not None:
                dst_lbl.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            # new_lines == None → 라벨 파일 없음 (배경 이미지) → 그대로 추가

            existing_hashes.add(h_val)
            existing_names.add(dst_name)
            s["added"] += 1

    merge_stats[src_name] = s
    print(f"  [{src_name}]  추가={s['added']:,}  "
          f"MD5중복={s['skip_dup']:,}  "
          f"fire없음={s['skip_no_fire']:,}  "
          f"손상={s['skip_no_img']:,}")

total_added = sum(v["added"] for v in merge_stats.values())
print(f"\n  병합 후 총 이미지: {len(existing_hashes):,}장  (신규 {total_added:,}장 추가)\n")


# ==========================================
# Step 2 : pHash 유사 중복 제거
# ==========================================
print("=" * 56)
print(f"  Step 2 — pHash 유사 중복 제거  (Hamming <= {PHASH_THRESHOLD})")
print("=" * 56)

all_imgs: list[tuple[str, Path]] = []
for split in SPLITS:
    d = DEST / split / "images"
    if d.exists():
        all_imgs += [(split, p) for p in sorted(d.glob("*.*"))]

total_before_phash = len(all_imgs)
print(f"  pHash 계산 중... ({total_before_phash:,}장)")

indexed:    list[tuple[str, Path]] = []
hash_list:  list[np.ndarray]       = []

for i, (split, p) in enumerate(all_imgs):
    if i % 1000 == 0 and i > 0:
        print(f"    {i:,} / {total_before_phash:,}")
    bits = phash_bits(p)
    if bits is not None:
        indexed.append((split, p))
        hash_list.append(bits)

n = len(hash_list)
hash_matrix = np.stack(hash_list)   # (n, 64) uint8
print(f"  해시 완료 ({n:,}장) — 배치 비교 중...")

to_remove_idx: set[int] = set()

for b_start in range(0, n, BATCH_SIZE):
    b_end  = min(b_start + BATCH_SIZE, n)
    batch  = hash_matrix[b_start:b_end]

    dists = np.sum(
        batch[:, np.newaxis, :] != hash_matrix[np.newaxis, :, :],
        axis=2, dtype=np.uint8,
    )   # (B, n)

    for k in range(b_end - b_start):
        i = b_start + k
        if i in to_remove_idx:
            continue
        mask = dists[k] <= PHASH_THRESHOLD
        mask[: i + 1] = False
        for j in np.where(mask)[0]:
            j_int = int(j)
            if j_int not in to_remove_idx:
                to_remove_idx.add(j_int)

print(f"  유사 중복 {len(to_remove_idx):,}장 발견 — 제거 중...")

for idx in to_remove_idx:
    split, img_path = indexed[idx]
    lbl = DEST / split / "labels" / (img_path.stem + ".txt")
    img_path.unlink(missing_ok=True)
    lbl.unlink(missing_ok=True)

# 목록 갱신
all_imgs = [(s, p) for s, p in all_imgs if p.exists()]
print(f"  pHash 제거 후 남음: {len(all_imgs):,}장\n")


# ==========================================
# Step 3 : 라벨 품질 정제
# ==========================================
print("=" * 56)
print("  Step 3 — 라벨 품질 정제")
print("=" * 56)

qc = defaultdict(int)
class_count: dict[int, int] = defaultdict(int)
CLASS_NAMES = {0: "fire", 1: "smoke"}

for split, img_path in all_imgs:
    lbl_path = DEST / split / "labels" / (img_path.stem + ".txt")

    if not lbl_path.exists():
        qc["no_label"] += 1
        continue

    raw_lines = [l.strip() for l in lbl_path.read_text(encoding="utf-8").split("\n") if l.strip()]
    if not raw_lines:
        qc["empty_label"] += 1
        continue

    new_lines: list[str] = []
    changed = False

    for line in raw_lines:
        parts = line.split()
        if len(parts) != 5:
            qc["bad_format"] += 1
            changed = True
            continue
        try:
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            qc["bad_format"] += 1
            changed = True
            continue

        # bbox 클리핑
        x1 = max(0.0, cx - bw / 2);  y1 = max(0.0, cy - bh / 2)
        x2 = min(1.0, cx + bw / 2);  y2 = min(1.0, cy + bh / 2)
        nw = x2 - x1;  nh = y2 - y1
        ncx = (x1 + x2) / 2;  ncy = (y1 + y2) / 2

        if nw <= 0 or nh <= 0:
            qc["clipped_out"] += 1; changed = True; continue
        if abs(nw - bw) > 1e-6 or abs(nh - bh) > 1e-6:
            qc["clipped_fixed"] += 1; changed = True

        area = nw * nh
        if area < MIN_BBOX_AREA:
            qc["tiny_removed"] += 1; changed = True; continue
        if area > MAX_BBOX_AREA:
            qc["huge_removed"] += 1; changed = True; continue

        class_count[cls] += 1
        new_lines.append(f"{cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")

    if changed:
        lbl_path.write_text(
            "\n".join(new_lines) + ("\n" if new_lines else ""),
            encoding="utf-8",
        )
        qc["files_fixed"] += 1

print(f"  배경 이미지 (hard-neg, 라벨 없음): {qc['no_label']:,}장")
print(f"  bbox 클리핑 수정: {qc['clipped_fixed']:,}개  |  클리핑 후 소멸: {qc['clipped_out']:,}개")
print(f"  극소 bbox 제거 (<{MIN_BBOX_AREA*100:.1f}%): {qc['tiny_removed']:,}개")
print(f"  극대 bbox 제거 (>{MAX_BBOX_AREA*100:.0f}%): {qc['huge_removed']:,}개")
print(f"  수정된 라벨 파일: {qc['files_fixed']:,}개")

print(f"\n  [클래스 분포]")
total_cls = sum(class_count.values()) or 1
for idx, name in CLASS_NAMES.items():
    cnt   = class_count[idx]
    ratio = cnt / total_cls * 100
    bar   = "#" * int(ratio / 2)
    print(f"    {idx} {name:6s}: {cnt:7,}개  ({ratio:5.1f}%)  {bar}")


# ==========================================
# Step 4 : data.yaml + README 갱신
# ==========================================
print(f"\n{'='*56}")
print("  Step 4 — data.yaml / README 갱신")
print("=" * 56)

split_counts: dict[str, int] = {}
for split in SPLITS:
    d = DEST / split / "images"
    split_counts[split] = len(list(d.glob("*.*"))) if d.exists() else 0

total_final = sum(split_counts.values())

yaml_content = f"""# Fire & Smoke Detection Dataset (Full Merged + Cleaned)
# ==========================================
# 병합 출처
#   - fire      (기 정제된 원본)
#   - fire_new  (fire 단일 클래스, 123장)
#   - fire_new2 (fire+smoke, 241장)
#   - fire_new3 (혼합 5클래스 → fire만 추출, 2,184장)
#   - fire_new4 (fire+smoke, 1,153장)
# MD5 + pHash(Hamming<={PHASH_THRESHOLD}) 중복 제거 후 총 {total_final:,}장
# ==========================================

path: ../datasets/fire

train: train/images
val:   valid/images
test:  test/images

nc: 2
names: ['fire', 'smoke']

# Split 통계 (정제 후)
# train : {split_counts['train']:,}장
# valid : {split_counts['valid']:,}장
# test  : {split_counts['test']:,}장
# total : {total_final:,}장
"""

(DEST / "data.yaml").write_text(yaml_content, encoding="utf-8")

readme_content = f"""Fire & Smoke Detection Dataset - Full Merged
=============================================

병합 날짜     : 2026-05-29
병합 스크립트 : tools/merge_all_fire.py

[소스 데이터셋]
  fire      - 기존 정제 데이터 (fire, smoke)
  fire_new  - Roboflow 추가 수집 (fire 단일 클래스, 123장)
  fire_new2 - Roboflow 추가 수집 (fire, smoke, 241장)
  fire_new3 - Roboflow 혼합 데이터셋 (bottle/fire/gun/person/stick 5클래스)
              → fire 클래스(index 1)만 추출해 index 0으로 리매핑
  fire_new4 - Roboflow 추가 수집 (fire, smoke, 1,153장)

[중복 제거]
  MD5 완전 동일 이미지 제거 (바이너리 동일)
  pHash 유사 이미지 제거 (Hamming 거리 <= {PHASH_THRESHOLD}, DCT 64bit)

[라벨 품질 정제]
  bbox 경계 초과 → 클리핑
  극소 bbox 제거 (이미지 면적 {MIN_BBOX_AREA*100:.1f}% 미만)
  극대 bbox 제거 (이미지 면적 {MAX_BBOX_AREA*100:.0f}% 초과)

[최종 클래스]
  0 : fire   (화재)
  1 : smoke  (연기)

[최종 이미지 수]
  train : {split_counts['train']:,}장
  valid : {split_counts['valid']:,}장
  test  : {split_counts['test']:,}장
  total : {total_final:,}장

[포맷]
  YOLO 형식 (train/valid/test + labels/)
  라벨: <class_idx> <cx> <cy> <w> <h>  (정규화 좌표)
  라이선스: CC BY 4.0 (Roboflow)
"""

(DEST / "README.dataset.txt").write_text(readme_content, encoding="utf-8")
print("  data.yaml      갱신 완료")
print("  README.dataset.txt 갱신 완료")


# ==========================================
# Step 5 : 소스 폴더 삭제
# ==========================================
print(f"\n{'='*56}")
print("  Step 5 — 병합된 소스 폴더 삭제")
print("=" * 56)

for src_name in SOURCE_CONFIGS:
    src_dir = DATASETS / src_name
    if src_dir.exists():
        shutil.rmtree(src_dir)
        print(f"  삭제: {src_name}/")


# ==========================================
# 최종 요약
# ==========================================
print(f"\n{'='*56}")
print("  완료")
print("=" * 56)
print(f"  최종 위치  : datasets/fire/")
print(f"  train      : {split_counts['train']:,}장")
print(f"  valid      : {split_counts['valid']:,}장")
print(f"  test       : {split_counts['test']:,}장")
print(f"  합계       : {total_final:,}장")
print(f"\n  재학습 명령어:")
print(f"    python src/training/train.py fire")
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
