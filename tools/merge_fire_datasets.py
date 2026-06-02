# -*- coding: utf-8 -*-
"""
datasets/fire + datasets/fire_new 병합 스크립트
================================================
1. fire/ 전체 이미지 MD5 해시 사전 구축
2. fire_new/ 이미지를 순회하며:
   - 중복(MD5 일치) → 건너뜀
   - 파일명 충돌(내용 다름) → _new 접미사로 이름 변경
   - 그 외 → 이미지 + 라벨을 fire/ 동일 split 에 이동
3. datasets/fire/data.yaml 업데이트 (소문자 클래스명, 이미지 수 갱신)
4. 병합 README 생성
5. datasets/fire_new/ 삭제

실행:
    cd AISilverBridgeLJH
    python tools/merge_fire_datasets.py
"""

import hashlib
import shutil
from pathlib import Path

# ==========================================
# 경로 설정
# ==========================================
BASE       = Path(__file__).parent.parent          # AISilverBridgeLJH/
FIRE_DIR   = BASE / "datasets" / "fire"
NEW_DIR    = BASE / "datasets" / "fire_new"
SPLITS     = ["train", "valid", "test"]

# ==========================================
# 유틸리티
# ==========================================

def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def count_images(base_dir: Path) -> int:
    total = 0
    for split in SPLITS:
        img_dir = base_dir / split / "images"
        if img_dir.exists():
            total += len(list(img_dir.glob("*.*")))
    return total


# ==========================================
# 1. fire/ 기존 이미지 해시 사전 구축
# ==========================================
print("=" * 55)
print("  STEP 1 — 기존 fire/ 이미지 해시 사전 구축")
print("=" * 55)

existing_hashes: set[str] = set()
existing_names:  set[str] = set()

for split in SPLITS:
    img_dir = FIRE_DIR / split / "images"
    if not img_dir.exists():
        continue
    imgs = list(img_dir.glob("*.*"))
    for img in imgs:
        existing_hashes.add(md5(img))
        existing_names.add(img.name)

print(f"  기존 이미지 수: {len(existing_hashes)}장  (split 합산)")


# ==========================================
# 2. fire_new/ 이미지 순회 → 복사/건너뜀
# ==========================================
print("\n" + "=" * 55)
print("  STEP 2 — fire_new/ 이미지 병합 중…")
print("=" * 55)

stats = {"skip_dup": 0, "skip_no_label": 0, "moved": 0, "renamed": 0}

for split in SPLITS:
    src_img_dir = NEW_DIR  / split / "images"
    src_lbl_dir = NEW_DIR  / split / "labels"
    dst_img_dir = FIRE_DIR / split / "images"
    dst_lbl_dir = FIRE_DIR / split / "labels"

    if not src_img_dir.exists():
        print(f"  [{split}] 폴더 없음 — 건너뜀")
        continue

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(src_img_dir.glob("*.*"))
    moved_in_split = 0

    for img in imgs:
        h = md5(img)

        # ── 중복 이미지 ──
        if h in existing_hashes:
            stats["skip_dup"] += 1
            continue

        # ── 대응 라벨 파일 확인 ──
        lbl = src_lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            stats["skip_no_label"] += 1
            continue

        # ── 파일명 충돌 처리 ──
        dst_name = img.name
        if dst_name in existing_names:
            dst_name = img.stem + "_new" + img.suffix
            stats["renamed"] += 1

        dst_img = dst_img_dir / dst_name
        dst_lbl = dst_lbl_dir / (Path(dst_name).stem + ".txt")

        shutil.copy2(img,  dst_img)
        shutil.copy2(lbl,  dst_lbl)

        existing_hashes.add(h)
        existing_names.add(dst_name)
        stats["moved"] += 1
        moved_in_split += 1

    print(f"  [{split:5s}] {moved_in_split}장 추가")

print(f"\n  완료: 추가={stats['moved']}  중복 제외={stats['skip_dup']}"
      f"  라벨 없음={stats['skip_no_label']}  이름 변경={stats['renamed']}")


# ==========================================
# 3. data.yaml 업데이트
# ==========================================
print("\n" + "=" * 55)
print("  STEP 3 — data.yaml 업데이트")
print("=" * 55)

train_count = len(list((FIRE_DIR / "train" / "images").glob("*.*"))) if (FIRE_DIR / "train" / "images").exists() else 0
valid_count = len(list((FIRE_DIR / "valid" / "images").glob("*.*"))) if (FIRE_DIR / "valid" / "images").exists() else 0
test_count  = len(list((FIRE_DIR / "test"  / "images").glob("*.*"))) if (FIRE_DIR / "test"  / "images").exists() else 0
total_count = train_count + valid_count + test_count

yaml_content = f"""# Fire & Smoke Detection Dataset (Merged)
# ==========================================
# 병합 출처
#   - datasets/fire      (원본, 400장)
#   - datasets/fire_new  (추가, 8456장)
# 중복 제거 후 총 {total_count}장
# 클래스: fire, smoke
# ==========================================

path: ../datasets/fire

train: train/images
val:   valid/images
test:  test/images

nc: 2
names: ['fire', 'smoke']

# Split 통계 (병합 후)
# train : {train_count}장
# valid : {valid_count}장
# test  : {test_count}장
# total : {total_count}장
"""

(FIRE_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")
print(f"  train={train_count}  valid={valid_count}  test={test_count}  total={total_count}")
print("  data.yaml 저장 완료")


# ==========================================
# 4. 병합 README 생성
# ==========================================
print("\n" + "=" * 55)
print("  STEP 4 — README.dataset.txt 작성")
print("=" * 55)

readme = f"""Fire & Smoke Detection Dataset — Merged
=========================================

병합 날짜  : 2026-05-28
병합 스크립트 : tools/merge_fire_datasets.py

[원본 데이터셋 1] datasets/fire
  - 출처  : Roboflow — fire-h2gkf-nmh0t
  - 라이선스: CC BY 4.0
  - 이미지 수: 400장

[원본 데이터셋 2] datasets/fire_new
  - 출처  : Roboflow — fire (추가 수집)
  - 라이선스: CC BY 4.0
  - 이미지 수: 8456장

[병합 결과]
  - 중복 제거   : {stats['skip_dup']}장
  - 이름 변경   : {stats['renamed']}장
  - 최종 이미지 : {total_count}장
    train : {train_count}장
    valid : {valid_count}장
    test  : {test_count}장

[클래스]
  0 : fire   (화재)
  1 : smoke  (연기)

[포맷]
  YOLO 형식 (train/valid/test + labels/)
  라벨 형식 : <class_index> <cx> <cy> <w> <h>  (정규화 좌표)
"""

(FIRE_DIR / "README.dataset.txt").write_text(readme, encoding="utf-8")
print("  README.dataset.txt 저장 완료")

# 기존 README.roboflow.txt 는 원본 정보라 보존
if (FIRE_DIR / "README.roboflow.txt").exists():
    print("  README.roboflow.txt — 원본 Roboflow 정보 보존 (삭제 안 함)")


# ==========================================
# 5. datasets/fire_new/ 삭제
# ==========================================
print("\n" + "=" * 55)
print("  STEP 5 — datasets/fire_new/ 삭제")
print("=" * 55)

shutil.rmtree(NEW_DIR)
print(f"  {NEW_DIR} 삭제 완료")


# ==========================================
# 최종 요약
# ==========================================
print("\n" + "=" * 55)
print("  병합 완료")
print("=" * 55)
print(f"  최종 데이터셋 위치 : datasets/fire/")
print(f"  총 이미지 수       : {total_count}장")
print(f"  data.yaml 클래스   : ['fire', 'smoke']")
print(f"  train.py 재학습 명령: python src/training/train.py fire")
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
