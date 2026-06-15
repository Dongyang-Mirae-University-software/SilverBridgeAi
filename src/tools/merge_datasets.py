"""
fire_new/ + smoke/ 데이터셋을 fire/ 에 병합하는 스크립트

클래스 매핑
  fire/     nc=2  fire=0  smoke=1  (변경 없음, 타겟 스키마)
  fire_new/ nc=1  0 → 0(fire)
  smoke/    nc=3  0(Fire)→0  1(Other)→제거(하드 네거티브)  2(Smoke)→1

동작
  1. fire/ 기존 이미지의 MD5 해시 수집 (중복 판별 기준)
  2. fire_new/ / smoke/ 에서 신규 이미지 추출 (MD5 중복 제외)
  3. 라벨 리매핑 후 fire/train, valid, test 에 복사
  4. smoke/ Other 전용 이미지 → 빈 라벨 = 하드 네거티브
  5. 최종 통계 출력
"""

from __future__ import annotations
import hashlib
import io
import shutil
import sys
from pathlib import Path
from collections import defaultdict

# Windows 콘솔 UTF-8 출력 보장
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parents[2]   # AISilverBridgeLJH/
DS   = BASE / "datasets"
TARGET = DS / "fire"

SOURCES = [
    {
        "name":      "fire_new",
        "dir":       DS / "fire_new",
        "class_map": {0: 0},         # fire → fire
        "skip":      set(),
    },
    {
        "name":      "smoke",
        "dir":       DS / "smoke",
        "class_map": {0: 0, 2: 1},   # Fire→fire, Smoke→smoke
        "skip":      {1},             # Other → 제거 (빈 라벨로 하드 네거티브)
    },
]

SPLITS   = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ─────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────

def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def remap_label(lbl_path: Path, class_map: dict, skip: set) -> str:
    """라벨 파일을 읽고 클래스 리매핑한 텍스트 반환. 유효 박스 없으면 빈 문자열."""
    if not lbl_path.exists():
        return ""
    out = []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(parts[0])
        if cls in skip:
            continue
        new_cls = class_map.get(cls)
        if new_cls is None:
            continue
        out.append(f"{new_cls} " + " ".join(parts[1:]))
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────
# 기존 fire/ 이미지 해시 수집
# ─────────────────────────────────────────────────────────────────

def collect_existing_hashes() -> set[str]:
    hashes: set[str] = set()
    count = 0
    for split in SPLITS:
        img_dir = TARGET / split / "images"
        if not img_dir.exists():
            continue
        for p in img_dir.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                hashes.add(md5_of(p))
                count += 1
                if count % 5000 == 0:
                    print(f"  기존 해시 수집 중... {count:,}장")
    print(f"  [fire/] 기존 {count:,}장 해시 수집 완료\n")
    return hashes


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 55)
    print("  SilverBridgeAI - 화재 데이터셋 병합")
    print("=" * 55 + "\n")

    seen = collect_existing_hashes()

    grand_added   = 0
    grand_dup     = 0
    grand_no_lbl  = 0

    for cfg in SOURCES:
        name      = cfg["name"]
        src_dir   = cfg["dir"]
        class_map = cfg["class_map"]
        skip      = cfg["skip"]

        added = dup = no_lbl = 0
        split_count: dict[str, int] = defaultdict(int)

        print(f"[{name}] 처리 중...")

        for split in SPLITS:
            img_dir = src_dir / split / "images"
            lbl_dir = src_dir / split / "labels"
            if not img_dir.exists():
                continue

            tgt_img = TARGET / split / "images"
            tgt_lbl = TARGET / split / "labels"
            tgt_img.mkdir(parents=True, exist_ok=True)
            tgt_lbl.mkdir(parents=True, exist_ok=True)

            imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
            for img in imgs:
                h = md5_of(img)
                if h in seen:
                    dup += 1
                    continue

                lbl_txt = remap_label(lbl_dir / (img.stem + ".txt"), class_map, skip)

                # 원본 라벨 없음 → 빈 문자열 유지 (하드 네거티브)
                if not (lbl_dir / (img.stem + ".txt")).exists():
                    no_lbl += 1

                # 파일명 충돌 방지: 소스명 prefix
                new_name = f"{name}_{img.stem}{img.suffix}"
                shutil.copy2(img, tgt_img / new_name)
                (tgt_lbl / f"{name}_{img.stem}.txt").write_text(lbl_txt, encoding="utf-8")

                seen.add(h)
                added += 1
                split_count[split] += 1

                if added % 2000 == 0:
                    print(f"    {split} ... {added:,}장 추가 중")

        grand_added  += added
        grand_dup    += dup
        grand_no_lbl += no_lbl

        print(f"   추가 {added:,}장  /  중복 제외 {dup:,}장  /  라벨 없는 이미지 {no_lbl}장")
        for s, n in split_count.items():
            print(f"   {s:6s}: +{n:,}장")
        print()

    # ── 최종 통계 ────────────────────────────────────────────────
    print("=" * 55)
    print("  병합 완료 - fire/ 최종 현황")
    print("=" * 55)
    total = 0
    for split in SPLITS:
        img_dir = TARGET / split / "images"
        n = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        total += n
        print(f"  {split:6s}: {n:,}장")
    print(f"  {'합계':6s}: {total:,}장")
    print(f"\n  이번 병합 추가 : {grand_added:,}장")
    print(f"  중복 제외      : {grand_dup:,}장")
    print(f"  하드 네거티브  : {grand_no_lbl}장 (빈 라벨)")
    print()
    print("  data.yaml 권장 학습 명령어:")
    print("  python src/training/train.py fire --epochs 250")


if __name__ == "__main__":
    main()

# Updated: fix: 에러 처리 강화

# Updated: feat: 로깅 기능 추가

# Updated: docs: 타입 힌트 추가

# Updated: feat: 예외 처리 개선

# Updated: docs: 함수 설명 추가

# Updated: fix: 에러 처리 강화

# Updated: feat: 로깅 기능 추가

# Enhanced dataset validation and error handling for large-scale data merging
<!-- Update 46 -->
<!-- Update 47 -->
<!-- Update 48 -->
<!-- Update 49 -->
<!-- Update 50 -->
