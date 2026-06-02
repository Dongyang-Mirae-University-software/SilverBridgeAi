# -*- coding: utf-8 -*-
"""
Knife Hard Negative 대규모 보강
================================
목표: Hard Negative 비율 9.5% → 25% 이상

[추가 이유별 카테고리]

★ 1순위 — 칼과 가장 헷갈리는 물체 (지금 0장)
  pen_pencil      - 볼펜·연필·샤프 클로즈업
  ruler_measure   - 자·줄자·각도기
  marker_pen      - 형광펜·보드마커·사인펜

★ 2순위 — 손에 쥔 긴 물체 (오감지의 주요 원인)
  hand_pen        - 손에 볼펜·연필 쥔 장면
  hand_stick      - 손에 막대기·지시봉 쥔 장면
  hand_utensil    - 손에 주방 도구 쥔 장면
  hand_tool       - 손에 드라이버·공구 쥔 장면

★ 3순위 — 강의실·학교 특화
  lecture_scene   - 교수·강사가 강의하는 장면
  study_desk      - 책상 위 필기구·교재
  whiteboard      - 칠판·화이트보드 앞 장면

★ 4순위 — 가정 환경 특화
  kitchen_safe    - 주방 (칼 없는 요리 장면)
  dining_table    - 식탁 (젓가락·숟가락·포크)
  sewing_craft    - 바느질·공예 (바늘·실·자)

★ 5순위 — 기타 칼 유사 외형
  metal_bar       - 금속 봉·파이프·막대
  credit_card     - 카드·통장 (얇고 날카로운 모서리)
  toothbrush      - 칫솔 (손잡이 형태 유사)
  comb_brush      - 빗·헤어브러시
"""

import cv2
import hashlib
import shutil
import numpy as np
from pathlib import Path
import yt_dlp

# ==========================================
# 경로
# ==========================================
BASE         = Path(__file__).parent.parent
KNIFE_DIR    = BASE / "datasets" / "knife"
DEST_DIR     = KNIFE_DIR / "train" / "images"
TEMP_DIR     = Path(__file__).parent / "knife_hn_extra"

TEMP_DIR.mkdir(exist_ok=True)
DEST_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 파라미터
# ==========================================
FRAME_INTERVAL     = 90
MAX_VIDEO_FRAMES   = 2700
MAX_FRAMES_PER_VID = 10
PHASH_MIN_DIST     = 10

# ==========================================
# 카테고리 — 우선순위 순
# ==========================================
CATEGORIES = [
    # ── 1순위: 볼펜·연필·자 ───────────────────────
    {
        "query":      "ytsearch5:pencil review drawing writing close up",
        "tag":        "pencil",
        "max_frames": 80,
    },
    {
        "query":      "ytsearch5:ballpoint pen fountain pen review test",
        "tag":        "pen_review",
        "max_frames": 80,
    },
    {
        "query":      "ytsearch5:ruler measuring tape stationery haul school",
        "tag":        "ruler",
        "max_frames": 70,
    },
    {
        "query":      "ytsearch5:marker whiteboard highlighter pen stationery",
        "tag":        "marker",
        "max_frames": 70,
    },
    # ── 2순위: 손에 쥔 긴 물체 ────────────────────
    {
        "query":      "ytsearch5:hand holding pen pencil writing notes",
        "tag":        "hand_pen",
        "max_frames": 80,
    },
    {
        "query":      "ytsearch5:teacher pointer stick pointing whiteboard lecture",
        "tag":        "hand_pointer",
        "max_frames": 70,
    },
    {
        "query":      "ytsearch5:hand screwdriver repair fixing diy tool",
        "tag":        "hand_tool",
        "max_frames": 60,
    },
    {
        "query":      "ytsearch5:cooking spatula wok stir fry hand holding",
        "tag":        "hand_spatula",
        "max_frames": 60,
    },
    # ── 3순위: 강의실·학교 ─────────────────────────
    {
        "query":      "ytsearch5:university professor lecture hall teaching class",
        "tag":        "lecture_hall",
        "max_frames": 80,
    },
    {
        "query":      "ytsearch5:student study desk notebook pencil case school",
        "tag":        "study_desk",
        "max_frames": 70,
    },
    {
        "query":      "ytsearch5:whiteboard marker drawing classroom teacher",
        "tag":        "whiteboard",
        "max_frames": 60,
    },
    # ── 4순위: 가정 환경 ───────────────────────────
    {
        "query":      "ytsearch5:cooking stir fry vegetables wok no knife",
        "tag":        "cooking_safe",
        "max_frames": 60,
    },
    {
        "query":      "ytsearch5:eating chopsticks fork spoon dining table meal",
        "tag":        "dining",
        "max_frames": 60,
    },
    {
        "query":      "ytsearch5:sewing needle thread fabric craft diy",
        "tag":        "sewing",
        "max_frames": 50,
    },
    # ── 5순위: 외형 유사 기타 ──────────────────────
    {
        "query":      "ytsearch5:metal rod bar pipe construction material",
        "tag":        "metal_bar",
        "max_frames": 50,
    },
    {
        "query":      "ytsearch5:toothbrush brushing teeth bathroom",
        "tag":        "toothbrush",
        "max_frames": 40,
    },
    {
        "query":      "ytsearch5:comb brush hair styling beauty",
        "tag":        "comb",
        "max_frames": 40,
    },
    {
        "query":      "ytsearch5:credit card wallet hands payment",
        "tag":        "card",
        "max_frames": 40,
    },
]


# ==========================================
# 유틸리티
# ==========================================
def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_bits(img: np.ndarray) -> np.ndarray | None:
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct  = cv2.dct(np.float32(gray))
    low  = dct[:8, :8].flatten()
    return (low > low.mean()).astype(np.uint8)


def phash_from_file(p: Path) -> np.ndarray | None:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return phash_bits(img) if img is not None else None


def is_too_similar(bits: np.ndarray, pool: list) -> bool:
    for h in pool:
        if int(np.sum(bits != h)) < PHASH_MIN_DIST:
            return True
    return False


# ==========================================
# 기존 Hard Negative pHash 풀 구축
# ==========================================
print("기존 Hard Negative pHash 풀 구축 중...")
existing_hashes: set[str] = set()
phash_pool: list[np.ndarray] = []

for p in DEST_DIR.glob("*.*"):
    existing_hashes.add(md5(p))
    if p.name.startswith("hardneg_"):
        bits = phash_from_file(p)
        if bits is not None:
            phash_pool.append(bits)

print(f"  기존 train 이미지 : {len(existing_hashes):,}장")
print(f"  기존 Hard Neg 풀  : {len(phash_pool):,}개\n")

total_saved = 0

# ==========================================
# 카테고리별 수집
# ==========================================
for cat in CATEGORIES:
    tag        = cat["tag"]
    max_frames = cat["max_frames"]
    cat_dir    = TEMP_DIR / tag
    cat_dir.mkdir(exist_ok=True)

    print(f"  [{tag}]  목표 {max_frames}장", end="  →  ")

    ydl_opts = {
        "outtmpl":      str(cat_dir / "%(id)s.%(ext)s"),
        "format":       "worstvideo[ext=mp4]/worst[ext=mp4]/worst",
        "match_filter": yt_dlp.utils.match_filter_func("duration < 600"),
        "noplaylist":   True,
        "quiet":        True,
        "no_warnings":  True,
        "ignoreerrors": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([cat["query"]])
    except Exception as e:
        print(f"오류: {e}")
        continue

    video_files = (list(cat_dir.glob("*.mp4"))
                   + list(cat_dir.glob("*.webm"))
                   + list(cat_dir.glob("*.mkv")))

    if not video_files:
        print("영상 없음")
        continue

    cat_saved = 0
    for vf in video_files:
        if cat_saved >= max_frames:
            break
        cap, count, v_saved = cv2.VideoCapture(str(vf)), 0, 0

        while cat_saved < max_frames and v_saved < MAX_FRAMES_PER_VID:
            ret, frame = cap.read()
            if not ret or count >= MAX_VIDEO_FRAMES:
                break
            if count % FRAME_INTERVAL == 0 and frame is not None:
                bits = phash_bits(frame)
                if bits is None or is_too_similar(bits, phash_pool):
                    count += 1; continue
                tmp = cat_dir / f"_tmp_{count}.jpg"
                cv2.imwrite(str(tmp), frame)
                h = md5(tmp)
                if h in existing_hashes:
                    tmp.unlink(missing_ok=True)
                else:
                    dest = DEST_DIR / f"hardneg_{tag}_{vf.stem}_{count:05d}.jpg"
                    shutil.move(str(tmp), str(dest))
                    existing_hashes.add(h)
                    phash_pool.append(bits)
                    cat_saved += 1; total_saved += 1; v_saved += 1
            count += 1

        cap.release()

    print(f"{cat_saved}장 수집")

shutil.rmtree(TEMP_DIR, ignore_errors=True)

# ==========================================
# 결과 요약
# ==========================================
all_imgs   = len(list(DEST_DIR.glob("*.*")))
hn_total   = len([p for p in DEST_DIR.glob("hardneg_*.jpg")])
hn_ratio   = hn_total / all_imgs * 100 if all_imgs else 0

print(f"\n{'='*54}")
print(f"  완료")
print(f"{'='*54}")
print(f"  이번 추가    : {total_saved}장")
print(f"  Hard Neg 누적: {hn_total}장  ({hn_ratio:.1f}%)")
print(f"  train 총합   : {all_imgs:,}장")
print()
print("  재학습 명령어:")
print("    python src/training/train.py knife --epochs 200")

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
