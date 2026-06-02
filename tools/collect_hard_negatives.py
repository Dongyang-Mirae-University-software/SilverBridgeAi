# -*- coding: utf-8 -*-
"""
Hard Negative 이미지 자동 수집 스크립트
========================================
화재처럼 보이지만 실제로는 화재가 아닌 장면을 YouTube에서 자동 수집.
수집된 프레임은 라벨 없이 datasets/fire/train/images/ 에 추가되어
모델이 오감지를 줄이도록 학습됩니다.

[수집 카테고리]
  sunset     - 노을 (주황·붉은 하늘 → 불 오인 방지)
  fog        - 안개·안개비 (연기 오인 방지)
  steam      - 수증기 (연기 오인 방지)
  city_light - 도시 야경·네온사인 (붉은 조명 오인 방지)
  dust       - 먼지·모래폭풍 (연기 오인 방지)

실행:
    cd AISilverBridgeLJH
    python tools/collect_hard_negatives.py
"""

import cv2
import hashlib
import shutil
import numpy as np
from pathlib import Path
import yt_dlp

# ==========================================
# 경로 설정
# ==========================================
BASE          = Path(__file__).parent.parent
TEMP_DIR      = Path(__file__).parent / "hard_neg_videos"
DEST_IMG_DIR  = BASE / "datasets" / "fire" / "train" / "images"

TEMP_DIR.mkdir(exist_ok=True)
DEST_IMG_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 수집 카테고리 설정
# ==========================================
# ytsearch{N}: 로 YouTube에서 N개 영상 검색
# max_frames  : 카테고리당 최대 저장 프레임 수
CATEGORIES = [
    {
        "query":      "ytsearch3:sunset golden hour timelapse 4k short",
        "tag":        "sunset",
        "max_frames": 80,
    },
    {
        "query":      "ytsearch3:morning fog mist forest nature short",
        "tag":        "fog",
        "max_frames": 60,
    },
    {
        "query":      "ytsearch3:steam cooking boiling water kitchen short",
        "tag":        "steam",
        "max_frames": 60,
    },
    {
        "query":      "ytsearch3:city night lights bokeh neon short",
        "tag":        "city_light",
        "max_frames": 60,
    },
    {
        "query":      "ytsearch2:dust storm desert sandstorm short",
        "tag":        "dust",
        "max_frames": 40,
    },
]

FRAME_INTERVAL     = 90    # 90프레임마다 추출 후보 (30fps 기준 3초에 1장)
MAX_VIDEO_FRAMES   = 2700  # 영상당 최대 2700프레임 읽기 (~90초)
MAX_FRAMES_PER_VID = 12    # 영상 1개당 최대 저장 프레임 수
PHASH_MIN_DIST     = 10    # Hamming 거리 < 이 값이면 유사 프레임으로 스킵


# ==========================================
# 유틸리티
# ==========================================
def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_bits(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct  = cv2.dct(np.float32(gray))
    low  = dct[:8, :8].flatten()
    return (low > low.mean()).astype(np.uint8)


def is_too_similar(bits: np.ndarray, pool: list) -> bool:
    for existing in pool:
        if int(np.sum(bits != existing)) < PHASH_MIN_DIST:
            return True
    return False


# 기존 이미지 MD5 수집 (중복 방지)
print("기존 train/images MD5 수집 중...")
existing_hashes: set[str] = set()
for p in DEST_IMG_DIR.glob("*.*"):
    existing_hashes.add(md5(p))
print(f"  기존 이미지 {len(existing_hashes):,}장\n")

total_saved = 0

# ==========================================
# 카테고리별 수집
# ==========================================
for cat in CATEGORIES:
    tag        = cat["tag"]
    max_frames = cat["max_frames"]
    cat_dir    = TEMP_DIR / tag
    cat_dir.mkdir(exist_ok=True)

    print("=" * 54)
    print(f"  [{tag}]  목표 {max_frames}장")
    print("=" * 54)

    # ── yt-dlp 설정 ──────────────────────────────
    ydl_opts = {
        "outtmpl":       str(cat_dir / "%(id)s.%(ext)s"),
        "format":        "worstvideo[ext=mp4]/worst[ext=mp4]/worst",
        "match_filter":  yt_dlp.utils.match_filter_func("duration < 600"),  # 10분 미만
        "noplaylist":    True,
        "quiet":         True,
        "no_warnings":   True,
        "ignoreerrors":  True,
    }

    # ── 영상 다운로드 ─────────────────────────────
    print(f"  영상 검색·다운로드 중...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([cat["query"]])
    except Exception as e:
        print(f"  다운로드 오류: {e}")

    # ── 다운로드된 영상에서 프레임 추출 ─────────────
    video_files = list(cat_dir.glob("*.mp4")) + list(cat_dir.glob("*.webm")) + list(cat_dir.glob("*.mkv"))
    if not video_files:
        print(f"  [!] 다운로드된 영상 없음 — 건너뜀\n")
        continue

    cat_saved   = 0
    cat_phashes: list = []

    for video_path in video_files:
        if cat_saved >= max_frames:
            break

        cap     = cv2.VideoCapture(str(video_path))
        count   = 0
        v_saved = 0

        while cat_saved < max_frames and v_saved < MAX_FRAMES_PER_VID:
            ret, frame = cap.read()
            if not ret or count >= MAX_VIDEO_FRAMES:
                break

            if count % FRAME_INTERVAL == 0 and frame is not None:
                bits = phash_bits(frame)

                # 이 카테고리 내 저장된 프레임과 너무 유사하면 스킵
                if is_too_similar(bits, cat_phashes):
                    count += 1
                    continue

                tmp = cat_dir / f"_tmp_{count}.jpg"
                cv2.imwrite(str(tmp), frame)
                h = md5(tmp)

                if h in existing_hashes:
                    tmp.unlink(missing_ok=True)
                else:
                    dest = DEST_IMG_DIR / f"hardneg_{tag}_{video_path.stem}_{count:05d}.jpg"
                    shutil.move(str(tmp), str(dest))
                    existing_hashes.add(h)
                    cat_phashes.append(bits)
                    cat_saved  += 1
                    total_saved += 1
                    v_saved    += 1

            count += 1

        cap.release()
        print(f"    {video_path.name[:40]:40s}  → {v_saved}장")

    print(f"  [{tag}] 합계 {cat_saved}장 저장\n")

# ==========================================
# 임시 폴더 정리
# ==========================================
shutil.rmtree(TEMP_DIR, ignore_errors=True)

# ==========================================
# 결과 요약
# ==========================================
final_count = len(list(DEST_IMG_DIR.glob("*.*")))

print("=" * 54)
print("  Hard Negative 수집 완료")
print("=" * 54)
print(f"  이번 수집: {total_saved}장")
print(f"  train/images 총합: {final_count:,}장")
print()
print("  * 수집된 이미지는 라벨 없이 추가됨")
print("  * 모델이 해당 장면을 '화재 없음'으로 학습")
print()
print("  재학습 명령어:")
print("    python src/training/train.py fire")

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
