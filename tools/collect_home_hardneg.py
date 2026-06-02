# -*- coding: utf-8 -*-
"""
독거노인 가정환경 특화 Hard Negative 추가 수집
==============================================
문제: 학습 데이터 44%가 어두운 환경 → 밝은 실내에서 오감지 위험

목표: 독거노인 집 안 카메라가 실제로 보게 될 장면들을 Hard Negative로 추가
  → 모델이 "이런 밝은 실내 장면 = 화재 없음"을 학습

[수집 카테고리]
  home_bright      - 밝은 실내 (거실·주방·침실) 일상 장면
  kitchen_bright   - 밝은 주방 (요리 중, 화재 없음)
  living_room      - 거실 TV·소파·일상
  electric_stove   - 전기레인지 인디케이터 (빨간 불빛, 화재 아님)
  heater_device    - 전기히터·온풍기 (붉은 코일, 화재 아님)
  warm_indoor_sun  - 창문으로 들어오는 햇빛 (노란·주황빛 실내)
  elderly_daily    - 노인 일상 활동 (TV 시청, 식사 등)
  microwave_oven   - 전자레인지·오븐 내부 빛
  red_lamp_shade   - 붉은 전등갓·조명 (화재 아님)
  stove_indicator  - 가스레인지 파란 불꽃? 아님 → 조리 중 증기
"""

import cv2
import hashlib
import shutil
import numpy as np
from pathlib import Path
import yt_dlp

BASE         = Path(__file__).parent.parent
FIRE_DIR     = BASE / "datasets" / "fire"
DEST_DIR     = FIRE_DIR / "train" / "images"
TEMP_DIR     = Path(__file__).parent / "home_hn_videos"

TEMP_DIR.mkdir(exist_ok=True)
DEST_DIR.mkdir(parents=True, exist_ok=True)

FRAME_INTERVAL     = 90
MAX_VIDEO_FRAMES   = 2700
MAX_FRAMES_PER_VID = 10
PHASH_MIN_DIST     = 10

CATEGORIES = [
    {
        "query":      "ytsearch5:bright living room interior daytime cozy home",
        "tag":        "home_bright",
        "max_frames": 80,
    },
    {
        "query":      "ytsearch5:kitchen cooking bright daylight no fire steam",
        "tag":        "kitchen_bright",
        "max_frames": 70,
    },
    {
        "query":      "ytsearch5:living room TV watching elderly senior daily life",
        "tag":        "living_room",
        "max_frames": 70,
    },
    {
        "query":      "ytsearch5:electric stove induction cooktop indicator light red",
        "tag":        "electric_stove",
        "max_frames": 50,
    },
    {
        "query":      "ytsearch5:electric heater fan warm coil glow indoor room",
        "tag":        "heater_glow",
        "max_frames": 50,
    },
    {
        "query":      "ytsearch5:sunlight window room morning afternoon warm golden",
        "tag":        "indoor_sunlight",
        "max_frames": 60,
    },
    {
        "query":      "ytsearch4:microwave oven inside light food heating",
        "tag":        "microwave",
        "max_frames": 40,
    },
    {
        "query":      "ytsearch4:red orange lamp shade light bedroom warm",
        "tag":        "warm_lamp",
        "max_frames": 40,
    },
    {
        "query":      "ytsearch5:senior elderly person home daily routine indoor",
        "tag":        "elderly_home",
        "max_frames": 70,
    },
    {
        "query":      "ytsearch4:cooking steam pot boil kitchen no fire visible",
        "tag":        "cooking_steam2",
        "max_frames": 50,
    },
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_bits(img: np.ndarray) -> np.ndarray | None:
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct  = cv2.dct(np.float32(gray))
    low  = dct[:8, :8].flatten()
    return (low > low.mean()).astype(np.uint8)


def phash_from_file(p: Path) -> np.ndarray | None:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return phash_bits(img) if img is not None else None


def is_too_similar(bits, pool):
    return any(int(np.sum(bits != h)) < PHASH_MIN_DIST for h in pool)


# 기존 pHash 풀 구축
print("기존 Hard Negative pHash 풀 구축 중...")
existing_hashes: set[str] = set()
phash_pool: list[np.ndarray] = []

for p in DEST_DIR.glob("*.*"):
    existing_hashes.add(md5(p))
    if p.name.startswith("hardneg_"):
        bits = phash_from_file(p)
        if bits is not None:
            phash_pool.append(bits)

print(f"  기존 train  : {len(existing_hashes):,}장")
print(f"  기존 Hard Neg 풀: {len(phash_pool):,}개\n")

total_saved = 0

for cat in CATEGORIES:
    tag = cat["tag"]; max_frames = cat["max_frames"]
    cat_dir = TEMP_DIR / tag; cat_dir.mkdir(exist_ok=True)

    print(f"  [{tag}]  목표 {max_frames}장", end="  →  ")

    ydl_opts = {
        "outtmpl":      str(cat_dir / "%(id)s.%(ext)s"),
        "format":       "worstvideo[ext=mp4]/worst[ext=mp4]/worst",
        "match_filter": yt_dlp.utils.match_filter_func("duration < 600"),
        "noplaylist":   True, "quiet": True, "no_warnings": True, "ignoreerrors": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([cat["query"]])
    except Exception as e:
        print(f"오류: {e}"); continue

    video_files = list(cat_dir.glob("*.mp4")) + list(cat_dir.glob("*.webm")) + list(cat_dir.glob("*.mkv"))
    if not video_files: print("영상 없음"); continue

    cat_saved = 0
    for vf in video_files:
        if cat_saved >= max_frames: break
        cap, count, v_saved = cv2.VideoCapture(str(vf)), 0, 0
        while cat_saved < max_frames and v_saved < MAX_FRAMES_PER_VID:
            ret, frame = cap.read()
            if not ret or count >= MAX_VIDEO_FRAMES: break
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
                    existing_hashes.add(h); phash_pool.append(bits)
                    cat_saved += 1; total_saved += 1; v_saved += 1
            count += 1
        cap.release()

    print(f"{cat_saved}장")

shutil.rmtree(TEMP_DIR, ignore_errors=True)

# 결과
hn_total = len([p for p in DEST_DIR.glob("hardneg_*.jpg")])
final    = len(list(DEST_DIR.glob("*.*")))
print(f"\n{'='*54}")
print(f"  이번 추가    : {total_saved}장")
print(f"  Hard Neg 누적: {hn_total}장  ({hn_total/final*100:.1f}%)")
print(f"  train 총합   : {final:,}장")

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
