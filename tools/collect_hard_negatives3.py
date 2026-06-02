# -*- coding: utf-8 -*-
"""
Hard Negative 3차 수집 — 전체 기존 이미지 pHash 비교 적용
==========================================================
기존 collect_hard_negatives*.py 와 차이점:
  - 수집 시작 전 기존 모든 hardneg 이미지의 pHash 풀을 구축
  - 새 프레임이 풀 내 어느 이미지와도 Hamming < PHASH_MIN_DIST 이면 스킵
  - 카테고리 내부뿐 아니라 전체 Hard Negative 에 대해 중복 방지
  - MD5 중복 방지도 동시 적용

[추가 카테고리 — 약한 부분 보강]
  projector2      - 다른 검색어로 프로젝터 추가 수집
  candle2         - 다른 검색어로 촛불 추가 수집
  red_led         - 빨간 LED·경광등·브레이크등
  indoor_morning  - 실내 아침빛 (창문 햇살·커튼 투과광)
  classroom2      - 다른 검색어로 강의실 추가 수집
  orange_nature   - 주황빛 자연 (단풍·황토·모래)
  tv_movie        - 영화 장면 (붉은 계열 화면)
  indoor_fire_dec - 벽난로·장식용 불 영상 (작은 통제된 불)
  factory_light   - 공장·창고 조명 (주황·노란 산업 조명)
  traffic_lights  - 교통 신호등 (빨간·주황)

실행:
    cd AISilverBridgeLJH
    python tools/collect_hard_negatives3.py
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
BASE         = Path(__file__).parent.parent
TEMP_DIR     = Path(__file__).parent / "hard_neg_videos3"
DEST_IMG_DIR = BASE / "datasets" / "fire" / "train" / "images"

TEMP_DIR.mkdir(exist_ok=True)
DEST_IMG_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 파라미터
# ==========================================
FRAME_INTERVAL     = 90    # 90프레임마다 후보 추출 (30fps → 3초에 1장)
MAX_VIDEO_FRAMES   = 2700  # 영상당 최대 읽기 프레임 수 (~90초)
MAX_FRAMES_PER_VID = 10    # 영상 1개당 최대 저장 수
PHASH_MIN_DIST     = 10    # 기존 전체 풀과 Hamming < 이 값이면 스킵

# ==========================================
# 카테고리
# ==========================================
CATEGORIES = [
    {
        "query":      "ytsearch5:projector presentation slides screen colorful",
        "tag":        "projector2",
        "max_frames": 60,
        "desc":       "프로젝터 화면 (추가)",
    },
    {
        "query":      "ytsearch5:candle flame christmas birthday dinner table",
        "tag":        "candle2",
        "max_frames": 60,
        "desc":       "촛불·장식 불꽃 (추가)",
    },
    {
        "query":      "ytsearch5:red LED light indicator warning blink",
        "tag":        "red_led",
        "max_frames": 60,
        "desc":       "빨간 LED·경광등",
    },
    {
        "query":      "ytsearch5:morning sunlight window room curtain rays",
        "tag":        "indoor_morning",
        "max_frames": 60,
        "desc":       "실내 아침 창문 햇살",
    },
    {
        "query":      "ytsearch5:university classroom lecture students chairs",
        "tag":        "classroom2",
        "max_frames": 60,
        "desc":       "강의실 내부 (추가)",
    },
    {
        "query":      "ytsearch5:autumn fall foliage orange red leaves forest",
        "tag":        "orange_nature",
        "max_frames": 60,
        "desc":       "단풍·주황빛 자연",
    },
    {
        "query":      "ytsearch5:movie scene red orange cinematic shot film",
        "tag":        "tv_movie",
        "max_frames": 60,
        "desc":       "영화·TV 붉은 화면",
    },
    {
        "query":      "ytsearch5:fireplace indoor cozy living room burning log",
        "tag":        "fireplace",
        "max_frames": 60,
        "desc":       "벽난로·장식 불 (통제된 환경)",
    },
    {
        "query":      "ytsearch5:warehouse factory workshop orange yellow light indoor",
        "tag":        "factory_light",
        "max_frames": 60,
        "desc":       "공장·창고 주황 조명",
    },
    {
        "query":      "ytsearch5:traffic light red stop car road intersection",
        "tag":        "traffic_light",
        "max_frames": 60,
        "desc":       "교통 신호등 (빨간)",
    },
    {
        "query":      "ytsearch5:orange sunset sky clouds dramatic",
        "tag":        "sunset2",
        "max_frames": 50,
        "desc":       "노을 (추가 다양성)",
    },
    {
        "query":      "ytsearch5:neon sign bar restaurant interior night",
        "tag":        "neon2",
        "max_frames": 50,
        "desc":       "네온 간판·조명 (추가)",
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


def phash_from_file(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return phash_bits(img) if img is not None else None


def is_too_similar(bits: np.ndarray, pool: list) -> bool:
    for h in pool:
        if int(np.sum(bits != h)) < PHASH_MIN_DIST:
            return True
    return False


# ==========================================
# 기존 Hard Negative pHash 풀 구축
# (새 이미지가 기존 전체와 비교됨)
# ==========================================
print("기존 Hard Negative pHash 풀 구축 중...")
global_phash_pool: list[np.ndarray] = []
existing_hashes:   set[str]         = set()

for p in sorted(DEST_IMG_DIR.glob("*.*")):
    existing_hashes.add(md5(p))
    # hard neg 이미지만 pHash 풀에 추가
    if p.name.startswith("hardneg_"):
        bits = phash_from_file(p)
        if bits is not None:
            global_phash_pool.append(bits)

print(f"  전체 train 이미지 : {len(existing_hashes):,}장")
print(f"  Hard Neg pHash 풀 : {len(global_phash_pool):,}개\n")

total_saved = 0

# ==========================================
# 카테고리별 수집
# ==========================================
for cat in CATEGORIES:
    tag        = cat["tag"]
    max_frames = cat["max_frames"]
    cat_dir    = TEMP_DIR / tag
    cat_dir.mkdir(exist_ok=True)

    print("=" * 56)
    print(f"  [{tag}]  {cat['desc']}  (목표 {max_frames}장)")
    print("=" * 56)

    ydl_opts = {
        "outtmpl":      str(cat_dir / "%(id)s.%(ext)s"),
        "format":       "worstvideo[ext=mp4]/worst[ext=mp4]/worst",
        "match_filter": yt_dlp.utils.match_filter_func("duration < 600"),
        "noplaylist":   True,
        "quiet":        True,
        "no_warnings":  True,
        "ignoreerrors": True,
    }

    print("  검색·다운로드 중...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([cat["query"]])
    except Exception as e:
        print(f"  오류: {e}")

    video_files = (list(cat_dir.glob("*.mp4"))
                   + list(cat_dir.glob("*.webm"))
                   + list(cat_dir.glob("*.mkv")))

    if not video_files:
        print(f"  [!] 다운로드 영상 없음\n")
        continue

    cat_saved = 0

    for vf in video_files:
        if cat_saved >= max_frames:
            break

        cap     = cv2.VideoCapture(str(vf))
        count   = 0
        v_saved = 0

        while cat_saved < max_frames and v_saved < MAX_FRAMES_PER_VID:
            ret, frame = cap.read()
            if not ret or count >= MAX_VIDEO_FRAMES:
                break

            if count % FRAME_INTERVAL == 0 and frame is not None:
                bits = phash_bits(frame)
                if bits is None:
                    count += 1
                    continue

                # 전체 Hard Neg 풀과 비교 (기존 + 이번 수집 포함)
                if is_too_similar(bits, global_phash_pool):
                    count += 1
                    continue

                tmp = cat_dir / f"_tmp_{count}.jpg"
                cv2.imwrite(str(tmp), frame)
                h = md5(tmp)

                if h in existing_hashes:
                    tmp.unlink(missing_ok=True)
                else:
                    dest = DEST_IMG_DIR / f"hardneg_{tag}_{vf.stem}_{count:05d}.jpg"
                    shutil.move(str(tmp), str(dest))
                    existing_hashes.add(h)
                    global_phash_pool.append(bits)   # 풀에 즉시 추가
                    cat_saved  += 1
                    total_saved += 1
                    v_saved    += 1

            count += 1

        cap.release()
        print(f"    {vf.name[:44]:44s} → {v_saved}장")

    print(f"  [{tag}] 합계 {cat_saved}장\n")

# ==========================================
# 임시 폴더 정리
# ==========================================
shutil.rmtree(TEMP_DIR, ignore_errors=True)

# ==========================================
# 결과 요약
# ==========================================
hardneg_count = len([p for p in DEST_IMG_DIR.glob("hardneg_*.jpg")])
total_count   = len(list(DEST_IMG_DIR.glob("*.*")))

print("=" * 56)
print("  3차 Hard Negative 수집 완료")
print("=" * 56)
print(f"  이번 수집      : {total_saved}장")
print(f"  Hard Neg 누적  : {hardneg_count}장")
print(f"  train 총합     : {total_count:,}장")
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
