# -*- coding: utf-8 -*-
"""
Hard Negative 2차 수집 — 실내(가정·강의실) 특화
=================================================
오감지 원인별 수집 대상:

[가정 환경]
  indoor_lamp      - 실내 전구·스탠드 (주황빛 → 불꽃 오인)
  kitchen_glow     - 주방 기기 빛 (오븐·토스터·전자레인지 불빛)
  red_objects      - 빨간·주황색 물체 (옷·가방·가구·포장)
  tv_screen        - TV·모니터 화면 (붉은 계열 화면 → 불꽃 오인)
  candle_decor     - 촛불·데코용 조명 (작은 불꽃 → 실제 화재와 혼동)

[강의실·학교 환경]
  projector        - 프로젝터 화면 (노란·붉은 슬라이드 → 오인)
  exit_sign        - 비상구 표지판 (빨간 불빛)
  classroom        - 강의실 내부 (형광등·창문 빛)
  red_backpack     - 빨간 가방·옷 착용한 사람

실행:
    cd AISilverBridgeLJH
    python tools/collect_hard_negatives2.py
"""

import cv2
import hashlib
import shutil
from pathlib import Path
import yt_dlp

# ==========================================
# 경로 설정
# ==========================================
BASE         = Path(__file__).parent.parent
TEMP_DIR     = Path(__file__).parent / "hard_neg_videos2"
DEST_IMG_DIR = BASE / "datasets" / "fire" / "train" / "images"

TEMP_DIR.mkdir(exist_ok=True)
DEST_IMG_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 수집 카테고리
# ==========================================
CATEGORIES = [
    # ── 가정 환경 ──────────────────────────────────
    {
        "query":      "ytsearch3:cozy room warm lamp light ambiance indoor",
        "tag":        "indoor_lamp",
        "max_frames": 80,
        "desc":       "실내 전구·스탠드 (주황빛)",
    },
    {
        "query":      "ytsearch3:toaster oven microwave glow kitchen appliance",
        "tag":        "kitchen_glow",
        "max_frames": 60,
        "desc":       "주방 기기 불빛",
    },
    {
        "query":      "ytsearch3:red orange objects home interior decor",
        "tag":        "red_objects",
        "max_frames": 60,
        "desc":       "빨간·주황 물체",
    },
    {
        "query":      "ytsearch3:tv screen movie red orange scene television",
        "tag":        "tv_screen",
        "max_frames": 60,
        "desc":       "TV·모니터 화면",
    },
    {
        "query":      "ytsearch2:candle light decoration room ambiance",
        "tag":        "candle_decor",
        "max_frames": 50,
        "desc":       "데코 촛불·조명",
    },
    # ── 강의실·학교 환경 ────────────────────────────
    {
        "query":      "ytsearch3:projector screen classroom presentation lecture",
        "tag":        "projector",
        "max_frames": 70,
        "desc":       "프로젝터 화면",
    },
    {
        "query":      "ytsearch3:emergency exit sign red light indoor building",
        "tag":        "exit_sign",
        "max_frames": 40,
        "desc":       "비상구·경고등",
    },
    {
        "query":      "ytsearch3:university lecture hall classroom students indoor",
        "tag":        "classroom",
        "max_frames": 70,
        "desc":       "강의실 내부",
    },
    {
        "query":      "ytsearch3:people red orange clothing backpack street indoor",
        "tag":        "red_clothing",
        "max_frames": 60,
        "desc":       "빨간·주황 옷·가방",
    },
    # ── 추가 공통 오감지 원인 ────────────────────────
    {
        "query":      "ytsearch3:street light night orange sodium lamp road",
        "tag":        "street_lamp",
        "max_frames": 60,
        "desc":       "가로등·나트륨등 (주황빛)",
    },
    {
        "query":      "ytsearch2:sunrise window indoor morning light curtain",
        "tag":        "window_light",
        "max_frames": 50,
        "desc":       "창문 통한 햇빛·노을빛",
    },
]

FRAME_INTERVAL      = 90    # 90프레임마다 1회 추출 후보 (30fps 기준 3초에 1장)
MAX_VIDEO_FRAMES    = 2700  # 영상당 최대 2700프레임까지 읽기 (~90초)
MAX_FRAMES_PER_VID  = 12    # 영상 1개당 최대 저장 프레임 수 (다양성 확보)
PHASH_MIN_DIST      = 10    # 이미 저장된 프레임과 Hamming 거리 < 이 값이면 스킵 (유사 프레임 제거)


# ==========================================
# 유틸리티
# ==========================================
def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_bits(img: "np.ndarray") -> "np.ndarray":
    """OpenCV ndarray → 64-bit pHash (DCT 기반)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct  = cv2.dct(np.float32(gray))
    low  = dct[:8, :8].flatten()
    return (low > low.mean()).astype(np.uint8)


def is_too_similar(bits: "np.ndarray", pool: list) -> bool:
    """pool 안의 기존 pHash 중 하나라도 Hamming 거리 < PHASH_MIN_DIST 이면 True"""
    for existing in pool:
        if int(np.sum(bits != existing)) < PHASH_MIN_DIST:
            return True
    return False


import numpy as np  # phash 사용을 위해 추가

# 기존 이미지 MD5 (중복 방지)
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

    print("  영상 검색·다운로드 중...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([cat["query"]])
    except Exception as e:
        print(f"  오류: {e}")

    video_files = (list(cat_dir.glob("*.mp4"))
                   + list(cat_dir.glob("*.webm"))
                   + list(cat_dir.glob("*.mkv")))

    if not video_files:
        print(f"  [!] 다운로드된 영상 없음\n")
        continue

    cat_saved   = 0
    cat_phashes: list = []   # 이 카테고리에서 저장한 pHash 풀 (유사 프레임 방지)

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

                # ① 이 카테고리 내 기존 저장 프레임과 시각적으로 너무 유사하면 스킵
                if is_too_similar(bits, cat_phashes):
                    count += 1
                    continue

                tmp = cat_dir / f"_tmp_{count}.jpg"
                cv2.imwrite(str(tmp), frame)
                h = md5(tmp)

                # ② MD5 완전 중복 스킵
                if h in existing_hashes:
                    tmp.unlink(missing_ok=True)
                else:
                    dest = DEST_IMG_DIR / f"hardneg_{tag}_{vf.stem}_{count:05d}.jpg"
                    shutil.move(str(tmp), str(dest))
                    existing_hashes.add(h)
                    cat_phashes.append(bits)
                    cat_saved  += 1
                    total_saved += 1
                    v_saved    += 1

            count += 1

        cap.release()
        print(f"    {vf.name[:42]:42s} → {v_saved}장")

    print(f"  [{tag}] 합계 {cat_saved}장\n")

# ==========================================
# 임시 폴더 정리
# ==========================================
shutil.rmtree(TEMP_DIR, ignore_errors=True)

# ==========================================
# 결과 요약
# ==========================================
final_count = len(list(DEST_IMG_DIR.glob("*.*")))
hardneg_count = len([p for p in DEST_IMG_DIR.glob("hardneg_*.jpg")])

print("=" * 56)
print("  2차 Hard Negative 수집 완료")
print("=" * 56)
print(f"  이번 수집    : {total_saved}장")
print(f"  Hard Neg 총합: {hardneg_count}장")
print(f"  train 총합   : {final_count:,}장")
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
