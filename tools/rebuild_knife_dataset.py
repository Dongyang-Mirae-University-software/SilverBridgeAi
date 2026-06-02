# -*- coding: utf-8 -*-
"""
Knife 데이터셋 재구성
=====================
Step 1 : pHash 유사 중복 제거 (기존 데이터)
Step 2 : 라벨 파일 정합성 검사 (파싱 불가 라인 제거)
Step 3 : Hard Negative 수집
         칼처럼 생겼지만 칼이 아닌 긴 물체들
           ruler_stationery - 자·연필·볼펜·형광펜
           scissors         - 가위
           kitchen_utensil  - 주방용 젓가락·주걱·국자 (비 칼)
           tools_metal      - 드라이버·스패너·망치
           pointer_stick    - 지시봉·막대기·우산
           chopsticks       - 젓가락 (식당·집)
           classroom_desk   - 강의실 책상 위 (칼 없는 일반 사물)
           home_kitchen     - 가정 주방 (칼 없는 장면)
Step 4 : data.yaml 갱신

실행:
    cd AISilverBridgeLJH
    python tools/rebuild_knife_dataset.py
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
KNIFE_DIR    = BASE / "datasets" / "knife"
TEMP_DIR     = Path(__file__).parent / "knife_hard_neg_videos"
SPLITS       = ["train", "valid", "test"]

TEMP_DIR.mkdir(exist_ok=True)

# ==========================================
# 파라미터
# ==========================================
PHASH_THRESHOLD    = 8
FRAME_INTERVAL     = 90
MAX_VIDEO_FRAMES   = 2700
MAX_FRAMES_PER_VID = 10
PHASH_MIN_DIST     = 10

# ==========================================
# Hard Negative 카테고리
# ==========================================
HARD_NEG_CATEGORIES = [
    {
        "query":      "ytsearch5:ruler pencil pen marker stationery desk table",
        "tag":        "ruler_stationery",
        "max_frames": 80,
        "desc":       "자·연필·볼펜 (책상 위)",
    },
    {
        "query":      "ytsearch5:scissors cutting paper craft sewing fabric",
        "tag":        "scissors",
        "max_frames": 70,
        "desc":       "가위",
    },
    {
        "query":      "ytsearch5:cooking spatula ladle spoon wok kitchen no knife",
        "tag":        "kitchen_utensil",
        "max_frames": 70,
        "desc":       "주방 도구 (칼 제외)",
    },
    {
        "query":      "ytsearch5:screwdriver wrench hammer tool repair workshop",
        "tag":        "tools_metal",
        "max_frames": 70,
        "desc":       "드라이버·스패너 등 공구",
    },
    {
        "query":      "ytsearch5:pointer stick presentation rod baton conductor",
        "tag":        "pointer_stick",
        "max_frames": 60,
        "desc":       "지시봉·막대기",
    },
    {
        "query":      "ytsearch5:chopsticks eating asian food restaurant table",
        "tag":        "chopsticks",
        "max_frames": 60,
        "desc":       "젓가락",
    },
    {
        "query":      "ytsearch5:classroom desk chair student university lecture daytime",
        "tag":        "classroom_desk",
        "max_frames": 80,
        "desc":       "강의실 (칼 없는 일반 장면)",
    },
    {
        "query":      "ytsearch5:home kitchen counter cooking no knife utensils",
        "tag":        "home_kitchen",
        "max_frames": 70,
        "desc":       "가정 주방 (칼 없는 장면)",
    },
    {
        "query":      "ytsearch4:umbrella cane walking stick pipe metal rod",
        "tag":        "stick_umbrella",
        "max_frames": 50,
        "desc":       "우산·지팡이·파이프",
    },
    {
        "query":      "ytsearch4:letter opener box cutter packaging cardboard open",
        "tag":        "box_cutter_bg",
        "max_frames": 40,
        "desc":       "커터칼·상자 (칼 라벨 없이 배경으로)",
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


def phash_bits(img) -> np.ndarray | None:
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct  = cv2.dct(np.float32(gray))
    low  = dct[:8, :8].flatten()
    return (low > low.mean()).astype(np.uint8)


def phash_from_file(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return phash_bits(img)


def is_too_similar(bits: np.ndarray, pool: list) -> bool:
    for h in pool:
        if int(np.sum(bits != h)) < PHASH_MIN_DIST:
            return True
    return False


# ==========================================
# Step 1: pHash 유사 중복 제거
# ==========================================
print("=" * 56)
print("  Step 1 — pHash 유사 중복 제거")
print("=" * 56)

all_imgs: list[tuple[str, Path]] = []
for split in SPLITS:
    d = KNIFE_DIR / split / "images"
    if d.exists():
        all_imgs += [(split, p) for p in sorted(d.glob("*.*"))]

total_before = len(all_imgs)
print(f"  시작 이미지: {total_before:,}장  |  pHash 계산 중...")

indexed:   list[tuple[str, Path]] = []
hash_list: list[np.ndarray]       = []

for i, (split, p) in enumerate(all_imgs):
    if i % 500 == 0 and i > 0:
        print(f"    {i:,} / {total_before:,}")
    bits = phash_from_file(p)
    if bits is not None:
        indexed.append((split, p))
        hash_list.append(bits)

n = len(hash_list)
hash_matrix = np.stack(hash_list)
print(f"  해시 완료 ({n:,}장)  →  배치 비교 중...")

to_remove_idx: set[int] = set()
BATCH = 128

for b_start in range(0, n, BATCH):
    b_end = min(b_start + BATCH, n)
    batch = hash_matrix[b_start:b_end]
    dists = np.sum(
        batch[:, np.newaxis, :] != hash_matrix[np.newaxis, :, :],
        axis=2, dtype=np.uint8,
    )
    for k in range(b_end - b_start):
        i = b_start + k
        if i in to_remove_idx:
            continue
        mask = dists[k] <= PHASH_THRESHOLD
        mask[: i + 1] = False
        for j in np.where(mask)[0]:
            if int(j) not in to_remove_idx:
                to_remove_idx.add(int(j))

for idx in to_remove_idx:
    split, img_path = indexed[idx]
    lbl = KNIFE_DIR / split / "labels" / (img_path.stem + ".txt")
    img_path.unlink(missing_ok=True)
    lbl.unlink(missing_ok=True)

all_imgs = [(s, p) for s, p in all_imgs if p.exists()]
print(f"  유사 중복 {len(to_remove_idx):,}장 제거  →  남음: {len(all_imgs):,}장\n")


# ==========================================
# Step 2: 라벨 정합성 검사
# ==========================================
print("=" * 56)
print("  Step 2 — 라벨 정합성 검사")
print("=" * 56)

bad_lines = 0
fixed_files = 0
empty_labels = 0

for split, img_path in all_imgs:
    lbl_path = KNIFE_DIR / split / "labels" / (img_path.stem + ".txt")
    if not lbl_path.exists():
        continue
    raw = lbl_path.read_text(encoding="utf-8").strip()
    if not raw:
        empty_labels += 1
        continue

    new_lines = []
    changed = False
    for line in raw.split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:        # 최소 5개 (class + 2쌍 좌표)
            bad_lines += 1
            changed = True
            continue
        try:
            int(parts[0])
            [float(x) for x in parts[1:]]
            new_lines.append(line.strip())
        except ValueError:
            bad_lines += 1
            changed = True

    if changed:
        lbl_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        fixed_files += 1

print(f"  빈 라벨 (배경 이미지): {empty_labels}개  ← 유지")
print(f"  파싱 오류 제거: {bad_lines}개  |  수정 파일: {fixed_files}개\n")


# ==========================================
# Step 3: Hard Negative 수집
# ==========================================
print("=" * 56)
print("  Step 3 — Hard Negative 수집 (칼 유사 물체)")
print("=" * 56)

# 기존 모든 이미지 MD5 + pHash 풀 구축
print("  기존 이미지 MD5 / pHash 풀 구축 중...")
existing_hashes: set[str] = set()
global_phash_pool: list[np.ndarray] = []

dest_img_dir = KNIFE_DIR / "train" / "images"
for p in dest_img_dir.glob("*.*"):
    existing_hashes.add(md5(p))
    if p.name.startswith("hardneg_"):
        bits = phash_from_file(p)
        if bits is not None:
            global_phash_pool.append(bits)

print(f"  기존 train 이미지: {len(existing_hashes):,}장")
print(f"  기존 Hard Neg 풀:  {len(global_phash_pool):,}개\n")

total_hn_saved = 0

for cat in HARD_NEG_CATEGORIES:
    tag        = cat["tag"]
    max_frames = cat["max_frames"]
    cat_dir    = TEMP_DIR / tag
    cat_dir.mkdir(exist_ok=True)

    print(f"  [{tag}]  {cat['desc']}  (목표 {max_frames}장)")

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
        print(f"    오류: {e}")

    video_files = (list(cat_dir.glob("*.mp4"))
                   + list(cat_dir.glob("*.webm"))
                   + list(cat_dir.glob("*.mkv")))

    if not video_files:
        print(f"    [!] 영상 없음\n")
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
                if bits is None or is_too_similar(bits, global_phash_pool):
                    count += 1
                    continue
                tmp = cat_dir / f"_tmp_{count}.jpg"
                cv2.imwrite(str(tmp), frame)
                h = md5(tmp)
                if h in existing_hashes:
                    tmp.unlink(missing_ok=True)
                else:
                    dest = dest_img_dir / f"hardneg_{tag}_{vf.stem}_{count:05d}.jpg"
                    shutil.move(str(tmp), str(dest))
                    existing_hashes.add(h)
                    global_phash_pool.append(bits)
                    cat_saved += 1; total_hn_saved += 1; v_saved += 1
            count += 1

        cap.release()
        print(f"    {vf.name[:44]:44s} → {v_saved}장")

    print(f"    [{tag}] 합계 {cat_saved}장\n")

shutil.rmtree(TEMP_DIR, ignore_errors=True)


# ==========================================
# Step 4: data.yaml 갱신
# ==========================================
print("=" * 56)
print("  Step 4 — data.yaml 갱신")
print("=" * 56)

split_counts: dict[str, int] = {}
for split in SPLITS:
    d = KNIFE_DIR / split / "images"
    split_counts[split] = len(list(d.glob("*.*"))) if d.exists() else 0

total_final = sum(split_counts.values())
hn_count    = len([p for p in (KNIFE_DIR/"train"/"images").glob("hardneg_*.jpg")])

yaml_content = f"""# Knife Detection Dataset (Rebuilt)
# ==========================================
# 원본: Roboflow knife-skqnq-yiuai v1 (CC BY 4.0)
# 재구성: pHash 유사 중복 제거 + Hard Negative 추가
#
# Hard Negative 카테고리 (라벨 없음):
#   ruler_stationery - 자·연필·볼펜
#   scissors         - 가위
#   kitchen_utensil  - 주방 도구 (칼 제외)
#   tools_metal      - 드라이버·스패너 등
#   pointer_stick    - 지시봉·막대기
#   chopsticks       - 젓가락
#   classroom_desk   - 강의실 (칼 없는 장면)
#   home_kitchen     - 가정 주방 (칼 없는 장면)
#
# total : {total_final:,}장  (Hard Neg: {hn_count}장)
# ==========================================

path: ../datasets/knife

train: train/images
val:   valid/images
test:  test/images

nc: 2
names: ['Knife', 'Knife_Handle']

# Split 통계
# train : {split_counts['train']:,}장
# valid : {split_counts['valid']:,}장
# test  : {split_counts['test']:,}장
# total : {total_final:,}장
"""

(KNIFE_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")

readme = f"""Knife Detection Dataset - Rebuilt
===================================

재구성 일자   : 2026-05-30
재구성 스크립트: tools/rebuild_knife_dataset.py

[원본]
  Roboflow knife-skqnq-yiuai v1 (CC BY 4.0)
  라벨 형식: YOLO Segmentation (polygon, ~39 좌표쌍)
  클래스: Knife(0), Knife_Handle(1)

[재구성 내용]
  pHash 유사 중복 제거 (Hamming <= {PHASH_THRESHOLD})
  Hard Negative 추가 (칼처럼 생긴 비-칼 물체, {hn_count}장)

[Hard Negative 카테고리]
  ruler_stationery : 자·연필·볼펜
  scissors         : 가위
  kitchen_utensil  : 주방 도구 (칼 제외)
  tools_metal      : 드라이버·스패너 등 공구
  pointer_stick    : 지시봉·막대기
  chopsticks       : 젓가락
  classroom_desk   : 강의실 배경 (칼 없음)
  home_kitchen     : 가정 주방 배경 (칼 없음)

[최종 이미지 수]
  train : {split_counts['train']:,}장
  valid : {split_counts['valid']:,}장
  test  : {split_counts['test']:,}장
  total : {total_final:,}장

[라이선스]
  CC BY 4.0 (Roboflow)
"""

(KNIFE_DIR / "README.dataset.txt").write_text(readme, encoding="utf-8")
print("  data.yaml / README 갱신 완료")


# ==========================================
# 최종 요약
# ==========================================
print(f"\n{'='*56}")
print("  완료")
print("=" * 56)
print(f"  원본       : {total_before:,}장")
print(f"  중복 제거  : {len(to_remove_idx):,}장")
print(f"  Hard Neg   : +{total_hn_saved:,}장")
print(f"  최종       : {total_final:,}장")
print(f"    train    : {split_counts['train']:,}장")
print(f"    valid    : {split_counts['valid']:,}장")
print(f"    test     : {split_counts['test']:,}장")
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
