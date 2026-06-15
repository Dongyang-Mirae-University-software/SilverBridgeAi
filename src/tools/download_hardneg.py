"""
화재 오탐(False Positive) 방지용 하드 네거티브 이미지 자동 수집
=============================================================
YouTube 에서 노을 / 안개 / 수증기 / 실내조명 등 영상을 다운로드 후 프레임 추출.
추출 프레임 -> datasets/fire/train/ 에 빈 라벨로 저장
 => YOLO 학습 시 "이런 장면은 화재/연기가 아님"을 학습

독거노인 시스템에서 주요 오탐 원인:
  sunset     : 노을/석양  (주황-빨간 하늘 -> 불 아님)
  fog        : 아침 안개  (연기처럼 보임 -> 연기 아님)
  steam      : 조리 수증기 (연기처럼 보임 -> 연기 아님)
  warmlight  : 실내 조명  (주황빛 -> 불 아님)
  autumn     : 가을 단풍  (주황-빨간 색조 -> 불 아님)
  lava       : 용암등     (천천히 움직이는 주황 -> 불 아님)

사용법:
    python src/tools/download_hardneg.py            # 기본 (검색어당 2개 영상)
    python src/tools/download_hardneg.py --count 3  # 더 많은 영상
    python src/tools/download_hardneg.py --dry-run  # 실제 다운 없이 목록만 확인
"""

from __future__ import annotations
import argparse
import hashlib
import io
import shutil
import subprocess
import sys
from pathlib import Path

# Windows 콘솔 UTF-8 출력
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import cv2
except ImportError:
    print("[오류] opencv-python 이 설치되지 않았습니다.")
    print("  pip install opencv-python")
    sys.exit(1)

BASE       = Path(__file__).resolve().parents[2]   # AISilverBridgeLJH/
TGT_IMG    = BASE / "datasets" / "fire" / "train" / "images"
TGT_LBL    = BASE / "datasets" / "fire" / "train" / "labels"
TMP_DIR    = BASE / "tmp_hardneg"

# yt-dlp 실행파일 경로 (venv 내부)
_scripts   = Path(sys.executable).parent
YTDLP_BIN  = _scripts / "yt-dlp.exe"
if not YTDLP_BIN.exists():
    YTDLP_BIN = _scripts / "yt-dlp"

# ─────────────────────────────────────────────────────────────
# 오탐 유형별 검색어
# ─────────────────────────────────────────────────────────────
QUERIES = [
    ("sunset",    "sunset golden hour timelapse outdoor"),
    ("fog",       "morning fog mist nature landscape"),
    ("steam",     "cooking steam kettle pot kitchen"),
    ("warmlight", "cozy warm lamp living room interior night"),
    ("autumn",    "autumn leaves forest colorful fall"),
    ("lava",      "lava lamp slow motion close up"),
]

FRAME_INTERVAL = 20   # N 프레임마다 1장 추출 (30fps -> 약 1.5 fps)
MAX_FRAMES     = 200  # 영상당 최대 추출 장수


# ─────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────

def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_existing_hashes() -> set[str]:
    """기존 fire/train 이미지 MD5 로드 (중복 방지)."""
    hashes: set[str] = set()
    if TGT_IMG.exists():
        for p in TGT_IMG.iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                hashes.add(md5_of(p))
    return hashes


def extract_frames(video_path: Path, tag: str, seen: set[str]) -> int:
    """영상에서 FRAME_INTERVAL 마다 1장씩 추출, 중복 제외 후 저장."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    [경고] 열기 실패: {video_path.name}")
        return 0

    added = frame_num = 0
    while added < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % FRAME_INTERVAL == 0:
            tmp_img = TMP_DIR / f"_tmp_{tag}_{frame_num}.jpg"
            cv2.imwrite(str(tmp_img), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            h = md5_of(tmp_img)
            if h in seen:
                tmp_img.unlink()
            else:
                name = f"hardneg_{tag}_{video_path.stem}_f{frame_num:06d}.jpg"
                dst  = TGT_IMG / name
                tmp_img.rename(dst)
                (TGT_LBL / (dst.stem + ".txt")).write_text("", encoding="utf-8")
                seen.add(h)
                added += 1
        frame_num += 1

    cap.release()
    return added


def download_and_extract(tag: str, query: str, n: int, seen: set[str]) -> int:
    """YouTube 검색 -> 다운로드 -> 프레임 추출."""
    print(f"\n  [{tag}] '{query}' ({n}개 영상)")

    tmp = TMP_DIR / tag
    tmp.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(YTDLP_BIN),
        f"ytsearch{n}:{query}",
        "-f", "worst[ext=mp4]/bestvideo[height<=480][ext=mp4]/worst",
        "--no-playlist",
        "--max-filesize", "80M",
        "--match-filter", "duration < 600",   # 10분 미만 영상만
        "-o", str(tmp / "%(id)s.%(ext)s"),
        "--quiet", "--no-warnings",
        "--no-check-certificates",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        print("    [경고] 다운로드 시간 초과 (180초)")
        return 0

    if result.returncode != 0 and result.stderr:
        print(f"    [경고] yt-dlp: {result.stderr[:120]}")

    added = 0
    video_files = [p for p in tmp.iterdir() if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi"}]
    if not video_files:
        print("    다운로드된 영상 없음 (검색 결과 없거나 제한)")
        return 0

    for vf in video_files:
        n_frames = extract_frames(vf, tag, seen)
        print(f"    {vf.name}: {n_frames}장 추출")
        added += n_frames
        vf.unlink()   # 영상 파일 삭제 (용량 절약)

    return added


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="하드 네거티브 이미지 수집")
    parser.add_argument("--count",   type=int, default=2,
                        help="검색어당 다운로드 영상 수 (기본 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="다운로드 없이 검색어 목록만 출력")
    args = parser.parse_args()

    print("=" * 55)
    print("  하드 네거티브 이미지 수집")
    print(f"  검색어당 영상 수: {args.count}")
    print(f"  저장 위치: datasets/fire/train/")
    print(f"  프레임 추출 간격: {FRAME_INTERVAL}프레임마다 1장")
    print("=" * 55)

    if args.dry_run:
        print("\n[DRY RUN] 다운로드 없이 검색어 목록만 출력:")
        for tag, q in QUERIES:
            print(f"  {tag:12s}: ytsearch{args.count}:{q}")
        return

    if not YTDLP_BIN.exists():
        print(f"[오류] yt-dlp 를 찾을 수 없습니다: {YTDLP_BIN}")
        sys.exit(1)

    TGT_IMG.mkdir(parents=True, exist_ok=True)
    TGT_LBL.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)

    print("\n기존 이미지 해시 로드 중...")
    seen = load_existing_hashes()
    print(f"기존 {len(seen):,}개 해시 로드 완료\n")

    total = 0
    for tag, query in QUERIES:
        n = download_and_extract(tag, query, args.count, seen)
        total += n

    # 임시 폴더 정리
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    print("\n" + "=" * 55)
    print(f"  총 하드 네거티브 추가: {total:,}장")
    print("  datasets/fire/train/ 에 저장 완료")
    print("  다음 학습 시 자동 반영됩니다.")
    print("=" * 55)


if __name__ == "__main__":
    main()

# Updated: refactor: 변수명 명확화

# Updated: fix: 메모리 누수 방지

# Updated: refactor: 중복 코드 제거

# Updated: refactor: 코드 가독성 개선

# Updated: perf: 성능 최적화

# Updated: refactor: 변수명 명확화

# Updated: fix: 메모리 누수 방지
<!-- Update 41 -->
<!-- Update 42 -->
<!-- Update 43 -->
<!-- Update 44 -->
<!-- Update 45 -->
<!-- Update 16 -->
<!-- Update 17 -->
<!-- Update 18 -->
<!-- Update 19 -->
<!-- Update 20 -->
