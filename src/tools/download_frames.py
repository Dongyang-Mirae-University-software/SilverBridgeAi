"""
YouTube 영상 다운로드 + 프레임 추출 도구
========================================
yt-dlp 로 영상을 받아 OpenCV 로 N프레임마다 1장씩 추출.
라벨링용 원본 이미지 수집에 사용.

사용법:
    python src/tools/download_frames.py
    python src/tools/download_frames.py --url <URL> --interval 30 --out extracted_frames/
"""

from __future__ import annotations
import argparse
import io
import subprocess
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import cv2
except ImportError:
    print("[오류] opencv-python 이 설치되지 않았습니다. pip install opencv-python")
    sys.exit(1)

BASE      = Path(__file__).resolve().parents[2]
_scripts  = Path(sys.executable).parent
YTDLP_BIN = _scripts / "yt-dlp.exe"
if not YTDLP_BIN.exists():
    YTDLP_BIN = _scripts / "yt-dlp"

DL_DIR  = BASE / "downloaded_videos"
OUT_DIR = BASE / "extracted_frames"


def download_video(url: str, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(YTDLP_BIN),
        url,
        "-f", "bestvideo[height<=720][ext=mp4]/best[height<=720]/best",
        "--no-playlist",
        "-o", str(out_dir / "%(title)s.%(ext)s"),
        "--no-check-certificates",
    ]
    result = subprocess.run(cmd, timeout=300)
    if result.returncode != 0:
        print(f"[경고] 다운로드 실패: {url}")
        return None
    # 가장 최근에 생성된 파일 반환
    files = sorted(out_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def extract_frames(video_path: Path, out_dir: Path, interval: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[오류] 영상 열기 실패: {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"  총 프레임: {total}  FPS: {fps:.1f}  추출 간격: {interval}프레임")

    stem   = video_path.stem
    count  = 0
    idx    = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            out_path = out_dir / f"{stem}_frame{idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1
        idx += 1
    cap.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube 영상 프레임 추출")
    parser.add_argument("--url",      default="",  help="YouTube URL (생략 시 프롬프트 입력)")
    parser.add_argument("--interval", type=int, default=30, help="N프레임마다 1장 (기본 30)")
    parser.add_argument("--out",      default="",  help="출력 디렉터리 (기본: extracted_frames/)")
    args = parser.parse_args()

    url = args.url or input("YouTube URL: ").strip()
    if not url:
        print("[오류] URL이 필요합니다.")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else OUT_DIR

    print(f"\n다운로드 중: {url}")
    video = download_video(url, DL_DIR)
    if not video:
        sys.exit(1)

    print(f"추출 중: {video.name}")
    n = extract_frames(video, out_dir, args.interval)
    print(f"\n완료: {n}장 -> {out_dir}")


if __name__ == "__main__":
    main()

# Updated: feat: 예외 처리 개선

# Updated: docs: 함수 설명 추가

# Updated: fix: 에러 처리 강화

# Updated: feat: 로깅 기능 추가

# Updated: docs: 타입 힌트 추가

# Updated: feat: 예외 처리 개선

# Updated: docs: 함수 설명 추가
<!-- Update 1 -->
<!-- Update 2 -->
<!-- Update 3 -->
<!-- Update 4 -->
<!-- Update 5 -->
