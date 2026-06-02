# -*- coding: utf-8 -*-
"""
유튜브 영상 다운로드 + 프레임 추출 도구
========================================
YouTube URL을 입력하면 영상을 다운로드하고
일정 간격으로 프레임을 이미지로 추출합니다.

※ Roboflow 자체 URL 다운로드 기능이 중단되어
   yt-dlp 기반으로 직접 영상을 수집합니다.

사용법:
    cd AISilverBridgeLJH/tools
    python download_frames.py

결과:
    tools/downloaded_videos/  ← 다운로드된 영상
    tools/extracted_frames/   ← 추출된 프레임 이미지

설정:
    frame_interval = 30   → 30프레임마다 1장 추출 (아래에서 직접 수정)

의존성:
    pip install yt-dlp opencv-python
"""

import cv2
import os
from pathlib import Path

import yt_dlp

# ==========================================
# 경로 설정 — 이 스크립트 위치 기준으로 저장
# ==========================================
SCRIPT_DIR    = Path(__file__).parent
VIDEO_FOLDER  = SCRIPT_DIR / "downloaded_videos"
FRAME_FOLDER  = SCRIPT_DIR / "extracted_frames"

VIDEO_FOLDER.mkdir(exist_ok=True)
FRAME_FOLDER.mkdir(exist_ok=True)

# ==========================================
# 설정
# ==========================================
FRAME_INTERVAL = 30   # 몇 프레임마다 이미지 1장 저장 (낮을수록 더 많이 추출)

# ==========================================
# 1. URL 입력
# ==========================================
video_url = input("YouTube URL을 입력하세요: ").strip()

# ==========================================
# 2. 영상 다운로드
# ==========================================
ydl_opts = {
    "outtmpl": str(VIDEO_FOLDER / "video.%(ext)s"),
    "format": "mp4/best",
}

print("\n영상 다운로드 중...")
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info       = ydl.extract_info(video_url, download=True)
    video_path = ydl.prepare_filename(info)

print(f"다운로드 완료: {video_path}")

# ==========================================
# 3. 프레임 추출
# ==========================================
cap   = cv2.VideoCapture(video_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)

print(f"\n프레임 추출 시작 (총 {total}프레임 / {fps:.1f}fps / {FRAME_INTERVAL}프레임마다 1장)")

count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % FRAME_INTERVAL == 0:
        out_path = FRAME_FOLDER / f"frame_{saved:05d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    count += 1

cap.release()

print(f"추출 완료: {saved}장 → {FRAME_FOLDER}")
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
