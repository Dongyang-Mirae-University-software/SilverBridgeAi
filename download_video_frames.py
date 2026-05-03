# 유튜브 url 입력하면 영상 자동 다운로드 + 프레임 추출
# roboflow의 url 입력하는 기능이 없어졌음

import cv2
import os
import yt_dlp

# 1. 유튜브 URL 입력
video_url = input("영상 URL을 입력하세요: ")

# 2. 저장 폴더
video_folder = "downloaded_videos"
frame_folder = "extracted_frames"

os.makedirs(video_folder, exist_ok=True)
os.makedirs(frame_folder, exist_ok=True)

# 3. 영상 다운로드 설정
ydl_opts = {
    "outtmpl": f"{video_folder}/video.%(ext)s",
    "format": "mp4/best",
}

# 4. 영상 다운로드
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(video_url, download=True)
    video_path = ydl.prepare_filename(info)

print("영상 다운로드 완료:", video_path)

# 5. 프레임 추출
cap = cv2.VideoCapture(video_path)

count = 0
saved = 0

# 몇 프레임마다 저장할지
frame_interval = 30

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if count % frame_interval == 0:
        frame_name = f"{frame_folder}/frame_{saved}.jpg"
        cv2.imwrite(frame_name, frame)
        saved += 1

    count += 1

cap.release()

print(f"프레임 추출 완료: {saved}장 저장됨")