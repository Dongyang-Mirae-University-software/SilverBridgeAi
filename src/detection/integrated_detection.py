"""
🚨 SilverBridge AI - 통합 실시간 감지 시스템

카메라로 촬영 + YOLO (Fire, Knife) + MediaPipe (Fall) + FastAPI 업로드 + 영상 저장
피보호자(독거노인)의 이상 감지 여부를 확인합니다.

실행:
    python integrated_detection.py

종료:
    ESC 키

저장 위치:
    output/
    ├── photo_001_YYYYMMDD_HHMMSS.jpg
    ├── clip_001_YYYYMMDD_HHMMSS.mp4
    └── detection_log_YYYYMMDD_HHMMSS.csv
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import csv
import sys
from pathlib import Path
from collections import deque
from datetime import datetime

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from src.integration.config import Config
from src.integration.fastapi_client import FastAPIClient


# ==================== 설정 ====================

class IntegratedDetectionConfig:
    def __init__(self):
        self.camera_index = 1  # 카메라 인덱스 (0=내장, 1=외부/DroidCam)
        self.frame_width = 1280
        self.frame_height = 720
        self.fps = 30

        # YOLO 설정
        self.fire_model_path = project_root / "models" / "fire_100ep" / "weights" / "best.pt"
        self.knife_model_path = project_root / "models" / "knife_500ep" / "weights" / "best.pt"
        self.yolo_conf = 0.3

        # MediaPipe Fall Detection 설정
        self.fall_confirm_sec = 1.3
        self.fall_window_sec = 2.0
        self.intentional_sitting_sec = 0.8
        self.rapid_fall_delta = 0.08

        # FastAPI 업로드 설정
        self.upload_interval = 5  # 5프레임마다 업로드
        self.upload_on_detection = True  # 감지 시 즉시 업로드

        # 영상 저장 설정
        self.output_dir = project_root / "output"
        self.output_dir.mkdir(exist_ok=True)
        self.pre_fall_sec = 5.0  # 낙상 전 5초
        self.post_fall_sec = 5.0  # 낙상 후 5초
        self.save_on_detection = True  # 감지 시 프레임 저장


# ==================== MediaPipe 초기화 ====================

def init_mediapipe():
    """MediaPipe Pose 초기화"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose, mp_pose


# ==================== 자세 분류 ====================

def classify_pose(angle, torso_ratio, hip_y):
    """자세 분류 (STANDING, SITTING, LYING)"""
    angle_from_vertical = abs(angle - 90)

    if angle_from_vertical < 30:
        return "STANDING"

    if angle_from_vertical > 35 or torso_ratio > 1.0 or hip_y > 0.75:
        return "LYING"

    return "SITTING"


def get_pose_info(landmarks):
    """MediaPipe 랜드마크에서 자세 정보 추출"""
    shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    hip_x = (landmarks[23].x + landmarks[24].x) / 2
    hip_y = (landmarks[23].y + landmarks[24].y) / 2

    dx = shoulder_x - hip_x
    dy = shoulder_y - hip_y

    angle = abs(math.degrees(math.atan2(dy, dx)))
    torso_ratio = abs(dx) / (abs(dy) + 0.0001)

    return angle, torso_ratio, hip_y


# ==================== 감지 이벤트 처리 ====================

class DetectionEvent:
    def __init__(self, event_type, confidence=None, bbox=None):
        self.timestamp = datetime.now()
        self.type = event_type  # 'fire', 'smoke', 'knife', 'fall'
        self.confidence = confidence
        self.bbox = bbox

    def __repr__(self):
        if self.confidence:
            return f"{self.type.upper()} ({self.confidence:.1%})"
        return self.type.upper()


# ==================== 메인 감지 시스템 ====================

class IntegratedDetectionSystem:
    def __init__(self):
        print("🚀 SilverBridge AI 통합 감지 시스템 초기화...\n")

        # 설정
        self.cfg = IntegratedDetectionConfig()

        # MediaPipe
        self.pose, self.mp_pose = init_mediapipe()
        self.mp_draw = mp.solutions.drawing_utils

        # YOLO 모델
        print("📦 YOLO 모델 로드 중...")
        self.model_fire = YOLO(str(self.cfg.fire_model_path))
        self.model_knife = YOLO(str(self.cfg.knife_model_path))
        print("   ✓ Fire/Smoke 모델 로드됨")
        print("   ✓ Knife 모델 로드됨")

        # FastAPI 클라이언트
        self.config = Config()
        self.fastapi_client = FastAPIClient(self.config)
        print(f"   ✓ FastAPI 클라이언트 준비 (URL: {self.config.fastapi_base_url})")

        # 카메라 초기화
        print("\n📷 카메라 초기화 중...")
        self.cap = cv2.VideoCapture(self.cfg.camera_index)
        if not self.cap.isOpened():
            print(f"❌ 카메라 {self.cfg.camera_index}을(를) 열 수 없습니다.")
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        print(f"   ✓ 카메라 {self.cfg.camera_index} 준비 완료 ({self.cfg.frame_width}x{self.cfg.frame_height}@{self.cfg.fps}fps)")

        # 상태 추적
        self.frame_count = 0
        self.prev_pose_state = None
        self.last_standing_frame = None
        self.fall_frame_count = 0
        self.prev_hip_y = None
        self.sitting_start_frame = None
        self.prev_sitting_duration = 0
        self.sitting_end_frame = 0

        # 감지 이벤트 버퍼
        self.detection_events = deque(maxlen=100)

        # 업로드 카운터
        self.upload_counter = 0

        # 저장 관련 변수
        self.event_count = 0
        self.pre_fall_frames = int(self.cfg.fps * self.cfg.pre_fall_sec)
        self.post_fall_frames = int(self.cfg.fps * self.cfg.post_fall_sec)
        self.frame_buffer = deque(maxlen=self.pre_fall_frames)  # 최근 5초 프레임

        # 후방 녹화 상태
        self.is_post_recording = False
        self.post_frames = []
        self.pre_frames_snapshot = []
        self.pending_event = None

        # CSV 로그 파일
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.cfg.output_dir / f"detection_log_{run_timestamp}.csv"
        self.log_file = open(self.log_path, "w", newline="", encoding="utf-8")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(["event_id", "datetime", "type", "confidence", "photo_file", "clip_file", "clip_sec"])

        # 비디오 저장 설정
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        print("\n✅ 초기화 완료!")
        print(f"📁 저장 위치: {self.cfg.output_dir}")
        print("=" * 60)
        print("🎥 감지 시작 (ESC로 종료)\n")

    def save_clip(self, frames, path):
        """영상 클립 저장"""
        if not frames:
            return False
        try:
            out = cv2.VideoWriter(str(path), self.fourcc, self.cfg.fps,
                                (self.cfg.frame_width, self.cfg.frame_height))
            for f in frames:
                out.write(f)
            out.release()
            return True
        except Exception as e:
            print(f"❌ 영상 저장 실패: {e}")
            return False

    def save_detection(self, detection, frame):
        """감지 이벤트 감지 (사진 저장 + 후방 녹화 시작)"""
        self.event_count += 1
        event_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 사진 저장
        photo_filename = f"photo_{self.event_count:03d}_{event_ts}_{detection.type}.jpg"
        photo_path = self.cfg.output_dir / photo_filename
        cv2.imwrite(str(photo_path), frame)
        print(f"   💾 사진: {photo_filename}")

        # 후방 녹화 시작 (감지 후 5초 동안 프레임 수집)
        self.pre_frames_snapshot = list(self.frame_buffer)
        self.is_post_recording = True
        self.post_frames = []
        self.pending_event = {
            'id': self.event_count,
            'ts': event_ts,
            'type': detection.type,
            'confidence': detection.confidence,
            'photo': photo_filename
        }

        print(f"   🎥 영상 녹화 시작 (5초)...")

    def finalize_recording(self):
        """후방 녹화 완료 → 영상 저장"""
        if not self.pending_event:
            return

        event = self.pending_event
        clip_filename = f"clip_{event['id']:03d}_{event['ts']}_{event['type']}.mp4"
        clip_path = self.cfg.output_dir / clip_filename

        all_frames = self.pre_frames_snapshot + self.post_frames
        clip_sec = len(all_frames) / self.cfg.fps

        success = self.save_clip(all_frames, clip_path)

        if success:
            print(f"   💾 영상: {clip_filename} ({clip_sec:.1f}초)")

            # CSV에 기록
            self.log_writer.writerow([
                f"{event['id']:03d}",
                event['ts'],
                event['type'],
                f"{event['confidence']:.1%}" if event['confidence'] else "N/A",
                event['photo'],
                clip_filename,
                f"{clip_sec:.1f}"
            ])
            self.log_file.flush()

        # 상태 초기화
        self.is_post_recording = False
        self.post_frames = []
        self.pre_frames_snapshot = []
        self.pending_event = None

    def detect_yolo(self, frame):
        """YOLO를 이용한 Fire, Smoke, Knife 감지"""
        detections = []

        # Fire/Smoke 감지
        results_fire = self.model_fire(frame, conf=self.cfg.yolo_conf, verbose=False)
        for detection in results_fire[0].boxes:
            cls_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            bbox = detection.xyxy[0].tolist()

            # cls_id: 0=fire, 1=smoke
            event_type = "fire" if cls_id == 0 else "smoke"
            event = DetectionEvent(event_type, confidence=confidence, bbox=bbox)
            detections.append(event)

            # 프레임에 그리기
            color = (0, 0, 255) if event_type == "fire" else (0, 255, 255)
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{event_type} {confidence:.1%}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Knife 감지
        results_knife = self.model_knife(frame, conf=self.cfg.yolo_conf, verbose=False)
        for detection in results_knife[0].boxes:
            confidence = float(detection.conf[0])
            bbox = detection.xyxy[0].tolist()

            event = DetectionEvent("knife", confidence=confidence, bbox=bbox)
            detections.append(event)

            # 프레임에 그리기
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Knife {confidence:.1%}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return detections

    def detect_fall(self, frame, landmarks):
        """MediaPipe를 이용한 Fall 감지"""
        if not landmarks:
            return None

        angle, torso_ratio, hip_y = get_pose_info(landmarks)
        current_pose = classify_pose(angle, torso_ratio, hip_y)

        # STANDING 갱신
        if current_pose == "STANDING":
            self.last_standing_frame = self.frame_count

        # SITTING 구간 추적
        if current_pose == "SITTING":
            if self.prev_pose_state != "SITTING":
                self.sitting_start_frame = self.frame_count
        elif self.prev_pose_state == "SITTING" and self.sitting_start_frame is not None:
            self.prev_sitting_duration = self.frame_count - self.sitting_start_frame
            self.sitting_end_frame = self.frame_count
            self.sitting_start_frame = None

        # 빠른 낙상 감지
        rapid_fall = False
        if self.prev_hip_y is not None and (hip_y - self.prev_hip_y) > self.cfg.rapid_fall_delta:
            rapid_fall = True
        self.prev_hip_y = hip_y

        # 낙상 조건 판정
        frames_since_standing = (
            self.frame_count - self.last_standing_frame
            if self.last_standing_frame is not None
            else float('inf')
        )

        fall_window_frames = int(self.cfg.fall_window_sec * self.cfg.fps)
        is_lying = (current_pose == "LYING")
        recently_fell = is_lying and (frames_since_standing < fall_window_frames)
        rapid_fall_and_lying = is_lying and rapid_fall

        # 의도적 눕기 판정
        intentional_sitting_frames = int(self.cfg.intentional_sitting_sec * self.cfg.fps)
        frames_since_sitting_ended = (
            self.frame_count - self.sitting_end_frame
            if self.sitting_end_frame > 0
            else float('inf')
        )
        was_sitting_deliberately = (
            is_lying
            and self.prev_sitting_duration >= intentional_sitting_frames
            and frames_since_sitting_ended < fall_window_frames
        )

        fall_condition = (recently_fell or rapid_fall_and_lying) and not was_sitting_deliberately

        self.prev_pose_state = current_pose

        if fall_condition:
            return DetectionEvent("fall", confidence=1.0)
        return None

    def process_frame(self, frame):
        """한 프레임 처리"""
        self.frame_count += 1

        # RGB 변환
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLO 감지
        yolo_detections = self.detect_yolo(frame)

        # MediaPipe 감지
        result = self.pose.process(rgb)
        fall_detection = None
        if result.pose_landmarks:
            self.mp_draw.draw_landmarks(frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            fall_detection = self.detect_fall(frame, result.pose_landmarks.landmark)

        # 모든 감지 이벤트 수집
        all_detections = yolo_detections
        if fall_detection:
            all_detections.append(fall_detection)

        # 프레임을 버퍼에 추가 (최근 5초 유지)
        self.frame_buffer.append(frame.copy())

        # 감지 이벤트 기록 및 저장
        for detection in all_detections:
            self.detection_events.append(detection)
            print(f"\n🚨 감지: {detection} [F{self.frame_count}]")

            # 감지 사진 + 후방 녹화 시작
            if self.cfg.save_on_detection:
                self.save_detection(detection, frame)

        # 후방 녹화 진행 중
        if self.is_post_recording:
            self.post_frames.append(frame.copy())

            # 5초 녹화 완료
            if len(self.post_frames) >= self.post_fall_frames:
                self.finalize_recording()

        # 프레임에 감지 정보 표시
        text_y = 30
        for detection in all_detections:
            text = str(detection)
            cv2.putText(frame, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            text_y += 35

        # FastAPI에 업로드
        self.upload_counter += 1
        should_upload = (
            (self.upload_counter >= self.cfg.upload_interval) or
            (self.cfg.upload_on_detection and len(all_detections) > 0)
        )

        if should_upload:
            for detection in all_detections:
                success, response = self.fastapi_client.upload_frame(
                    frame=frame,
                    detection_type=detection.type,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                )
                if success:
                    print(f"   ✓ FastAPI 업로드 성공")
                else:
                    print(f"   ✗ FastAPI 업로드 실패: {response.get('message', 'Unknown')}")
            self.upload_counter = 0

        # 프레임 정보 추가
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Events: {len(self.detection_events)}", (10, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def run(self):
        """메인 루프"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ 프레임 읽기 실패")
                break

            # 프레임 처리
            frame = self.process_frame(frame)

            # 화면 표시
            cv2.imshow("SilverBridge AI - 통합 감지", frame)

            # ESC 키로 종료
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print("\n✋ 감지 중지...")
                break

        # 진행 중인 녹화 완료 (부분)
        if self.is_post_recording and (self.pre_frames_snapshot or self.post_frames):
            event = self.pending_event
            clip_filename = f"clip_{event['id']:03d}_{event['ts']}_{event['type']}_partial.mp4"
            clip_path = self.cfg.output_dir / clip_filename

            all_frames = self.pre_frames_snapshot + self.post_frames
            clip_sec = len(all_frames) / self.cfg.fps

            self.save_clip(all_frames, clip_path)
            print(f"   💾 영상 (부분): {clip_filename} ({clip_sec:.1f}초)")

            # CSV에 기록
            self.log_writer.writerow([
                f"{event['id']:03d}",
                event['ts'],
                event['type'],
                f"{event['confidence']:.1%}" if event['confidence'] else "N/A",
                event['photo'],
                clip_filename,
                f"{clip_sec:.1f}"
            ])

        # 정리
        self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
        self.log_file.close()

        # 통계 출력
        print("\n" + "=" * 60)
        print("📊 감지 통계")
        print("=" * 60)
        print(f"총 프레임: {self.frame_count}")
        print(f"감지된 이벤트: {len(self.detection_events)}")

        for event_type in ["fire", "smoke", "knife", "fall"]:
            count = sum(1 for e in self.detection_events if e.type == event_type)
            if count > 0:
                print(f"  - {event_type.upper()}: {count}건")

        print(f"\n📁 저장 위치:")
        print(f"  - {self.cfg.output_dir}")
        print(f"  - 사진: photo_*.jpg")
        print(f"  - 영상: clip_*.mp4 (앞뒤 5초씩)")
        print(f"  - 로그: {self.log_path.name}")
        print("=" * 60)


# ==================== 실행 ====================

if __name__ == "__main__":
    system = IntegratedDetectionSystem()
    system.run()

# Updated: feat: 예외 처리 개선

# Updated: docs: 함수 설명 추가

# Updated: fix: 에러 처리 강화

# Updated: feat: 로깅 기능 추가

# Updated: docs: 타입 힌트 추가

# Updated: feat: 예외 처리 개선

# Updated: docs: 함수 설명 추가

# Real-time multi-class detection with confidence thresholding optimization
<!-- Update 41 -->
<!-- Update 42 -->
<!-- Update 43 -->
<!-- Update 44 -->
<!-- Update 45 -->
