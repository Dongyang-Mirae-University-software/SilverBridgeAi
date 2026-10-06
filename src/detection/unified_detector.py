#!/usr/bin/env python3
"""
Unified Threat Detection System
Fire/Smoke + Knife + Fall Detection
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from .detectors import FallDetector

class UnifiedDetector:
    """모든 위협을 감지하는 통합 감지기"""

    def __init__(self,
                 fire_model_path=None,
                 knife_model_path=None,
                 fire_confidence=0.92,
                 knife_confidence=0.95,
                 fall_angle_threshold=60,
                 fall_duration_sec=2.5):
        """
        Args:
            fire_model_path: Fire/Smoke 모델 경로
            knife_model_path: Knife 모델 경로
            fire_confidence: Fire 신뢰도 임계값
            knife_confidence: Knife 신뢰도 임계값
            fall_angle_threshold: Fall 감지 각도 임계값
            fall_duration_sec: Fall 감지 지속 시간
        """

        # YOLO 모델 로드
        self.fire_model = None
        self.knife_model = None

        if fire_model_path and Path(fire_model_path).exists():
            self.fire_model = YOLO(fire_model_path)

        if knife_model_path and Path(knife_model_path).exists():
            self.knife_model = YOLO(knife_model_path)

        # Fall 감지기
        self.fall_detector = FallDetector(
            fall_angle_threshold=fall_angle_threshold,
            fall_duration_sec=fall_duration_sec
        )

        # 신뢰도 임계값
        self.fire_confidence = fire_confidence
        self.knife_confidence = knife_confidence

        # 프레임 타이밍
        self.frame_time = 0

    def detect_fire_smoke(self, frame):
        """Fire/Smoke 감지"""
        if self.fire_model is None:
            return {'fire': [], 'smoke': []}

        results = self.fire_model(frame, conf=self.fire_confidence, verbose=False)

        detections = {'fire': [], 'smoke': []}

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf)
                    class_id = int(box.cls)
                    bbox = box.xyxy[0].cpu().numpy()

                    if conf >= self.fire_confidence:
                        class_name = result.names[class_id]
                        detections[class_name.lower()].append({
                            'bbox': bbox,
                            'confidence': conf,
                            'class': class_name
                        })

        return detections

    def detect_knife(self, frame):
        """Knife 감지"""
        if self.knife_model is None:
            return {'knife': []}

        results = self.knife_model(frame, conf=self.knife_confidence, verbose=False)

        detections = {'knife': []}

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf)
                    class_id = int(box.cls)
                    bbox = box.xyxy[0].cpu().numpy()

                    if conf >= self.knife_confidence:
                        class_name = result.names[class_id]
                        detections['knife'].append({
                            'bbox': bbox,
                            'confidence': conf,
                            'class': class_name
                        })

        return detections

    def detect_fall(self, frame, current_time=None):
        """Fall 감지"""
        result = self.fall_detector.detect(frame, current_time)

        return {
            'fall': {
                'detected': result['is_falling'],
                'confidence': result['confidence'],
                'body_angle': result['body_angle'],
                'duration': result['fall_duration'],
                'has_motion': result['has_motion']
            }
        }

    def detect(self, frame, current_time=None):
        """
        모든 위협 감지

        Returns:
            {
                'fire': [...],
                'smoke': [...],
                'knife': [...],
                'fall': {...}
            }
        """

        detections = {}

        # Fire/Smoke
        fire_detections = self.detect_fire_smoke(frame)
        detections.update(fire_detections)

        # Knife
        knife_detections = self.detect_knife(frame)
        detections.update(knife_detections)

        # Fall
        fall_detections = self.detect_fall(frame, current_time)
        detections.update(fall_detections)

        return detections

    def draw_detections(self, frame, detections):
        """감지 결과 시각화"""

        h, w = frame.shape[:2]

        # Fire 표시 (빨강)
        for det in detections.get('fire', []):
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Fire {det['confidence']:.2f}",
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Smoke 표시 (주황)
        for det in detections.get('smoke', []):
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, f"Smoke {det['confidence']:.2f}",
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Knife 표시 (노랑)
        for det in detections.get('knife', []):
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Knife {det['confidence']:.2f}",
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Fall 표시 (초록/빨강)
        fall_info = detections.get('fall', {})
        if fall_info.get('detected'):
            color = (0, 0, 255)  # 빨강 (위험)
            cv2.putText(frame, f"FALL DETECTED! {fall_info['confidence']:.2f}",
                       (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, f"Angle: {fall_info['body_angle']:.1f}°, Duration: {fall_info['duration']:.1f}s",
                       (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            # Fall 정보 표시 (모니터링)
            cv2.putText(frame, f"Fall: Normal (Angle: {fall_info['body_angle']:.1f}°)",
                       (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def release(self):
        """리소스 해제"""
        self.fall_detector.pose.close()
