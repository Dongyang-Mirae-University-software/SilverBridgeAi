#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import ImageGrab

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("🔥 화재 감지 - 실시간 화면 캡처")
print("=" * 60)

# 모델 로드
print("\n📦 모델 로드 중...")
model = YOLO('models/fire_smoke/best.pt')
print("✅ 모델 로드 완료")

# 모델 클래스 정보
print(f"\n📊 모델 정보:")
print(f"   클래스: {model.names}")
print(f"   총 클래스 수: {len(model.names)}")

print("\n📺 화면 캡처 시작...")
print("   (종료: Q 키)\n")

frame_count = 0

try:
    while True:
        # 화면 실시간 캡처
        screenshot = ImageGrab.grab()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # 해상도 조정 (1280x720으로 스케일링)
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            new_w = 1280
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        # 화재 감지
        results = model(frame, conf=0.5)
        annotated_frame = results[0].plot()

        # 감지 결과 텍스트 추가
        frame_count += 1
        detections = len(results[0].boxes)
        cv2.putText(annotated_frame, f'Detections: {detections}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Frame: {frame_count}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 감지된 클래스 표시
        if detections > 0:
            for idx, box in enumerate(results[0].boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = box.conf[0].item()
                status_text = f'{class_name}: {confidence:.2%}'
                cv2.putText(annotated_frame, status_text,
                           (10, 90 + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 화면에 표시
        cv2.imshow('🔥 Fire Detection - Real-time Screen Capture', annotated_frame)

        # 종료 (Q 키)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n✅ 테스트 완료!")
            break

except KeyboardInterrupt:
    print("\n⛔ 사용자 중단")
except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
finally:
    cv2.destroyAllWindows()
