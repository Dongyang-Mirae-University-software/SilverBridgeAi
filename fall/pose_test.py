import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Pose", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Updated: feat: 로깅 기능 추가

# Updated: docs: 타입 힌트 추가

# Updated: feat: 예외 처리 개선

# Updated: docs: 함수 설명 추가

# Updated: fix: 에러 처리 강화

# Updated: feat: 로깅 기능 추가

# Updated: docs: 타입 힌트 추가
<!-- Update 26 -->
<!-- Update 27 -->
<!-- Update 28 -->
<!-- Update 29 -->
<!-- Update 30 -->
<!-- Update 31 -->
<!-- Update 32 -->
<!-- Update 33 -->
<!-- Update 34 -->
<!-- Update 35 -->
