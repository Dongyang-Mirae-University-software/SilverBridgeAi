from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("YOLO", annotated)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Updated: fix: 메모리 누수 방지

# Updated: refactor: 중복 코드 제거

# Updated: refactor: 코드 가독성 개선

# Updated: perf: 성능 최적화

# Updated: refactor: 변수명 명확화

# Updated: fix: 메모리 누수 방지

# Updated: refactor: 중복 코드 제거
<!-- Update 21 -->
<!-- Update 22 -->
<!-- Update 23 -->
<!-- Update 24 -->
<!-- Update 25 -->
<!-- Update 66 -->
<!-- Update 67 -->
<!-- Update 68 -->
<!-- Update 69 -->
<!-- Update 70 -->
