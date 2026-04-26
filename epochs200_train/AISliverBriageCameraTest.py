import cv2
from ultralytics import YOLO

model_fire = YOLO(r"C:/Users/happy/SilverBridgeAI/AISilverBridgeLJH/epochs30_train/firetrain/weights/best.pt")
model_knife = YOLO(r"C:/Users/happy/SilverBridgeAI/AISilverBridgeLJH/epochs200_train/knifetrain/weights/best.pt")

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results_fire = model_fire(frame)
    results_knife = model_knife(frame)

    img = frame.copy()

    # Fire / Smoke 감지 박스 그리기
    for box in results_fire[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        class_name = model_fire.names[cls_id]  # fire 또는 smoke

        if conf < 0.5:
            continue

        if class_name == "fire":
            color = (0, 0, 255)      # 빨간색
        elif class_name == "smoke":
            color = (255, 0, 0)      # 파란색
        else:
            color = (0, 255, 255)    # 노란색

        label = f"{class_name} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # Knife 감지 박스 그리기
    for box in results_knife[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        class_name = model_knife.names[cls_id]

        if conf < 0.5:
            continue

        label = f"{class_name} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Fire + Smoke + Knife Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()