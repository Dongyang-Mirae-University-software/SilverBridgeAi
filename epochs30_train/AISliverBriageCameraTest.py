import cv2
from ultralytics import YOLO

model_fire = YOLO(r"C:/Users/happy/SilverBridgeAI/AISilverBridgeLJH/epochs30_train/firetrain/weights/best.pt")
model_knife = YOLO(r"C:/Users/happy/SilverBridgeAI/AISilverBridgeLJH/epochs30_train/knifetrain/weights/best.pt")

cap = cv2.VideoCapture(1)

while True:
	ret, frame = cap.read()
	if not ret:
		break

	results_fire = model_fire(frame)
	results_knife = model_knife(frame)

	img = results_fire[0].plot()

	for box in results_fire[0].boxes:
		x1, y1, x2, y2 = map(int, box.xyxy[0])
		conf = float(box.conf[0])
		cls = int(box.cls[0])

		label = f"knife {conf:.2f}"

		cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
		cv2.putText(img, label, (x1, y1-10),
			  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

	cv2.imshow("Fire + Knife Detection", img)

	if cv2.waitKey(1) & 0xFF == 27 : # ESC 누르면 종료
		break


cap.release()
cv2.destroyAllWindows()