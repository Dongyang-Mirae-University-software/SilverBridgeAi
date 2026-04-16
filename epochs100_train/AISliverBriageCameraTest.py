import cv2
from ultralytics import YOLO

model = YOLO(r"C:/Users/happy/SilverBridgeAI/AISilverBridgeLJH/epochs100_train/train100/weights/best.pt")
cap = cv2.VideoCapture(1)

while True:
	ret, frame = cap.read()
	if not ret:
		break

	results = model(frame)
	img = results[0].plot()

	cv2.imshow("Frie Detection", img)

	if cv2.waitKey(1) & 0xFF == 27 : # ESC 누르면 종료
		break


cap.release()
cv2.destroyAllWindows()