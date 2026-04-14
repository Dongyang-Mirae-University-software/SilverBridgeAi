import cv2

print("=== CAP_DSHOW 없이 스캔 ===")
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"인덱스 {i}: 열림 - 해상도 {int(cap.get(3))}x{int(cap.get(4))}")
        if ret:
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
    else:
        print(f"인덱스 {i}: 없음")
    cap.release()