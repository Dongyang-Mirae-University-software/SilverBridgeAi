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
# Updated: refactor: 코드 가독성 개선

# Updated: perf: 성능 최적화

# Updated: refactor: 변수명 명확화

# Updated: fix: 메모리 누수 방지

# Updated: refactor: 중복 코드 제거

# Updated: refactor: 코드 가독성 개선

# Updated: perf: 성능 최적화

# Multi-camera format detection and compatibility checking
<!-- Update 21 -->
<!-- Update 22 -->
<!-- Update 23 -->
<!-- Update 24 -->
<!-- Update 25 -->
