# 🔗 SilverBridge AI - 백엔드 통합 모듈

SilverBridge AI의 실시간 감지 시스템을 Java Spring Boot 백엔드와 연동하는 모듈입니다.

## 📦 모듈 구성

### 1. `config.py` - 설정 관리
백엔드 API URL, 토큰 저장 경로, 모델 경로 등을 관리합니다.

```python
from integration.config import Config

config = Config()
print(config.api_base_url)  # http://localhost:8080
print(config.confidence_threshold)  # 0.3
```

**환경변수**:
- `SILVERBRIDGE_API_URL`: 백엔드 API URL (기본값: http://localhost:8080)
- `SILVERBRIDGE_API_TIMEOUT`: API 호출 타임아웃 (기본값: 30초)
- `DETECTION_CONF`: 감지 신뢰도 임계값 (기본값: 0.3)
- `AUTO_REPORT`: 자동 보고 여부 (기본값: true)

### 2. `auth_client.py` - JWT 인증 관리
로그인, 토큰 관리, 로그아웃을 처리합니다.

```python
from integration.auth_client import AuthClient
from integration.config import Config

config = Config()
auth = AuthClient(config)

# 로그인
success, msg = auth.signin("user@example.com", "password")
if success:
    print(f"✓ {msg}")
    print(f"User ID: {auth.user_id}")
    print(f"Token: {auth.access_token[:20]}...")

# 상태 확인
if auth.is_authenticated():
    user = auth.get_user_info()
    print(f"로그인: {user['name']} ({user['role']})")

# 토큰 자동 갱신
success, msg = auth.refresh_access_token()

# 로그아웃
success, msg = auth.logout()
```

**저장 경로**:
- 토큰: `AISilverBridgeLJH/config/tokens.json`
- 사용자 정보: `AISilverBridgeLJH/config/user.json`

### 3. `api_client.py` - 백엔드 API 클라이언트
백엔드와의 모든 API 통신을 처리합니다.

```python
from integration.api_client import APIClient
from integration.auth_client import AuthClient
from integration.config import Config

config = Config()
auth = AuthClient(config)
auth.signin("user@example.com", "password")

api = APIClient(auth, config)

# 프로필 조회
success, profile = api.get_my_profile()
if success:
    print(f"이름: {profile['name']}")
    print(f"역할: {profile['role']}")

# 알림 설정 조회
success, settings = api.get_notification_settings()
if success:
    print(f"FCM 푸시: {settings['fcm']}")
    print(f"SMS: {settings['sms']}")

# 알림 설정 수정
success, msg = api.update_notification_settings(fcm=True, sms=False, email=True)

# 보호자-피보호자 연결 확인
if api.check_guardian_role():
    success, wards = api.get_my_wards()  # 내 피보호자 조회
    
elif api.check_active_ward_role():
    success, guardians = api.get_my_guardians()  # 내 보호자 조회
    success, pending = api.get_pending_requests()  # 대기 중인 요청

# FCM 토큰 등록 (푸시 알림 수신)
success, msg = api.register_fcm_token("firebase_fcm_token_here")

# 공지사항 조회
success, announcements = api.get_announcements()
```

### 4. `detection_reporter.py` - 감지 이벤트 보고
감지 결과를 로컬에 기록하고 보호자에게 알림을 보냅니다.

```python
from integration.detection_reporter import DetectionReporter, DetectionType
from integration.api_client import APIClient

api = APIClient(auth, config)
reporter = DetectionReporter(api, config)

# 감지 이벤트 보고
event = reporter.report_detection(
    detection_type=DetectionType.KNIFE,
    confidence=0.95,
    frame_number=1234,
    metadata={"location": "living_room", "bbox": [100, 150, 200, 300]},
)
# → 로컬에 기록되고, 보호자들에게 알림이 발송됨

# 다른 감지 유형들
reporter.report_detection(DetectionType.FIRE, 0.87, frame_number)
reporter.report_detection(DetectionType.SMOKE, 0.76, frame_number)
reporter.report_detection(DetectionType.FALL, 0.92, frame_number)

# 감지 이력 조회
history = reporter.get_detection_history(detection_type="knife", limit=100)
for event in history:
    print(f"{event['timestamp']}: {event['type']} ({event['confidence']:.1%})")

# 감지 통계
summary = reporter.get_detection_summary()
print(f"오늘 감지: {summary['total']}건")
print(f"  - Fire: {summary['by_type']['fire']}건")
print(f"  - Knife: {summary['by_type']['knife']}건")
print(f"  - Fall: {summary['by_type']['fall']}건")
print(f"평균 신뢰도: {summary['average_confidence']:.1%}")

# 감지 로그 출력
reporter.print_detection_log(limit=20)
```

**로그 저장 경로**:
- `AISilverBridgeLJH/logs/detections/detections_YYYY-MM-DD.jsonl`
- JSONL 형식 (한 줄 = 한 이벤트)

---

## 🚀 실제 사용 예제

### 기본 통합 (multi_detection.py)

```python
from integration.config import Config
from integration.auth_client import AuthClient
from integration.api_client import APIClient
from integration.detection_reporter import DetectionReporter, DetectionType

# 초기화
config = Config()
auth = AuthClient(config)

# 로그인 확인 (저장된 토큰이 없으면 로그인)
if not auth.is_authenticated():
    email = input("📧 이메일: ")
    password = input("🔐 비밀번호: ")
    success, msg = auth.signin(email, password)
    if not success:
        print(f"❌ 로그인 실패: {msg}")
        exit(1)

api = APIClient(auth, config)
reporter = DetectionReporter(api, config)

# 모델 로드 및 감지 루프
from ultralytics import YOLO

model_knife = YOLO(str(config.knife_model_path))
model_fire = YOLO(str(config.fire_model_path))

# 웹캠 처리
import cv2

cap = cv2.VideoCapture(1)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Knife 감지
    results_knife = model_knife(frame, conf=config.confidence_threshold, verbose=False)
    for detection in results_knife[0].boxes:
        confidence = float(detection.conf[0])
        if confidence > config.confidence_threshold:
            reporter.report_detection(
                DetectionType.KNIFE,
                confidence=confidence,
                frame_number=frame_count,
                metadata={"bbox": detection.xyxy[0].tolist()},
            )

    # Fire 감지
    results_fire = model_fire(frame, conf=config.confidence_threshold, verbose=False)
    for detection in results_fire[0].boxes:
        cls_id = int(detection.cls[0])
        confidence = float(detection.conf[0])
        detection_type = DetectionType.FIRE if cls_id == 0 else DetectionType.SMOKE

        if confidence > config.confidence_threshold:
            reporter.report_detection(
                detection_type,
                confidence=confidence,
                frame_number=frame_count,
                metadata={"bbox": detection.xyxy[0].tolist()},
            )

    # 프레임 표시 (생략)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# 로그아웃
auth.logout()
cap.release()
cv2.destroyAllWindows()

# 통계 출력
reporter.print_detection_log()
```

---

## 📝 설정 파일 형식

### `config/tokens.json`
```json
{
  "accessToken": "eyJhbGciOiJIUzI1NiJ9...",
  "refreshToken": "eyJhbGciOiJIUzI1NiJ9...",
  "userId": "abc123",
  "savedAt": "2026-06-01T10:30:00.123456"
}
```

### `config/user.json`
```json
{
  "userId": "abc123",
  "email": "user@example.com",
  "name": "김철수",
  "role": "WARD",
  "loginAt": "2026-06-01T10:30:00.123456"
}
```

### `logs/detections/detections_2026-06-01.jsonl`
```jsonl
{"timestamp": "2026-06-01T10:30:05.123", "type": "knife", "confidence": 0.95, "frame_number": 1234, "metadata": {"bbox": [100, 150, 200, 300]}, "user_id": "abc123"}
{"timestamp": "2026-06-01T10:30:10.456", "type": "fire", "confidence": 0.87, "frame_number": 1290, "metadata": {}, "user_id": "abc123"}
```

---

## 🔄 API 응답 흐름

### 감지 발생 → 보호자 알림

```
AI (Knife 감지)
  ↓
report_detection() → 로컬 JSONL 저장
  ↓
_notify_guardians() → get_my_guardians() → API 호출
  ↓
각 보호자 조회 완료
  ↓
get_notification_settings() → 알림 설정 확인
  ↓
FCM/SMS 알림 준비 (향후 구현)
```

---

## ⚠️ 주의사항

1. **로그인 필수**: 모든 API 호출 전에 `auth.signin()`이 필요합니다
2. **토큰 자동 갱신**: 토큰 만료 시 자동 갱신됩니다
3. **네트워크 에러**: 백엔드 응답 실패 시 로컬 JSONL에만 기록됩니다
4. **보호자 역할 확인**: `api.check_guardian_role()`와 `api.check_active_ward_role()`로 확인하세요

---

## 📚 참고 자료

- [백엔드 API 문서](https://github.com/Dongyang-Mirae-University-software/SilverBridgeBe)
- [JWT 인증](https://tools.ietf.org/html/rfc7519)
- [Firebase Cloud Messaging](https://firebase.google.com/docs/cloud-messaging)

---

**마지막 업데이트**: 2026-06-01
