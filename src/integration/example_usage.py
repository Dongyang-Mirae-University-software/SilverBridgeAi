"""
SilverBridge AI - 백엔드 통합 사용 예제

이 파일은 auth_client, api_client, detection_reporter를 어떻게 사용하는지 보여줍니다.
"""

from config import Config
from auth_client import AuthClient
from api_client import APIClient
from detection_reporter import DetectionReporter, DetectionType


def main():
    print("🚀 SilverBridge AI - 백엔드 통합 예제\n")

    # 1️⃣ 설정 로드
    print("=" * 60)
    print("1️⃣ 설정 초기화")
    print("=" * 60)
    config = Config()
    print(f"✓ API URL: {config.api_base_url}")
    print(f"✓ 신뢰도 임계값: {config.confidence_threshold}")
    print()

    # 2️⃣ 인증
    print("=" * 60)
    print("2️⃣ 로그인")
    print("=" * 60)
    auth = AuthClient(config)

    if auth.is_authenticated():
        user_info = auth.get_user_info()
        print(f"✓ 이전 세션: {user_info['name']} ({user_info['role']})")
    else:
        email = input("📧 이메일 입력: ").strip()
        password = input("🔐 비밀번호 입력: ").strip()

        success, msg = auth.signin(email, password)
        print(f"{'✓' if success else '✗'} {msg}")

        if not success:
            return

    user_info = auth.get_user_info()
    print(f"✓ 로그인 사용자: {user_info['name']} (ID: {user_info['userId']})")
    print(f"✓ 역할: {user_info['role']}")
    print()

    # 3️⃣ API 클라이언트
    print("=" * 60)
    print("3️⃣ API 클라이언트 초기화")
    print("=" * 60)
    api = APIClient(auth, config)
    print("✓ API 클라이언트 준비 완료")
    print()

    # 4️⃣ 사용자 정보 조회
    print("=" * 60)
    print("4️⃣ 내 프로필 조회")
    print("=" * 60)
    success, profile = api.get_my_profile()
    if success:
        print(f"✓ 이름: {profile.get('name')}")
        print(f"✓ 이메일: {profile.get('email')}")
        print(f"✓ 전화: {profile.get('phone')}")
        print(f"✓ 역할: {profile.get('role')}")
    else:
        print(f"✗ 프로필 조회 실패: {profile.get('message')}")
    print()

    # 5️⃣ 알림 설정 확인
    print("=" * 60)
    print("5️⃣ 알림 설정 확인")
    print("=" * 60)
    success, settings = api.get_notification_settings()
    if success:
        print(f"✓ FCM 푸시: {'활성' if settings.get('fcm') else '비활성'}")
        print(f"✓ SMS: {'활성' if settings.get('sms') else '비활성'}")
        print(f"✓ 이메일: {'활성' if settings.get('email') else '비활성'}")
        print(f"✓ 카카오 알림톡: {'활성' if settings.get('kakaoAlimtalk') else '비활성'}")
    else:
        print(f"✗ 알림 설정 조회 실패: {settings.get('message')}")
    print()

    # 6️⃣ 역할별 연결 정보
    print("=" * 60)
    print("6️⃣ 역할에 따른 연결 정보")
    print("=" * 60)

    is_guardian = api.check_guardian_role()
    is_ward = api.check_active_ward_role()

    if is_guardian:
        print("🛡️ 보호자 역할")
        success, wards = api.get_my_wards()
        if success:
            print(f"✓ 피보호자: {len(wards)}명")
            for ward in wards:
                print(f"   - {ward.get('name')} ({ward.get('status')})")
        else:
            print(f"✗ 피보호자 조회 실패")

    elif is_ward:
        print("👤 피보호자 역할")
        success, guardians = api.get_my_guardians()
        if success:
            print(f"✓ 보호자: {len(guardians)}명")
            for guardian in guardians:
                print(f"   - {guardian.get('name')} ({guardian.get('address')})")
        else:
            print(f"✗ 보호자 조회 실패")

        success, pending = api.get_pending_requests()
        if success and pending:
            print(f"⏳ 대기 중인 요청: {len(pending)}건")
            for req in pending:
                print(f"   - {req.get('name')} ({req.get('relation')})")
    print()

    # 7️⃣ 감지 이벤트 보고
    print("=" * 60)
    print("7️⃣ 감지 이벤트 보고 (예시)")
    print("=" * 60)
    reporter = DetectionReporter(api, config)

    # 예시: Knife 감지
    event = reporter.report_detection(
        detection_type=DetectionType.KNIFE,
        confidence=0.95,
        frame_number=1234,
        metadata={"location": "living_room", "bbox": [100, 150, 200, 300]},
    )
    print(f"✓ 감지 보고 완료:")
    print(f"   - 유형: {event['type'].upper()}")
    print(f"   - 신뢰도: {event['confidence']:.1%}")
    print(f"   - 타임스탐프: {event['timestamp']}")
    print()

    # 8️⃣ 감지 통계
    print("=" * 60)
    print("8️⃣ 감지 통계")
    print("=" * 60)
    summary = reporter.get_detection_summary()
    print(f"✓ 오늘 감지: {summary['total']}건")
    for dtype, count in summary["by_type"].items():
        if count > 0:
            print(f"   - {dtype.upper()}: {count}건")
    if summary["total"] > 0:
        print(f"✓ 평균 신뢰도: {summary['average_confidence']:.1%}")
    print()

    # 9️⃣ 감지 로그 출력
    print("=" * 60)
    print("9️⃣ 감지 로그")
    print("=" * 60)
    reporter.print_detection_log(limit=10)

    # 🔟 로그아웃
    print("=" * 60)
    print("🔟 로그아웃")
    print("=" * 60)
    success, msg = auth.logout()
    print(f"{'✓' if success else '✗'} {msg}")


if __name__ == "__main__":
    main()

# Updated: fix: 메모리 누수 방지

# Updated: refactor: 중복 코드 제거

# Updated: refactor: 코드 가독성 개선

# Updated: perf: 성능 최적화

# Updated: refactor: 변수명 명확화

# Updated: fix: 메모리 누수 방지

# Updated: refactor: 중복 코드 제거
<!-- Update 1 -->
<!-- Update 2 -->
<!-- Update 3 -->
<!-- Update 4 -->
<!-- Update 5 -->
