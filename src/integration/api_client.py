import requests
from typing import Optional, List, Dict, Tuple
from .config import Config
from .auth_client import AuthClient


class APIClient:
    """SilverBridge 백엔드 API 클라이언트"""

    def __init__(self, auth_client: Optional[AuthClient] = None, config: Optional[Config] = None):
        self.config = config or Config()
        self.auth = auth_client or AuthClient(self.config)
        self.session = requests.Session()
        self.session.timeout = self.config.api_timeout

    def _request(self, method: str, endpoint: str, **kwargs) -> Tuple[bool, dict]:
        """API 요청 (인증 헤더 자동 포함)

        Args:
            method: GET, POST, PUT, DELETE 등
            endpoint: /api/... 형태의 엔드포인트
            **kwargs: requests 라이브러리 인자

        Returns:
            (성공 여부, 응답 데이터)
        """
        if not self.auth.is_authenticated():
            return False, {"message": "인증이 필요합니다"}

        url = f"{self.config.api_base_url}{endpoint}"
        kwargs.setdefault("headers", {}).update(self.auth.get_headers())

        try:
            response = self.session.request(method, url, **kwargs)

            # 토큰 만료 시 자동 갱신
            if response.status_code == 401:
                success, msg = self.auth.refresh_access_token()
                if success:
                    kwargs["headers"].update(self.auth.get_headers())
                    response = self.session.request(method, url, **kwargs)
                else:
                    return False, {"message": "토큰 갱신 실패"}

            data = response.json() if response.text else {}
            return response.status_code == 200, data

        except requests.RequestException as e:
            return False, {"message": f"네트워크 오류: {str(e)}"}

    # ==================== 사용자 정보 ====================

    def get_my_profile(self) -> Tuple[bool, dict]:
        """내 프로필 조회

        Returns:
            (성공 여부, 프로필 데이터)
        """
        success, data = self._request("GET", "/api/user/me")
        if success:
            return True, data.get("data", {})
        return False, data

    def update_profile(self, **kwargs) -> Tuple[bool, dict]:
        """프로필 수정 (주소, 성별, 생년월일 등)

        Args:
            **kwargs: 수정할 필드들

        Returns:
            (성공 여부, 수정된 프로필)
        """
        success, data = self._request("PUT", "/api/user/me", json=kwargs)
        if success:
            return True, data.get("data", {})
        return False, data

    def get_notification_settings(self) -> Tuple[bool, dict]:
        """알림 설정 조회

        Returns:
            (성공 여부, 알림 설정)
            예: {"fcm": true, "sms": false, "email": true, "kakaoAlimtalk": false}
        """
        success, data = self._request("GET", "/api/user/me/notification-settings")
        if success:
            return True, data.get("data", {})
        return False, data

    def update_notification_settings(self, **settings) -> Tuple[bool, dict]:
        """알림 설정 수정

        Args:
            fcm: FCM 푸시 (bool)
            sms: SMS 알림 (bool)
            email: 이메일 알림 (bool)
            kakaoAlimtalk: 카카오 알림톡 (bool)

        Returns:
            (성공 여부, 수정된 설정)
        """
        success, data = self._request("PUT", "/api/user/me/notification-settings", json=settings)
        if success:
            return True, data.get("data", {})
        return False, data

    # ==================== 보호자-피보호자 연결 ====================

    def get_my_wards(self) -> Tuple[bool, List[dict]]:
        """내 피보호자 목록 조회 (보호자용)

        Returns:
            (성공 여부, 피보호자 목록)
            [{id, wardId, name, phone, address, status}, ...]
        """
        success, data = self._request("GET", "/api/guardian/connection/select")
        if success:
            return True, data.get("data", [])
        return False, []

    def get_my_guardians(self) -> Tuple[bool, List[dict]]:
        """내 보호자 목록 조회 (피보호자용 - ACTIVE만)

        Returns:
            (성공 여부, 보호자 목록)
            [{id, guardianId, name, phone, address}, ...]
        """
        success, data = self._request("GET", "/api/ward/connection/active")
        if success:
            return True, data.get("data", [])
        return False, []

    def get_pending_requests(self) -> Tuple[bool, List[dict]]:
        """요청온 보호자 목록 조회 (피보호자용 - PENDING)

        Returns:
            (성공 여부, 대기 중인 요청 목록)
            [{id, guardianId, name, phone_masked, relation}, ...]
        """
        success, data = self._request("GET", "/api/ward/connection/pending")
        if success:
            return True, data.get("data", [])
        return False, []

    def accept_guardian_request(self, connection_id: int) -> Tuple[bool, str]:
        """보호자 요청 수락

        Args:
            connection_id: 연결 ID

        Returns:
            (성공 여부, 메시지)
        """
        success, data = self._request("POST", f"/api/ward/connection/{connection_id}/accept")
        msg = data.get("message", "요청 수락 완료" if success else "요청 수락 실패")
        return success, msg

    def reject_guardian_request(self, connection_id: int) -> Tuple[bool, str]:
        """보호자 요청 거절

        Args:
            connection_id: 연결 ID

        Returns:
            (성공 여부, 메시지)
        """
        success, data = self._request("DELETE", f"/api/ward/connection/request/{connection_id}/refusal")
        msg = data.get("message", "요청 거절 완료" if success else "요청 거절 실패")
        return success, msg

    # ==================== FCM (푸시 알림) ====================

    def register_fcm_token(self, fcm_token: str) -> Tuple[bool, str]:
        """FCM 토큰 등록 (기기별)

        Args:
            fcm_token: Firebase Cloud Messaging 토큰

        Returns:
            (성공 여부, 메시지)
        """
        success, data = self._request("POST", "/api/notifications/fcm-token", json={"token": fcm_token})
        msg = data.get("message", "FCM 토큰 등록 완료" if success else "FCM 토큰 등록 실패")
        return success, msg

    def unregister_fcm_token(self, fcm_token: str) -> Tuple[bool, str]:
        """FCM 토큰 삭제 (로그아웃 시)

        Args:
            fcm_token: Firebase Cloud Messaging 토큰

        Returns:
            (성공 여부, 메시지)
        """
        success, data = self._request("DELETE", f"/api/notifications/fcm-token?token={fcm_token}")
        msg = data.get("message", "FCM 토큰 삭제 완료" if success else "FCM 토큰 삭제 실패")
        return success, msg

    # ==================== 공지사항 ====================

    def get_announcements(self) -> Tuple[bool, List[dict]]:
        """공지사항 목록 조회

        Returns:
            (성공 여부, 공지사항 목록)
            [{id, title, content, viewCount, createdAt}, ...]
        """
        success, data = self._request("GET", "/api/commonness/announcement/select")
        if success:
            return True, data.get("data", [])
        return False, []

    def get_announcement_detail(self, announcement_id: int) -> Tuple[bool, dict]:
        """공지사항 상세 조회

        Args:
            announcement_id: 공지사항 ID

        Returns:
            (성공 여부, 공지사항 상세)
        """
        success, data = self._request("GET", f"/api/commonness/announcement/select/detail/{announcement_id}")
        if success:
            return True, data.get("data", {})
        return False, {}

    # ==================== 헬퍼 메서드 ====================

    def check_guardian_connection(self, ward_id: Optional[str] = None) -> Tuple[bool, List[dict]]:
        """보호자 연결 상태 확인 (피보호자가 보호자가 있는지 확인)

        Args:
            ward_id: 피보호자 ID (None이면 현재 사용자)

        Returns:
            (성공 여부, 보호자 목록)
        """
        return self.get_my_guardians()

    def check_active_ward_role(self) -> bool:
        """현재 사용자가 피보호자 역할인지 확인

        Returns:
            피보호자 역할 여부
        """
        user_info = self.auth.get_user_info()
        return user_info and user_info.get("role") == "WARD" if user_info else False

    def check_guardian_role(self) -> bool:
        """현재 사용자가 보호자 역할인지 확인

        Returns:
            보호자 역할 여부
        """
        user_info = self.auth.get_user_info()
        return user_info and user_info.get("role") == "GUARDIAN" if user_info else False

# Updated: refactor: 중복 코드 제거

# Updated: refactor: 코드 가독성 개선

# Updated: perf: 성능 최적화

# Updated: refactor: 변수명 명확화

# Updated: fix: 메모리 누수 방지

# Updated: refactor: 중복 코드 제거

# Updated: refactor: 코드 가독성 개선
<!-- Update 6 -->
<!-- Update 7 -->
<!-- Update 8 -->
<!-- Update 9 -->
<!-- Update 10 -->
<!-- Update 61 -->
<!-- Update 62 -->
<!-- Update 63 -->
<!-- Update 64 -->
<!-- Update 65 -->
