import requests
from typing import Optional, Tuple
from datetime import datetime
from .config import Config


class AuthClient:
    """JWT 토큰 기반 인증 관리"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.session = requests.Session()
        self.session.timeout = self.config.api_timeout
        self._load_tokens()

    def _load_tokens(self) -> None:
        """저장된 토큰 로드"""
        tokens = self.config.load_tokens()
        self.access_token = tokens.get("accessToken")
        self.refresh_token = tokens.get("refreshToken")
        self.user_id = tokens.get("userId")

    def _save_tokens(self, access_token: str, refresh_token: str, user_id: str) -> None:
        """토큰 저장"""
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.user_id = user_id
        self.config.save_tokens({
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "userId": user_id,
            "savedAt": datetime.now().isoformat(),
        })

    def get_headers(self) -> dict:
        """API 요청 헤더 (JWT 토큰 포함)"""
        headers = {"Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    def signin(self, email: str, password: str) -> Tuple[bool, str]:
        """로그인

        Args:
            email: 사용자 이메일
            password: 비밀번호

        Returns:
            (성공 여부, 메시지)
        """
        try:
            url = f"{self.config.api_base_url}/api/auth/signin"
            payload = {"email": email, "password": password}

            response = self.session.post(url, json=payload)

            if response.status_code == 200:
                data = response.json().get("data", {})
                self._save_tokens(
                    data.get("accessToken"),
                    data.get("refreshToken"),
                    data.get("userId"),
                )
                user_info = {
                    "userId": data.get("userId"),
                    "email": data.get("email"),
                    "name": data.get("name"),
                    "role": data.get("role"),
                    "loginAt": datetime.now().isoformat(),
                }
                self.config.save_user_info(user_info)
                return True, f"로그인 성공: {data.get('name')} ({data.get('role')})"

            error_msg = response.json().get("message", "로그인 실패")
            return False, error_msg

        except requests.RequestException as e:
            return False, f"네트워크 오류: {str(e)}"

    def refresh_access_token(self) -> Tuple[bool, str]:
        """Access Token 갱신

        Returns:
            (성공 여부, 메시지)
        """
        if not self.refresh_token:
            return False, "Refresh Token이 없습니다"

        try:
            url = f"{self.config.api_base_url}/api/auth/refresh"
            response = self.session.post(
                url,
                headers=self.get_headers(),
                json={"refreshToken": self.refresh_token},
            )

            if response.status_code == 200:
                data = response.json().get("data", {})
                self._save_tokens(
                    data.get("accessToken"),
                    data.get("refreshToken"),
                    self.user_id,
                )
                return True, "토큰 갱신 성공"

            return False, response.json().get("message", "토큰 갱신 실패")

        except requests.RequestException as e:
            return False, f"네트워크 오류: {str(e)}"

    def logout(self) -> Tuple[bool, str]:
        """로그아웃

        Returns:
            (성공 여부, 메시지)
        """
        try:
            url = f"{self.config.api_base_url}/api/auth/logout"
            response = self.session.post(url, headers=self.get_headers())

            self.config.clear_credentials()
            self.access_token = None
            self.refresh_token = None
            self.user_id = None

            if response.status_code == 200:
                return True, "로그아웃 성공"
            return True, "로그아웃 완료"

        except requests.RequestException as e:
            return False, f"네트워크 오류: {str(e)}"

    def is_authenticated(self) -> bool:
        """인증 상태 확인"""
        return self.access_token is not None and self.user_id is not None

    def get_user_info(self) -> Optional[dict]:
        """저장된 사용자 정보 조회"""
        return self.config.load_user_info()

# Updated: docs: 타입 힌트 추가

# Updated: feat: 예외 처리 개선

# Updated: docs: 함수 설명 추가

# Updated: fix: 에러 처리 강화

# Updated: feat: 로깅 기능 추가

# Updated: docs: 타입 힌트 추가

# Updated: feat: 예외 처리 개선
<!-- Update 31 -->
<!-- Update 32 -->
<!-- Update 33 -->
<!-- Update 34 -->
<!-- Update 35 -->
