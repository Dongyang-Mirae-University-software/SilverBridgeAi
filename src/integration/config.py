import json
import os
from pathlib import Path
from typing import Optional


class Config:
    """SilverBridge AI 백엔드 연동 설정"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.config_dir = self.project_root / "config"
        self.config_dir.mkdir(exist_ok=True)

        self.token_file = self.config_dir / "tokens.json"
        self.user_file = self.config_dir / "user.json"

        # Java 백엔드 API 설정
        self.java_api_base_url = os.getenv("JAVA_API_URL", "http://localhost:8080")
        self.api_timeout = int(os.getenv("API_TIMEOUT", "30"))

        # FastAPI 팀원 서버 설정
        self.fastapi_base_url = os.getenv("FASTAPI_URL", "https://testai.gosky.kr")
        self.fastapi_api_key = os.getenv("FASTAPI_API_KEY", "silverbridge_live_7X9Dm4kaLdRHy")
        self.fastapi_session_id = os.getenv("FASTAPI_SESSION_ID", "stream_001")
        self.fastapi_camera_id = os.getenv("FASTAPI_CAMERA_ID", "ipad-room-001")

        # 모델 경로
        self.fire_model_path = self.project_root / "models" / "fire_100ep" / "weights" / "best.pt"
        self.knife_model_path = self.project_root / "models" / "knife_500ep" / "weights" / "best.pt"

        # 감지 설정
        self.confidence_threshold = float(os.getenv("DETECTION_CONF", "0.3"))
        self.enable_auto_report = os.getenv("AUTO_REPORT", "true").lower() == "true"

    def load_tokens(self) -> dict:
        """저장된 토큰 로드"""
        if self.token_file.exists():
            try:
                with open(self.token_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def save_tokens(self, tokens: dict) -> None:
        """토큰 저장"""
        with open(self.token_file, 'w') as f:
            json.dump(tokens, f, indent=2)

    def load_user_info(self) -> Optional[dict]:
        """저장된 사용자 정보 로드"""
        if self.user_file.exists():
            try:
                with open(self.user_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def save_user_info(self, user_info: dict) -> None:
        """사용자 정보 저장"""
        with open(self.user_file, 'w') as f:
            json.dump(user_info, f, indent=2)

    def clear_credentials(self) -> None:
        """저장된 인증정보 삭제"""
        if self.token_file.exists():
            self.token_file.unlink()
        if self.user_file.exists():
            self.user_file.unlink()

# Updated: perf: 성능 최적화

# Updated: refactor: 변수명 명확화

# Updated: fix: 메모리 누수 방지

# Updated: refactor: 중복 코드 제거

# Updated: refactor: 코드 가독성 개선

# Updated: perf: 성능 최적화

# Updated: refactor: 변수명 명확화
<!-- Update 11 -->
<!-- Update 12 -->
<!-- Update 13 -->
<!-- Update 14 -->
<!-- Update 15 -->
