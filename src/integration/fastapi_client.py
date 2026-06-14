import requests
import cv2
import numpy as np
from typing import Tuple, Optional
from io import BytesIO
from .config import Config


class FastAPIClient:
    """팀원 FastAPI 서버 (testai.gosky.kr) 통신"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.session = requests.Session()
        self.session.timeout = self.config.api_timeout
        self.session.headers.update({
            "X-API-Key": self.config.fastapi_api_key,
        })

    def upload_frame(
        self,
        frame: np.ndarray,
        detection_type: Optional[str] = None,
        confidence: Optional[float] = None,
        bbox: Optional[list] = None,
    ) -> Tuple[bool, dict]:
        """감지된 프레임을 FastAPI 서버에 업로드

        Args:
            frame: OpenCV 이미지 (BGR)
            detection_type: 감지 유형 (fire, smoke, knife, fall)
            confidence: 신뢰도 (0.0 ~ 1.0)
            bbox: 바운딩박스 [[x1, y1, x2, y2], ...]

        Returns:
            (성공 여부, 응답 데이터)
        """
        try:
            # 이미지를 JPEG로 인코딩
            success, jpeg_data = cv2.imencode(".jpg", frame)
            if not success:
                return False, {"message": "이미지 인코딩 실패"}

            # 메타데이터 추가
            metadata = {}
            if detection_type:
                metadata["detection_type"] = detection_type
            if confidence is not None:
                metadata["confidence"] = confidence
            if bbox:
                metadata["bbox"] = bbox

            # 업로드
            url = (
                f"{self.config.fastapi_base_url}/api/v1/stream-sessions/"
                f"{self.config.fastapi_session_id}/frame"
            )

            files = {"file": ("frame.jpg", BytesIO(jpeg_data.tobytes()), "image/jpeg")}
            data = {
                "camera_id": self.config.fastapi_camera_id,
                **metadata,
            }

            response = self.session.post(url, files=files, data=data)

            if response.status_code in [200, 201]:
                return True, response.json() if response.text else {"status": "uploaded"}

            return False, response.json() if response.text else {"message": "업로드 실패"}

        except requests.RequestException as e:
            return False, {"message": f"네트워크 오류: {str(e)}"}
        except Exception as e:
            return False, {"message": f"오류: {str(e)}"}

    def create_session(self, camera_id: Optional[str] = None) -> Tuple[bool, dict]:
        """스트림 세션 생성

        Args:
            camera_id: 카메라 ID (기본값: config의 camera_id 사용)

        Returns:
            (성공 여부, 세션 데이터)
        """
        try:
            url = f"{self.config.fastapi_base_url}/api/v1/stream-sessions"
            payload = {
                "camera_id": camera_id or self.config.fastapi_camera_id,
            }

            response = self.session.post(url, json=payload)

            if response.status_code in [200, 201]:
                data = response.json().get("data", {})
                session_id = data.get("session_id")
                if session_id:
                    self.config.fastapi_session_id = session_id
                return True, data

            return False, response.json() if response.text else {"message": "세션 생성 실패"}

        except requests.RequestException as e:
            return False, {"message": f"네트워크 오류: {str(e)}"}

    def get_live_streams(self) -> Tuple[bool, list]:
        """라이브 스트림 목록 조회

        Returns:
            (성공 여부, 스트림 목록)
        """
        try:
            url = f"{self.config.fastapi_base_url}/api/v1/live-streams"
            response = self.session.get(url)

            if response.status_code == 200:
                data = response.json().get("data", [])
                return True, data if isinstance(data, list) else [data]

            return False, []

        except requests.RequestException as e:
            print(f"❌ 스트림 목록 조회 실패: {e}")
            return False, []

    def get_latest_analysis(self, session_id: Optional[str] = None) -> Tuple[bool, dict]:
        """최신 AI 분석 결과 조회

        Args:
            session_id: 세션 ID (기본값: config의 session_id 사용)

        Returns:
            (성공 여부, 분석 결과)
        """
        try:
            session_id = session_id or self.config.fastapi_session_id
            url = f"{self.config.fastapi_base_url}/api/v1/live-streams/{session_id}/latest-analysis"
            response = self.session.get(url)

            if response.status_code == 200:
                return True, response.json().get("data", {})

            return False, response.json() if response.text else {}

        except requests.RequestException as e:
            return False, {"message": f"네트워크 오류: {str(e)}"}

    def test_connection(self) -> Tuple[bool, str]:
        """연결 테스트

        Returns:
            (성공 여부, 메시지)
        """
        try:
            url = f"{self.config.fastapi_base_url}/api/v1/live-streams"
            response = self.session.get(url, timeout=5)

            if response.status_code == 200:
                return True, f"✓ 연결 성공 ({self.config.fastapi_base_url})"
            else:
                return False, f"✗ 상태 코드: {response.status_code}"

        except requests.ConnectionError:
            return False, f"✗ 연결 실패: {self.config.fastapi_base_url}"
        except requests.Timeout:
            return False, "✗ 타임아웃"
        except Exception as e:
            return False, f"✗ 오류: {str(e)}"

# Updated: feat: 로깅 기능 추가

# Updated: docs: 타입 힌트 추가

# Updated: feat: 예외 처리 개선

# Updated: docs: 함수 설명 추가

# Updated: fix: 에러 처리 강화

# Updated: feat: 로깅 기능 추가

# Updated: docs: 타입 힌트 추가
<!-- Update 36 -->
<!-- Update 37 -->
<!-- Update 38 -->
<!-- Update 39 -->
<!-- Update 40 -->
