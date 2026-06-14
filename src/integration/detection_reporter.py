import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from enum import Enum
from .config import Config
from .api_client import APIClient


class DetectionType(Enum):
    """감지 유형"""
    FIRE = "fire"
    SMOKE = "smoke"
    KNIFE = "knife"
    FALL = "fall"


class DetectionReporter:
    """감지 이벤트 보고 및 로깅"""

    def __init__(self, api_client: APIClient, config: Optional[Config] = None):
        self.api = api_client
        self.config = config or Config()
        self.log_dir = self.config.project_root / "logs" / "detections"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.daily_log_file = self._get_daily_log_file()

    def _get_daily_log_file(self) -> Path:
        """오늘자 로그 파일 경로"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"detections_{today}.jsonl"

    def report_detection(
        self,
        detection_type: DetectionType,
        confidence: float,
        frame_number: int,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """감지 이벤트 보고

        Args:
            detection_type: 감지 유형 (FIRE, SMOKE, KNIFE, FALL)
            confidence: 신뢰도 (0.0 ~ 1.0)
            frame_number: 프레임 번호
            metadata: 추가 정보 (위치, 바운딩박스 등)

        Returns:
            감지 이벤트 데이터
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": detection_type.value,
            "confidence": confidence,
            "frame_number": frame_number,
            "metadata": metadata or {},
            "user_id": self.api.auth.user_id,
        }

        # 로컬에 기록
        self._log_detection(event)

        # 보호자에게 알림 (백그라운드)
        if self.config.enable_auto_report:
            self._notify_guardians(event)

        return event

    def _log_detection(self, event: Dict) -> None:
        """감지 이벤트를 로컬 JSONL로 기록

        Args:
            event: 감지 이벤트 데이터
        """
        try:
            with open(self.daily_log_file, "a") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except IOError as e:
            print(f"❌ 감지 로그 저장 실패: {e}")

    def _notify_guardians(self, event: Dict) -> None:
        """보호자들에게 알림 발송

        Args:
            event: 감지 이벤트 데이터
        """
        if not self.api.auth.is_authenticated():
            return

        # 현재 사용자가 피보호자인지 확인
        if not self.api.check_guardian_connection():
            return

        success, guardians = self.api.get_my_guardians()
        if not success or not guardians:
            print("⚠️ 보호자 목록 조회 실패 또는 보호자 없음")
            return

        # 알림 설정 조회
        success, settings = self.api.get_notification_settings()
        if not success:
            print("⚠️ 알림 설정 조회 실패")
            return

        # 각 보호자에게 알림 (현재는 로그만, 향후 구현)
        detection_type = event["type"].upper()
        confidence = event["confidence"]
        message = f"🚨 {detection_type} 감지됨 (신뢰도: {confidence:.1%})"

        print(f"\n📢 보호자 알림 전송 준비:")
        print(f"   메시지: {message}")
        print(f"   보호자: {len(guardians)}명")
        if settings.get("fcm"):
            print(f"   ✓ FCM 푸시 활성화")
        if settings.get("sms"):
            print(f"   ✓ SMS 활성화")
        if settings.get("email"):
            print(f"   ✓ 이메일 활성화")

    def get_detection_history(self, detection_type: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """감지 이력 조회

        Args:
            detection_type: 필터링할 감지 유형 (None이면 전체)
            limit: 조회할 최대 개수

        Returns:
            감지 이벤트 목록
        """
        events = []
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"detections_{today}.jsonl"

        if not log_file.exists():
            return events

        try:
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if detection_type is None or event["type"] == detection_type:
                            events.append(event)
                    except json.JSONDecodeError:
                        continue

            return events[-limit:] if len(events) > limit else events
        except IOError:
            return events

    def get_detection_summary(self) -> Dict:
        """오늘 감지 통계

        Returns:
            {
                "total": 감지 총 건수,
                "by_type": {
                    "fire": 화재,
                    "smoke": 연기,
                    "knife": 칼,
                    "fall": 낙상
                },
                "average_confidence": 평균 신뢰도
            }
        """
        events = self.get_detection_history()

        if not events:
            return {
                "total": 0,
                "by_type": {"fire": 0, "smoke": 0, "knife": 0, "fall": 0},
                "average_confidence": 0,
            }

        by_type = {
            "fire": sum(1 for e in events if e["type"] == "fire"),
            "smoke": sum(1 for e in events if e["type"] == "smoke"),
            "knife": sum(1 for e in events if e["type"] == "knife"),
            "fall": sum(1 for e in events if e["type"] == "fall"),
        }

        avg_confidence = sum(e["confidence"] for e in events) / len(events)

        return {
            "total": len(events),
            "by_type": by_type,
            "average_confidence": avg_confidence,
        }

    def print_detection_log(self, limit: int = 20) -> None:
        """최근 감지 로그 출력

        Args:
            limit: 출력할 최대 개수
        """
        events = self.get_detection_history(limit=limit)

        if not events:
            print("📭 감지 이력이 없습니다")
            return

        print("\n" + "=" * 70)
        print(f"📊 최근 감지 이력 ({len(events)}건)")
        print("=" * 70)

        for i, event in enumerate(events, 1):
            timestamp = event["timestamp"]
            det_type = event["type"].upper()
            confidence = event["confidence"]
            print(f"{i:2}. [{timestamp}] {det_type:6} | 신뢰도: {confidence:6.1%}")

        summary = self.get_detection_summary()
        print("\n" + "-" * 70)
        print(f"📈 오늘 통계: 총 {summary['total']}건")
        for dtype, count in summary["by_type"].items():
            if count > 0:
                print(f"   - {dtype.upper()}: {count}건")
        print(f"   - 평균 신뢰도: {summary['average_confidence']:.1%}")
        print("=" * 70 + "\n")

# Updated: docs: 함수 설명 추가

# Updated: fix: 에러 처리 강화

# Updated: feat: 로깅 기능 추가

# Updated: docs: 타입 힌트 추가

# Updated: feat: 예외 처리 개선

# Updated: docs: 함수 설명 추가

# Updated: fix: 에러 처리 강화
<!-- Update 16 -->
<!-- Update 17 -->
<!-- Update 18 -->
<!-- Update 19 -->
<!-- Update 20 -->
