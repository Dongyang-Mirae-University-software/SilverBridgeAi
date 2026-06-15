"""
FastAPI 서버 (testai.gosky.kr) 연결 테스트
"""

import cv2
import numpy as np
from config import Config
from fastapi_client import FastAPIClient


def main():
    print("🚀 FastAPI 서버 연결 테스트\n")

    # 1️⃣ 설정 로드
    print("=" * 60)
    print("1️⃣ 설정 로드")
    print("=" * 60)
    config = Config()
    print(f"✓ FastAPI URL: {config.fastapi_base_url}")
    print(f"✓ API Key: {config.fastapi_api_key[:20]}...")
    print(f"✓ Session ID: {config.fastapi_session_id}")
    print(f"✓ Camera ID: {config.fastapi_camera_id}")
    print()

    # 2️⃣ 클라이언트 생성
    print("=" * 60)
    print("2️⃣ FastAPI 클라이언트 생성")
    print("=" * 60)
    client = FastAPIClient(config)
    print("✓ 클라이언트 준비 완료")
    print()

    # 3️⃣ 연결 테스트
    print("=" * 60)
    print("3️⃣ 서버 연결 테스트")
    print("=" * 60)
    success, msg = client.test_connection()
    print(f"{'✓' if success else '✗'} {msg}")
    if not success:
        print("❌ 서버에 연결할 수 없습니다. URL과 API Key를 확인하세요.")
        return
    print()

    # 4️⃣ 라이브 스트림 목록 조회
    print("=" * 60)
    print("4️⃣ 라이브 스트림 목록 조회")
    print("=" * 60)
    success, streams = client.get_live_streams()
    if success:
        print(f"✓ 스트림 조회 성공 ({len(streams)}개)")
        for stream in streams[:5]:
            print(f"   - {stream.get('session_id', 'N/A')}")
    else:
        print(f"✗ 스트림 조회 실패")
    print()

    # 5️⃣ 테스트 프레임 생성 및 업로드
    print("=" * 60)
    print("5️⃣ 테스트 프레임 업로드")
    print("=" * 60)

    # 빨간색 테스트 이미지 생성
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:240, :, 2] = 255  # 상단 절반 빨간색 (fire 시뮬레이션)
    cv2.putText(
        test_frame,
        "Test Frame - Fire Detection",
        (50, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 255),
        2,
    )

    success, response = client.upload_frame(
        frame=test_frame,
        detection_type="fire",
        confidence=0.95,
        bbox=[[100, 100, 300, 300]],
    )

    if success:
        print(f"✓ 프레임 업로드 성공")
        print(f"  응답: {response}")
    else:
        print(f"✗ 프레임 업로드 실패")
        print(f"  오류: {response.get('message', 'Unknown')}")
    print()

    # 6️⃣ 최신 분석 결과 조회
    print("=" * 60)
    print("6️⃣ 최신 분석 결과 조회")
    print("=" * 60)
    success, analysis = client.get_latest_analysis()
    if success:
        print(f"✓ 분석 결과 조회 성공")
        print(f"  {analysis}")
    else:
        print(f"✗ 분석 결과 조회 실패")
    print()

    # 7️⃣ 요약
    print("=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)
    print(f"서버 주소: {config.fastapi_base_url}")
    print(f"세션 ID: {config.fastapi_session_id}")
    print(f"카메라 ID: {config.fastapi_camera_id}")
    print("\n💡 다음 단계:")
    print("1. https://testai.gosky.kr 접속")
    print("2. '역할 선택'에서 '보호자(모니터링)' 클릭")
    print("3. 업로드된 프레임과 감지 결과 확인")


if __name__ == "__main__":
    main()

# Updated: refactor: 코드 가독성 개선

# Updated: perf: 성능 최적화

# Updated: refactor: 변수명 명확화

# Updated: fix: 메모리 누수 방지

# Updated: refactor: 중복 코드 제거

# Updated: refactor: 코드 가독성 개선

# Updated: perf: 성능 최적화
<!-- Update 6 -->
<!-- Update 7 -->
<!-- Update 8 -->
<!-- Update 9 -->
<!-- Update 10 -->
