#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
import json
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 키 가져오기
API_KEY = os.getenv("TOGETHER_API_KEY")

if not API_KEY:
    print("❌ 오류: TOGETHER_API_KEY가 설정되지 않았습니다.")
    print("   .env 파일을 확인하세요.")
    exit(1)

print(f"✅ API 키 로드 완료 (키 길이: {len(API_KEY)})")

# EXAONE 모델로 감정 분석
def analyze_emotion_exaone(text):
    """EXAONE을 사용한 감정 분석"""

    url = "https://api.together.xyz/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""다음 텍스트의 감정을 분석해줘.

텍스트: "{text}"

응답은 JSON 형식으로만 해줘 (다른 설명 없이):
{{"emotion": "positive|negative|neutral", "confidence": 0.0~1.0}}
"""

    data = {
        "model": "lgai/EXAONE-3.5-32B-Instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }

    try:
        print(f"\n📤 요청 중... (모델: EXAONE-3.5-32B-Instruct)")
        response = requests.post(url, headers=headers, json=data, timeout=30)

        print(f"📊 상태 코드: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"✅ 응답 성공!")
            print(f"📝 EXAONE 응답: {content}")
            return content
        else:
            error_msg = response.json()
            print(f"❌ API 오류 ({response.status_code}):")
            print(f"   {error_msg}")
            return None

    except requests.exceptions.Timeout:
        print(f"❌ 타임아웃: 요청이 30초를 초과했습니다.")
        return None
    except requests.exceptions.ConnectionError:
        print(f"❌ 연결 오류: 인터넷 연결을 확인하세요.")
        return None
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return None

# 메인 테스트
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 EXAONE 감정 분석 테스트")
    print("=" * 60)

    test_texts = [
        "오늘 날씨 정말 좋네!",
        "완전 화났어",
        "그냥 그런 날이야"
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n[테스트 {i}/3]")
        print(f"입력: '{text}'")
        analyze_emotion_exaone(text)
        print("-" * 60)

    print("\n✅ 테스트 완료!")
