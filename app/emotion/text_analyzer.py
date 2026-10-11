import re
from typing import Literal
from pydantic import BaseModel
import ollama

MODEL = "exaone3.5:2.4b"

AMBIGUOUS_HINT_PATTERN = r"글쎄|모르겠|(^|[.,!?…]\s*)몰라|아무 생각|그냥 있었|그냥 하루가"
AMBIGUOUS_HINT = re.compile(AMBIGUOUS_HINT_PATTERN)

class Signal(BaseModel):
    emotion: Literal["positive", "neutral", "negative", "unknown"]
    response_evasion: bool
    social_contact: Literal["mentioned", "none", "unknown"]
    relevance: Literal["relevant", "irrelevant", "unknown"]
    ambiguous: bool
    evidence: str

ANALYZE_PROMPT = """당신은 독거노인 돌봄 서비스의 답변 분석기입니다.
보호자가 낸 질문과 어르신의 답변을 읽고, 규칙에 따라 JSON으로만 답합니다.

[규칙]
- 어르신의 '답변'에 직접 드러난 내용만 판단합니다. 근거가 없으면 반드시 unknown 또는 false 로 둡니다. 추측하지 마세요.
- emotion: 아래 순서로 판단하세요.
  1) 기분 좋은 단어(맛있다, 좋다, 재밌다, 즐겁다, 기쁘다, 편안하다, 행복하다)가 있으면 positive
  2) 기분 나쁜 단어(심심하다, 답답하다, 외롭다, 슬프다, 속상하다, 서운하다, 힘들다)가 있으면 negative
  3) 사람과 즐거운 활동을 같이 했으면(같이 점심 먹었어, 산책, 모임, 길게 통화 등) positive
  4) 사람과 연락이 없었다는 표현(아무도 안 와, 연락이 없다, 찾아오는 사람이 없다)이 있으면 negative
  5) 긍정과 부정이 함께 있거나, 사실만 말했으면 neutral
  6) 판단이 애매하면 unknown. 감정을 추측하지 마세요.
  - "먹었어", "다녀왔어", "나갔다 왔어", "했어" 같은 평범한 활동은 사실일 뿐입니다. 즐거움을 나타내는 단어가 없으면 neutral 입니다.
  - "혼자"는 사실로만 봅니다. 감정 단어(심심하다, 외롭다 등)가 함께 있을 때만 negative 입니다.
- ambiguous 가 true 이면 emotion 은 반드시 unknown 으로 쓰세요
- response_evasion: 답을 거부하거나("그 얘기는 하기 싫어"), 화제를 명확히 돌리는 표현이 있을 때만 true. 엉뚱한 답은 relevance 로만 표시하고 response_evasion 은 false. 의욕이 없다는 말만으로는 false
- social_contact: 오늘 사람을 만났거나 대화했다고 말하면 mentioned, 혼자 지낸다고 말하면 none, 계획만 말하거나 언급이 없으면 unknown
- relevance: 답변이 질문에 대한 답이면 relevant, 질문과 전혀 상관없는 말이면 irrelevant, 판단할 수 없으면 unknown
- ambiguous: 같은 말이 여러 뜻으로 읽힐 수 있으면 true (예: "아무것도 하기 싫어"는 무기력일 수도, 그냥 한 말일 수도 있음). 뜻이 분명하면 false
- evidence: 판단 근거가 된 답변 일부 (없으면 빈 문자열)

[예시]
질문: 오늘 점심은 드셨어요?
답변: 응, 손주가 와서 같이 먹었어. 너무 좋았지.
답: {"emotion":"positive","response_evasion":false,"social_contact":"mentioned","relevance":"relevant","ambiguous":false,"evidence":"손주가 와서 같이 먹었어"}

질문: 잠은 잘 주무세요?
답변: 그 얘기는 하기 싫어. 날씨나 얘기하자.
답: {"emotion":"neutral","response_evasion":true,"social_contact":"unknown","relevance":"irrelevant","ambiguous":false,"evidence":"그 얘기는 하기 싫어"}

질문: 오늘 점심은 드셨어요?
답변: 혼자 맛있는 거 먹었어.
답: {"emotion":"positive","response_evasion":false,"social_contact":"none","relevance":"relevant","ambiguous":false,"evidence":"맛있는 거 먹었어"}

질문: 요즘 누구 만나셨어요?
답변: 아들이 와서 같이 점심 먹었어.
답: {"emotion":"positive","response_evasion":false,"social_contact":"mentioned","relevance":"relevant","ambiguous":false,"evidence":"아들이 와서 같이 점심 먹었어"}

질문: 오늘 점심은 드셨어요?
답변: 텔레비전에 나온 연예인이 참 잘생겼더라.
답: {"emotion":"neutral","response_evasion":false,"social_contact":"unknown","relevance":"irrelevant","ambiguous":false,"evidence":""}

질문: 오늘 하루 어떠셨어요?
답변: 글쎄 모르겠어.
답: {"emotion":"neutral","response_evasion":false,"social_contact":"unknown","relevance":"relevant","ambiguous":true,"evidence":""}

질문: 요즘 누구 만나셨어요?
답변: 요즘 통 입맛이 없고 아무도 안 와.
답: {"emotion":"negative","response_evasion":false,"social_contact":"none","relevance":"relevant","ambiguous":false,"evidence":"아무도 안 와"}
"""

def analyze_answer(question: str, answer: str) -> Signal:
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": ANALYZE_PROMPT},
            {"role": "user", "content": f"질문: {question}\n답변: {answer}"},
        ],
        format=Signal.model_json_schema(),
        options={"temperature": 0},
    )
    signal = Signal.model_validate_json(resp.message.content)
    if AMBIGUOUS_HINT.search(answer):
        signal = signal.model_copy(update={"ambiguous": True})
    if signal.ambiguous and signal.emotion != "unknown":
        signal = signal.model_copy(update={"emotion": "unknown"})
    return signal

if __name__ == "__main__":
    print(analyze_answer("요즘 식사는 잘 하세요?", "입맛이 없어서 거의 못 먹어."))
