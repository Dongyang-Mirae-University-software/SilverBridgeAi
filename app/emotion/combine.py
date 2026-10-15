def _combine(sig, face: dict | None) -> dict:
    text_neg = sig.emotion == "negative" or sig.response_evasion

    if face is None:
        level = "관찰" if text_neg else "양호"
        return {"level": level, "note": "표정 정보 없음 — 답변만으로 판단"}

    face_neg = face["negative_ratio"] >= 0.5
    if text_neg and face_neg:
        return {"level": "주의", "note": "답변과 표정 모두 부정적"}
    if sig.emotion == "positive" and face_neg:
        return {"level": "관찰", "note": "말은 긍정적이나 표정은 부정적 (불일치)"}
    if text_neg or face_neg:
        return {"level": "관찰", "note": "한쪽에서만 부정 신호"}
    return {"level": "양호", "note": "특이사항 없음"}

def combine(sig, face: dict | None) -> dict:
    result = _combine(sig, face)
    if result["level"] == "양호" and sig.relevance == "irrelevant":
        return {"level": "관찰", "note": "질문과 맞지 않는 답변"}
    return result
