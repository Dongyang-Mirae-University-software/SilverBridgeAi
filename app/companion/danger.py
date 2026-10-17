DANGER_KEYWORDS = ["쓰러졌", "숨이 안", "가슴이 아파", "도와줘", "죽고 싶"]

def is_danger(text: str) -> bool:
    return any(k in text for k in DANGER_KEYWORDS)
