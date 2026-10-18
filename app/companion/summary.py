import ollama
MODEL = "exaone3.5:2.4b"

def summarize(messages: list[dict]) -> str:
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "다음 대화에서 실제로 언급된 사실만 3줄 이내로 요약하세요. 대화에 없는 내용은 절대 쓰지 마세요."},
            {"role": "user", "content": text},
        ],
        options={"temperature": 0},
    )
    return resp.message.content
