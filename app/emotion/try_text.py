from app.emotion.text_analyzer import analyze_answer


def main():
    print("질문과 답변을 입력하세요. 질문을 빈 줄로 두면 끝납니다.")
    while True:
        question = input("\n질문> ").strip()
        if not question:
            break
        answer = input("답변> ").strip()
        print(analyze_answer(question, answer).model_dump_json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
