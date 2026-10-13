import csv
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path
from app.emotion.text_analyzer import analyze_answer

TESTSET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "testset_draft.csv"

def read_rows(path: Path) -> list[dict]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp949")
    return list(csv.DictReader(io.StringIO(text)))

def norm(value: str) -> str:
    return value.strip().lower()

FIELDS = ["emotion", "response_evasion", "social_contact", "relevance", "ambiguous"]

def expected_value(row: dict, field: str) -> str:
    return norm(row.get(field, ""))

def predicted_value(sig, field: str) -> str:
    value = getattr(sig, field)
    if isinstance(value, bool):
        return "true" if value else "false"
    return norm(str(value))

rows = read_rows(TESTSET)
correct = Counter()
checked = Counter()
by_type = defaultdict(lambda: Counter())
type_total = Counter()
mismatches = defaultdict(list)
lenient_ok = 0
lenient_total = 0
answered = 0
answered_correct = 0
total_rows = 0

for row in rows:
    if not row.get("answer", "").strip():
        continue
    sig = analyze_answer(row["question"], row["answer"])
    kind = row.get("유형", "")
    type_total[kind] += 1
    total_rows += 1

    got_emotion = predicted_value(sig, "emotion")
    exp_emotion = expected_value(row, "emotion")
    if got_emotion != "unknown":
        answered += 1
        if got_emotion == exp_emotion:
            answered_correct += 1

    allowed = [v.strip().lower() for v in row.get("emotion_허용", "").split("|") if v.strip()]
    if allowed:
        lenient_total += 1
        if got_emotion in allowed:
            lenient_ok += 1
    for field in FIELDS:
        exp = expected_value(row, field)
        if not exp:
            continue
        checked[field] += 1
        got = predicted_value(sig, field)
        if got == exp:
            correct[field] += 1
            by_type[kind][field] += 1
        else:
            mismatches[field].append(f"[{kind}] {row['answer']} → {got} (정답: {exp})")

print(f"파일: {TESTSET.name}, 문장 수: {sum(type_total.values())}")
print("\n[항목별 정확도]")
for field in FIELDS:
    if checked[field]:
        print(f"  {field}: {correct[field]}/{checked[field]}")

print("\n[허용 정답 기준 (애매 문장만)]")
if lenient_total:
    print(f"  emotion 허용 정답 안에 든 비율: {lenient_ok}/{lenient_total}")

print("\n[답변 비율]")
print(f"  emotion을 unknown이 아닌 값으로 답한 비율: {answered}/{total_rows}")
if answered:
    print(f"  답한 문장 중 엄격 기준 정답: {answered_correct}/{answered}")

print("\n[유형별 emotion 정확도]")
for kind, total in type_total.items():
    print(f"  {kind}: {by_type[kind]['emotion']}/{total}")

print("\n[틀린 문장]")
for field in FIELDS:
    for line in mismatches[field]:
        print(f"  ({field}) {line}")
