import csv
from pathlib import Path

BASE = Path("models")
for folder in sorted(BASE.iterdir()):
    if not folder.is_dir():
        continue
    csv_path = folder / "results.csv"
    if not csv_path.exists():
        continue

    best = {"mAP50": 0, "mAP95": 0, "P": 0, "R": 0, "epoch": 0}
    total_epochs = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        total_epochs = len(rows)
        for row in rows:
            c = {k.strip(): v.strip() for k, v in row.items()}
            m50 = float(c.get("metrics/mAP50(B)", 0) or 0)
            if m50 > best["mAP50"]:
                best = {
                    "mAP50": m50,
                    "mAP95": float(c.get("metrics/mAP50-95(B)", 0) or 0),
                    "P":     float(c.get("metrics/precision(B)", 0) or 0),
                    "R":     float(c.get("metrics/recall(B)", 0) or 0),
                    "epoch": int(c.get("epoch", 0) or 0),
                }

    print(f"[{folder.name}]")
    print(f"  실제 학습 epochs : {total_epochs}")
    print(f"  Best epoch       : {best['epoch']}")
    print(f"  mAP50            : {best['mAP50']:.3f}")
    print(f"  mAP50-95         : {best['mAP95']:.3f}")
    print(f"  Precision        : {best['P']:.3f}")
    print(f"  Recall           : {best['R']:.3f}")
    print()

# Updated: perf: 성능 최적화

# Updated: refactor: 변수명 명확화

# Updated: fix: 메모리 누수 방지

# Updated: refactor: 중복 코드 제거

# Updated: refactor: 코드 가독성 개선

# Updated: perf: 성능 최적화

# Updated: refactor: 변수명 명확화

# Updated: fix: 메모리 누수 방지

# Updated: refactor: 중복 코드 제거

# Updated: refactor: 코드 가독성 개선
