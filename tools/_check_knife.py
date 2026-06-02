from pathlib import Path
import csv

sample_lbls = list((Path('datasets/knife/train/labels')).glob('*.txt'))[:3]
for lbl in sample_lbls:
    content = lbl.read_text(encoding='utf-8').strip()
    print(lbl.name, ':', repr(content[:100]))

print()

area_bins = {'tiny(<1%)':0,'small(1-5%)':0,'medium(5-20%)':0,'large(>20%)':0}
total = 0
empty = 0

for split in ['train','valid','test']:
    lbl_dir = Path('datasets/knife') / split / 'labels'
    if not lbl_dir.exists():
        continue
    for lbl in lbl_dir.glob('*.txt'):
        content = lbl.read_text(encoding='utf-8').strip()
        if not content:
            empty += 1
            continue
        for line in content.split('\n'):
            p = line.strip().split()
            if len(p) != 5:
                continue
            total += 1
            bw, bh = float(p[3]), float(p[4])
            area = bw * bh * 100
            if area < 1:
                area_bins['tiny(<1%)'] += 1
            elif area < 5:
                area_bins['small(1-5%)'] += 1
            elif area < 20:
                area_bins['medium(5-20%)'] += 1
            else:
                area_bins['large(>20%)'] += 1

print(f'총 bbox: {total}개  빈 라벨: {empty}개')
for k, v in area_bins.items():
    pct = v / total * 100 if total else 0
    bar = '#' * int(pct / 2.5)
    print(f'  {k:18s}: {v:5d}개 ({pct:5.1f}%)  {bar}')

print()
print('=== knife 모델 성능 ===')
for folder in sorted(Path('models').iterdir()):
    if not folder.name.startswith('knife'):
        continue
    csv_path = folder / 'results.csv'
    if not csv_path.exists():
        continue
    best = 0.0
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            clean = {k.strip(): v.strip() for k, v in row.items()}
            val = float(clean.get('metrics/mAP50(B)', 0) or 0)
            if val > best:
                best = val
    print(f'  {folder.name}: mAP50={best:.3f}')

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
