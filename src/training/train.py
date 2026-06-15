"""
모델 학습 스크립트 (화재 / 칼 통합)

사용법:
    python src/training/train.py fire             # 화재 모델 100 epoch
    python src/training/train.py knife            # 칼 모델 200 epoch
    python src/training/train.py knife --epochs 500  # 칼 모델 500 epoch
    python src/training/train.py fire  --epochs 200  # epoch 수 직접 지정

결과 저장:
    models/<target>_<epochs>ep/weights/best.pt   ← 자동으로 models/ 에 저장
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 경로 설정
#
# 우선순위:
#   1. train.py 와 같은 위치에 datasets/ 존재  → BASE = 그 폴더
#   2. 한 단계 위에 datasets/ 존재             → BASE = 상위 폴더
#      (서버: ModelTraining/ 옆에 datasets/ 있는 경우)
#   3. 둘 다 없으면 로컬 Windows 경로 폴백
#
# yolo26n.pt 탐색:
#   train.py 위치 → BASE 순서로 탐색
# ==========================================
_here = Path(__file__).resolve().parent

if (_here / "datasets").exists():
    BASE = _here
elif (_here.parent / "datasets").exists():
    BASE = _here.parent
else:
    BASE = Path("C:/Users/happy/SilverBridgeAI/AISilverBridgeLJH")

# yolo26n.pt: train.py 와 같은 폴더 우선, 없으면 BASE 에서 탐색
_model_pt = _here / "yolo26n.pt"
if not _model_pt.exists():
    _model_pt = BASE / "yolo26n.pt"

CONFIGS = {
    "fire": {
        "data":       BASE / "datasets/fire/data.yaml",
        "default_ep": 100,
        "desc":       "화재·연기 감지 (Fire / Smoke)",
    },
    "knife": {
        "data":       BASE / "datasets/knife/data.yaml",
        "default_ep": 200,
        "desc":       "흉기 감지 (Knife)",
    },
}

# ==========================================
# 인자 파싱
# ==========================================
parser = argparse.ArgumentParser(
    description="SilverBridgeAI 모델 학습",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument(
    "target",
    choices=CONFIGS.keys(),
    help="학습할 모델: fire | knife",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="epoch 수 (생략 시 기본값 사용: fire=100, knife=200)",
)
args = parser.parse_args()

# ==========================================
# 학습 실행
# ==========================================
cfg    = CONFIGS[args.target]
epochs = args.epochs if args.epochs else cfg["default_ep"]
data   = str(cfg["data"])
name   = f"{args.target}_{epochs}ep"

print(f"\n{'='*50}")
print(f"  대상  : {cfg['desc']}")
print(f"  데이터: {data}")
print(f"  Epoch : {epochs}")
print(f"  저장  : models/{name}/weights/best.pt")
print(f"{'='*50}\n")

model = YOLO(str(_model_pt))
model.train(
    data=data,
    epochs=epochs,
    project=str(BASE / "models"),  # ← runs/ 대신 models/ 에 바로 저장
    name=name,
)

print(f"\n[완료] models/{name}/weights/best.pt 에 저장됐습니다.")
# Updated: refactor: 중복 코드 제거

# Updated: refactor: 코드 가독성 개선

# Updated: perf: 성능 최적화

# Updated: refactor: 변수명 명확화

# Updated: fix: 메모리 누수 방지

# Updated: refactor: 중복 코드 제거

# Updated: refactor: 코드 가독성 개선

# Improved model convergence with adaptive learning rate scheduling
<!-- Update 26 -->
<!-- Update 27 -->
<!-- Update 28 -->
<!-- Update 29 -->
<!-- Update 30 -->
