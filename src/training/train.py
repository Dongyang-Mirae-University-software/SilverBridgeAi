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
# ==========================================
BASE = Path("C:/Users/happy/SilverBridgeAI/AISilverBridgeLJH")

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

model = YOLO(str(BASE / "yolo26n.pt"))
model.train(
    data=data,
    epochs=epochs,
    project=str(BASE / "models"),  # ← runs/ 대신 models/ 에 바로 저장
    name=name,
)

print(f"\n[완료] models/{name}/weights/best.pt 에 저장됐습니다.")