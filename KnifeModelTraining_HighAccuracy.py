#!/usr/bin/env python3
"""
Knife Detection Model Training - High Accuracy Version
목표: 정확도 높이고, 오탐/오감지 최소화, 정밀도 극대화

실행 방법 (서버):
    cd /home/apps/SilverBridgeSky/SilverBridgeJH/ModelTraining
    source venv/bin/activate
    nohup python3 KnifeModelTraining_HighAccuracy.py > knife_training.log 2>&1 &
    tail -f knife_training.log

"""

import os
import sys
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import torch

# ============================================================================
# 설정 (서버 환경에 맞게 수정)
# ============================================================================

# 1. 데이터셋 경로 (서버 경로)
DATASET_PATH = Path("/home/apps/SilverBridgeSky/SilverBridgeJH/new_datasets/knife")
DATA_YAML = DATASET_PATH / "data.yaml"

# 2. 학습 결과 저장 경로
OUTPUT_BASE = Path("/home/apps/SilverBridgeSky/SilverBridgeJH/ModelTraining")
RUNS_DIR = OUTPUT_BASE / "runs/detect"
MODELS_DIR = OUTPUT_BASE / "models/knife"

# 3. 기본 모델 (경량: YOLOv8n, 고정확도: YOLOv8s)
BASE_MODEL = "yolov8n.pt"  # 또는 "yolov8s.pt" (더 정확하지만 느림)

# 4. 학습 설정 (고정확도 최적화)
TRAINING_CONFIG = {
    "epochs": 300,              # 충분히 오래 학습
    "batch": 16,                # 안정적인 배치 크기
    "imgsz": 640,               # 이미지 크기 (640 또는 832)
    "patience": 60,             # Early stopping patience
    "save": True,               # 모델 저장
    "device": 0,                # GPU 0번 (여러 개면 [0,1,2,...])
    "pretrained": True,         # 사전학습 가중치 사용
    "optimizer": "SGD",         # SGD (안정적), Adam (빠름)
    "lr0": 0.01,                # Initial learning rate
    "lrf": 0.01,                # Final learning rate
    "momentum": 0.937,          # Momentum
    "weight_decay": 0.0005,     # L2 정규화
    "warmup_epochs": 3.0,       # Warmup epochs
    "warmup_momentum": 0.8,
    "box": 7.5,                 # Box loss gain
    "cls": 0.5,                 # Cls loss gain
    "hsv_h": 0.015,             # HSV H augmentation
    "hsv_s": 0.7,               # HSV S augmentation
    "hsv_v": 0.4,               # HSV V augmentation
    "degrees": 10.0,            # Rotation
    "translate": 0.1,           # Translation
    "scale": 0.5,               # Scale
    "flipud": 0.0,              # Flip upside down
    "fliplr": 0.5,              # Flip left-right
    "mosaic": 1.0,              # Mosaic augmentation
    "mixup": 0.1,               # Mixup augmentation
    "copy_paste": 0.0,
    "seed": 42,                 # 재현성
    "verbose": True,
    "plots": True,              # 학습 곡선 저장
    "exist_ok": False,
}

# ============================================================================
# 함수
# ============================================================================

def log_message(msg):
    """로그 메시지 출력 (타임스탐프 포함)"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def check_environment():
    """환경 확인"""
    log_message("=" * 70)
    log_message("환경 확인")
    log_message("=" * 70)

    # CUDA 확인
    log_message(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_message(f"GPU 개수: {torch.cuda.device_count()}")
        log_message(f"GPU 이름: {torch.cuda.get_device_name(0)}")

    # 데이터셋 확인
    if not DATA_YAML.exists():
        log_message(f"ERROR: data.yaml 없음: {DATA_YAML}")
        sys.exit(1)
    log_message(f"✓ data.yaml 경로: {DATA_YAML}")

    # 디렉터리 생성
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_message(f"✓ 출력 디렉터리 생성")

def train_model():
    """모델 학습"""
    log_message("\n" + "=" * 70)
    log_message("모델 학습 시작 (고정확도 최적화)")
    log_message("=" * 70)

    # 모델 로드
    log_message(f"기본 모델 로드: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    # 학습 실행
    log_message(f"학습 설정:")
    for key, value in TRAINING_CONFIG.items():
        log_message(f"  {key}: {value}")

    log_message("\n학습 중...")
    results = model.train(
        data=str(DATA_YAML),
        project=str(RUNS_DIR.parent),
        name="knife_v2",
        **TRAINING_CONFIG
    )

    return results

def evaluate_model():
    """모델 평가"""
    log_message("\n" + "=" * 70)
    log_message("모델 평가")
    log_message("=" * 70)

    # 최고 성능 모델 로드
    best_model_path = RUNS_DIR / "knife_v2" / "weights" / "best.pt"
    if not best_model_path.exists():
        log_message(f"ERROR: 모델 파일 없음: {best_model_path}")
        return None

    model = YOLO(str(best_model_path))

    # Val 평가
    log_message("Val 데이터셋 평가 중...")
    val_results = model.val(
        data=str(DATA_YAML),
        device=0,
    )

    # 메트릭 출력
    log_message("\n평가 결과:")
    log_message(f"  Precision: {val_results.results_dict.get('metrics/precision(B)', 'N/A')}")
    log_message(f"  Recall: {val_results.results_dict.get('metrics/recall(B)', 'N/A')}")
    log_message(f"  mAP50: {val_results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    log_message(f"  mAP50-95: {val_results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    return val_results

def save_final_model():
    """최종 모델 저장"""
    log_message("\n" + "=" * 70)
    log_message("최종 모델 저장")
    log_message("=" * 70)

    src_model = RUNS_DIR / "knife_v2" / "weights" / "best.pt"
    dst_model = MODELS_DIR / "best.pt"

    if src_model.exists():
        import shutil
        shutil.copy(str(src_model), str(dst_model))
        log_message(f"✓ 모델 저장: {dst_model}")
        log_message(f"  파일 크기: {dst_model.stat().st_size / (1024*1024):.2f} MB")
    else:
        log_message(f"ERROR: 모델 파일 없음: {src_model}")

def main():
    """메인 실행"""
    try:
        log_message("\n" + "=" * 70)
        log_message("Knife Detection Model Training - High Accuracy")
        log_message("=" * 70)

        # 환경 확인
        check_environment()

        # 모델 학습
        train_model()

        # 모델 평가
        evaluate_model()

        # 최종 모델 저장
        save_final_model()

        log_message("\n" + "=" * 70)
        log_message("학습 완료!")
        log_message("=" * 70)
        log_message(f"결과 저장 위치: {RUNS_DIR / 'knife_v2'}")
        log_message(f"최종 모델: {MODELS_DIR / 'best.pt'}")

    except Exception as e:
        log_message(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
