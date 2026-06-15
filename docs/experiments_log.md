# 실험 결과 로그

## 현재 사용 모델

| 감지 대상 | 모델 경로 | mAP50 |
|-----------|-----------|-------|
| Fire / Smoke | `experiments/fire_100ep/weights/best.pt` | 0.581 |
| Knife | `Train/Knife500/weights/best.pt` | 0.900 |

## 학습 실험 아카이브

| 폴더 | 모델 | Epochs | mAP50 | 비고 |
|------|------|--------|-------|------|
| `experiments/fire_100ep/` | Fire (yolo26n) | 100 | **0.581** | ✅ 현재 사용 중 |
| `Train/Knife/` | Knife (yolo26n) | 200 | 0.895 | 이전 버전 |
| `Train/Knife500/` | Knife (yolo26n) | 500 | **0.900** | ✅ 현재 사용 중 |

## 향후 개선 방향

- **Fire 모델**: 현재 mAP50=0.58로 낮음. 데이터셋 확충 후 200~500 epoch 재학습 권장.
- **Knife 모델**: mAP50=0.90 충분히 높음. 유지.
<!-- Update 46 -->
<!-- Update 47 -->
<!-- Update 48 -->
<!-- Update 49 -->
<!-- Update 50 -->
<!-- Update 56 -->
<!-- Update 57 -->
<!-- Update 58 -->
<!-- Update 59 -->
<!-- Update 60 -->
