# 🚨 CareAI - AI 감지 모듈 (SilverBridgeAI2)

> 생명과 직결된 **화재·연기·흉기 실시간 감지 AI 모델**  
> Ultralytics YOLOv8n 기반 객체 감지 시스템

---

## 🔴 **핵심 문제점 (지금 당장 해결해야 할 것!)**

| 항목 | 현황 | 목표 | 상태 |
| --- | --- | --- | --- |
| **화재(Fire) 정확도** | **Precision 0.5455** ⚠️ | ≥ 0.92 | 🔴 심각 |
| **흉기(Knife) 정확도** | **Precision 0.84** | ≥ 0.95 | 🟠 부족 |
| **Hard Negative 부재** | Fire: 5개, Knife: 0개 ❌ | 30-35% 필요 | 🔴 심각 |
| **라벨링 오류** | Fire 극소 박스 16.1% | 0% | 🟠 수정필요 |

**문제 분석:**
- ❌ **Fire**: 빨간 옷, 빨간 조명 구분 못 함 (색상만으로 인식)
- ❌ **Knife**: 포크, 나뭇가지와 구분 못 함 (형태만으로 인식)
- ❌ **Hard Negative 부족**: 모델이 "진짜 특징" 아닌 색상/형태로만 학습

---

## ⚠️ **TODO 리스트 (우선순위순)**

### 1️⃣ **즉시 처리 (이번 주내)**
- [ ] **Hard Negative 데이터 추가**
  - [ ] Fire: Red Clothing 200장, Red Light 150장, Sunset 150장, Red Objects 200장 (총 700장)
  - [ ] Knife: Fork 100장, Stick 150장, Spoon 80장, Scissors 100장, Metal Pointed 170장 (총 700장)
  - [ ] 모든 Hard Negative를 각 데이터셋의 `hard_neg/` 폴더에 배치

- [ ] **Fire 극소 박스 라벨 정리**
  - [ ] 극소 박스 16.1% (1,147개 박스) 검토 및 수정
  - [ ] 라벨링 오류 제거 또는 재라벨링
  - [ ] `validate_fire_dataset.py` 재실행으로 검증

### 2️⃣ **모델 재학습 (Hard Negative 포함)**
- [ ] Fire/Smoke 모델 재학습
  ```bash
  python FireSmokeModelTraining_HighAccuracy.py \
    --epochs 200 \
    --batch 16 \
    --augment True
  ```
  - 목표: Precision ≥ 0.92, Recall ≥ 0.88

- [ ] Knife 모델 재학습
  ```bash
  python KnifeModelTraining_HighAccuracy.py \
    --epochs 200 \
    --batch 16
  ```
  - 목표: Precision ≥ 0.95, Recall ≥ 0.90

### 3️⃣ **성능 검증**
- [ ] 재학습 후 모델 평가
  ```bash
  python -c "from ultralytics import YOLO; \
    model = YOLO('models/fire_smoke/best.pt'); \
    metrics = model.val(data='data/fire_smoke/data.yaml')"
  ```
- [ ] Confusion Matrix 분석 (어느 클래스가 실패하는지)
- [ ] 임계값 튜닝 (Precision-Recall 최적화)

### 4️⃣ **실시간 감지 테스트**
- [ ] `test_screen_detection.py` 실행 (Fire/Smoke)
- [ ] `test_knife_detection.py` 실행 (Knife)
- [ ] 실제 환경에서 오탐(False Positive) 확인

### 5️⃣ **GitHub 업로드**
- [ ] 학습된 모델 (`models/*/best.pt`) 저장
- [ ] 코드 최종 검토
- [ ] `git add` → `git commit` → `git push origin developing_new`
- [ ] GitHub에서 PR 생성 후 `main` 머지

### 6️⃣ **향후 개선 사항**
- [ ] Fall Detection (낙상 감지) 추가 개발
- [ ] 실시간 카메라 스트림 처리 최적화
- [ ] 멀티 GPU 학습 지원
- [ ] 모바일 배포 (TFLite 변환)

---

## 📊 **프로젝트 개요**

### 감지 대상 및 성능 목표

| 클래스 | 감지 대상 | 기술 | Precision | Recall | 상태 |
| --- | --- | --- | --- | --- | --- |
| 🔥 Fire | 실제 화재 | YOLOv8n | ≥ 0.92 | ≥ 0.88 | 🔴 현재 0.5455 |
| 💨 Smoke | 연기 | YOLOv8n | ≥ 0.90 | ≥ 0.85 | 🟠 미지정 |
| 🔪 Knife | 흉기(칼) | YOLOv8n | ≥ 0.95 | ≥ 0.90 | 🟠 현재 0.84 |
| 👤 Fall | 낙상 | MediaPipe | ≥ 0.90 | ≥ 0.88 | ⚪ 미개발 |

---

## 🔥 **Fire/Smoke 감지 - 현황**

### 데이터셋 상태

```
✅ 데이터셋 검증 완료
   - Train: 7,152장 (Positive: ?, Hard Neg: 5개)
   - Val:   200장 (다른 영상 출처)
   - Test:  200장 (완전히 독립적)

❌ 문제점:
   1. Hard Negative 극도로 부족 (5개만!)
      → 빨간 옷, 빨간 조명과 화재 구분 못 함
   2. 극소 박스 16.1% (1,147개)
      → 라벨링 오류 의심 (크기 < 32x32 픽셀)
   3. 색상 기반 오분류
      → 석양, 빨간 차, 케첩도 화재로 감지 가능
```

### 학습 결과 (현재)

```python
Model: YOLOv8n
Epochs: 150
Batch Size: 16
Image Size: 640x640

결과:
  Precision: 0.5455 ❌ (목표 0.92)
  Recall:    ? (목표 0.88)
  mAP50:     ? (목표 0.88)
  
해석: "거의 모든 감지가 false positive!"
      → Hard Negative 부족 + 라벨링 오류
```

### 다음 단계

1. **Hard Negative 700장 추가** (목표: 32% 비율)
   ```
   hard_neg/
   ├── red_clothing/      (200장)
   ├── red_light/         (150장)
   ├── sunset/            (150장)
   └── red_objects/       (200장)
   ```

2. **극소 박스 라벨 검토**
   - `validate_fire_dataset.py` 실행
   - 16.1%의 박스 수동 검토
   - 라벨링 오류 수정

3. **모델 재학습** (Hard Negative 포함)
   ```bash
   python FireSmokeModelTraining_HighAccuracy.py --epochs 200
   ```

---

## 🔪 **Knife 감지 - 현황**

### 데이터셋 상태

```
✅ 데이터셋 검증 완료
   - Train: 4,167장 (Positive: 1,600, Hard Neg: 0개!)
   - Val:   200장 (다른 촬영자)
   - Test:  200장 (완전히 독립적)

❌ 심각한 문제:
   1. Hard Negative 0개 ❌❌❌
      → 포크 = 칼로 인식할 가능성 높음
   2. 데이터 다양성 부족
      → 다양한 환경에서 테스트 필요
```

### 학습 결과 (현재)

```python
Model: YOLOv8n
Epochs: 150
Batch Size: 16

결과:
  Precision: 0.84 ⚠️ (목표 0.95)
  Recall:    ? (목표 0.90)
  mAP50:     ? (목표 0.92)
  
해석: "포크나 나뭇가지를 칼로 인식할 수 있음"
```

### 다음 단계

1. **Hard Negative 700장 추가** (목표: 28% 비율)
   ```
   hard_neg/
   ├── fork/              (100장)
   ├── stick/             (150장)
   ├── spoon/             (80장)
   ├── scissors/          (100장)
   └── metal_pointed/     (170장)
   ```

2. **모델 재학습** (Hard Negative 포함)
   ```bash
   python KnifeModelTraining_HighAccuracy.py --epochs 200
   ```

---

## 📈 **성능 목표 vs 현황**

### 요약표

| 항목 | 목표 | 현재 | 차이 | 우선순위 |
| --- | --- | --- | --- | --- |
| Fire Precision | 0.92 | 0.5455 | **-0.3745** 📉 | 🔴 1순위 |
| Knife Precision | 0.95 | 0.84 | -0.11 | 🟠 2순위 |
| Hard Neg (%) | 32% | 0.1% | **-31.9%** ⚠️ | 🔴 1순위 |
| Knife Hard Neg | 700 | 0 | **-700** ❌ | 🔴 1순위 |

---

## 🎯 **해결 전략**

### 즉시 해결 (1-2주)

1. **Hard Negative 데이터 수집**
   - Fire: 빨간 옷, 빨간 조명, 석양, 빨간 물체 각 100-200장
   - Knife: 포크, 나뭇가지, 가위, 숟가락 각 100-170장
   - ✅ 스크립트 완성됨 (`download_red_objects.py` 등)

2. **라벨링 오류 정리**
   - Fire 극소 박스 16.1% 검토
   - `validate_fire_dataset.py` 재실행으로 자동 정리

3. **모델 재학습**
   - Hard Negative 포함해서 150-200 epochs
   - 배치 크기: 16
   - 이미지 크기: 640x640

### 검증 (1주)

1. **성능 평가**
   ```bash
   python -c "from ultralytics import YOLO; \
     model = YOLO('models/fire_smoke/best.pt'); \
     metrics = model.val()"
   ```

2. **임계값 최적화**
   - Precision ≥ 0.92, Recall ≥ 0.88 만족하는 confidence 값 찾기
   - `notebooks/03_threshold_tuning.ipynb` 참조

3. **실제 테스트**
   - 화면 캡처 실시간 감지
   - 다양한 환경에서 오탐 확인

---

## 📁 **폴더 구조**

```
SilverBridgeAI2/
│
├── 📊 데이터셋 (학습 데이터)
│   ├── data/
│   │   ├── fire_smoke/
│   │   │   ├── images/
│   │   │   │   ├── train/          (7,152장 - Positive + Hard Neg 5개)
│   │   │   │   ├── val/            (200장)
│   │   │   │   └── test/           (200장)
│   │   │   ├── labels/
│   │   │   ├── data.yaml           (설정)
│   │   │   └── split_log.txt       (영상→프레임 매핑)
│   │   │
│   │   └── knife/
│   │       ├── images/
│   │       │   ├── train/          (4,167장 - Positive + Hard Neg 0개)
│   │       │   ├── val/            (200장)
│   │       │   └── test/           (200장)
│   │       ├── labels/
│   │       ├── data.yaml
│   │       └── split_log.txt
│   │
│   └── fall/                       (미구현)
│       └── videos/
│
├── 🤖 학습된 모델
│   ├── models/
│   │   ├── fire_smoke/
│   │   │   ├── best.pt             (현재 모델 - Precision 0.5455)
│   │   │   ├── metrics.json
│   │   │   └── training_log.csv
│   │   │
│   │   ├── knife/
│   │   │   ├── best.pt             (현재 모델 - Precision 0.84)
│   │   │   ├── metrics.json
│   │   │   └── training_log.csv
│   │   │
│   │   └── weights_backup/         (이전 모델)
│   │
│   └── 🎓 학습 로그
│       └── runs/                   (YOLOv8 학습 결과)
│
├── 📝 학습 스크립트
│   ├── FireSmokeModelTraining_HighAccuracy.py
│   ├── KnifeModelTraining_HighAccuracy.py
│   └── src/training/               (모듈화 버전)
│
├── 🔍 검증 & 테스트
│   ├── validate_fire_dataset.py    ✅ 완료
│   ├── validate_knife_dataset.py   ✅ 완료
│   ├── test_screen_detection.py    (실시간 Fire/Smoke)
│   ├── test_knife_detection.py     (실시간 Knife)
│   └── src/evaluation/
│
├── 🛠️ Hard Negative 수집 (스크립트)
│   ├── download_red_objects.py
│   ├── download_red_objects_v2.py
│   ├── download_red_clothing_simple.py
│   ├── integrate_hard_negatives.py
│   └── ... (총 8개 스크립트)
│
├── ⚙️ 설정 파일
│   ├── data.yaml (각 데이터셋)
│   ├── .gitignore
│   ├── requirements.txt
│   └── CLAUDE.md                   (자세한 가이드)
│
└── 📚 문서 & 정보
    ├── README.md                   (이 파일)
    └── docs/                       (더 자세한 가이드)
```

---

## 🔧 **핵심 설정 파일**

### `data/fire_smoke/data.yaml`
```yaml
train: images/train
val: images/val
test: images/test

nc: 3
names: ['fire', 'other', 'smoke']
```

### `data/knife/data.yaml`
```yaml
train: images/train
val: images/val
test: images/test

nc: 1
names: ['knife']
```

### `requirements.txt`
```
ultralytics>=8.0.0      # YOLO 모델
opencv-python>=4.8.0    # 영상 처리
torch>=2.0.0            # PyTorch (YOLO 백엔드)
albumentations>=1.3.0   # 데이터 증강
numpy scipy scikit-learn # 수치 계산
```

---

## 📊 **학습 명령어**

### Fire/Smoke 모델

```bash
python FireSmokeModelTraining_HighAccuracy.py \
  --epochs 200 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

**예상 결과 (재학습 후):**
- Precision: ≥ 0.92 ✅
- Recall: ≥ 0.88 ✅
- 학습 시간: ~2-3시간 (GPU)

### Knife 모델

```bash
python KnifeModelTraining_HighAccuracy.py \
  --epochs 200 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

**예상 결과 (재학습 후):**
- Precision: ≥ 0.95 ✅
- Recall: ≥ 0.90 ✅
- 학습 시간: ~1.5-2시간 (GPU)

---

## 🚀 **실시간 감지 테스트**

### Fire/Smoke 감지 (화면 캡처)

```bash
python test_screen_detection.py
# 종료: Q 키
```

**기능:**
- 실시간 화면 캡처
- Fire/Smoke 감지 표시
- 감지 개수 및 신뢰도 표시

### Knife 감지 (화면 캡처)

```bash
python test_knife_detection.py
# 종료: Q 키
```

---

## 🔐 **Git 브랜치 전략**

```
main (기본 브랜치)
  ↑
  └─ developing_new (현재 작업 브랜치)
                ↓
         (Hard Negative 추가)
         (모델 재학습)
         (성능 검증)
                ↓
          PR → main 머지
```

### 커밋 명령어

```bash
# 파일 추가
git add -A

# 커밋
git commit -m "feat: Hard Negative 데이터 추가 및 모델 재학습

- Fire: Hard Negative 700장 추가 (red_clothing, red_light, sunset, red_objects)
- Knife: Hard Negative 700장 추가 (fork, stick, spoon, scissors, metal_pointed)
- 라벨링 오류 정리 (극소 박스 제거)
- 모델 재학습: Fire Precision 0.5455→0.92, Knife Precision 0.84→0.95"

# Push
git push -u origin developing_new
```

---

## 📞 **현재 상태**

| 항목 | 상태 |
| --- | --- |
| Git Repository | ✅ https://github.com/Dongyang-Mirae-University-software/SilverBridgeAi.git |
| Branch | ✅ developing_new |
| 파일 복구 | ✅ 모두 복구됨 |
| Hard Negative 수집 | ⚠️ 스크립트 완성, 실행 필요 |
| 모델 성능 | 🔴 개선 필요 (Fire 0.5455, Knife 0.84) |
| 다음 단계 | Hard Negative 추가 + 모델 재학습 |

---

## 💡 **핵심 포인트 (잊지 말기!)**

1. **Hard Negative가 핵심이다!**
   - Fire: 빨간 옷, 빨간 조명 구분 필수
   - Knife: 포크, 나뭇가지 구분 필수
   - 색상/형태 기반 오분류 방지

2. **데이터 누수 주의**
   - 같은 영상의 프레임이 Train/Val/Test에 섞이면 안 됨!
   - 영상 단위로 분리

3. **높은 임계값 사용**
   - Fire: 0.92 (기본값 0.5보다 훨씬 높음)
   - Knife: 0.95 (가장 높음)
   - 안전 시스템이므로 오탐보다 보수적이어야 함

---

**마지막 업데이트:** 2026-09-25  
**프로젝트:** CareAI - AI 감지 모듈  
**상태:** 개발 진행 중 (Hard Negative 추가 단계)  
**담당:** Jaehehe (skarndaudwls@gmail.com)
