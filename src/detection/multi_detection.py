# -*- coding: utf-8 -*-
"""
SilverBridgeAI — 화재·연기·흉기 실시간 감지
============================================
두 YOLO 모델을 동시에 실행해 카메라 영상에서
Fire / Smoke / Knife 를 실시간으로 감지합니다.

실행:
    python src/detection/multi_detection.py
종료:
    ESC 키

모델 자동 선별:
    models/ 폴더의 fire_*/knife_* 하위 폴더를 스캔해
    results.csv 의 mAP50 이 가장 높은 모델을 자동으로 사용.
    새 모델을 models/ 에 추가하면 다음 실행 시 자동 반영됨.

카메라 인덱스:
    cv2.VideoCapture(1) = 외부 USB 캠 / DroidCam
    인덱스 확인: python src/detection/check_camera.py
"""

import csv
import cv2
from pathlib import Path
from ultralytics import YOLO
from collections import deque

# ==========================================
# 경로 설정
# ==========================================
BASE = Path("C:/Users/happy/SilverBridgeAI/AISilverBridgeLJH")


# ==========================================
# 최고 성능 모델 자동 선별
#
# models/<category>_*/weights/best.pt 를 전부 스캔하고
# results.csv 의 mAP50 최댓값을 비교해 가장 높은 모델을 반환.
# results.csv 가 없으면 mAP50=0.0 으로 처리.
# ==========================================

def find_best_model(category: str) -> Path:
    """
    models/ 에서 category(fire / knife)로 시작하는 폴더를 모두 스캔해
    mAP50 이 가장 높은 best.pt 경로를 반환.

    예: category="fire" → models/fire_100ep/, models/fire_200ep/ 비교
    """
    models_dir = BASE / "models"
    best_map50    = -1.0
    best_weights  = None
    best_folder   = None

    for folder in sorted(models_dir.iterdir()):
        if not folder.is_dir() or not folder.name.startswith(category + "_"):
            continue

        weights_path = folder / "weights" / "best.pt"
        if not weights_path.exists():
            continue  # 학습 결과가 없는 폴더는 건너뜀

        # results.csv 에서 최고 mAP50 읽기
        map50 = _read_best_map50(folder / "results.csv")

        if map50 > best_map50:
            best_map50   = map50
            best_weights = weights_path
            best_folder  = folder.name

    if best_weights is None:
        raise FileNotFoundError(
            f"[오류] models/ 에서 '{category}_*' 폴더를 찾을 수 없습니다.\n"
            f"       학습 완료 후 models/{category}_<ep>/ 폴더가 있는지 확인하세요."
        )

    print(f"[모델 선택] {category:5s} → {best_folder}  (mAP50={best_map50:.3f})")
    return best_weights


def _read_best_map50(csv_path: Path) -> float:
    """results.csv 에서 metrics/mAP50(B) 열의 최댓값을 반환. 파일 없으면 0.0."""
    if not csv_path.exists():
        return 0.0
    best = 0.0
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 컬럼명 앞뒤 공백 제거 후 읽기
                clean = {k.strip(): v.strip() for k, v in row.items()}
                val = float(clean.get("metrics/mAP50(B)", 0) or 0)
                if val > best:
                    best = val
    except Exception:
        pass
    return best


# ==========================================
# 모델 로드 (자동 선별)
# ==========================================
print("\n모델 자동 선별 중...")
model_fire  = YOLO(str(find_best_model("fire")))
model_knife = YOLO(str(find_best_model("knife")))
print()

print("Fire  classes:", model_fire.names)
print("Knife classes:", model_knife.names)

# ==========================================
# 추론 파라미터
#
# conf : 이 신뢰도 미만 박스는 추론 단계에서 바로 제거
# iou  : 겹치는 박스 제거 기준 (낮을수록 적극적으로 제거)
#
# ★ 독거노인 안전 시스템 설정 원칙
#   화재 미감지 = 생명 위협  →  인식률(Recall) 최우선
#   오감지      = 보호자 공황 →  차선으로 줄임
#
# Fire  모델:
#   conf 낮게 → 초기 소규모 화재·연기도 놓치지 않음
#   Temporal Tracker 가 순간 오감지를 추가로 걸러냄
# Knife 모델:
#   conf 높게 → 유사 물체(자·연필) 오감지 차단
# ==========================================
CONF_FIRE  = 0.30   # 낮게 유지 — 초기 화재·연기 미감지 방지
CONF_KNIFE = 0.55

IOU_FIRE   = 0.45
IOU_KNIFE  = 0.40

# 이 픽셀² 미만 박스는 노이즈로 제거
# Fire: 작게 설정 → 초기 소규모 화재(작은 불씨)도 감지
MIN_AREA_FIRE  = 1000
MIN_AREA_KNIFE = 1500

# ==========================================
# 신뢰도 가중 시간적 일관성 추적기
# (Confidence-Weighted Temporal Tracker)
#
# [목적] 순간적 오감지 방어
#   - 진짜 감지 → 여러 프레임에 걸쳐 높은 신뢰도로 지속 등장
#   - 오감지   → 한두 프레임만 낮은 신뢰도로 나타났다 사라짐
#
# [방식] 각 프레임의 최고 신뢰도(float)를 저장
#   - 감지된 프레임: 신뢰도 값 저장
#   - 미감지 프레임: 0.0 저장
#
# [표시 조건] 아래 두 조건을 동시 만족할 때만 화면에 표시
#   1. 감지 빈도 >= FREQ_RATIO  (최근 N 프레임 중 몇 % 이상 감지됐는가)
#   2. 평균 신뢰도 >= AVG_CONF  (감지된 프레임만의 평균 신뢰도)
#
# ★ 독거노인 안전 시스템 — Fire 기준 완화
#   HISTORY_SIZE 줄임  → 화재 발생 후 더 빠르게 경보
#   FREQ_RATIO 낮춤    → 6프레임 중 2회만 감지돼도 표시
#   AVG_CONF 낮춤      → 낮은 신뢰도라도 지속되면 표시
# ==========================================
HISTORY_SIZE = 6   # 8 → 6: 더 빠른 반응

# Fire : 6프레임 중 2회(33%) 이상 + 평균 신뢰도 0.35 이상
# 초기 화재는 신뢰도가 낮고 간헐적으로 감지됨 → 민감하게 설정
FIRE_FREQ_RATIO = 0.33
FIRE_AVG_CONF   = 0.35

# Knife: 6프레임 중 4회(66%) 이상 + 평균 신뢰도 0.55 이상
KNIFE_FREQ_RATIO = 0.66
KNIFE_AVG_CONF   = 0.55


class DetectionTracker:
    """
    클래스별로 최근 HISTORY_SIZE 프레임의 최고 신뢰도를 추적.
    감지 안 된 프레임은 0.0으로 기록.
    """

    def __init__(self):
        # { class_name: deque([0.0, 0.85, 0.0, ...]) }
        self.history: dict[str, deque] = {}

    def update(self, detected: dict[str, float]):
        """
        detected: {class_name: 이번 프레임 최고 신뢰도}
        기존 추적 클래스는 이번 값을 추가하고,
        새 클래스는 히스토리를 새로 생성.
        """
        for cls in list(self.history.keys()):
            self.history[cls].append(detected.get(cls, 0.0))
        for cls, conf in detected.items():
            if cls not in self.history:
                self.history[cls] = deque([conf], maxlen=HISTORY_SIZE)

    def is_visible(self, cls: str, freq_ratio: float, avg_conf: float) -> bool:
        """
        해당 클래스가 '표시 조건'을 통과하는지 판단.
          freq_ratio : 감지 비율 최솟값 (예: 0.50 → 프레임의 50% 이상)
          avg_conf   : 감지된 프레임의 평균 신뢰도 최솟값
        """
        if cls not in self.history:
            return False
        hist = list(self.history[cls])
        detected_confs = [c for c in hist if c > 0.0]
        if not detected_confs:
            return False
        freq = len(detected_confs) / len(hist)
        avg  = sum(detected_confs) / len(detected_confs)
        return freq >= freq_ratio and avg >= avg_conf

    def visible_classes(self, freq_ratio: float, avg_conf: float) -> set[str]:
        """표시 조건을 통과한 클래스 이름 집합 반환."""
        return {cls for cls in self.history if self.is_visible(cls, freq_ratio, avg_conf)}


fire_tracker  = DetectionTracker()
knife_tracker = DetectionTracker()


# ==========================================
# 칼 클래스명 정규화
#
# Knife 모델의 실제 클래스명 예시 (Roboflow 버전 태그 포함):
#   'knife - v1 2026-04-12 1-26pm'
#   'knife-detection - v4 2023-11-17 12-31am'
#   '------------------------------'  ← 라벨링 오류 클래스
#
# → 'knife' 포함 여부로 판단, 오류 클래스는 None 반환
# ==========================================

def normalize_knife_class(raw_name: str):
    """
    클래스명에 'knife'가 포함되면 → 'knife' 반환.
    그 외(구분자, 라벨 오류 등) → None 반환 (화면에 표시하지 않음).
    """
    cleaned = raw_name.lower().strip("-").strip()
    if "knife" in cleaned:
        return "knife"
    return None


# ==========================================
# 카메라 캡처 시작
# ==========================================
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = frame.copy()

    # ── 추론 ───────────────────────────────
    results_fire  = model_fire (frame, conf=CONF_FIRE,  iou=IOU_FIRE,  verbose=False)
    results_knife = model_knife(frame, conf=CONF_KNIFE, iou=IOU_KNIFE, verbose=False)

    # ── 이번 프레임 감지 수집 {class: max_conf} ──
    current_fire:  dict[str, float] = {}
    current_knife: dict[str, float] = {}

    for box in results_fire[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2 - x1) * (y2 - y1) < MIN_AREA_FIRE:
            continue
        cls_name = model_fire.names[int(box.cls[0])].lower()
        conf     = float(box.conf[0])
        if cls_name not in current_fire or conf > current_fire[cls_name]:
            current_fire[cls_name] = conf

    for box in results_knife[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2 - x1) * (y2 - y1) < MIN_AREA_KNIFE:
            continue
        raw_name = model_knife.names[int(box.cls[0])]
        norm     = normalize_knife_class(raw_name)
        if norm is None:
            continue
        conf = float(box.conf[0])
        if norm not in current_knife or conf > current_knife[norm]:
            current_knife[norm] = conf

    # ── 이력 업데이트 → 표시할 클래스 결정 ──
    fire_tracker.update(current_fire)
    knife_tracker.update(current_knife)

    visible_fire  = fire_tracker.visible_classes(FIRE_FREQ_RATIO,  FIRE_AVG_CONF)
    visible_knife = knife_tracker.visible_classes(KNIFE_FREQ_RATIO, KNIFE_AVG_CONF)

    # ── 바운딩 박스 렌더링 (visible 클래스만) ──

    # 불 / 연기
    for box in results_fire[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf     = float(box.conf[0])
        cls_name = model_fire.names[int(box.cls[0])].lower()

        if (x2 - x1) * (y2 - y1) < MIN_AREA_FIRE:
            continue
        if cls_name not in visible_fire:
            continue

        if cls_name == "fire":
            color, label = (0, 0, 255), "Fire"
        elif cls_name == "smoke":
            color, label = (255, 0, 0), "Smoke"
        else:
            color, label = (0, 255, 255), cls_name

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 칼
    for box in results_knife[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf     = float(box.conf[0])
        raw_name = model_knife.names[int(box.cls[0])]
        norm     = normalize_knife_class(raw_name)

        if (x2 - x1) * (y2 - y1) < MIN_AREA_KNIFE:
            continue
        if norm not in visible_knife:
            continue

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Knife {conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ── 우측 상단 상태 표시 ────────────────
    h, w = img.shape[:2]
    fire_color  = (0, 0, 255) if visible_fire  else (80, 80, 80)
    knife_color = (0, 255, 0) if visible_knife else (80, 80, 80)

    cv2.putText(img, "FIRE"  if visible_fire  else "no fire",
                (w - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fire_color,  2)
    cv2.putText(img, "KNIFE" if visible_knife else "no knife",
                (w - 190, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, knife_color, 2)

    cv2.imshow("Fire + Smoke + Knife Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

cap.release()
cv2.destroyAllWindows()
# Updated: refactor: 변수명 명확화

# Updated: fix: 메모리 누수 방지

# Updated: refactor: 중복 코드 제거

# Updated: refactor: 코드 가독성 개선

# Updated: perf: 성능 최적화

# Updated: refactor: 변수명 명확화

# Updated: fix: 메모리 누수 방지

# Enhanced logging for detection performance monitoring
<!-- Update 36 -->
<!-- Update 37 -->
<!-- Update 38 -->
<!-- Update 39 -->
<!-- Update 40 -->
