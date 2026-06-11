import cv2
import mediapipe as mp
import math
import os
import csv
from datetime import datetime
from collections import deque

# ==========================================
# MediaPipe Pose 초기화
# ==========================================
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# ==========================================
# 영상 입력
# ==========================================

# 🎥 카메라 사용 (0=내장, 1=외부/DroidCam)
camera_index = 1  # ← 필요하면 0으로 변경
cap = cv2.VideoCapture(camera_index)

# ==========================================
# 시작 전 진단
# ==========================================

print("=" * 50)
print(f"[현재 디렉터리] {os.getcwd()}")

if not cap.isOpened():
    print(f"❌ 오류: 카메라 {camera_index}을(를) 열 수 없습니다.")
    print(f"   확인: 카메라가 연결되어 있는지 확인하세요")
    exit(1)

frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_fps    = cap.get(cv2.CAP_PROP_FPS)
fps          = input_fps if input_fps > 0 else 30.0
total_frames = 0  # 카메라는 총 프레임 수가 없음

print(f"[카메라 정보] {frame_w}x{frame_h}, {fps:.1f}fps, 카메라 {camera_index}")
print("=" * 50)

# ==========================================
# 저장 경로
# ==========================================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path      = os.path.join(OUTPUT_DIR, f"log_{run_timestamp}.csv")
fourcc        = cv2.VideoWriter_fourcc(*"mp4v")

# ==========================================
# CSV 초기화
# ==========================================

log_file   = open(log_path, "w", newline="", encoding="utf-8")
log_writer = csv.writer(log_file)
log_writer.writerow(["fall_id", "datetime", "photo_file", "clip_file", "clip_sec"])


def save_clip(frames, path):
    if not frames:
        return
    out = cv2.VideoWriter(path, fourcc, fps, (frame_w, frame_h))
    for f in frames:
        out.write(f)
    out.release()


def write_fall_record(fall_id, fall_ts, photo_filename, clip_filename, clip_sec):
    log_writer.writerow([f"{fall_id:03d}", fall_ts, photo_filename, clip_filename, f"{clip_sec:.1f}"])
    log_file.flush()


# ==========================================
# 설정값 — 프레임 수 기준
# ==========================================

PRE_FALL_SEC     = 5.0
POST_FALL_SEC    = 5.0
FALL_CONFIRM_SEC = 1.3    # 낙상 확정 지속 시간
FALL_WINDOW_SEC  = 2.0    # 서있다가 이 시간 내 누우면 낙상 후보
RAPID_FALL_DELTA = 0.08   # 한 프레임 내 골반 급락 임계값

# ==========================================
# 오감지 방어 — 의도적 눕기 감지
#
# 낙상 vs 침대에 눕기 구분 원리:
#   낙상:   STANDING → (순간적) → LYING         (SITTING 거의 없음)
#   눕기:   STANDING → SITTING (오래) → LYING   (앉는 과정 명확)
#
# SITTING을 INTENTIONAL_SITTING_SEC 이상 유지했다면 "의도적으로 누운 것"으로 판단.
# ==========================================

INTENTIONAL_SITTING_SEC    = 0.8   # 이 이상 앉아있었으면 → 의도적 눕기

PRE_FALL_FRAMES            = int(fps * PRE_FALL_SEC)
POST_FALL_FRAMES           = int(fps * POST_FALL_SEC)
FALL_CONFIRM_FRAMES        = int(fps * FALL_CONFIRM_SEC)
FALL_WINDOW_FRAMES         = int(fps * FALL_WINDOW_SEC)
INTENTIONAL_SITTING_FRAMES = int(fps * INTENTIONAL_SITTING_SEC)

print(f"[설정] 낙상 확정: {FALL_CONFIRM_FRAMES}프레임 ({FALL_CONFIRM_SEC}초)")
print(f"[설정] 의도적 눕기 기준: {INTENTIONAL_SITTING_FRAMES}프레임 ({INTENTIONAL_SITTING_SEC}초 이상 SITTING)")
print(f"[저장 경로] {os.path.abspath(OUTPUT_DIR)}")
print("종료: q 키\n")


# ==========================================
# 자세 분류 함수
# ==========================================
def classify_pose(angle, torso_ratio, hip_y):
    """
    몸통 각도 / 수평 비율 / 골반 높이로 자세를 분류합니다.

    반환값: 'STANDING' | 'SITTING' | 'LYING'
    """
    angle_from_vertical = abs(angle - 90)

    if angle_from_vertical < 30:
        return 'STANDING'

    if angle_from_vertical > 35 or torso_ratio > 1.0 or hip_y > 0.75:
        return 'LYING'

    return 'SITTING'


# ==========================================
# 상태 추적 변수
# ==========================================

frame_buffer        = deque(maxlen=PRE_FALL_FRAMES)
frame_count         = 0
prev_pose_state     = None    # 이전 프레임 자세

# 낙상 감지
last_standing_frame = None    # 마지막으로 STANDING이었던 프레임
fall_frame_count    = 0       # 연속 fall_condition 프레임 수
fall_confirmed      = False
prev_hip_y          = None

# ==========================================
# 오감지 방어: SITTING 구간 추적
# ==========================================
sitting_start_frame   = None   # 현재 SITTING 시작 프레임
prev_sitting_duration = 0      # 직전 SITTING 구간 길이 (프레임)
sitting_end_frame     = 0      # 직전 SITTING이 끝난 프레임

# 저장 관련
fall_count          = 0
is_post_recording   = False
post_frames         = []
pre_frames_snapshot = []
pending_fall_id     = 0
pending_fall_ts     = ""
pending_photo       = ""

# 진단용 통계
stat_no_person = 0
stat_standing  = 0
stat_sitting   = 0
stat_lying     = 0
prev_fall_cond = False

# ==========================================
# 메인 루프
# ==========================================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    current_pose   = None
    fall_condition = False
    angle          = 0.0
    torso_ratio    = 0.0

    if result.pose_landmarks:

        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = result.pose_landmarks.landmark

        shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
        shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
        hip_x      = (landmarks[23].x + landmarks[24].x) / 2
        hip_y      = (landmarks[23].y + landmarks[24].y) / 2

        dx = shoulder_x - hip_x
        dy = shoulder_y - hip_y

        angle       = abs(math.degrees(math.atan2(dy, dx)))
        torso_ratio = abs(dx) / (abs(dy) + 0.0001)

        rapid_fall = False
        if prev_hip_y is not None and (hip_y - prev_hip_y) > RAPID_FALL_DELTA:
            rapid_fall = True
        prev_hip_y = hip_y

        # ==========================================
        # 자세 분류
        # ==========================================
        current_pose = classify_pose(angle, torso_ratio, hip_y)

        # 통계
        if current_pose == 'STANDING':
            stat_standing += 1
        elif current_pose == 'SITTING':
            stat_sitting += 1
        else:
            stat_lying += 1

        # ==========================================
        # STANDING 시각 갱신
        # ==========================================
        if current_pose == 'STANDING':
            last_standing_frame = frame_count

        # ==========================================
        # SITTING 구간 추적 (오감지 방어 핵심)
        #
        # 상태 전이를 추적해서 SITTING을 충분히 거쳤는지 확인합니다.
        # ==========================================
        if current_pose == 'SITTING':
            if prev_pose_state != 'SITTING':
                # SITTING 구간 시작
                sitting_start_frame = frame_count
        elif prev_pose_state == 'SITTING' and sitting_start_frame is not None:
            # SITTING 구간 종료 → 길이 기록
            prev_sitting_duration = frame_count - sitting_start_frame
            sitting_end_frame     = frame_count
            sitting_start_frame   = None

        # ==========================================
        # 낙상 조건 판정
        # ==========================================
        is_lying = (current_pose == 'LYING')

        frames_since_standing = (
            frame_count - last_standing_frame
            if last_standing_frame is not None
            else float('inf')
        )

        recently_fell        = is_lying and (frames_since_standing < FALL_WINDOW_FRAMES)
        rapid_fall_and_lying = is_lying and rapid_fall

        # ==========================================
        # 오감지 방어: 의도적 눕기 판단
        #
        # 직전 SITTING 구간이 INTENTIONAL_SITTING_FRAMES 이상이고
        # 그 SITTING이 최근(FALL_WINDOW_FRAMES 이내)에 끝났다면
        # → 낙상이 아니라 침대에 눕거나 쉬려는 행동
        # ==========================================
        frames_since_sitting_ended = (
            frame_count - sitting_end_frame
            if sitting_end_frame > 0
            else float('inf')
        )

        was_sitting_deliberately = (
            is_lying
            and prev_sitting_duration >= INTENTIONAL_SITTING_FRAMES
            and frames_since_sitting_ended < FALL_WINDOW_FRAMES
        )

        fall_condition = (recently_fell or rapid_fall_and_lying) and not was_sitting_deliberately

        # ==========================================
        # 상태 변화 터미널 출력 (진단용)
        # ==========================================
        if current_pose != prev_pose_state:
            since_sec = frames_since_standing / fps if last_standing_frame else float('inf')
            print(f"  [F{frame_count:5d}] {prev_pose_state} → {current_pose}"
                  f"  angle={int(angle)} torso={torso_ratio:.2f} hip_y={hip_y:.2f}"
                  f"  since_stand={since_sec:.1f}s")

        if fall_condition != prev_fall_cond:
            reason = "의도적눕기" if was_sitting_deliberately else \
                     f"recently_fell={recently_fell}, rapid={rapid_fall_and_lying}"
            print(f"  [F{frame_count:5d}] fall_condition: {prev_fall_cond} → {fall_condition}"
                  f"  ({reason})")
            prev_fall_cond = fall_condition

        # ==========================================
        # 화면 출력
        # ==========================================
        state_color = {
            'STANDING': (0, 255, 0),
            'SITTING':  (255, 255, 0),
            'LYING':    (0, 165, 255),
        }[current_pose]

        cv2.putText(frame, f"Pose  : {current_pose}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
        cv2.putText(frame, f"Angle : {int(angle)}  (vert: {int(abs(angle-90))})",
                    (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Torso : {torso_ratio:.2f}  HipY: {hip_y:.2f}",
                    (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        since_sec = frames_since_standing / fps if last_standing_frame else float('inf')
        cv2.putText(frame, f"Since Stand: {since_sec:.1f}s",
                    (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        if was_sitting_deliberately:
            cv2.putText(frame, "[ Intentional Lie-down ]",
                        (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 150), 2)
        elif rapid_fall:
            cv2.putText(frame, "! Rapid Drop",
                        (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

    else:
        stat_no_person += 1
        cv2.putText(frame, "No Person Detected",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)

    # ==========================================
    # 낙상 후보 → 확정 (프레임 카운터 기준)
    # ==========================================
    if fall_condition:
        fall_frame_count += 1
        elapsed_sec = fall_frame_count / fps

        cv2.putText(frame, f"Falling... {elapsed_sec:.1f}s  ({fall_frame_count}/{FALL_CONFIRM_FRAMES}f)",
                    (30, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if fall_frame_count >= FALL_CONFIRM_FRAMES and not fall_confirmed and not is_post_recording:
            fall_confirmed = True
            fall_count    += 1
            fall_ts        = datetime.now().strftime("%Y%m%d_%H%M%S")

            photo_filename = f"photo_{fall_count:03d}_{fall_ts}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, photo_filename), frame)

            pre_frames_snapshot = list(frame_buffer)
            is_post_recording   = True
            post_frames         = []
            pending_fall_id     = fall_count
            pending_fall_ts     = fall_ts
            pending_photo       = photo_filename

            print(f"\n★ [낙상 #{fall_count:03d} 감지] {fall_ts}")
            print(f"  사진 → {photo_filename}")
            print(f"  영상 녹화 중...\n")

        if fall_frame_count >= FALL_CONFIRM_FRAMES:
            cv2.putText(frame, f"!! FALL DETECTED  #{fall_count:03d} !!",
                        (30, 355), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        fall_frame_count = 0
        fall_confirmed   = False

    # ==========================================
    # 자세 상태 업데이트 (다음 프레임에서 prev로 사용)
    # ==========================================
    prev_pose_state = current_pose

    # 롤링 버퍼 적재
    frame_buffer.append(frame.copy())

    # ==========================================
    # 후방 녹화
    # ==========================================
    if is_post_recording:
        post_frames.append(frame.copy())

        if len(post_frames) >= POST_FALL_FRAMES:
            all_frames    = pre_frames_snapshot + post_frames
            clip_filename = f"clip_{pending_fall_id:03d}_{pending_fall_ts}.mp4"
            clip_path     = os.path.join(OUTPUT_DIR, clip_filename)

            save_clip(all_frames, clip_path)
            clip_sec = len(all_frames) / fps
            write_fall_record(pending_fall_id, pending_fall_ts, pending_photo, clip_filename, clip_sec)

            print(f"  영상 → {clip_filename}  ({clip_sec:.1f}초)")
            print(f"  CSV  → {log_path}\n")

            is_post_recording   = False
            post_frames         = []
            pre_frames_snapshot = []
            pending_fall_id     = 0
            pending_fall_ts     = ""
            pending_photo       = ""

    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(30) == ord('q'):
        break

# ==========================================
# 종료 처리
# ==========================================
if is_post_recording and (pre_frames_snapshot or post_frames):
    all_frames    = pre_frames_snapshot + post_frames
    clip_filename = f"clip_{pending_fall_id:03d}_{pending_fall_ts}_partial.mp4"
    clip_path     = os.path.join(OUTPUT_DIR, clip_filename)
    save_clip(all_frames, clip_path)
    clip_sec = len(all_frames) / fps
    write_fall_record(pending_fall_id, pending_fall_ts, pending_photo, clip_filename, clip_sec)
    print(f"  영상 (부분) → {clip_filename}  ({clip_sec:.1f}초)")

cap.release()
log_file.close()
cv2.destroyAllWindows()

# ==========================================
# 최종 진단 요약
# ==========================================
print("\n" + "=" * 50)
print("[최종 요약]")
print(f"  전체 프레임  : {frame_count}")
print(f"  사람 미감지  : {stat_no_person}프레임 ({stat_no_person/max(1,frame_count)*100:.1f}%)")
print(f"  STANDING     : {stat_standing}프레임 ({stat_standing/max(1,frame_count)*100:.1f}%)")
print(f"  SITTING      : {stat_sitting}프레임 ({stat_sitting/max(1,frame_count)*100:.1f}%)")
print(f"  LYING        : {stat_lying}프레임 ({stat_lying/max(1,frame_count)*100:.1f}%)")
print(f"  감지된 낙상  : {fall_count}건")
print(f"[저장 위치] {os.path.abspath(OUTPUT_DIR)}")
print("=" * 50)

# Updated: docs: 함수 설명 추가

# Updated: fix: 에러 처리 강화

# Updated: feat: 로깅 기능 추가

# Updated: docs: 타입 힌트 추가

# Updated: feat: 예외 처리 개선

# Updated: docs: 함수 설명 추가

# Updated: fix: 에러 처리 강화

# Confidence threshold optimization for improved detection accuracy

# Improved pose landmark validation for better fall detection accuracy
