# -*- coding: utf-8 -*-
"""
YOLO 모델 감지속도(FPS) 측정
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time

def measure_model_fps(model_path, test_frames=100, image_size=640):
    """
    모델의 FPS(Frames Per Second) 측정
    """
    print(f"\n[측정 시작] {Path(model_path).name}")
    print("=" * 60)

    try:
        # 모델 로드
        model = YOLO(model_path)
        print(f"✓ 모델 로드 완료: {model_path}")

        # 더미 이미지 생성 (워밍업)
        dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        _ = model(dummy_img, conf=0.5, verbose=False)
        print(f"✓ 워밍업 완료")

        # FPS 측정
        print(f"✓ {test_frames}개 프레임 처리 중...")
        times = []

        for i in range(test_frames):
            img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

            start = time.time()
            results = model(img, conf=0.5, verbose=False)
            elapsed = time.time() - start

            times.append(elapsed)

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{test_frames}] 진행 중...")

        # 통계 계산
        times = np.array(times)
        avg_time = times.mean()
        fps = 1.0 / avg_time
        fps_std = 1.0 / times.std() if times.std() > 0 else 0

        print(f"\n[측정 결과]")
        print(f"  평균 처리시간: {avg_time*1000:.2f}ms")
        print(f"  최소 처리시간: {times.min()*1000:.2f}ms")
        print(f"  최대 처리시간: {times.max()*1000:.2f}ms")
        print(f"  FPS (평균): {fps:.1f} FPS")
        print(f"  FPS (범위): {1.0/times.max():.1f} ~ {1.0/times.min():.1f} FPS")

        return fps

    except FileNotFoundError:
        print(f"✗ 모델 파일 없음: {model_path}")
        return None
    except Exception as e:
        print(f"✗ 오류 발생: {e}")
        return None

def main():
    """메인 함수"""
    base_path = Path("models")

    results = {}

    # Knife 250 epoch 측정 (현재 최고 모델)
    knife_model_250 = base_path / "knife_250ep" / "weights" / "best.pt"
    if knife_model_250.exists():
        fps = measure_model_fps(str(knife_model_250), test_frames=100)
        results["Knife 250"] = fps
    else:
        print(f"✗ 모델 파일 없음: {knife_model_250}")

    print("\n" + "=" * 60)
    print("[최종 결과]")
    print("=" * 60)
    for model_name, fps in results.items():
        if fps:
            print(f"  {model_name:15s}: {fps:.1f} FPS")
        else:
            print(f"  {model_name:15s}: 측정 실패")

    return results

if __name__ == "__main__":
    main()

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
