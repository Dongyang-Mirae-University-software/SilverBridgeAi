Fire & Smoke Detection Dataset - Full Merged
=============================================

병합 날짜     : 2026-05-29
병합 스크립트 : tools/merge_all_fire.py

[소스 데이터셋]
  fire      - 기존 정제 데이터 (fire, smoke)
  fire_new  - Roboflow 추가 수집 (fire 단일 클래스, 123장)
  fire_new2 - Roboflow 추가 수집 (fire, smoke, 241장)
  fire_new3 - Roboflow 혼합 데이터셋 (bottle/fire/gun/person/stick 5클래스)
              → fire 클래스(index 1)만 추출해 index 0으로 리매핑
  fire_new4 - Roboflow 추가 수집 (fire, smoke, 1,153장)

[중복 제거]
  MD5 완전 동일 이미지 제거 (바이너리 동일)
  pHash 유사 이미지 제거 (Hamming 거리 <= 8, DCT 64bit)

[라벨 품질 정제]
  bbox 경계 초과 → 클리핑
  극소 bbox 제거 (이미지 면적 0.3% 미만)
  극대 bbox 제거 (이미지 면적 90% 초과)

[최종 클래스]
  0 : fire   (화재)
  1 : smoke  (연기)

[최종 이미지 수]
  train : 5,519장
  valid : 811장
  test  : 434장
  total : 6,764장

[포맷]
  YOLO 형식 (train/valid/test + labels/)
  라벨: <class_idx> <cx> <cy> <w> <h>  (정규화 좌표)
  라이선스: CC BY 4.0 (Roboflow)
