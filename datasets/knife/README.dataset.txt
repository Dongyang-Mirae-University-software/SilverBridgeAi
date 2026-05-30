Knife Detection Dataset - Rebuilt
===================================

재구성 일자   : 2026-05-30
재구성 스크립트: tools/rebuild_knife_dataset.py

[원본]
  Roboflow knife-skqnq-yiuai v1 (CC BY 4.0)
  라벨 형식: YOLO Segmentation (polygon, ~39 좌표쌍)
  클래스: Knife(0), Knife_Handle(1)

[재구성 내용]
  pHash 유사 중복 제거 (Hamming <= 8)
  Hard Negative 추가 (칼처럼 생긴 비-칼 물체, 220장)

[Hard Negative 카테고리]
  ruler_stationery : 자·연필·볼펜
  scissors         : 가위
  kitchen_utensil  : 주방 도구 (칼 제외)
  tools_metal      : 드라이버·스패너 등 공구
  pointer_stick    : 지시봉·막대기
  chopsticks       : 젓가락
  classroom_desk   : 강의실 배경 (칼 없음)
  home_kitchen     : 가정 주방 배경 (칼 없음)

[최종 이미지 수]
  train : 2,117장
  valid : 132장
  test  : 59장
  total : 2,308장

[라이선스]
  CC BY 4.0 (Roboflow)
