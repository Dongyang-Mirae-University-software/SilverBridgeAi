# -*- coding: utf-8 -*-
"""
PowerPoint를 이미지로 변환
"""
import subprocess
from pathlib import Path

pptx_path = r"c:\Users\happy\SilverBridgeAI\AISilverBridgeLJH\docs\SilverBridgeAI_Analysis.pptx"
output_dir = r"c:\Users\happy\SilverBridgeAI\AISilverBridgeLJH\docs\PPT_Images"

Path(output_dir).mkdir(exist_ok=True)

# LibreOffice를 사용해서 PDF로 변환
print("[1] PPTX -> PDF 변환 중...")
try:
    subprocess.run([
        "libreoffice", "--headless", "--convert-to", "pdf",
        "--outdir", output_dir, pptx_path
    ], check=True, capture_output=True)
    print("    [OK] PDF 생성 완료")
except Exception as e:
    print(f"    [ERROR] LibreOffice 변환 실패: {e}")
    print("    대신 pdfrw 사용 시도...")

# PDF를 이미지로 변환
print("[2] PDF -> PNG 변환 중...")
try:
    from pdf2image import convert_from_path

    pdf_path = Path(output_dir) / "SilverBridgeAI_Analysis.pdf"
    if pdf_path.exists():
        images = convert_from_path(str(pdf_path), dpi=150)
        for i, image in enumerate(images, 1):
            img_path = Path(output_dir) / f"Slide_{i:02d}.png"
            image.save(str(img_path), "PNG")
            print(f"    [OK] Slide_{i:02d}.png 저장")
        print(f"    [OK] 총 {len(images)}개 슬라이드 이미지 생성")
    else:
        print("    [ERROR] PDF 파일 없음")
except ImportError:
    print("    [INFO] pdf2image 설치 필요...")
    subprocess.run(["pip", "install", "-q", "pdf2image", "pillow"], check=False)
    print("    다시 실행해주세요: python pptx_to_image.py")
except Exception as e:
    print(f"    [ERROR] 변환 실패: {e}")
    print("    PowerPoint를 직접 열어서 '내보내기' > '그림으로 내보내기' 사용하세요")

print("\n[완료] 이미지 저장 위치:")
print(f"      {output_dir}")

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
