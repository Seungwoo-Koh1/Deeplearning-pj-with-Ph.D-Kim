import os
import sys
import torch
import zipfile
import numpy as np
from PIL import Image
from collections import OrderedDict

# 'colorizers' 디렉토리를 모듈 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'colorizers'))

from colorizers import eccv16, siggraph17
from util import load_img, preprocess_img, postprocess_tens

def prepare_image(image_path):
    """이미지를 모델에 맞는 형식으로 준비"""
    img = Image.open(image_path)
    
    # 그레이스케일로 변환
    if img.mode != 'L':
        img = img.convert('L')
    
    # numpy 배열로 변환
    img_array = np.array(img)
    
    # shape 확인 및 차원 추가
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array, img_array, img_array], axis=2)
    
    return img_array

def get_output_filename(index):
    """시퀀셜한 출력 파일 이름 생성"""
    return f"TEST_{str(index).zfill(3)}.png"

def colorize_images_in_dir(input_dir, output_dir, model_type="eccv16"):
    """디렉토리 내의 이미지들을 색상화하는 함수"""
    if model_type == "eccv16":
        colorizer = eccv16(pretrained=True).eval()
    elif model_type == "siggraph17":
        colorizer = siggraph17(pretrained=True).eval()
    else:
        raise ValueError("Invalid model type. Choose 'eccv16' or 'siggraph17'.")

    os.makedirs(output_dir, exist_ok=True)
    
    processed_files = OrderedDict()
    processed_count = 0
    failed_count = 0

    # 입력 디렉토리의 모든 이미지 파일을 정렬된 순서로 가져오기
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

    print(f"\n총 처리할 이미지 수: {len(image_files)}\n")

    for index, img_file in enumerate(image_files):
        input_path = os.path.join(input_dir, img_file)
        output_filename = get_output_filename(index)
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            # 이미지 처리
            img_array = prepare_image(input_path)
            tens_l_orig, tens_l_rs = preprocess_img(img_array, HW=(256, 256))
            
            with torch.no_grad():
                out_ab = colorizer(tens_l_rs).cpu()
            
            out_img = postprocess_tens(tens_l_orig, out_ab)
            out_img = Image.fromarray((out_img * 255).astype('uint8'))
            out_img.save(output_path)
            
            # 처리 성공 기록
            processed_files[img_file] = output_path
            processed_count += 1
            print(f"처리 완료 ({processed_count}/{len(image_files)}): {output_filename}")
            
        except Exception as e:
            failed_count += 1
            print(f"처리 실패 ({failed_count} 번째 실패): {img_file}")
            print(f"에러 메시지: {str(e)}")
            
            # 디버깅 정보 출력
            try:
                img = Image.open(input_path)
                print(f"이미지 정보 - 모드: {img.mode}, 크기: {img.size}")
            except Exception as debug_e:
                print(f"이미지 정보 확인 실패: {str(debug_e)}")
    
    # 처리 결과 요약
    print(f"\n처리 완료 요약:")
    print(f"성공: {processed_count}/{len(image_files)} 이미지")
    print(f"실패: {failed_count}/{len(image_files)} 이미지")
    
    return processed_count, failed_count, processed_files

def zip_directory(directory, zip_path):
    """디렉토리를 ZIP 파일로 압축"""
    if os.path.isdir(zip_path):
        zip_path = os.path.join(zip_path, 'colorized_images.zip')
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in sorted(files):  # 파일을 정렬하여 순서대로 추가
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory)
                zipf.write(file_path, arcname)
                print(f"ZIP에 추가: {arcname}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch Image Colorization and Compression")
    parser.add_argument("-i", "--input_dir", type=str, required=True, 
                        help="Path to the input directory containing grayscale images.")
    parser.add_argument("-o", "--output_zip", type=str, required=True, 
                        help="Path to save the output ZIP file.")
    parser.add_argument("-m", "--model", type=str, choices=["eccv16", "siggraph17"],
                        default="eccv16",
                        help="Choose the colorization model: 'eccv16' or 'siggraph17'")

    args = parser.parse_args()

    # 임시 출력 디렉토리
    output_dir = "colorized_temp"

    # 이미지 처리
    processed, failed, processed_files = colorize_images_in_dir(
        args.input_dir, output_dir, args.model
    )

    if processed > 0:
        # ZIP 파일 생성
        print(f"\nZIP 파일 생성 중...")
        zip_directory(output_dir, args.output_zip)
        print(f"ZIP 파일 생성 완료: {args.output_zip}")
    else:
        print("\n처리된 이미지가 없어 ZIP 파일을 생성하지 않습니다.")

    # 임시 디렉토리 정리
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print("\n임시 파일 정리 완료")