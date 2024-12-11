import os
import sys
import torch


# 'colorizers' 디렉토리를 모듈 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'colorizers'))

from colorizers import eccv16, siggraph17
from util import load_img, preprocess_img, postprocess_tens
from PIL import Image
import argparse


def colorize_image(input_path, output_path, model_type="eccv16"):
    # Load the selected colorizer
    if model_type == "eccv16":
        colorizer = eccv16(pretrained=True).eval()
    elif model_type == "siggraph17":
        colorizer = siggraph17(pretrained=True).eval()
    else:
        raise ValueError("Invalid model type. Choose 'eccv16' or 'siggraph17'.")

    # Load and preprocess the image
    img = load_img(input_path)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))

    # Colorize
    with torch.no_grad():
        out_ab = colorizer(tens_l_rs).cpu()

    # Postprocess and save the output
    out_img = postprocess_tens(tens_l_orig, out_ab)
    out_img = Image.fromarray((out_img * 255).astype('uint8'))
    out_img.save(output_path)
    print(f"Colorized image saved at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Colorization using ECCV16 and SIGGRAPH17 models.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input grayscale image.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the colorized output image.")
    parser.add_argument("-m", "--model", type=str, choices=["eccv16", "siggraph17"], default="eccv16",
                        help="Choose the colorization model to use: 'eccv16' or 'siggraph17'. Default is 'eccv16'.")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        exit(1)

    # Colorize the image
    colorize_image(args.input, args.output, args.model)
