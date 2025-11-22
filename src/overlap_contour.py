import os
import argparse
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('root_dir', type=str)
parser.add_argument('--erode_size', type=int, default=7)
args = parser.parse_args()

if os.path.exists(os.path.join(args.root_dir, 'line_drawing')):
    subdirs = [args.root_dir]
else:
    subdirs = [os.path.join(args.root_dir, x) for x in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, x))]
    subdirs = [x for x in subdirs if os.path.exists(os.path.join(x, 'line_drawing')) and os.path.exists(os.path.join(x, 'line_drawing_processed'))]

for subdir in tqdm(subdirs):
    line_drawing_dir = os.path.join(subdir, 'line_drawing')
    image_names = [x for x in os.listdir(line_drawing_dir) if x.endswith('.png')]
    line_drawing_processed_dir = os.path.join(subdir, 'line_drawing_processed')
    save_dir = os.path.join(subdir, 'line_drawing_overlap')

    os.makedirs(save_dir, exist_ok=True)

    if len(os.listdir(save_dir)) == len(image_names):
        print(f'{save_dir} already processed, skipping...')
        continue

    for image_name in image_names:
        line_drawing_image_path = os.path.join(line_drawing_dir, image_name)
        line_drawing_processed_image_path = os.path.join(line_drawing_processed_dir, image_name)

        line_drawing_processed_gray = np.array(Image.open(line_drawing_processed_image_path).convert('L')) # (H, W)
        line_drawing_gray = np.array(Image.open(line_drawing_image_path).convert('L')) # (H, W)

        # for position where line_drawing_processed_image is 255 but line_drawing_image is < 128, set to red (255, 0, 0)
        condition = (line_drawing_processed_gray == 255) & (line_drawing_gray < 32)
        output_image = np.array(Image.open(line_drawing_image_path).convert('L')) # (H, W)
        output_image = np.where(output_image > 128, 255, 0).astype(np.uint8)
        output_image = np.stack([output_image, output_image, output_image], axis=2)
        output_image[condition] = [255, 0, 0] # red
        
        # Save the result
        save_path = os.path.join(save_dir, image_name)
        Image.fromarray(output_image).save(save_path)


        