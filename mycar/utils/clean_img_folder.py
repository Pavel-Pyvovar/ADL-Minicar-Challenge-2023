import os
import shutil

import cv2

# IMG_DIR = "/gpfs/space/home/pyvovar/repos/ADL-Minicar-Challenge-2023/mycar/data/stop_sign/tub_1_22-12-04/images"
IMG_DIR = "/gpfs/space/home/pyvovar/repos/ADL-Minicar-Challenge-2023/mycar/data/steering/tub_3_22-10-25/images"

if __name__=="__main__":
    cnt = 0
    for img_name in os.listdir(IMG_DIR):
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            cnt += 1
            # print(f"Corrupt image: {img_path}!")
        print(f"{cnt} corrupt images")