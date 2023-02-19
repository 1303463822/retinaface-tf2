import os
import numpy as np
import cv2
from tqdm import tqdm

dir_origin_path = "E:\dataset\webface\WebFace260M"
dir_save_path1 = "E:\Dataset\dataset\lfw\lfw3_1680"
dir_save_path2 = "E:\Dataset\dataset\lfw\lfw4_1680"
img_path = os.listdir(dir_origin_path)
for img_names in tqdm(img_path):
    try:
        image_list = os.listdir(os.path.join(dir_origin_path, img_names))
        if len(image_list) >= 2:
            # print(image_list[0])
            image_path1 = os.path.join(dir_origin_path, img_names, image_list[0])
            image1 = cv2.imread(image_path1)
            # print(image_path1)
            image_path2 =os.path.join(dir_origin_path, img_names, image_list[1])
            image2 = cv2.imread(image_path2)
            if not os.path.exists(dir_save_path1):
                os.makedirs(dir_save_path1)
            if not os.path.exists(dir_save_path2):
                os.makedirs(dir_save_path2)
            cv2.imwrite(os.path.join(dir_save_path1, image_list[0]), image1)
            cv2.imwrite(os.path.join(dir_save_path2, image_list[1]), image2)
    except:
        pass

    # if img_names.lower().endswith(('1.bmp', '.dib', '.png', '0001.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
    #     image_path  = os.path.join(dir_origin_path, img_names)
    #     # print(image_path)
    #     image       = cv2.imread(image_path)
    #     # print(np.array(image).shape)
    #     if not os.path.exists(dir_save_path):
    #         os.makedirs(dir_save_path)
    #     cv2.imwrite(os.path.join(dir_save_path, img_names), image)

