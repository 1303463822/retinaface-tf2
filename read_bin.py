import cv2
import numpy as np


def read2byte(path):
    '''
    图片转二进制
    path：图片路径
    byte_data：二进制数据
    '''
    with open(path, "rb") as f:
        byte_data = f.read()
    return byte_data


def byte2numpy(byte_data):
    '''
    byte转numpy矩阵/cv格式
    byte_data：二进制数据
    image : numpy矩阵/cv格式图片
    '''
    image = np.asarray(bytearray(byte_data), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

if __name__ == "__main__":
    # image_path = "img/111.jpg"
    # byte_data = read2byte(image_path)
    # print(byte_data)
    # image = byte2numpy(byte_data)
    # cv2.imshow("after", image)
    # cv2.waitKey(0)
    for line in open("E:\dataset\Asian-Celeb（亚洲名人数据集）/faces_glintasia/agedb_30.bin", 'rb'):
        print(line)
        # line = bytes(line, 'utf-8')
        image = byte2numpy("line")
        cv2.imshow("after", image)
        cv2.waitKey(1)
