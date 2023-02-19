import datetime
import os

import cv2
import numpy as np

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic,  QtWidgets
from tqdm import tqdm

from retinaface import Retinaface


class MyWindow(QWidget):

    def __init__(self):
        super(MyWindow, self).__init__()
        self.w = None
        self.ui = None
        self.directory = None
        self.init_ui()

    def init_ui(self):
        # 初始化窗口
        self.ui = uic.loadUi("Face_decetion.ui")
        self.choose_btn = self.ui.pushButton_2
        self.start_btn = self.ui.pushButton
        self.determine_btn = self.ui.pushButton_3
        self.text_label = self.ui.label
        self.pgb = self.ui.progressBar
        self.number_edit = self.ui.lineEdit


        self.text_label.setFrameShape(QtWidgets.QFrame.Box)
        self.text_label.setFrameShadow(QtWidgets.QFrame.Raised)
        self.text_label.setFrameShape(QFrame.Box)
        self.text_label.setText("选择需要检测的文件夹")
        self.number_edit.setPlaceholderText("输入检测图片数量")

        # 绑定信号和槽函数
        self.choose_btn.clicked.connect(self.choose_file)
        self.start_btn.clicked.connect(lambda: self.start_detect(dir_origin_path=self.directory))
        self.determine_btn.clicked.connect(self.determine_number)

    def choose_file(self):
        self.directory = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        directory1 = self.directory.split('/')[-1]
        if directory1 == "":
            self.text_label.setText("选择需要检测的文件夹")
        else:
            self.text_label.setText("所选文件夹为："+directory1)
        return self.directory

    def determine_number(self):
        self.number = int(self.number_edit.text())
        print(self.number)
        return self.number

    def start_detect(self, dir_origin_path):
        retinaface = Retinaface()
        number = str(self.number)
        dir_save_path = dir_origin_path + number + "_out"
        img_paths = os.listdir(dir_origin_path)
        time1 = datetime.datetime.now()
        count = 0
        # for img_names in tqdm(img_paths):
        # for img_name in os.listdir(dir_origin_path):
        for img_name in tqdm(os.listdir(dir_origin_path)):
            count += 1
            if count > self.number:
                break
            elif img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.jfif', '.ppm', '.tif', '.tiff', '.webp')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = cv2.imread(image_path)

                # print(image)
                try:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    r_image = retinaface.detect_image(image)
                    r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
                    img_name = str(img_name.split(".")[0]) + ".jpg"
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    cv2.imwrite(os.path.join(dir_save_path, img_name), r_image)
                except:
                    print(img_name)
                    print(image)
                    print("图片损坏")


        time2 = datetime.datetime.now()
        print(time2 - time1)
