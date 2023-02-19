import os
import time
import datetime

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from skimage import transform as trans

from nets.retinaface import RetinaFace
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50, cfg_re101
from utils.utils import letterbox_image
from utils.utils_bbox import BBoxUtility, retinaface_correct_boxes


#------------------------------------#
#   请注意主干网络与预训练权重的对应
#   即注意修改model_path和backbone
#------------------------------------#
class Retinaface(object):
    _defaults = {
        #---------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择损失较低的即可。
        #---------------------------------------------------------------------#
        "model_path"        : 'model_data/retinaface_mobilenet025.h5',
        #---------------------------------------------------------------------#
        #   所使用的的主干网络：mobilenet、resnet50、resnet101
        #---------------------------------------------------------------------#
        "backbone"          : 'mobilenet',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.8,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.45,
        #----------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
        #   tf2代码中主干为mobilenet时存在小bug，当输入图像的宽高不为32的倍数
        #   会导致检测结果偏差，主干为resnet50不存在此问题。
        #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
        #----------------------------------------------------------------------#
        "input_shape"       : [640, 640, 3],
        #---------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        #---------------------------------------------------------------------#
        "letterbox_image"   : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Retinaface
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   不同主干网络的config信息
        #---------------------------------------------------#
        if self.backbone == "mobilenet":
            self.cfg = cfg_mnet
        elif self.backbone == "resnet50":
            self.cfg = cfg_re50
        else:
            self.cfg = cfg_re101

        #---------------------------------------------------#
        #   工具箱和先验框的生成
        #---------------------------------------------------#
        self.bbox_util  = BBoxUtility(nms_thresh=self.nms_iou)
        self.anchors    = Anchors(self.cfg, image_size=(self.input_shape[0], self.input_shape[1])).get_anchors()
        self.generate()

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'tensorflow.keras model or weights must be a .h5 file.'

        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        self.retinaface = RetinaFace(self.cfg, self.backbone)
        self.retinaface.load_weights(self.model_path)
        print('{} model, anchors loaded.'.format(self.model_path))

    @tf.function
    def get_pred(self, photo):
        preds = self.retinaface(photo, training=False)
        return preds

    def zero(self, x):
        if x < 0:
            x = 0
        return x

    # 获取最大人脸预测框
    def get_max_fram(self, x):
        results = list(x)
        for i in range(len(results)):
            size = (results[i][2] - results[i][0]) * (results[i][3] - results[i][1])  # 计算人脸预测框尺寸
            results[i] = np.append(results[i], size)
        results = sorted(results, key=lambda x: x[15], reverse=True)
        results = results[:1]
        return results

    # 获得图片中心人脸预测框
    def get_middle_fram(self, x, old_image):
        results = list(x)
        img_size = old_image.shape
        middle_height = int(img_size[0] / 2)
        middle_width = int(img_size[1] / 2)
        # cv2.circle(old_image, (middle_width, middle_height), 1, (0, 0, 255), 4)
        for i in range(len(results)):
            fram_height_middle = int((results[i][3]+results[i][1])/2)
            fram_width_middle = int((results[i][2] + results[i][0])/2)
            # cv2.circle(old_image, (fram_width_middle, fram_height_middle), 1, (0, 0, 255), 4)
            distance = pow(pow(fram_height_middle-middle_height, 2)+pow(fram_width_middle-middle_width, 2), 0.5)
            results[i] = np.append(results[i], distance)
        results = sorted(results, key=lambda x: x[15])
        results = results[:1]
        for b in results:
            b  = list(map(self.zero, map(int, b)))
            landmark = np.array(
                [[b[5], b[6]], [b[7], b[8]], [b[9], b[10]], [b[11], b[12]], [b[13], b[14]]]
            )
        return results, landmark

    # 获得仿射变换矩阵
    def estimate_norm(self, landmark):
        arcface_src = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
             [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32)
        tform = trans.SimilarityTransform()
        # lmk_tran = np.insert(landmark, 2, values=np.ones(5), axis=1)
        tform.estimate(landmark, arcface_src)
        M = tform.params[0:2, :]
        return M

    def norm_crop(self, old_image, landmark, image_size=112):
        M = self.estimate_norm(landmark)
        warped = cv2.warpAffine(old_image, M, (image_size, image_size), borderValue=0.0)
        return warped


    def get_crop_image(self, results, old_image):
        for b in results:
            b = list(map(self.zero, map(int, b)))
            # cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            # cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            # cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            # cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            # cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            # cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
            # print(b[0], b[1], b[2], b[3], b[4])
            # 获取原人脸预测框的坐标
            left_up = np.array([b[0], b[1]])
            left_down = np.array([b[0], b[3]])
            right_up = np.array([b[2], b[1]])
            right_down = np.array([b[2], b[3]])
            height_adjustment = (b[3] - b[1])/14
            width_adjustment = (b[2] - b[0])/8
            # crop_image = old_image[int(self.zero(min(left_up[1], right_up[1]) - height_adjustment)):int(max(left_down[1], right_down[1]) + height_adjustment),
            #                     int(self.zero(min(left_up[0], left_down[0]) - width_adjustment)):int(max(right_up[0], right_down[0]) + width_adjustment)]
            # 获取人眼坐标,计算人眼向量
            left_eye = np.array([b[5], b[6]])
            right_eye = np.array([b[7], b[8]])
            dp = left_eye - right_eye
            # 计算图片矫正角度
            if dp[0] == 0:
                angle = np.arctan(dp[1] / (dp[0] + 1e-5))
            else:
                angle = np.arctan(dp[1] / dp[0])
            # ---------------------------------------------------#
            #   自适应图片边框大小
            # ---------------------------------------------------#
            # 获得图像的大小，宽高
            img_size = old_image.shape
            height = img_size[0]
            width = img_size[1]
            t1 = datetime.datetime.now()
            mat = cv2.getRotationMatrix2D((width*0.5, height*0.5), angle=+angle * 180 / np.pi, scale=1)
            t2 = datetime.datetime.now()
            cos = np.abs(mat[0, 0])
            sin = np.abs(mat[0, 1])
            # 计算新宽高,更新仿射变换矩阵
            new_width = width * cos + height * sin
            new_height = width * sin + height * cos
            mat[0, 2] += (new_width - width) * 0.5    # 代表中心点水平平移的距离
            mat[1, 2] += (new_height - height) * 0.5  # 代表中心点竖直平移的距离
            rot_image = cv2.warpAffine(old_image, mat, (int(new_width), int(new_height)))

            t3 = datetime.datetime.now()
            org_image_center = (np.array(old_image.shape[:2][::-1]) - 1) / 2          # 原图片的中心
            rot_image_center = (np.array(rot_image.shape[:2][::-1]) - 1) / 2    # 旋转后图片的中心
            R = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])  # 旋转矩阵
            # 计算旋转后人脸预测框的坐标
            left_up = np.dot(R, left_up - org_image_center) + rot_image_center
            left_down = np.dot(R, left_down - org_image_center) + rot_image_center
            right_up = np.dot(R, right_up - org_image_center) + rot_image_center
            right_down = np.dot(R, right_down - org_image_center) + rot_image_center
            # 截取人脸
            crop_image = rot_image[int(self.zero(min(left_up[1], right_up[1]))):int(max(left_down[1], right_down[1])-height_adjustment),
                         int(self.zero(min(left_up[0], left_down[0])-width_adjustment)):int(max(right_up[0], right_down[0])+width_adjustment)]
            # print("*" * 50)
            # print(t2 - t1)
            # print("*" * 50)
            # print(t3 - t2)
            # crop_image = rot_image[int(min(left_up[1], right_up[1]) - (b[3] - b[1]) / 15):int(max(left_down[1], right_down[1]) + (b[3] - b[1]) / 15),
            #              int(min(left_up[0], left_down[0]) - (b[2] - b[0]) / 12):int(max(right_up[0], right_down[0]) + (b[2] - b[0]) / 12)]
            return crop_image

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
        old_image   = image.copy()
        #---------------------------------------------------#
        #   把图像转换成numpy的形式
        #---------------------------------------------------#
        image       = np.array(image, np.float32)
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        #---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        #---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        #---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
        #---------------------------------------------------------#
        #   图片预处理，归一化。
        #---------------------------------------------------------#
        photo   = np.expand_dims(preprocess_input(image), 0)
        
        #---------------------------------------------------------#
        #   传入网络进行预测
        #---------------------------------------------------------#
        preds = self.get_pred(photo)
        preds = [pred.numpy() for pred in preds]
        #-----------------------------------------------------------#
        #   将预测结果进行解码
        #-----------------------------------------------------------#
        results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)

        #---------------------------------------------------------#
        #   如果没有检测到物体，则返回原图
        #---------------------------------------------------------#
        if len(results) <= 0:
            return old_image

        results = np.array(results)
        #---------------------------------------------------------#
        #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
        #---------------------------------------------------------#
        if self.letterbox_image:
            results = retinaface_correct_boxes(results, np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
        results[:, :4] = results[:, :4] * scale
        results[:, 5:] = results[:, 5:] * scale_for_landmarks
        # results = results[0:1]  # 获得置信度最大的人脸预测框
        # 获得最大人脸预测框
        # results = self.get_max_fram(results)
        # 获得图片中心人脸预测框
        results, landmark = self.get_middle_fram(results, old_image)
        # 获得截取人脸
        wrap = self.norm_crop(old_image, landmark, image_size=112)
        # crop_image = self.get_cropping_img(results, old_image)
        # t1 = datetime.datetime.now()
        # crop_image = self.get_crop_image(results, old_image)
        # t2 = datetime.datetime.now()
        # print(t2 - t1)
        return wrap

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------#
        #   把图像转换成numpy的形式
        #---------------------------------------------------#
        image       = np.array(image, np.float32)
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        #---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
        #---------------------------------------------------------#
        #   图片预处理，归一化。
        #---------------------------------------------------------#
        photo   = np.expand_dims(preprocess_input(image), 0)
        
        #---------------------------------------------------------#
        #   传入网络进行预测
        #---------------------------------------------------------#
        preds   = self.get_pred(photo)
        preds   = [pred.numpy() for pred in preds]
        #---------------------------------------------------------#
        #   将预测结果进行解码
        #---------------------------------------------------------#
        results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)

        t1 = time.time()
        for _ in range(test_interval):
            #---------------------------------------------------------#
            #   传入网络进行预测
            #---------------------------------------------------------#
            preds   = self.get_pred(photo)
            preds   = [pred.numpy() for pred in preds]
            #---------------------------------------------------------#
            #   将预测结果进行解码
            #---------------------------------------------------------#
            results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def get_map_txt(self, image):
        #---------------------------------------------------#
        #   把图像转换成numpy的形式
        #---------------------------------------------------#
        image       = np.array(image, np.float32)
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        #---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        #---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        #---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
            
        #---------------------------------------------------------#
        #   图片预处理，归一化。
        #---------------------------------------------------------#
        photo   = np.expand_dims(preprocess_input(image), 0)
        
        #---------------------------------------------------------#
        #   传入网络进行预测
        #---------------------------------------------------------#
        preds   = self.get_pred(photo)
        preds   = [pred.numpy() for pred in preds]

        #---------------------------------------------------------#
        #   将预测结果进行解码
        #---------------------------------------------------------#
        results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold = self.confidence)

        #--------------------------------------#
        #   如果没有检测到物体，则返回原图
        #--------------------------------------#
        if len(results) <= 0:
            return np.array([])

        results = np.array(results)
        #---------------------------------------------------------#
        #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
        #---------------------------------------------------------#
        if self.letterbox_image:
            results = retinaface_correct_boxes(results, np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
        
        results[:, :4] = results[:, :4] * scale
        results[:, 5:] = results[:, 5:] * scale_for_landmarks

        return results
