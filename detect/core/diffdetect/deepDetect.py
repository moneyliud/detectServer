import configparser
import cv2
import math
import numpy as np
import os.path
import random
import sys
import torch
from copy import deepcopy

import detect.core.diffdetect.detectUtil as detectUtil
from detect.models.common import DetectMultiBackend
from detect.utils.augmentations import (letterbox)
from detect.utils.general import (Profile, check_img_size, non_max_suppression, scale_boxes)
from detect.utils.torch_utils import select_device
from detect.utils.metrics import box_iou

cv2.ocl.setUseOpenCL(False)


class DeepDetect:
    def __init__(self, min_shape=1000):
        self.dir = ""
        self.M = None
        self.orb = cv2.ORB_create(2000)
        self.show_img = False
        self.save_img = False
        self.min_shape = min_shape
        self.img1 = None
        self.img2 = None
        self.img1_org = None
        self.img2_org = None
        self.h = None
        self.w = None
        self.detect_range1 = None
        self.detect_range2 = None
        # 最小差异点数量
        self.key_point_size = 0
        self.mask1 = None
        self.mask2 = None
        self.mtx = None
        self.dist = None
        self.image_size = [640, 640]  # inference size (height, width)
        self.conf_threshold = 0.25  # confidence threshold
        self.iou_threshold = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.model_path = None
        self.model = None
        self.stride = 32
        self.result_img = None
        self.line_width = None
        self.same_color = (11, 255, 11)  # bgr
        self.diff_color = (10, 10, 255)  # bgr
        self.class_filter = None
        self.class_names = None
        self.draw_label = True
        try:
            self.read_config()
            # self.load_deep_model()
        except Exception as e:
            print(e)
            pass

    def load_deep_model(self):
        device = select_device('cpu')
        self.model = DetectMultiBackend(self.model_path, device=device, dnn=False, data=None, fp16=False)
        # stride, names, pt = self.model.stride, self.model.names, self.model.pt
        # self.image_size = check_img_size(self.image_size, s=stride)  # check image size
        # # Run inference
        # bs = 1  # batch_size
        # self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *self.image_size))  # warmup

    def read_config(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        camera_items = dict(config.items('Camera'))
        cnn_items = dict(config.items('CNN'))
        if camera_items:
            mtx_tmp = camera_items['mtx']
            dist_tmp = camera_items['dist']
            self.mtx = np.float32(mtx_tmp.split(",")).reshape(3, 3)
            self.dist = np.float32([dist_tmp.split(",")])
        if cnn_items:
            self.conf_threshold = np.float32(cnn_items['conf_threshold'])
            self.iou_threshold = np.float32(cnn_items['iou_threshold'])
            self.model_path = cnn_items['model_path']
            self.class_filter = np.int32(cnn_items['class_filter'].split(","))
            self.class_names = cnn_items['class_name'].split(",")

    def detect(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.line_width = max(round(sum(img1.shape) / 2 * 0.0015), 2)
        # 原始图像深拷贝
        self.img1_org = self.img1.copy()
        self.img2_org = self.img2.copy()

        # 图像畸变矫正
        if self.mtx is not None and self.dist is not None:
            self.img1 = self.__distort_image(self.img1)
            self.img2 = self.__distort_image(self.img2)
            self.__output_img("distort1", self.img1)
            self.__output_img("distort12", self.img2)

        gray_tolerance = 35
        self.detect_range1, self.mask1 = detectUtil.find_image_primary_area(self.img1,
                                                                            tolerance=gray_tolerance)
        self.detect_range2, self.mask2 = detectUtil.find_image_primary_area(self.img2,
                                                                            tolerance=gray_tolerance)
        self.__output_img("detect_range1", self.detect_range1)
        self.__output_img("detect_range2", self.detect_range2)
        self.__output_img("mask1", self.mask1)

        # 设定关键点的尺度
        self.key_point_size = int(self.img1.shape[1] * 0.0025)
        # 计算单应性变换矩阵
        self.M, mask = self.__calculate_convert_matrix()
        self.h, self.w = self.img1_org.shape[:2]
        # 图像2变换对齐
        self.img2 = cv2.warpPerspective(self.img2, np.linalg.inv(self.M), (self.w, self.h))

        result1 = self.__detect_one_image(self.img1)
        result2 = self.__detect_one_image(self.img2)
        # print(result1, result2)
        same_obj, diff_obj = self.__find_difference(result1, result2)
        # print(same_obj, diff_obj)
        out_img = self.__draw_result(same_obj, diff_obj)
        all_result = diff_obj
        self.__output_img("result", out_img)
        return out_img, all_result

    def __find_difference(self, result1, result2):
        used_flag1 = [False] * len(result1)
        used_flag2 = [False] * len(result2)
        for i in range(len(result1)):
            for j in range(len(result2)):
                iou_result = box_iou(result1[i][0:4].unsqueeze(0), result2[j][0:4].unsqueeze(0)).squeeze(0)
                if not used_flag2[j] and iou_result > self.iou_threshold:
                    used_flag1[i] = True
                    used_flag2[j] = True
                    break
        same_obj, diff_obj = [], []
        for i in range(len(result1)):
            if used_flag1[i]:
                same_obj.append(result1[i])
            else:
                diff_obj.append(result1[i])
        for i in range(len(result2)):
            if not used_flag2[i]:
                diff_obj.append(result2[i])
        return same_obj, diff_obj

    def __draw_result(self, same_obj, diff_obj):
        self.result_img = self.__get_contrast_img()
        for i in same_obj:
            self.__draw_box(i, self.same_color)
        for i in diff_obj:
            self.__draw_box(i, self.diff_color)
        return self.result_img

    def __draw_box(self, box, color):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.result_img, p1, p2, color, thickness=self.line_width, lineType=cv2.LINE_AA)
        label = self.class_names[int(box[-1])]
        self.__draw_label(p1, p2, label, color)
        p3 = (p1[0] + self.img1.shape[1], p1[1])
        p4 = (p2[0] + self.img1.shape[1], p2[1])
        self.__draw_label(p3, p4, label, color)
        cv2.rectangle(self.result_img, p3, p4, color, thickness=self.line_width, lineType=cv2.LINE_AA)

    def __draw_label(self, p1, p2, label, color):
        if label:
            tf = int(max(self.line_width * 3 / 2, 1))  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=self.line_width / 2, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.result_img, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.result_img,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        self.line_width / 2,
                        (255, 255, 255),
                        thickness=tf,
                        lineType=cv2.LINE_AA)

    def __detect_one_image(self, im0):
        return torch.tensor([])
        im = letterbox(im0, self.image_size, stride=self.stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        with dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = self.model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, self.class_filter, False,
                                       max_det=self.max_det)

        result = []
        det = pred[0]
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            return det
        else:
            return []
        pass

    def __calculate_hsv_diff(self, hsv_filled1, hsv_diff1, hsv_filled2, hsv_diff2, channel):
        array1 = hsv_filled1[np.where(hsv_filled1[:, :, channel] != 0)][:, channel]
        array2 = hsv_filled2[np.where(hsv_filled2[:, :, channel] != 0)][:, channel]
        array3 = hsv_diff1[np.where(hsv_diff1[:, :, channel] != 0)][:, channel]
        array4 = hsv_diff2[np.where(hsv_diff2[:, :, channel] != 0)][:, channel]
        # 如果填充区域基本上为黑色，直接过滤掉不要
        if len(array1) < 10 or len(array2) < 10:
            return 0, 0, 0, 0
        filled_mean1 = np.mean(array1) if len(array1) > 0 else 0
        filled_mean2 = np.mean(array2) if len(array2) > 0 else 0
        diff_mean1 = np.mean(array3) if len(array3) > 0 else 0
        diff_mean2 = np.mean(array4) if len(array4) > 0 else 0
        # 图像1 h通道均值差异
        minus1 = abs(filled_mean1 - diff_mean1)
        # 图像2 h通道均值差异
        minus2 = abs(filled_mean2 - diff_mean2)
        return abs(minus1 - minus2), abs(filled_mean2 - filled_mean1), filled_mean1, filled_mean2

    # 计算图像单应性变换矩阵
    def __calculate_convert_matrix(self):
        # 检测关键点
        detect_type = "orb"
        matches = []
        if detect_type == "orb":
            # orb
            kp1, des1 = self.orb.detectAndCompute(self.img1, None)
            kp2, des2 = self.orb.detectAndCompute(self.img2, None)
        else:
            # sift
            kp1, des1 = self.sift.detectAndCompute(self.img1, None)
            kp2, des2 = self.sift.detectAndCompute(self.img2, None)
        # 根据主体识别范围进行过滤
        kp1, des1 = detectUtil.sift_key_point_filter(kp1, des1, self.detect_range1)
        kp2, des2 = detectUtil.sift_key_point_filter(kp2, des2, self.detect_range2)

        # 关键点匹配
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=6)
        search_params = dict(checks=10)

        if detect_type == "orb":
            matches = cv2.BFMatcher(cv2.NORM_HAMMING).match(des1, des2)
        else:
            # sift
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

        good = []
        if detect_type == "orb":
            for i in matches:
                if i.distance < 50:
                    good.append(i)
        else:
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    # if m.distance < 0.7 * n.distance:
                    good.append(m)

        # 把good中的左右点分别提出来找单应性变换
        pts_src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 单应性变换
        M, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
        if self.show_img or self.save_img:
            self.__show_sift_image(pts_src, pts_dst, mask)
        if M is None:
            M = np.diag([1, 1, 1])
        return M, mask

    # 显示SIFT特征点对比图像
    def __show_sift_image(self, pts_src, pts_dst, mask):
        # 输出图片初始化
        height = max(self.img1.shape[0], self.img2.shape[0])
        width = self.img1.shape[1] + self.img1.shape[1]
        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[0:self.img1.shape[0], 0:self.img1.shape[1]] = self.img1
        output[0:self.img2.shape[0], self.img2.shape[1]:] = self.img2[:]

        # 把点画出来
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

        for i in range(len(mask)):

            left = pts_src[i][0]
            right = pts_dst[i][0]
            colormap_idx = int((left[0] - self.img1.shape[1] * .5 + left[1] - self.img1.shape[0] * .5) * 256. / (
                    self.img1.shape[0] * .5 + self.img1.shape[1] * .5))

            if mask[i] == 1:
                color = tuple(map(int, _colormap[colormap_idx, 0, :]))
                # 只展示部分匹配对
                if i % 1 == 0:
                    cv2.circle(output, (int(pts_src[i][0][0]), int(pts_src[i][0][1])), 2, color, 2)
                    cv2.circle(output, (int(pts_dst[i][0][0]) + self.img1.shape[1], int(pts_dst[i][0][1])), 2, color, 2)
                    cv2.line(output, (int(pts_src[i][0][0]), int(pts_src[i][0][1])),
                             (int(pts_dst[i][0][0] + self.img1.shape[1]), int(pts_dst[i][0][1])), color, 1, 0)

        # 匹配结果输出
        outputN = cv2.resize(output, (int(self.img1.shape[1] * 2), int(self.img1.shape[0])),
                             interpolation=cv2.INTER_CUBIC)

        self.__output_img('SIFT_result', outputN)
        pass

    def __get_contrast_img(self):
        heightO = max(self.img1.shape[0], self.img2.shape[0])
        widthO = self.img1.shape[1] + self.img2.shape[1]
        contrast_img = np.zeros((heightO, widthO, 3), dtype=np.uint8)
        contrast_img[0:self.img1.shape[0], 0:self.img1.shape[1]] = self.img1
        contrast_img[0:self.img1.shape[0], self.img1.shape[1]:] = self.img2[:]
        return contrast_img

    def __convert_point_to_key_point(self, points):
        key_point = []
        point_index = []
        for i in range(len(points)):
            key_point.append(cv2.KeyPoint(points[i][0][0], points[i][0][1], self.key_point_size))
            point_index.append(i)
        return key_point, point_index

    def __output_img(self, name, img, flag=0):
        if self.show_img:
            cv2.imshow(name, img)
        if self.save_img or flag == 2:
            dir_path = os.path.dirname(os.path.abspath(__file__)) + "\\" + self.dir + "\\"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            # print(dir_path + name + ".jpg")
            cv2.imwrite(dir_path + name + ".jpg", img)

    def __distort_image(self, img):
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))  # 自由比例参数
        dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
        # dst = cv2.undistort(img, self.mtx, self.dist, None, None)
        return dst
