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
        self.diff_color1 = (73, 73, 255)  # bgr 漏装颜色 红
        self.diff_color2 = (0, 126, 227)  # bgr 多装颜色 橙色
        self.class_filter = None
        self.class_names = None
        self.draw_label = True
        try:
            self.read_config()
            self.load_deep_model()
        except Exception as e:
            print(e)
            pass

    def load_deep_model(self):
        # device = select_device('cpu')
        device = torch.device('cpu')
        self.model = DetectMultiBackend(self.model_path, device=device, dnn=False, data=None, fp16=False)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.image_size = check_img_size(self.image_size, s=stride)  # check image size
        # Run inference
        bs = 1  # batch_size
        self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *self.image_size))  # warmup

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

    def detect_one(self, img):
        img_org = img.copy()
        gray_tolerance = 35
        detect_range, mask = detectUtil.find_image_primary_area(img, tolerance=gray_tolerance)
        img_r = img_org.copy()
        img_r[np.where(detect_range[:, :] == 0)] = [0, 0, 0]
        result = self.__detect_one_image(img_r)
        return result

    def detect(self, img1, img2, base_label=None):
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

        img1_r = self.img1.copy()
        img2_r = self.img2.copy()
        img1_r[np.where(self.detect_range1[:, :] == 0)] = [0, 0, 0]
        img2_r[np.where(self.detect_range2[:, :] == 0)] = [0, 0, 0]
        result1 = base_label
        if base_label is None or len(base_label) == 0:
            result1 = self.__detect_one_image(img1_r)
        result2 = self.__detect_one_image(img2_r)
        # print(result1, result2)
        same_obj, diff_obj1, diff_obj2 = self.__find_difference(result1, result2)
        # print(same_obj, diff_obj)
        out_img = self.__draw_result(same_obj, diff_obj1, diff_obj2)
        all_result = diff_obj1 + diff_obj2
        self.__output_img("result", out_img)
        return out_img, all_result

    def __detail_detect(self, detect_area):
        x1, y1, x2, y2 = int(detect_area[0]), int(detect_area[1]), int(detect_area[2]), int(detect_area[3])
        # sub_img1, sub_img2 = np.zeros((x2 - x1, y2 - y1, 3)), np.zeros((x2 - x1, y2 - y1, 3))
        sub_img1 = self.img1[y1:y2, x1:x2, :]
        sub_img2 = self.img2[y1:y2, x1:x2, :]
        self.__output_img("sub_img1", sub_img1)
        self.__output_img("sub_img2", sub_img2)
        img1_gray = cv2.cvtColor(deepcopy(sub_img1), cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(deepcopy(sub_img2), cv2.COLOR_BGR2GRAY)
        # canny边缘提取
        img1_contours = cv2.Canny(img1_gray, 50, 80)
        img2_contours = cv2.Canny(img2_gray, 50, 80)
        img1_gray = cv2.bilateralFilter(img1_gray, 5, 75, 75)
        img2_gray = cv2.bilateralFilter(img2_gray, 5, 75, 75)
        # 图像1减图像2
        sub = cv2.subtract(img1_gray, img2_gray)
        # 图像2减图象1
        sub2 = cv2.subtract(img2_gray, img1_gray)
        # 查找相减后的图像梯度
        grad_dir1 = detectUtil.calculate_grad_direction(sub)
        grad_dir2 = detectUtil.calculate_grad_direction(sub2)
        sub_contours = cv2.Canny(sub, 80, 140)
        sub_contours2 = cv2.Canny(sub2, 80, 140)
        # 查找所有轮廓，并以列表形式给出(RETR_LIST),轮廓包含所有点（CHAIN_APPROX_NONE）,不使用近似
        contour_list1 = cv2.findContours(sub_contours, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        contour_list2 = cv2.findContours(sub_contours2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        contour_list1 = detectUtil.remove_duplicated_contours(contour_list1)
        contour_list2 = detectUtil.remove_duplicated_contours(contour_list2)
        contours = self.__same_contour_filter(contour_list1, grad_dir1, img1_contours, img2_contours)
        contours2 = self.__same_contour_filter(contour_list2, grad_dir2, img2_contours, img1_contours)
        return len(contours) > 0 or len(contours2) > 0
        pass

    def __same_contour_filter(self, contours, grad_dir, contour_img1, contour_img2):
        max_dis = int(math.sqrt(contour_img1.shape[0] ** 2 + contour_img1.shape[1] ** 2))
        contours_nearest_dis1 = []
        contours_nearest_dis2 = []
        threshold = 8
        for i in range(len(contours)):
            l_len = len(contours[i])
            contours_nearest_dis1.append(np.array([max_dis] * l_len))
            contours_nearest_dis2.append(np.array([max_dis] * l_len))
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                x = contours[i][j][0][0]
                y = contours[i][j][0][1]
                find_flag = np.full(2, False)
                step = 0
                while not (find_flag == True).all():
                    step_x = int(step * grad_dir[y, x][0])
                    step_y = int(step * grad_dir[y, x][1])
                    dis = math.sqrt(step_x ** 2 + step_y ** 2)
                    searchxy = np.zeros([4, 4], dtype=np.int32)
                    contour_imgs = [contour_img1, contour_img2]
                    contours_nearest_dis_list = [contours_nearest_dis1, contours_nearest_dis2]
                    # 延法矢和垂直于法矢共4个方向搜索最短距离
                    searchxy[0] = [x + step_x, y + step_y, x - step_x, y - step_y]
                    searchxy[1] = [x + step_y, y + step_x, x - step_y, y - step_x]
                    for k in range(len(find_flag)):
                        if not find_flag[k]:
                            find_flag_ret = detectUtil.find_nearest_point(searchxy[0][0], searchxy[0][1],
                                                                          searchxy[0][2], searchxy[0][3],
                                                                          contour_imgs[k])
                            find_flag[k] |= find_flag_ret
                            if find_flag_ret:
                                contours_nearest_dis_list[k][i][j] = dis
                            find_flag_ret = detectUtil.find_nearest_point(searchxy[1][0], searchxy[1][1],
                                                                          searchxy[1][2], searchxy[1][3],
                                                                          contour_imgs[k])
                            find_flag[k] |= find_flag_ret
                            if find_flag_ret:
                                contours_nearest_dis_list[k][i][j] = dis

                    step += 1
        diff_contours = []
        for i in range(len(contours)):
            diff_flag = False
            while True:
                # 太短或面积太小的边缘不采用梯度查找的方式，直接在周边搜索，过滤孔位、墩头等微小偏差，如果没有差异直接判断为没有差异
                if self.__short_contours_filter(contours[i], contour_img1, contour_img2, len_threshold=50,
                                                search_range=8, area_threshold=100):
                    break
                # 选择出图1图2均值相差太大的边缘，直接列为差异边缘
                if self.__mean_value_filter(contours_nearest_dis1[i], contours_nearest_dis2[i], threshold):
                    diff_flag = True
                    break
                break
            if diff_flag:
                diff_contours.append(contours[i])
        return diff_contours

    def __mean_value_filter(self, nearest_dis1, nearest_dis2, threshold=8):
        mean1 = np.mean(nearest_dis1)
        mean2 = np.mean(nearest_dis2)
        if mean2 > threshold or mean1 > threshold:
            return True
        return False

    def __short_contours_filter(self, contour, contour_img1, contour_img2, len_threshold=10, area_threshold=25,
                                search_range=7):
        contour_near_flag_tmp = np.full((len(contour), 2), False)
        contour_near_flag = [False] * len(contour)
        p_list = np.reshape(contour, (-1, 2))
        min_x = np.min(p_list[:, 0])
        max_x = np.max(p_list[:, 0])
        min_y = np.min(p_list[:, 1])
        max_y = np.max(p_list[:, 1])
        area = (max_x - min_x) * (max_y - min_y)
        if len(contour) <= len_threshold or area <= area_threshold:
            for i in range(len(contour)):
                x = contour[i][0][0]
                y = contour[i][0][1]
                for step in range(0, search_range):
                    flag1, flag2 = detectUtil.search_exist_point(x - step, x + step, y - step, y + step, contour_img1,
                                                                 contour_img2)
                    contour_near_flag_tmp[i][0] |= flag1
                    contour_near_flag_tmp[i][1] |= flag2
                    contour_near_flag[i] = contour_near_flag_tmp[i][0] and contour_near_flag_tmp[i][1]
                    if contour_near_flag[i]:
                        break
            t_len = len(np.where(contour_near_flag)[0])
            if t_len / len(contour) > 0.5:
                return True
            else:
                return False
        else:
            return False

    def __find_difference(self, result1, result2):
        # [x1 ,y1 ,x2 ,y2 ,conf , class, auto_detect 自动识别 ,enable 是否生效]
        compare_flag1 = [0] * len(result1)
        compare_flag2 = [0] * len(result2)
        result1 = torch.tensor(result1)
        result2 = torch.tensor(result2)
        for i in range(len(result1)):
            for j in range(len(result2)):
                iou_result = box_iou(result1[i][0:4].unsqueeze(0), result2[j][0:4].unsqueeze(0)).squeeze(0)
                if iou_result > self.iou_threshold:
                    # 若该标签不生效,状态置为2
                    if not result1[i][-1]:
                        compare_flag1[i] = 2
                        compare_flag2[j] = 2
                    else:
                        compare_flag1[i] = 1
                        compare_flag2[j] = 1
        same_obj, diff_obj_1, diff_obj_2 = [], [], []
        for i in range(len(result1)):
            if compare_flag1[i] == 1:
                same_obj.append(result1[i])
            elif compare_flag1[i] == 0:
                # 如果不是自动识别的特征，用传统算法做进一步处理：
                if result1[i][6] == 0:
                    detail_ret = self.__detail_detect(result1[i])
                    # 如果不一致
                    if detail_ret:
                        diff_obj_1.append(result1[i])
                    else:
                        same_obj.append(result1[i])
                else:
                    diff_obj_1.append(result1[i])
        for i in range(len(result2)):
            if compare_flag2[i] == 0:
                diff_obj_2.append(result2[i])
        return same_obj, diff_obj_1, diff_obj_2

    def __draw_result(self, same_obj, diff_obj1, diff_obj2):
        self.result_img = self.__get_contrast_img()
        for i in same_obj:
            self.__draw_box(i, self.same_color)
        for i in diff_obj1:
            self.__draw_box(i, self.diff_color1)
        for i in diff_obj2:
            self.__draw_box(i, self.diff_color2)
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
        # return torch.tensor([])
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
            # [x1 ,y1 ,x2 ,y2 ,conf , class, auto_detect 自动识别 ,enable 是否生效]
            det = np.concatenate((det, np.ones((det.shape[0], 2), dtype=int)), axis=1)
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
