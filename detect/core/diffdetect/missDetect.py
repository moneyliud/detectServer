import sys

import os.path
import cv2
import numpy as np
from copy import deepcopy
import math
import random
import detect.core.diffdetect.detectUtil as detectUtil

cv2.ocl.setUseOpenCL(False)


class MissDetect:
    def __init__(self, threshold=400, min_shape=1000):
        self.hsv_image2 = None
        self.hsv_image1 = None
        self.img2_warp = None
        self.img2_org_warp = None
        self.img1_kmeans_res_org = None
        self.img2_kmeans_res_org = None
        self.dir = ""
        self.img2_contours = None
        self.img1_contours = None
        self.M = None
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.orb = cv2.ORB_create(2000)
        self.detectDensity = 2
        self.show_img = False
        self.save_img = False
        self.threshold = threshold
        self.min_shape = min_shape
        self.img1 = None
        self.img2 = None
        self.img1_org = None
        self.img2_org = None
        self.img2_gray = None
        self.img1_gray = None
        self.h = None
        self.w = None
        self.detect_range1 = None
        self.detect_range2 = None
        # 最小差异点数量
        self.least_diff_point_num = 7
        self.key_point_size = 0
        self.mask1 = None
        self.mask2 = None
        self.remove_dark1 = None
        self.remove_dark2 = None
        self.mtx = np.array([[1.41198791e+03, 0.00000000e+00, 6.80102996e+02],
                             [0.00000000e+00, 1.41208078e+03, 9.04382079e+02],
                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        self.dist = np.array([[-2.92432027e-03, 2.73220082e-01, 1.15161400e-03, 4.61783815e-04,
                               -6.13711620e-01]])

    def detect(self, img1, img2):
        self.img1 = img1
        self.img2 = img2

        self.min_shape = 1800
        if self.img1.shape[0] < self.img1.shape[1]:
            self.img1 = cv2.resize(self.img1, dsize=(
                self.min_shape, int(self.min_shape * self.img1.shape[0] / self.img1.shape[1])))
            self.img2 = cv2.resize(self.img2, dsize=(
                self.min_shape, int(self.min_shape * self.img1.shape[0] / self.img1.shape[1])))
        else:
            self.img1 = cv2.resize(self.img1, dsize=(
                int(self.min_shape * self.img1.shape[1] / self.img1.shape[0]), self.min_shape))
            self.img2 = cv2.resize(self.img2, dsize=(
                int(self.min_shape * self.img1.shape[1] / self.img1.shape[0]), self.min_shape))

        # 原始图像深拷贝
        self.img1_org = self.img1.copy()
        self.img2_org = self.img2.copy()

        # 图像畸变矫正
        # self.detect_range1 = self.__distort_image(self.detect_range1)
        # self.detect_range2 = self.__distort_image(self.detect_range2)
        self.img1 = self.__distort_image(self.img1)
        self.img2 = self.__distort_image(self.img2)
        self.__output_img("distort1", self.img1)
        self.__output_img("distort12", self.img2)

        # 计算特征识别范围,SIFT计算单应性变换矩阵要用,并获取hsv图中的h色相通道图
        # primary_color1, self.detect_range1, self.hsv_image1, self.mask1 = detectUtil.find_image_primary_area(self.img1,
        #                                                                                                    hsv_tolerance=35)
        # primary_color2, self.detect_range2, self.hsv_image2, self.mask2 = detectUtil.find_image_primary_area(self.img2,
        #                                                                                                    hsv_tolerance=35)
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
        self.img2_org_warp = cv2.warpPerspective(self.img2_org, np.linalg.inv(self.M), (self.w, self.h))
        # 图像亮度统一
        self.img1, self.img2, self.hsv_image1, self.hsv_image2 = detectUtil.fit_image_brightness(self.img1, self.img2,
                                                                                                 self.detect_range1)
        self.__output_img("hsv_image1", self.hsv_image1)
        self.__output_img("hsv_image2", self.hsv_image2)
        # kmeans聚类分离图像主体 要放在图像2变换对齐后
        # self.img1_kmeans_res_org, self.img2_kmeans_res_org = self.__kmeans_cluster_img()
        # self.__output_img("img1_kmeans_res_org", self.img1_kmeans_res_org)
        # self.__output_img("img2_kmeans_res_org", self.img2_kmeans_res_org)
        # 转换出灰度图像
        self.img1_gray = cv2.cvtColor(deepcopy(self.img1), cv2.COLOR_BGR2GRAY)
        self.img2_gray = cv2.cvtColor(deepcopy(self.img2), cv2.COLOR_BGR2GRAY)

        for i in range(self.img1_gray.shape[0]):
            for j in range(self.img1_gray.shape[1]):
                if self.detect_range1[i, j] == 0:
                    self.img1_gray[i, j] = 0
                    self.img2_gray[i, j] = 0

        tmp1, self.remove_dark1 = cv2.threshold(self.img1_gray, gray_tolerance, 255, cv2.THRESH_BINARY)
        tmp2, self.remove_dark2 = cv2.threshold(self.img2_gray, gray_tolerance, 255, cv2.THRESH_BINARY)
        # canny边缘提取
        self.img1_contours = cv2.Canny(self.img1_gray, 50, 80)
        self.img2_contours = cv2.Canny(self.img2_gray, 50, 80)
        # 通过图像相减计算不同边缘
        difference_points = self.calculate_different_contours_by_sub()
        # 设置dbscan聚类最大距离
        self.detect_point_interval = 10
        out_img, all_result = self.__generate_result_img(difference_points, difference_points)
        self.__output_img("img1_contours", self.img1_contours)
        self.__output_img("img2_contours", self.img2_contours)
        return out_img, all_result

    def detect_old(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.min_shape = 1800
        if self.img1.shape[0] < self.img1.shape[1]:
            self.img1 = cv2.resize(self.img1, dsize=(
                self.min_shape, int(self.min_shape * self.img1.shape[0] / self.img1.shape[1])))
            self.img2 = cv2.resize(self.img2, dsize=(
                self.min_shape, int(self.min_shape * self.img1.shape[0] / self.img1.shape[1])))
        else:
            self.img1 = cv2.resize(self.img1, dsize=(
                int(self.min_shape * self.img1.shape[1] / self.img1.shape[0]), self.min_shape))
            self.img2 = cv2.resize(self.img2, dsize=(
                int(self.min_shape * self.img1.shape[1] / self.img1.shape[0]), self.min_shape))

        # 原始图像深拷贝
        self.img1_org = deepcopy(self.img1)
        self.img2_org = deepcopy(self.img2)

        # 计算特征识别范围,SIFT计算单应性变换矩阵要用,并获取hsv图中的h色相通道图
        # primary_color1, self.detect_range1, self.hsv_image1, self.mask1 = detectUtil.find_image_primary_area(self.img1,
        #                                                                                                    hsv_tolerance=35)
        # primary_color2, self.detect_range2, self.hsv_image2, self.mask2 = detectUtil.find_image_primary_area(self.img2,
        #                                                                                                    hsv_tolerance=35)

        self.detect_range1, self.mask1, self.hsv_image1 = detectUtil.find_image_primary_area(self.img1,
                                                                                             gray_tolerance=35)
        self.detect_range2, self.mask2, self.hsv_image2 = detectUtil.find_image_primary_area(self.img2,
                                                                                             gray_tolerance=35)
        # self.__output_img("detect_range1", self.detect_range1)
        # self.__output_img("detect_range2", self.detect_range2)
        # self.__output_img("hsv_image1", self.hsv_image1)
        # self.__output_img("hsv_image2", self.hsv_image2)
        # self.__output_img("mask1", self.mask1)

        # 设定关键点的尺度
        self.key_point_size = int(self.img1.shape[1] * 0.0025)
        # 计算单应性变换矩阵
        self.M, mask = self.__calculate_convert_matrix()

        # 图像2变换对齐
        self.img2 = cv2.warpPerspective(self.img2, np.linalg.inv(self.M), (self.w, self.h))
        self.img2_org_warp = cv2.warpPerspective(self.img2_org, np.linalg.inv(self.M), (self.w, self.h))
        # 转换出灰度图像
        img1_gray = cv2.cvtColor(deepcopy(self.img1), cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(deepcopy(self.img2), cv2.COLOR_BGR2GRAY)
        sub = cv2.subtract(img1_gray, img2_gray)
        for i in range(img1_gray.shape[0]):
            for j in range(img1_gray.shape[1]):
                if self.detect_range1[i, j] == 0:
                    sub[i, j] = 0
        grad_dir1 = detectUtil.calculate_grad_direction(sub)
        self.__output_img("sub", sub)
        sub2 = cv2.subtract(img2_gray, img1_gray)
        self.__output_img("sub2", sub2)
        sub_contours = cv2.Canny(sub, 100, 200)
        sub_contours_new = detectUtil.draw_direction_vector(sub_contours, grad_dir1)
        # 查找所有轮廓，并以列表形式给出(RETR_LIST),轮廓包含所有点（CHAIN_APPROX_NONE）,不使用近似
        contour_list1 = cv2.findContours(sub_contours, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        contours = detectUtil.same_contour_filter(contour_list1, img1_gray, img2_gray)
        self.__output_img("sub_contours", sub_contours)
        self.__output_img("sub_contours_new", sub_contours_new)
        sub_contours2 = cv2.Canny(sub2, 100, 200)
        self.__output_img("sub_contours2", sub_contours2)

        # kmeans聚类分离图像主体
        img1_kmeans_res, img2_kmeans_res = self.__kmeans_cluster_img()
        self.img1_kmeans_res_org = deepcopy(img1_kmeans_res)
        self.img2_kmeans_res_org = deepcopy(img2_kmeans_res)

        # 计算差异识别范围
        # img_detect_range = self.__calculate_detect_range(img1_kmeans_res)

        # 高斯滤波
        img1_gray = cv2.GaussianBlur(img1_gray, (3, 3), 1)
        img2_gray = cv2.GaussianBlur(img2_gray, (3, 3), 1)

        # canny边缘提取
        self.img1_gray = cv2.Canny(img1_gray, 50, 80)
        self.img2_gray = cv2.Canny(img2_gray, 50, 80)

        # contours = cv2.findContours(img1_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        self.img1_contours = self.img1_gray
        self.img2_contours = self.img2_gray

        sub_sub_contours = np.zeros((sub_contours.shape[0], sub_contours.shape[1]), dtype=np.uint8)
        for i in range(sub_contours.shape[0]):
            for j in range(sub_contours.shape[1]):
                if sub_contours[i][j] == 255:
                    sub_sub_contours[i][j] = sub_contours[i][j] - self.img1_contours[i][j]
        self.__output_img("sub_sub_contours", sub_sub_contours)

        self.h, self.w = self.img2.shape[:2]

        # self.xor_contours = self.__xor_image_gray(self.img1_contours, self.img2_contours)
        # self.__output_img("xor_contours", self.xor_contours)

        # 计算待分析差异的点位
        diff_left, diff_right = self.__calculate_difference_point(self.detect_range1)
        out_img = self.__generate_result_img(diff_left, diff_right)
        # 差异检测结果输出结果 ********************************************************
        return out_img, diff_left

    def __kmeans_cluster_img(self):
        # kmeans聚类分离主体和背景
        img1_array1 = np.float32(self.img1.reshape((-1, 3)))
        c = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # kmeans聚类分为主体和背景两类
        ret, label, center = cv2.kmeans(img1_array1, 2, None, c, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        label_color = [(0, 0, 0), (255, 255, 255)]
        first_color = deepcopy(res[0])
        # 计算主体和背景颜色灰度高的作为背景赋黑色(0, 0, 0)，灰度低的作为主体赋白色(255, 255, 255)
        label_org_color, label_index = detectUtil.get_label_org_color(label, res)
        for i in range(len(label)):
            res[i] = label_color[label_index[label[i][0]]]

        img_kmeans_res = res.reshape((self.img1_org.shape))
        img_kmeans_res = detectUtil.erode_dilate(img_kmeans_res, (3, 2), (3, 2))

        # 图像2与图像1的主图和背景颜色做匹配
        img2_array = np.float32(self.img2.reshape((-1, 3)))
        c = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # kmeans聚类分为主体和背景两类
        ret_2, label_2, center_2 = cv2.kmeans(img2_array, 2, None, c, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center_2)
        res_2 = center[label_2.flatten()]
        label_color_index_2 = detectUtil.match_label_color(label_org_color, label_2, res_2)
        for i in range(len(label_2)):
            res_2[i] = label_color[label_color_index_2[label_2[i][0]]]

        img2_kmeans_res = res_2.reshape((self.img2_org.shape))
        img2_kmeans_res = detectUtil.erode_dilate(img2_kmeans_res, (3, 2), (3, 2))
        return img_kmeans_res, img2_kmeans_res

    def calculate_different_contours_by_sub(self):
        img1_gray = cv2.bilateralFilter(self.img1_gray, 5, 75, 75)
        img2_gray = cv2.bilateralFilter(self.img2_gray, 5, 75, 75)
        # 中值滤波
        # img1_gray = cv2.medianBlur(img1_gray, 3, 1)
        # img2_gray = cv2.medianBlur(img2_gray, 3, 1)
        # # 高斯滤波
        # img1_gray = cv2.GaussianBlur(img1_gray, (5, 5), 1)
        # img2_gray = cv2.GaussianBlur(img2_gray, (5, 5), 1)
        # 图像1减图像2
        sub = cv2.subtract(img1_gray, img2_gray)
        # 图像2减图象1
        sub2 = cv2.subtract(img2_gray, img1_gray)
        self.__output_img("sub", sub)
        self.__output_img("sub2", sub2)

        # 检测范围外的区域置为0
        for i in range(img1_gray.shape[0]):
            for j in range(img1_gray.shape[1]):
                if self.detect_range1[i, j] == 0:
                    sub[i, j] = 0
                    sub2[i, j] = 0

        # 查找相减后的图像梯度
        grad_dir1 = detectUtil.calculate_grad_direction(sub)
        grad_dir2 = detectUtil.calculate_grad_direction(sub2)

        # 查找相减后的图像边缘
        sub_contours = cv2.Canny(sub, 80, 140)
        sub_contours2 = cv2.Canny(sub2, 80, 140)
        self.__output_img("sub_contours", sub_contours)
        self.__output_img("sub_contours2", sub_contours2)

        # sub_contours_new = detectUtil.draw_direction_vector(sub_contours, grad_dir1)
        # 查找所有轮廓，并以列表形式给出(RETR_LIST),轮廓包含所有点（CHAIN_APPROX_NONE）,不使用近似
        contour_list1 = cv2.findContours(sub_contours, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        contour_list2 = cv2.findContours(sub_contours2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

        # dbscan查找轮廓 有点慢
        # contour_list1 = detectUtil.dbscan_find_contours(sub_contours, eps=1.5, min_samples=2)
        # contour_list2 = detectUtil.dbscan_find_contours(sub_contours2, eps=1.5, min_samples=2)

        contour_list1 = detectUtil.remove_duplicated_contours(contour_list1)
        contour_list2 = detectUtil.remove_duplicated_contours(contour_list2)
        contours = self.__same_contour_filter(contour_list1, grad_dir1, self.img1_contours, self.img2_contours)
        contours2 = self.__same_contour_filter(contour_list2, grad_dir2, self.img2_contours, self.img1_contours)
        difference_points = []
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                difference_points.append([contours[i][j][0][0], contours[i][j][0][1]])
        for i in range(len(contours2)):
            for j in range(len(contours2[i])):
                difference_points.append([contours2[i][j][0][0], contours2[i][j][0][1]])
        self.show_diff_contour_img(contours, "diff_contours_img1")
        self.show_diff_contour_img(contours2, "diff_contours_img2")
        return difference_points

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
                if self.__short_contours_filter(contours[i], contour_img1, contour_img2, len_threshold=20,
                                                search_range=7, area_threshold=50):
                    break
                # 选择出图1图2均值相差太大的边缘，直接列为差异边缘
                if self.__mean_value_filter(contours_nearest_dis1[i], contours_nearest_dis2[i], threshold):
                    diff_flag = True
                    break
                # 如果边缘相差小，且长度较长，不列为差异边缘
                if self.__long_edge_filter(contours[i]):
                    break
                # 如果与图1图2边缘都不是特别相近，列为差异边缘
                # if self.__most_near_filter(contours_nearest_dis1[i], contours_nearest_dis2[i], 6, 0.4):
                #     diff_flag = True
                #     break
                # 如果边缘重合过多触发kmeans判断，防止漏项
                # if self.__kmeans_contour_filter(contours[i], i):
                #     diff_flag = True
                #     break
                # 根据HSV图像找出轮廓内外部差异，若轮廓内外部前后差异大，记为差异轮廓
                # break
                if self.__hsv_hull_filter(contours[i], str(i)):
                    diff_flag = True
                    break
                break
            if diff_flag:
                diff_contours.append(contours[i])
        return diff_contours

    # 根据HSV图像找出轮廓内外部差异，若轮廓内外部前后差异大，记为差异轮廓
    def __hsv_hull_filter(self, contour, postfix):
        hull = cv2.convexHull(contour)
        # 找出轮廓的外接矩形
        min_x, min_y, max_x, max_y = detectUtil.find_contour_rect(hull)
        # 扩大范围
        alpha = 2.5
        left, right, top, down = detectUtil.get_larger_rect(min_x, min_y, max_x, max_y, self.detect_range1, alpha)
        # 截取局部图像
        sub_image = np.zeros((int(down - top), int(right - left)), dtype=np.uint8)
        hull_sub = deepcopy(hull)
        for i in range(len(hull)):
            for j in range(len(hull[i])):
                hull_sub[i][j][0] = hull[i][j][0] - left
                hull_sub[i][j][1] = hull[i][j][1] - top
        hull_filled = cv2.drawContours(np.zeros(sub_image.shape), [hull_sub], -1, (255, 255, 255), cv2.FILLED)
        # 边缘膨胀
        hull_filled_large = cv2.dilate(hull_filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)),
                                       borderType=cv2.BORDER_CONSTANT,
                                       borderValue=0)
        hull_diff = np.zeros(sub_image.shape, dtype=np.uint8)
        # 计算扩大区域和原始区域的差异部分
        for i in range(len(hull_filled)):
            for j in range(len(hull_filled[i])):
                if hull_filled_large[i, j] != hull_filled[i, j]:
                    hull_diff[i, j] = 255
        sub_image_1 = self.hsv_image1[top:down, left:right]
        sub_image_2 = self.hsv_image2[top:down, left:right]
        dark1 = self.remove_dark1[top:down, left:right]
        dark2 = self.remove_dark2[top:down, left:right]
        # 消除图像死黑部分，排除hsv噪点
        detectUtil.set_zero_by_mask(sub_image_1, dark1)
        detectUtil.set_zero_by_mask(sub_image_2, dark2)

        filled1 = detectUtil.get_image_by_mask(sub_image_1, hull_filled)
        filled2 = detectUtil.get_image_by_mask(sub_image_2, hull_filled)
        diff1 = detectUtil.get_image_by_mask(sub_image_1, hull_diff)
        diff2 = detectUtil.get_image_by_mask(sub_image_2, hull_diff)

        # 计算rgb差异
        # mean_rgb = np.zeros((4, 3), np.uint8)
        # image_list = [filled1, filled2, diff1, diff2]
        # for i in range(len(mean_rgb)):
        #     mean_rgb[i] = detectUtil.mean_rgb_img(image_list[i])
        #     detectUtil.set_nonzero_color(image_list[i], mean_rgb[i])
        # rgb_diff1 = detectUtil.calculate_rgb_diff(mean_rgb[1], mean_rgb[0])
        # rgb_diff2 = detectUtil.calculate_rgb_diff(mean_rgb[3], mean_rgb[2])
        # print(postfix, rgb_diff1, rgb_diff2)

        # 计算hsv差异
        hsv_diff1, filled_diff1, mean1_h, mean2_h = self.__calculate_hsv_diff(filled1, diff1, filled2, diff2, 0)
        hsv_diff2, filled_diff2, mean1_s, mean2_s = self.__calculate_hsv_diff(filled1, diff1, filled2, diff2, 1)
        hsv_diff3, filled_diff3, mean1_v, mean2_v = self.__calculate_hsv_diff(filled1, diff1, filled2, diff2, 2)

        self.__output_img(postfix + "filled1", filled1)
        self.__output_img(postfix + "filled2", filled2)
        self.__output_img(postfix + "diff1", diff1)
        self.__output_img(postfix + "diff2", diff2)

        # if abs(rgb_diff1 - rgb_diff2) > 100:
        #     return True
        # else:
        #     return False

        # print(postfix, hsv_diff1, hsv_diff2, hsv_diff3, mean1_h, mean1_s, mean1_v)
        # print(postfix, filled_diff1, filled_diff2, filled_diff3)

        ret1 = np.array([hsv_diff1, hsv_diff2, hsv_diff3])
        ret2 = np.array([filled_diff1, filled_diff2, filled_diff3])
        max1 = np.max(ret1)
        max2 = np.max(ret2)
        # 内外差异大，并且前后差异大，黄色角片黄色背景的，只判断前后差异，不判断内外差异.v亮度都要小于200
        if max1 > 30 and max2 > 45 and max1 + max2 > 100 \
                and mean2_v < 200 and mean1_v < 200 \
                or (self.__is_yellow(mean1_h, mean1_s, mean1_v)
                    and max2 > 40 and self.__is_contour_near_rect(contour, 2, 0.6)):
            return True

    def __is_yellow(self, h, s, v):
        return 10 <= h <= 38 and s >= 43 and v >= 46

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

    def __kmeans_contour_filter(self, contour, index):
        # 找出轮廓的外接矩形
        min_x, min_y, max_x, max_y = detectUtil.find_contour_rect(contour)
        detect_point_interval = self.detectDensity * self.key_point_size  # 监测点间隔
        # 扩大对比范围系数
        points = []
        alpha = 1.5
        left, right, top, down = detectUtil.get_larger_rect(min_x, min_y, max_x, max_y, self.detect_range1, alpha)
        i = left
        while left <= i < right:
            j = top
            while top <= j < down:
                points.append([[i, j]])
                j += detect_point_interval
                pass
            i += detect_point_interval
            pass
        pass
        same_points, diff_points, is_point_diff = self.__sift_diff_comparator(points, self.img1_kmeans_res_org,
                                                                              self.img2_kmeans_res_org)
        same_points = np.expand_dims(same_points, 1)
        self.__show_difference_point(same_points, same_points, diff_points, diff_points,
                                     name="kmeans_sift_difference" + str(index), )
        if len(diff_points) == 0:
            return False
        result = detectUtil.dbscan_circle(diff_points, detect_point_interval * 2.5, min_samples=7)
        if len(result) > 0:
            return True
        else:
            return False
        pass

    def __long_edge_filter(self, contour):
        p_list = np.reshape(contour, (-1, 2))
        times = 8
        min_x = np.min(p_list[:, 0])
        max_x = np.max(p_list[:, 0])
        min_y = np.min(p_list[:, 1])
        max_y = np.max(p_list[:, 1])
        width = max_x - min_x
        height = max_y - min_y
        counter_len = len(contour)
        # 长宽比过大,判断为边缘线条不对比
        if (width * times < height or height * times < width) and counter_len > 30:
            return True
        if self.__is_contour_near_rect(contour):
            if counter_len > 200:
                return True
            else:
                return False
        else:
            if counter_len > 80:
                return True
            else:
                return False

    def __is_contour_near_rect(self, contour, edge_width_threshold=3, percent_threshold=0.8):
        min_x, min_y, max_x, max_y = detectUtil.find_contour_rect(contour)
        rect_len = (max_x - min_x + 1) * 2 + (max_y - min_y + 1) * 2 - 4
        point_map = {}
        count = 0
        for i in range(len(contour)):
            point = contour[i][0]
            point_key = str(point[0]) + "-" + str(point[1])
            if point_key not in point_map \
                    and (detectUtil.is_point_in_rect(point, min_x, min_y, min_x + edge_width_threshold, max_y) \
                         or detectUtil.is_point_in_rect(point, max_x - edge_width_threshold, min_y, max_x, max_y) \
                         or detectUtil.is_point_in_rect(point, min_x, min_y, max_x, min_y + edge_width_threshold) \
                         or detectUtil.is_point_in_rect(point, min_x, max_y - edge_width_threshold, max_x, max_y)):
                count += 1
                point_map[point_key] = True
        if count / rect_len > percent_threshold:
            return True
        else:
            return False

    def __mean_value_filter(self, nearest_dis1, nearest_dis2, threshold=8):
        mean1 = np.mean(nearest_dis1)
        mean2 = np.mean(nearest_dis2)
        if mean2 > threshold or mean1 > threshold:
            return True
        return False

    def __most_near_filter(self, nearest_dis1, nearest_dis2, dis_threshold=4, threshold=0.6):
        z1 = np.where(nearest_dis1 < dis_threshold)
        z2 = np.where(nearest_dis2 < dis_threshold)
        zlen1 = len(z1[0])
        zlen2 = len(z2[0])
        # z0 = np.where(nearest_dis2 < 1)
        # if len(z0) / len(nearest_dis1) > 0.9:
        #     return False
        if zlen1 / len(nearest_dis1) < threshold or zlen2 / len(nearest_dis2) < threshold:
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

    def show_diff_contour_img(self, contours, name="diff_contours_img"):
        diff_contours_img = np.zeros(self.img1_gray.shape, np.uint8)
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                diff_contours_img[contours[i][j][0][1], contours[i][j][0][0]] = 255
        self.__output_img(name, diff_contours_img)

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

    # def __calculate_detect_range(self, img1_kmeans_res):
    #     # kmeas聚类灰度图像获取，侵蚀膨胀填充空洞，canny算子识别边缘
    #     img_kmeans_gray = cv2.cvtColor(deepcopy(img1_kmeans_res), cv2.COLOR_BGR2GRAY)
    #     img_kmeans_gray = detectUtil.erode_dilate(img_kmeans_gray, (6, 4), (16, 12))
    #     img_kmeans_gray = cv2.Canny(img_kmeans_gray, 50, 100)
    #     img1_contours = cv2.findContours(img_kmeans_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #     max_area = 0
    #     max_contour = None
    #     # 查找零件的最大边缘作为轮廓
    #     for contour in img1_contours:
    #         cur_area = len(contour)
    #         if cur_area > max_area:
    #             max_area = cur_area
    #             max_contour = contour
    #     # 计算最大边缘的凸包
    #     hull = cv2.convexHull(max_contour)
    #     # 填充凸包作为识别范围
    #     img_detect_range = cv2.drawContours(np.zeros(img1_kmeans_res.shape), [hull], -1, (255, 255, 255), cv2.FILLED)
    #     self.__output_img('detect_range', img_detect_range)
    #     return img_detect_range

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

    # 计算待用SIFT算法检测的点位
    def __calculate_sift_point(self, detect_range):
        # 图像的长宽
        height, width, channel = self.img1.shape

        xMinLeft = 0
        xMaxLeft = width
        yMinLeft = 0
        yMaxLeft = height

        # 检测范围确定
        self.detect_point_interval = self.detectDensity * self.key_point_size  # 监测点间隔
        searchWidth = int((xMaxLeft - xMinLeft) / self.detect_point_interval - 2)
        searchHeight = int((yMaxLeft - yMinLeft) / self.detect_point_interval - 2)
        searchNum = searchWidth * searchHeight
        point_src = np.float32([[0] * 2] * searchNum * 1).reshape(-1, 1, 2)
        for i in range(searchWidth):
            for j in range(searchHeight):
                sx = xMinLeft + i * self.detect_point_interval + self.key_point_size
                sy = yMinLeft + j * self.detect_point_interval + self.key_point_size
                if (detect_range[sy][sx] == (255, 255, 255)).all():
                    point_src[i + j * searchWidth][0][
                        0] = xMinLeft + i * self.detect_point_interval + self.key_point_size
                    point_src[i + j * searchWidth][0][
                        1] = yMinLeft + j * self.detect_point_interval + self.key_point_size

        point_dst = deepcopy(point_src)

        # 转换成KeyPoint类型
        kp_src, point_src_index = self.__convert_point_to_key_point(point_src)
        kp_dst, point_dst_index = self.__convert_point_to_key_point(point_dst)

        return point_src, point_dst, point_src_index, point_dst_index, kp_src, kp_dst

    # 计算差异点
    def __calculate_difference_point(self, detect_range):
        # 计算待用SIFT算法检测的点位
        point_src, point_dst, point_src_index, point_dst_index, kp_src, kp_dst = self.__calculate_sift_point(
            detect_range)
        # 计算轮廓图的关键点的SIFT描述子
        keypoints_image1, descriptors_image1 = self.sift.compute(self.img1_contours, kp_src)
        keypoints_image2, descriptors_image2 = self.sift.compute(self.img2_contours, kp_dst)
        self.__output_img('contours_img1', self.img1_contours)
        self.__output_img('contours_img2', self.img2_contours)

        # 计算Kmeans聚类后主体图像的SIFT描述子
        ret, self.img2_kmeans_res_org = cv2.threshold(self.img2_kmeans_res_org, 70, 255, cv2.THRESH_BINARY)
        keypoints_image1_2, descriptors_image1_2 = self.sift.compute(self.img1_kmeans_res_org, kp_src)
        keypoints_image2_2, descriptors_image2_2 = self.sift.compute(self.img2_kmeans_res_org, kp_dst)
        self.__output_img('Kmeans_img1', self.img1_kmeans_res_org)
        self.__output_img('Kmeans_img2', self.img2_kmeans_res_org)

        # xor_kmeans = self.__xor_image_gray(self.img1_kmeans_res_org, self.img2_kmeans_res_org)
        # and_result = self.__and_image_gray(self.xor_contours, xor_kmeans)
        # self.__output_img("xor_kmeans", xor_kmeans)
        # self.__output_img("and_result", and_result)
        # 差异点
        diff_left = []
        diff_right = []
        same_point_src = []
        same_point_dst = []

        # 分析差异
        for i in range(len(kp_src)):
            now_threshold = self.threshold
            difference = 0
            for j in range(128):
                d = abs(descriptors_image1[i][j] - descriptors_image2[i][j])
                difference = difference + d * d
            difference = math.sqrt(difference)

            difference2 = 0
            for j in range(128):
                d = abs(descriptors_image1_2[i][j] - descriptors_image2_2[i][j])
                difference2 = difference2 + d * d
            difference2 = math.sqrt(difference2)

            difference = min(difference, difference2)

            # 右图关键点位置不超出范围
            src_point = point_src[point_src_index[i]]
            dst_point = point_dst[point_dst_index[i]]
            if (dst_point[0][1] >= 0) & (dst_point[0][0] >= 0):
                if difference <= now_threshold:
                    same_point_src.append(src_point)
                    same_point_dst.append(dst_point)
                else:
                    diff_left.append([src_point[0][0], src_point[0][1]])
                    diff_right.append([dst_point[0][0], dst_point[0][1]])
        self.__show_difference_point(same_point_src, same_point_dst, diff_left, diff_right)
        return diff_left, diff_right
        pass

    def __get_contrast_img(self):
        heightO = max(self.img1_org.shape[0], self.img2_org_warp.shape[0])
        widthO = self.img1_org.shape[1] + self.img2_org_warp.shape[1]
        contrast_img = np.zeros((heightO, widthO, 3), dtype=np.uint8)
        contrast_img[0:self.img1_org.shape[0], 0:self.img1_org.shape[1]] = self.img1
        contrast_img[0:self.img2_org_warp.shape[0], self.img1_org.shape[1]:] = self.img2[:]
        return contrast_img

    def __show_difference_point(self, same_point_src, same_point_dst, diff_left, diff_right, name="difference"):
        # 把差异点画出来
        # 生成空白对比图
        contrast_img = self.__get_contrast_img()

        # 相同点为绿色
        for i in range(len(same_point_src)):
            cv2.circle(contrast_img, (int(same_point_src[i][0][0]), int(same_point_src[i][0][1])), 1, (0, 255, 0), 2)
            cv2.circle(contrast_img, (int(same_point_dst[i][0][0] + self.w), int(same_point_dst[i][0][1])), 1,
                       (0, 255, 0), 2)

        # 不同点为红色
        for i in range(len(diff_left)):
            cv2.circle(contrast_img, (int(diff_left[i][0]), int(diff_left[i][1])), 1, (0, 0, 255), 2)
            cv2.circle(contrast_img, (int(diff_right[i][0] + self.w), int(diff_right[i][1])), 1, (0, 0, 255), 2)

        self.__output_img(name, contrast_img)

    def __generate_result_img(self, diff_left, diff_right):
        # output2
        output_cluster_circle = self.__get_contrast_img()
        # dbscan聚类，将聚类个数大于7个差一点的类找出来并画出
        if len(diff_left) == 0:
            return output_cluster_circle, []
        all_result = detectUtil.dbscan_circle(diff_left, self.detect_point_interval * 2.5, self.least_diff_point_num)
        # all_result = self.__hsv_diff_filter(all_result)

        for i in range(len(all_result)):
            cv2.circle(output_cluster_circle, (int(all_result[i][0]), int(all_result[i][1])),
                       int(np.sqrt(all_result[i][2])) * 7,
                       (255, 255, 255), 2)
            cv2.circle(output_cluster_circle, (int(all_result[i][0]) + self.w, int(all_result[i][1])),
                       int(np.sqrt(all_result[i][2])) * 7, (255, 255, 0), 2)

        # 输出结果
        self.__output_img('detect_result', output_cluster_circle)
        return output_cluster_circle, all_result

    def __sift_diff_comparator(self, points, image1, image2):
        key_points, key_points_index = self.__convert_point_to_key_point(points)
        keypoints_hsv_image1, descriptors_hsv_image1 = self.sift.compute(image1, key_points)
        keypoints_hsv_image2, descriptors_hsv_image2 = self.sift.compute(image2, key_points)

        is_point_diff = []
        diff_points = []
        same_points = []
        # 分析差异
        for i in range(len(points)):
            difference = 0
            for j in range(128):
                d = abs(descriptors_hsv_image1[i][j] - descriptors_hsv_image2[i][j])
                difference = difference + d * d
            difference = math.sqrt(difference)

            # 右图关键点位置不超出范围
            if difference <= self.threshold:
                # 无差异
                is_point_diff.append(False)
                same_points.append([points[i][0][0], points[i][0][1]])
            else:
                # 存在差异
                diff_points.append([points[i][0][0], points[i][0][1]])
                is_point_diff.append(True)
        return same_points, diff_points, is_point_diff

    # 根据Hsv图像的h通道计算sift差异
    def __hsv_diff_filter(self, circles):
        points = []
        points_index = 0
        circle_points_map = {}
        for k in range(len(circles)):
            left = max(circles[k][0] - circles[k][2], 0)
            right = min(circles[k][0] + circles[k][2], self.detect_range1.shape[1])
            top = max(circles[k][1] - circles[k][2], 0)
            down = min(circles[k][1] + circles[k][2], self.detect_range1.shape[0])
            i = left
            while left <= i < right:
                j = top
                while top <= j < down:
                    # 找处所有圆内的点
                    # print(i, j, circles[k][0], circles[k][1], circles[k][2],
                    #       math.sqrt(math.pow(j - circles[k][1], 2) + math.pow(i - circles[k][0], 2)))
                    if math.sqrt(math.pow(j - circles[k][1], 2) + math.pow(i - circles[k][0], 2)) \
                            <= circles[k][2] / 1.3 + 1 and (self.detect_range1[j][i] == (255, 255, 255)).all():
                        points.append([[i, j]])
                        if k not in circle_points_map:
                            circle_points_map[k] = []
                        circle_points_map[k].append(points_index)
                        points_index += 1
                    j += self.detect_point_interval
                i += self.detect_point_interval
        key_points, key_points_index = self.__convert_point_to_key_point(points)

        mask2_warp = cv2.warpPerspective(self.mask2, np.linalg.inv(self.M), (self.w, self.h))
        self.__output_img("mask2_warp", mask2_warp)
        same_points, diff_points, is_point_diff = self.__sift_diff_comparator(key_points, self.mask1, mask2_warp)
        keypoints_hsv_image1, descriptors_hsv_image1 = self.sift.compute(self.mask1, key_points)
        keypoints_hsv_image2, descriptors_hsv_image2 = self.sift.compute(mask2_warp, key_points)
        self.__show_difference_point(same_points, same_points, diff_points, diff_points, "hsv_difference")
        circles_difference_count = [0] * (len(circles))
        for k in circle_points_map:
            for i in circle_points_map[k]:
                if is_point_diff[i]:
                    circles_difference_count[k] += 1
        ret_circles = []
        for i in range(len(circles)):
            if circles_difference_count[i] >= self.least_diff_point_num:
                ret_circles.append(circles[i])
        return ret_circles

    def __xor_image_gray(self, image1, image2):
        ret = np.zeros((image1.shape[0], image1.shape[1]), np.uint8)
        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                if not (image1[i][j] == image2[i][j]).all():
                    ret[i][j] = 255
        return ret

    def __and_image_gray(self, image1, image2):
        ret = np.zeros((image1.shape[0], image1.shape[1]), np.uint8)
        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                if (image1[i][j] == image2[i][j]).all() and (image1[i][j] == 255).all():
                    ret[i][j] = 255
        return ret

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
