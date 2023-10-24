import time
import serial
import cv2
import numpy as np
from .lidaGenerator import LidaFile
from detect.core.diffdetect import detectUtil
from .Point3DToLida import Point3DToLida
from .PointAddStrategy import SinglePointStrategy


class LaserProjector:
    def __init__(self):
        self.empty_lida_file = None
        self.min_send_interval = 1
        self.last_send_time = 0
        self.serial_connect = serial.Serial("COM3", 115200)
        self.projector_in_mtx = None
        self.camera_project_trans_mtx = None
        self.camera_mtx = np.array([[1.79697103e+03, 0.00000000e+00, 1.16728650e+03],
                                    [0.00000000e+00, 1.79632799e+03, 8.53307321e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        self.__init_empty_lida()
        print("laser initialized!")
        pass

    def __init_empty_lida(self):
        lida_file = LidaFile()
        lida_file.name = "empty"
        lida_file.company = "buz141"
        lida_file.new_frame()
        lida_file.add_point(0, 0, 0, 0, False)
        self.empty_lida_file = lida_file.to_bytes()
        pass

    def generate_point3d_lida(self, points, trans_mat, contour=None):
        converter = Point3DToLida()
        converter.projector_in_mtx = self.projector_in_mtx
        converter.camera_project_trans_mtx = self.camera_project_trans_mtx
        converter.camera_trans_mtx = trans_mat
        strategy = SinglePointStrategy()
        converter.end_blank_repeat_num = 1
        converter.tiny_contour_repeat_num = 5
        # strategy.point_size = 300
        converter.set_strategy(strategy)
        if points is not None:
            for point in points:
                converter.add_point(point)
            print(len(points))
        if contour is not None:
            converter.add_contour(contour, [255, 255, 255])
        converter.new_frame(0)
        return converter.to_bytes()
        pass

    def project_image(self, image):
        self.empty_projection()
        time.sleep(1)
        if image is not None:
            mask = detectUtil.calculate_gray_primary_area(image, gray_tolerance=35)
            img1_contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            max_contour = detectUtil.find_max_n_contours(img1_contours, 1)[0]
            image_range = cv2.drawContours(np.zeros((image.shape[0], image.shape[1])), [max_contour], -1,
                                           (255, 255, 255),
                                           0)
            cv2.imwrite("image_range.jpg", image_range)
            in_mtx_inv = np.linalg.inv(self.camera_mtx)
            camera_ext = np.array([[9.94094767e-01, 3.40490091e-02, 1.03035236e-01, -4.12571276e+02],
                                   [-3.54787638e-02, 9.99297477e-01, 1.20751364e-02, -2.26786862e+02],
                                   [-1.02551705e-01, -1.56593927e-02, 9.94604409e-01, 1.71140716e+03],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
            projector_inner = np.array([[1.38267219e+05, 0.00000000e+00, 4.03611555e+04],
                                        [0.00000000e+00, 1.61803734e+05, 2.05332974e+04],
                                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
            camera_projector_trans_mat = np.array([[9.99813559e-01, 9.29488249e-03, -1.69248756e-02, -2.50059599e+02],
                                                   [-9.43177830e-03, 9.99923304e-01, -8.02666016e-03, 9.55884566e+01],
                                                   [1.68489707e-02, 8.18479534e-03, 9.99824545e-01, 7.83251266e+02],
                                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
            self.projector_in_mtx = projector_inner
            self.camera_project_trans_mtx = camera_projector_trans_mat
            r = camera_ext[0:3, 0:3]
            r_inv = np.linalg.inv(r)
            tvec = camera_ext[:, 3].reshape(-1)[:3]
            max_contour.reshape(-1, 2)
            word_point_list = []
            for point in max_contour:
                point = np.append(point, 1)
                word_point = in_mtx_inv.dot(point)
                word_point = r_inv.dot(word_point)
                p = tvec.reshape(-1)
                d = r_inv.dot(p)
                s = d[2] / word_point[2]
                word_point = word_point * s - d
                word_point = np.append(word_point, 1.0)
                word_point_list.append(word_point)
                pass
            # file_bytes = self.generate_point3d_lida(word_point_list, camera_ext, None)
            file_bytes = self.generate_point3d_lida(None, camera_ext, word_point_list)
            file = open("target.ild", 'wb')
            file.write(file_bytes)
            self.__send_bytes(file_bytes)
        pass

    def empty_projection(self):
        self.__send_bytes(self.empty_lida_file)

    def __send_bytes(self, image_bytes):
        file_len = len(image_bytes)
        buffer_size = 2048
        interval = time.time() - self.last_send_time
        print("send")
        if self.serial_connect.isOpen() and interval > self.min_send_interval:
            self.serial_connect.write(bytes([0xAA, 0xBB]))
            self.serial_connect.write(file_len.to_bytes(4, "little"))
            write_len = 0
            while write_len < file_len:
                end = (write_len + buffer_size) if write_len + buffer_size < file_len else file_len
                self.serial_connect.write(image_bytes[write_len:end])
                # print(end)
                write_len += buffer_size
            self.serial_connect.flush()
            self.last_send_time = time.time()
