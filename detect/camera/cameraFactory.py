import string
from detect.mvcamera import MVCamera
from detect.camera.cameraInterface import CameraInterface
from typing import List


class CameraFactory:
    def __init__(self):
        self.camera_class_list: List[CameraInterface] = [MVCamera.MVCamera]
        pass

    def get_camera(self) -> (object, CameraInterface):
        init_flag = False
        ret_e = ""
        for camera_class in self.camera_class_list:
            try:
                camera = camera_class()
                camera.open()
                return 0, camera
            except Exception as e:
                ret_e += str(e)
                pass
        if not init_flag:
            return "摄像头初始化失败:\n" + ret_e, None
