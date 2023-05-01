from detect.camera.cameraInterface import *
from detect.mvcamera.MVGigE import *


class MVCamera(CameraInterface):
    def __init__(self):
        super().__init__()
        self.__hCam = None
        self.__img = None

    def open(self):
        self.init_camera()
        r, self.__img = MVGetImgBuf(self.__hCam)

        if self.__img is None:
            MVCloseCam(self.__hCam)
            raise Exception('caemera image init error: ', r)

        if MVStartGrabWindow(self.__hCam) != MVST_SUCCESS:
            MVCloseCam(self.__hCam)
            raise Exception("MVStartGrabWindow error")
        pass

    def init_camera(self):
        r, self.__hCam = MVOpenCamByIndex(0)  # 根据相机的索引返回相机句柄

        if self.__hCam == 0:
            if r == MVST_ACCESS_DENIED:
                raise Exception('无法打开相机，可能正被别的软件控制!')
            else:
                raise Exception('无法打开相机!')
        return self.__hCam

    def get_image(self):
        res, id = MVGetSampleGrabBuf(self.__hCam, self.__img, 50)
        if res == MVST_SUCCESS:
            return self.__img
        else:
            return None

    def close(self):
        MVStopGrab(self.__hCam)  # 停止采集
        MVCloseCam(self.__hCam)
