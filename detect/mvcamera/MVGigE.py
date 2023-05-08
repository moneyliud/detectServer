from ctypes import *
from detect.mvcamera.GigECamera_Types import *
import numpy as np
import os
import ctypes

os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))
MVGigE = windll.LoadLibrary('MVGigE')


def MVInfo2Img(info):
    stFrameInfo = cast(info, POINTER(MV_IMAGE_INFO)).contents
    w = stFrameInfo.nSizeX
    h = stFrameInfo.nSizeY
    print(stFrameInfo.nImageSizeAcq,w,h)
    image = np.ctypeslib.as_array(stFrameInfo.pImageBuffer, shape=(h, w))
    return image, stFrameInfo.nBlockId


def MVGetImgBuf(hCam):
    r, w = MVGetWidth(hCam)
    if r != MVST_SUCCESS:
        return r, None

    r, h = MVGetHeight(hCam)
    if r != MVST_SUCCESS:
        return r, None

    r, pixelformat = MVGetPixelFormat(hCam)
    if r != MVST_SUCCESS:
        return r, None

    if pixelformat == PixelFormat_Mono8:
        img = np.zeros((h, w), dtype=np.uint8)
    elif pixelformat == PixelFormat_Mono16:
        img = np.zeros((h, w), dtype=np.uint16)
    elif pixelformat in [PixelFormat_BayerBG8, PixelFormat_BayerRG8, PixelFormat_BayerGB8, PixelFormat_BayerGR8]:
        img = np.zeros((h, w, 3), dtype=np.uint8)
    elif pixelformat in [PixelFormat_BayerBG16, PixelFormat_BayerRG16, PixelFormat_BayerGB16, PixelFormat_BayerGR16]:
        img = np.zeros((h, w, 3), dtype=np.uint16)
    else:
        # print('未知pixel format : {:#X}'.format(pixelformat))
        return MVST_ERROR, None

    return MVST_SUCCESS, img


# MVInitLib()
def MVInitLib():
    MVGigE.MVInitLib.restype = c_int
    res = MVGigE.MVInitLib()
    return res


# MVTerminateLib()
def MVTerminateLib():
    MVGigE.MVTerminateLib.restype = c_int
    res = MVGigE.MVTerminateLib()
    return res


# MVUpdateCameraList()
def MVUpdateCameraList():
    MVGigE.MVUpdateCameraList.restype = c_int
    res = MVGigE.MVUpdateCameraList()
    return res


# MVGetNumOfCameras(int* pNumCams)
def MVGetNumOfCameras():
    MVGigE.MVGetNumOfCameras.argtype = (c_void_p)
    MVGigE.MVGetNumOfCameras.restype = c_int
    pNumCams = c_int()
    res = MVGigE.MVGetNumOfCameras(byref(pNumCams))
    return res, pNumCams.value


# MVGetCameraInfo(unsigned char idx, MVCamInfo* pCamInfo)
def MVGetCameraInfo(idx):
    MVGigE.MVGetCameraInfo.argtype = (c_ubyte, c_void_p)
    MVGigE.MVGetCameraInfo.restype = c_int
    pCamInfo = MVCamInfo()
    res = MVGigE.MVGetCameraInfo(c_ubyte(idx), byref(pCamInfo))
    return res, pCamInfo


# MVOpenCamByIndex(unsigned char idx, HANDLE* hCam)
def MVOpenCamByIndex(idx):
    MVGigE.MVOpenCamByIndex.argtype = (c_ubyte, c_void_p)
    MVGigE.MVOpenCamByIndex.restype = c_int
    hCam = c_uint64()
    res = MVGigE.MVOpenCamByIndex(c_ubyte(idx), byref(hCam))
    return res, hCam.value


# MVOpenCamByUserDefinedName(char* name, HANDLE* hCam)
def MVOpenCamByUserDefinedName(name):
    MVGigE.MVOpenCamByUserDefinedName.argtype = (c_char_p, c_void_p)
    MVGigE.MVOpenCamByUserDefinedName.restype = c_int
    cname = (c_char * 16)()
    hCam = c_uint64()
    print('len: ', len(name))
    for i in range(len(name)):
        cname[i] = c_char(name[i])

    res = MVGigE.MVOpenCamByUserDefinedName(cname, byref(hCam))
    return res, hCam.value


# MVOpenCamByIP( char *ip,HANDLE *hCam )
def MVOpenCamByIP(ip):
    MVGigE.MVOpenCamByIP.argtype = (c_char_p, c_void_p)
    MVGigE.MVOpenCamByIP.restype = c_int
    # ip = c_char()
    hCam = c_uint64()
    res = MVGigE.MVOpenCamByIP(ip.encode('ascii'), byref(hCam))
    return res, hCam.value


# MVCloseCam(HANDLE hCam)
def MVCloseCam(hCam):
    MVGigE.MVCloseCam.argtype = (c_uint64)
    MVGigE.MVCloseCam.restype = c_int
    res = MVGigE.MVCloseCam(c_uint64(hCam))
    return res


# MVGetWidth(HANDLE hCam, int* pWidth)
def MVGetWidth(hCam):
    MVGigE.MVGetWidth.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetWidth.restype = c_int
    pWidth = c_int()
    res = MVGigE.MVGetWidth(c_uint64(hCam), byref(pWidth))
    return res, pWidth.value


# MVGetWidthRange(HANDLE hCam, int* pWidthMin, int* pWidthMax)
def MVGetWidthRange(hCam):
    MVGigE.MVGetWidthRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetWidthRange.restype = c_int
    pWidthMin = c_int()
    pWidthMax = c_int()
    res = MVGigE.MVGetWidthRange(c_uint64(hCam), byref(pWidthMin), byref(pWidthMax))
    return res, pWidthMin.value, pWidthMax.value


# MVGetWidthInc(HANDLE hCam, int* pWidthInc)
def MVGetWidthInc(hCam):
    MVGigE.MVGetWidthInc.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetWidthInc.restype = c_int
    pWidthInc = c_int()
    res = MVGigE.MVGetWidthInc(c_uint64(hCam), byref(pWidthInc))
    return res, pWidthInc.value


# MVSetWidth(HANDLE hCam, int nWidth)
def MVSetWidth(hCam, nWidth):
    MVGigE.MVSetWidth.argtype = (c_uint64, c_int)
    MVGigE.MVSetWidth.restype = c_int
    res = MVGigE.MVSetWidth(c_uint64(hCam), c_int(nWidth))
    return res


# MVGetHeight(HANDLE hCam, int* pHeight)
def MVGetHeight(hCam):
    MVGigE.MVGetHeight.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetHeight.restype = c_int
    pHeight = c_int()
    res = MVGigE.MVGetHeight(c_uint64(hCam), byref(pHeight))
    return res, pHeight.value


# MVGetHeightRange(HANDLE hCam, int* pHeightMin, int* pHeightMax)
def MVGetHeightRange(hCam):
    MVGigE.MVGetHeightRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetHeightRange.restype = c_int
    pHeightMin = c_int()
    pHeightMax = c_int()
    res = MVGigE.MVGetHeightRange(c_uint64(hCam), byref(pHeightMin), byref(pHeightMax))
    return res, pHeightMin.value, pHeightMax.value


# MVSetHeight(HANDLE hCam, int nHeight)
def MVSetHeight(hCam, nHeight):
    MVGigE.MVSetHeight.argtype = (c_uint64, c_int)
    MVGigE.MVSetHeight.restype = c_int
    res = MVGigE.MVSetHeight(c_uint64(hCam), c_int(nHeight))
    return res


# MVGetOffsetX(HANDLE hCam, int* pOffsetX)
def MVGetOffsetX(hCam):
    MVGigE.MVGetOffsetX.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetOffsetX.restype = c_int
    pOffsetX = c_int()
    res = MVGigE.MVGetOffsetX(c_uint64(hCam), byref(pOffsetX))
    return res, pOffsetX.value


# MVGetOffsetXRange(HANDLE hCam, int* pOffsetXMin, int* pOffsetXMax)
def MVGetOffsetXRange(hCam):
    MVGigE.MVGetOffsetXRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetOffsetXRange.restype = c_int
    pOffsetXMin = c_int()
    pOffsetXMax = c_int()
    res = MVGigE.MVGetOffsetXRange(c_uint64(hCam), byref(pOffsetXMin), byref(pOffsetXMax))
    return res, pOffsetXMin.value, pOffsetXMax.value


# MVSetOffsetX(HANDLE hCam, int nOffsetX)
def MVSetOffsetX(hCam, nOffsetX):
    MVGigE.MVSetOffsetX.argtype = (c_uint64, c_int)
    MVGigE.MVSetOffsetX.restype = c_int
    res = MVGigE.MVSetOffsetX(c_uint64(hCam), c_int(nOffsetX))
    return res


# MVGetOffsetY(HANDLE hCam, int* pOffsetY)
def MVGetOffsetY(hCam):
    MVGigE.MVGetOffsetY.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetOffsetY.restype = c_int
    pOffsetY = c_int()
    res = MVGigE.MVGetOffsetY(c_uint64(hCam), byref(pOffsetY))
    return res, pOffsetY.value


# MVGetOffsetYRange(HANDLE hCam, int* pOffsetYMin, int* pOffsetYMax)
def MVGetOffsetYRange(hCam):
    MVGigE.MVGetOffsetYRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetOffsetYRange.restype = c_int
    pOffsetYMin = c_int()
    pOffsetYMax = c_int()
    res = MVGigE.MVGetOffsetYRange(c_uint64(hCam), byref(pOffsetYMin), byref(pOffsetYMax))
    return res, pOffsetYMin.value, pOffsetYMax.value


# MVSetOffsetY(HANDLE hCam, int nOffsetY)
def MVSetOffsetY(hCam, nOffsetY):
    MVGigE.MVSetOffsetY.argtype = (c_uint64, c_int)
    MVGigE.MVSetOffsetY.restype = c_int
    res = MVGigE.MVSetOffsetY(c_uint64(hCam), c_int(nOffsetY))
    return res


# MVGetPixelFormat(HANDLE hCam, MV_PixelFormatEnums* pPixelFormat)
def MVGetPixelFormat(hCam):
    MVGigE.MVGetPixelFormat.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetPixelFormat.restype = c_int
    pPixelFormat = c_uint()
    res = MVGigE.MVGetPixelFormat(c_uint64(hCam), byref(pPixelFormat))
    return res, pPixelFormat.value


# MVGetSensorTaps(HANDLE hCam, SensorTapsEnums* pSensorTaps)
def MVGetSensorTaps(hCam):
    MVGigE.MVGetSensorTaps.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetSensorTaps.restype = c_int
    pSensorTaps = c_uint()
    res = MVGigE.MVGetSensorTaps(c_uint64(hCam), byref(pSensorTaps))
    return res, pSensorTaps.value


# MVGetGain(HANDLE hCam, double* pGain)
def MVGetGain(hCam):
    MVGigE.MVGetGain.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetGain.restype = c_int
    pGain = c_double()
    res = MVGigE.MVGetGain(c_uint64(hCam), byref(pGain))
    return res, pGain.value


# MVGetGainRange(HANDLE hCam, double* pGainMin, double* pGainMax)
def MVGetGainRange(hCam):
    MVGigE.MVGetGainRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetGainRange.restype = c_int
    pGainMin = c_double()
    pGainMax = c_double()
    res = MVGigE.MVGetGainRange(c_uint64(hCam), byref(pGainMin), byref(pGainMax))
    return res, pGainMin.value, pGainMax.value


# MVSetGain(HANDLE hCam, double fGain)
def MVSetGain(hCam, fGain):
    MVGigE.MVSetGain.argtype = (c_uint64, c_double)
    MVGigE.MVSetGain.restype = c_int
    res = MVGigE.MVSetGain(c_uint64(hCam), c_double(fGain))
    return res


# MVSetGainTaps(HANDLE hCam, double fGain, int nTap)
def MVSetGainTaps(hCam, fGain, nTap):
    MVGigE.MVSetGainTaps.argtype = (c_uint64, c_double, c_int)
    MVGigE.MVSetGainTaps.restype = c_int
    res = MVGigE.MVSetGainTaps(c_uint64(hCam), c_double(fGain), c_int(nTap))
    return res


# MVGetGainTaps(HANDLE hCam, double* pGain, int nTap)
def MVGetGainTaps(hCam, nTap):
    MVGigE.MVGetGainTaps.argtype = (c_uint64, c_void_p, c_int)
    MVGigE.MVGetGainTaps.restype = c_int
    pGain = c_double()
    res = MVGigE.MVGetGainTaps(c_uint64(hCam), byref(pGain), c_int(nTap))
    return res, pGain.value


# MVGetGainRangeTaps(HANDLE hCam, double* pGainMin, double* pGainMax, int nTap)
def MVGetGainRangeTaps(hCam, nTap):
    MVGigE.MVGetGainRangeTaps.argtype = (c_uint64, c_void_p, c_void_p, c_int)
    MVGigE.MVGetGainRangeTaps.restype = c_int
    pGainMin = c_double()
    pGainMax = c_double()
    res = MVGigE.MVGetGainRangeTaps(c_uint64(hCam), byref(pGainMin), byref(pGainMax), c_int(nTap))
    return res, pGainMin.value, pGainMax.value


# MVGetWhiteBalance(HANDLE hCam, double* pRed, double* pGreen, double* pBlue)
def MVGetWhiteBalance(hCam):
    MVGigE.MVGetWhiteBalance.argtype = (c_uint64, c_void_p, c_void_p, c_void_p)
    MVGigE.MVGetWhiteBalance.restype = c_int
    pRed = c_double()
    pGreen = c_double()
    pBlue = c_double()
    res = MVGigE.MVGetWhiteBalance(c_uint64(hCam), byref(pRed), byref(pGreen), byref(pBlue))
    return res, pRed.value, pGreen.value, pBlue.value


# MVGetWhiteBalanceRange(HANDLE hCam, double* pMin, double* pMax)
def MVGetWhiteBalanceRange(hCam):
    MVGigE.MVGetWhiteBalanceRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetWhiteBalanceRange.restype = c_int
    pMin = c_double()
    pMax = c_double()
    res = MVGigE.MVGetWhiteBalanceRange(c_uint64(hCam), byref(pMin), byref(pMax))
    return res, pMin.value, pMax.value


# MVSetWhiteBalance(HANDLE hCam, double fRed, double fGreen, double fBlue)
def MVSetWhiteBalance(hCam, fRed, fGreen, fBlue):
    MVGigE.MVSetWhiteBalance.argtype = (c_uint64, c_double, c_double, c_double)
    MVGigE.MVSetWhiteBalance.restype = c_int
    res = MVGigE.MVSetWhiteBalance(c_uint64(hCam), c_double(fRed), c_double(fGreen), c_double(fBlue))
    return res


# MVGetGainBalance(HANDLE hCam, int* pBalance)
def MVGetGainBalance(hCam):
    MVGigE.MVGetGainBalance.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetGainBalance.restype = c_int
    pBalance = c_int()
    res = MVGigE.MVGetGainBalance(c_uint64(hCam), byref(pBalance))
    return res, pBalance.value


# MVSetGainBalance(HANDLE hCam, int nBalance)
def MVSetGainBalance(hCam, nBalance):
    MVGigE.MVSetGainBalance.argtype = (c_uint64, c_int)
    MVGigE.MVSetGainBalance.restype = c_int
    res = MVGigE.MVSetGainBalance(c_uint64(hCam), c_int(nBalance))
    return res


# MVGetExposureTime(HANDLE hCam, double* pExposuretime)
def MVGetExposureTime(hCam):
    MVGigE.MVGetExposureTime.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetExposureTime.restype = c_int
    pExposuretime = c_double()
    res = MVGigE.MVGetExposureTime(c_uint64(hCam), byref(pExposuretime))
    return res, pExposuretime.value


# MVGetExposureTimeRange(HANDLE hCam, double* pExpMin, double* pExpMax)
def MVGetExposureTimeRange(hCam):
    MVGigE.MVGetExposureTimeRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetExposureTimeRange.restype = c_int
    pExpMin = c_double()
    pExpMax = c_double()
    res = MVGigE.MVGetExposureTimeRange(c_uint64(hCam), byref(pExpMin), byref(pExpMax))
    return res, pExpMin.value, pExpMax.value


# MVSetExposureTime(HANDLE hCam,double nExp_us)
def MVSetExposureTime(hCam, nExp_us):
    MVGigE.MVSetExposureTime.argtype = (c_uint64, c_double)
    MVGigE.MVSetExposureTime.restype = c_int
    res = MVGigE.MVSetExposureTime(c_uint64(hCam), c_double(nExp_us))
    return res


# MVGetFrameRateRange(HANDLE hCam, double* pFpsMin, double* pFpsMax)
def MVGetFrameRateRange(hCam):
    MVGigE.MVGetFrameRateRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetFrameRateRange.restype = c_int
    pFpsMin = c_double()
    pFpsMax = c_double()
    res = MVGigE.MVGetFrameRateRange(c_uint64(hCam), byref(pFpsMin), byref(pFpsMax))
    return res, pFpsMin.value, pFpsMax.value


# MVGetFrameRate(HANDLE hCam, double* fFPS)
def MVGetFrameRate(hCam):
    MVGigE.MVGetFrameRate.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetFrameRate.restype = c_int
    fFPS = c_double()
    res = MVGigE.MVGetFrameRate(c_uint64(hCam), byref(fFPS))
    return res, fFPS.value


# MVSetFrameRate(HANDLE hCam, double fps)
def MVSetFrameRate(hCam, fps):
    MVGigE.MVSetFrameRate.argtype = (c_uint64, c_double)
    MVGigE.MVSetFrameRate.restype = c_int
    res = MVGigE.MVSetFrameRate(c_uint64(hCam), c_double(fps))
    return res


# MVStartGrab(HANDLE hCam, MVStreamCB StreamCB, long nUserVal)
def MVStartGrab(hCam, StreamCB, nUserVal):
    MVGigE.MVStartGrab.argtype = (c_uint64, MVStreamCB, c_ulong)
    MVGigE.MVStartGrab.restype = c_int
    res = MVGigE.MVStartGrab(c_uint64(hCam), StreamCB, c_ulong(nUserVal))
    return res


# MVStopGrab(HANDLE hCam)
def MVStopGrab(hCam):
    MVGigE.MVStopGrab.argtype = (c_uint64)
    MVGigE.MVStopGrab.restype = c_int
    res = MVGigE.MVStopGrab(c_uint64(hCam))
    return res


# MVGetTriggerMode(HANDLE hCam, TriggerModeEnums* pMode)
def MVGetTriggerMode(hCam):
    MVGigE.MVGetTriggerMode.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetTriggerMode.restype = c_int
    pMode = c_uint()
    res = MVGigE.MVGetTriggerMode(c_uint64(hCam), byref(pMode))
    return res, pMode.value


# MVSetTriggerMode(HANDLE hCam, TriggerModeEnums mode)
def MVSetTriggerMode(hCam, mode):
    MVGigE.MVSetTriggerMode.argtype = (c_uint64, c_uint)
    MVGigE.MVSetTriggerMode.restype = c_int
    res = MVGigE.MVSetTriggerMode(c_uint64(hCam), mode)
    return res


# MVGetTriggerSource(HANDLE hCam, TriggerSourceEnums* pSource)
def MVGetTriggerSource(hCam):
    MVGigE.MVGetTriggerSource.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetTriggerSource.restype = c_int
    pSource = c_uint()
    res = MVGigE.MVGetTriggerSource(c_uint64(hCam), byref(pSource))
    return res, pSource.value


# MVSetTriggerSource(HANDLE hCam, TriggerSourceEnums source)
def MVSetTriggerSource(hCam, source):
    MVGigE.MVSetTriggerSource.argtype = (c_uint64, c_uint)
    MVGigE.MVSetTriggerSource.restype = c_int
    res = MVGigE.MVSetTriggerSource(c_uint64(hCam), source)
    return res


# MVGetTriggerActivation(HANDLE hCam, TriggerActivationEnums* pAct)
def MVGetTriggerActivation(hCam):
    MVGigE.MVGetTriggerActivation.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetTriggerActivation.restype = c_int
    pAct = c_uint()
    res = MVGigE.MVGetTriggerActivation(c_uint64(hCam), byref(pAct))
    return res, pAct.value


# MVSetTriggerActivation(HANDLE hCam, TriggerActivationEnums act)
def MVSetTriggerActivation(hCam, act):
    MVGigE.MVSetTriggerActivation.argtype = (c_uint64, c_uint)
    MVGigE.MVSetTriggerActivation.restype = c_int
    res = MVGigE.MVSetTriggerActivation(c_uint64(hCam), act)
    return res


# MVGetTriggerDelay(HANDLE hCam, uint32_t* pDelay_us)
def MVGetTriggerDelay(hCam):
    MVGigE.MVGetTriggerDelay.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetTriggerDelay.restype = c_int
    pDelay_us = c_uint()
    res = MVGigE.MVGetTriggerDelay(c_uint64(hCam), byref(pDelay_us))
    return res, pDelay_us.value


# MVGetTriggerDelayRange(HANDLE hCam, uint32_t* pMin, uint32_t* pMax)
def MVGetTriggerDelayRange(hCam):
    MVGigE.MVGetTriggerDelayRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetTriggerDelayRange.restype = c_int
    pMin = c_uint()
    pMax = c_uint()
    res = MVGigE.MVGetTriggerDelayRange(c_uint64(hCam), byref(pMin), byref(pMax))
    return res, pMin.value, pMax.value


# MVSetTriggerDelay(HANDLE hCam, uint32_t nDelay_us)
def MVSetTriggerDelay(hCam, nDelay_us):
    MVGigE.MVSetTriggerDelay.argtype = (c_uint64, c_uint)
    MVGigE.MVSetTriggerDelay.restype = c_int
    res = MVGigE.MVSetTriggerDelay(c_uint64(hCam), c_uint(nDelay_us))
    return res


# MVTriggerSoftware(HANDLE hCam)
def MVTriggerSoftware(hCam):
    MVGigE.MVTriggerSoftware.argtype = (c_uint64)
    MVGigE.MVTriggerSoftware.restype = c_int
    res = MVGigE.MVTriggerSoftware(c_uint64(hCam))
    return res


# MVGetStrobeSource(HANDLE hCam, LineSourceEnums* pSource)
def MVGetStrobeSource(hCam):
    MVGigE.MVGetStrobeSource.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetStrobeSource.restype = c_int
    pSource = c_uint()
    res = MVGigE.MVGetStrobeSource(c_uint64(hCam), byref(pSource))
    return res, pSource.value


# MVSetStrobeSource(HANDLE hCam, LineSourceEnums source)
def MVSetStrobeSource(hCam, source):
    MVGigE.MVSetStrobeSource.argtype = (c_uint64, c_uint)
    MVGigE.MVSetStrobeSource.restype = c_int
    res = MVGigE.MVSetStrobeSource(c_uint64(hCam), source)
    return res


# MVGetStrobeInvert(HANDLE hCam, bool* pInvert)
def MVGetStrobeInvert(hCam):
    MVGigE.MVGetStrobeInvert.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetStrobeInvert.restype = c_int
    pInvert = c_bool()
    res = MVGigE.MVGetStrobeInvert(c_uint64(hCam), byref(pInvert))
    return res, pInvert.value


# MVSetStrobeInvert(HANDLE hCam, bool bInvert)
def MVSetStrobeInvert(hCam, bInvert):
    MVGigE.MVSetStrobeInvert.argtype = (c_uint64, c_bool)
    MVGigE.MVSetStrobeInvert.restype = c_int
    res = MVGigE.MVSetStrobeInvert(c_uint64(hCam), c_bool(bInvert))
    return res


# MVGetUserOutputValue0(HANDLE hCam, bool* pSet)
def MVGetUserOutputValue0(hCam):
    MVGigE.MVGetUserOutputValue0.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetUserOutputValue0.restype = c_int
    pSet = c_bool()
    res = MVGigE.MVGetUserOutputValue0(c_uint64(hCam), byref(pSet))
    return res, pSet.value


# MVSetUserOutputValue0(HANDLE hCam, bool bSet)
def MVSetUserOutputValue0(hCam, bSet):
    MVGigE.MVSetUserOutputValue0.argtype = (c_uint64, c_bool)
    MVGigE.MVSetUserOutputValue0.restype = c_int
    res = MVGigE.MVSetUserOutputValue0(c_uint64(hCam), c_bool(bSet))
    return res


# MVSetHeartbeatTimeout(HANDLE hCam, unsigned long nTimeOut);//unit m
def MVSetHeartbeatTimeout(hCam, nTimeOut):
    MVGigE.MVSetHeartbeatTimeout.argtype = (c_uint64, c_ulong)
    MVGigE.MVSetHeartbeatTimeout.restype = c_int
    res = MVGigE.MVSetHeartbeatTimeout(c_uint64(hCam), c_ulong(nTimeOut))
    return res


# MVGetPacketSize(HANDLE hCam, unsigned int* pPacketSize)
def MVGetPacketSize(hCam):
    MVGigE.MVGetPacketSize.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetPacketSize.restype = c_int
    pPacketSize = c_uint()
    res = MVGigE.MVGetPacketSize(c_uint64(hCam), byref(pPacketSize))
    return res, pPacketSize.value


# MVGetPacketSizeRange(HANDLE hCam, unsigned int* pMin, unsigned int* pMax)
def MVGetPacketSizeRange(hCam):
    MVGigE.MVGetPacketSizeRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetPacketSizeRange.restype = c_int
    pMin = c_uint()
    pMax = c_uint()
    res = MVGigE.MVGetPacketSizeRange(c_uint64(hCam), byref(pMin), byref(pMax))
    return res, pMin.value, pMax.value


# MVSetPacketSize(HANDLE hCam, unsigned int nPacketSize)
def MVSetPacketSize(hCam, nPacketSize):
    MVGigE.MVSetPacketSize.argtype = (c_uint64, c_uint)
    MVGigE.MVSetPacketSize.restype = c_int
    res = MVGigE.MVSetPacketSize(c_uint64(hCam), c_uint(nPacketSize))
    return res


# MVGetPacketDelay(HANDLE hCam, unsigned int* pDelay_us)
def MVGetPacketDelay(hCam):
    MVGigE.MVGetPacketDelay.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetPacketDelay.restype = c_int
    pDelay_us = c_uint()
    res = MVGigE.MVGetPacketDelay(c_uint64(hCam), byref(pDelay_us))
    return res, pDelay_us.value


# MVGetPacketDelayRange(HANDLE hCam, unsigned int* pMin, unsigned int* pMax)
def MVGetPacketDelayRange(hCam):
    MVGigE.MVGetPacketDelayRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetPacketDelayRange.restype = c_int
    pMin = c_uint()
    pMax = c_uint()
    res = MVGigE.MVGetPacketDelayRange(c_uint64(hCam), byref(pMin), byref(pMax))
    return res, pMin.value, pMax.value


# MVSetPacketDelay(HANDLE hCam, unsigned int nDelay_us)
def MVSetPacketDelay(hCam, nDelay_us):
    MVGigE.MVSetPacketDelay.argtype = (c_uint64, c_uint)
    MVGigE.MVSetPacketDelay.restype = c_int
    res = MVGigE.MVSetPacketDelay(c_uint64(hCam), c_uint(nDelay_us))
    return res


# MVGetTimerDelay(HANDLE hCam, uint32_t* pDelay)
def MVGetTimerDelay(hCam):
    MVGigE.MVGetTimerDelay.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetTimerDelay.restype = c_int
    pDelay = c_uint()
    res = MVGigE.MVGetTimerDelay(c_uint64(hCam), byref(pDelay))
    return res, pDelay.value


# MVGetTimerDelayRange(HANDLE hCam, uint32_t* pMin, uint32_t* pMax)
def MVGetTimerDelayRange(hCam):
    MVGigE.MVGetTimerDelayRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetTimerDelayRange.restype = c_int
    pMin = c_uint()
    pMax = c_uint()
    res = MVGigE.MVGetTimerDelayRange(c_uint64(hCam), byref(pMin), byref(pMax))
    return res, pMin.value, pMax.value


# MVSetTimerDelay(HANDLE hCam, uint32_t nDelay)
def MVSetTimerDelay(hCam, nDelay):
    MVGigE.MVSetTimerDelay.argtype = (c_uint64, c_uint)
    MVGigE.MVSetTimerDelay.restype = c_int
    res = MVGigE.MVSetTimerDelay(c_uint64(hCam), c_uint(nDelay))
    return res


# MVGetTimerDuration(HANDLE hCam, uint32_t* pDuration)
def MVGetTimerDuration(hCam):
    MVGigE.MVGetTimerDuration.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetTimerDuration.restype = c_int
    pDuration = c_uint()
    res = MVGigE.MVGetTimerDuration(c_uint64(hCam), byref(pDuration))
    return res, pDuration.value


# MVGetTimerDurationRange(HANDLE hCam, uint32_t* pMin, uint32_t* pMax)
def MVGetTimerDurationRange(hCam):
    MVGigE.MVGetTimerDurationRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetTimerDurationRange.restype = c_int
    pMin = c_uint()
    pMax = c_uint()
    res = MVGigE.MVGetTimerDurationRange(c_uint64(hCam), byref(pMin), byref(pMax))
    return res, pMin.value, pMax.value


# MVSetTimerDuration(HANDLE hCam, uint32_t nDuration)
def MVSetTimerDuration(hCam, nDuration):
    MVGigE.MVSetTimerDuration.argtype = (c_uint64, c_uint)
    MVGigE.MVSetTimerDuration.restype = c_int
    res = MVGigE.MVSetTimerDuration(c_uint64(hCam), c_uint(nDuration))
    return res


# MVBayerToBGR(HANDLE hCam, void *psrc,void *pdst,unsigned int dststep,unsigned int width,unsigned int height,MV_PixelFormatEnums pixelformat,bool bMultiCores=FALSE)
def MVBayerToBGR(hCam, psrc, dststep, width, height, pixelformat, bMultiCores=FALSE):
    MVGigE.MVBayerToBGR.argtype = (c_uint64, c_void_p, c_void_p, c_uint, c_uint, c_uint, c_uint, c_bool)
    MVGigE.MVBayerToBGR.restype = c_int
    pdst = (c_ubyte * dststep * height)()
    res = MVGigE.MVBayerToBGR(c_uint64(hCam), psrc, byref(pdst), c_uint(dststep), c_uint(width), c_uint(height),
                              pixelformat, c_bool(bMultiCores))
    return res, pdst


# MVBayerToBGR16(HANDLE hCam, void *psrc,void *pdst,unsigned int dststep,unsigned int width,unsigned int height,MV_PixelFormatEnums pixelformat )
def MVBayerToBGR16(hCam, dststep, width, height, pixelformat):
    MVGigE.MVBayerToBGR16.argtype = (c_uint64, c_void_p, c_void_p, c_uint, c_uint, c_uint, c_uint)
    MVGigE.MVBayerToBGR16.restype = c_int
    psrc = c_uint()
    pdst = c_uint()
    res = MVGigE.MVBayerToBGR16(c_uint64(hCam), byref(psrc), byref(pdst), c_uint(dststep), c_uint(width),
                                c_uint(height), pixelformat)
    return res, psrc.value, pdst.value


# MVBayerToRGB(HANDLE hCam, void *psrc,void *pdst,unsigned int dststep,unsigned int width,unsigned int height,MV_PixelFormatEnums pixelformat,bool bMultiCores=FALSE)
def MVBayerToRGB(hCam, psrc, dststep, width, height, pixelformat, bMultiCores=FALSE):
    MVGigE.MVBayerToRGB.argtype = (c_uint64, c_void_p, c_void_p, c_uint, c_uint, c_uint, c_uint, c_bool)
    MVGigE.MVBayerToRGB.restype = c_int
    pdst = (c_ubyte * dststep * height)()
    res = MVGigE.MVBayerToRGB(c_uint64(hCam), psrc, byref(pdst), c_uint(dststep), c_uint(width), c_uint(height),
                              pixelformat, c_bool(bMultiCores))
    return res, pdst


# MVBayerToRGB16(HANDLE hCam, void *psrc,void *pdst,unsigned int dststep,unsigned int width,unsigned int height,MV_PixelFormatEnums pixelformat )
def MVBayerToRGB16(hCam, dststep, width, height, pixelformat):
    MVGigE.MVBayerToRGB16.argtype = (c_uint64, c_void_p, c_void_p, c_uint, c_uint, c_uint, c_uint)
    MVGigE.MVBayerToRGB16.restype = c_int
    psrc = c_uint()
    pdst = c_uint()
    res = MVGigE.MVBayerToRGB16(c_uint64(hCam), byref(psrc), byref(pdst), c_uint(dststep), c_uint(width),
                                c_uint(height), pixelformat)
    return res, psrc.value, pdst.value


# MVImageBayerToBGR(HANDLE hCam, MV_IMAGE_INFO* pInfo, MVImage* pImage)
def MVImageBayerToBGR(hCam, pInfo):
    MVGigE.MVImageBayerToBGR.argtype = (c_uint64, POINTER(MV_IMAGE_INFO), c_void_p)
    MVGigE.MVImageBayerToBGR.restype = c_int
    pImage = MVImage()
    res = MVGigE.MVImageBayerToBGR(c_uint64(hCam), pInfo, byref(pImage))
    return res, pImage


# MVInfo2Image(HANDLE hCam, MV_IMAGE_INFO* pInfo, MVImage* pImage)
def MVInfo2Image(hCam):
    MVGigE.MVInfo2Image.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVInfo2Image.restype = c_int
    pInfo = c_uint()
    pImage = c_uint()
    res = MVGigE.MVInfo2Image(c_uint64(hCam), byref(pInfo), byref(pImage))
    return res, pInfo.value, pImage.value


# MVImageBayerToBGREx( HANDLE hCam,MV_IMAGE_INFO *pInfo,MVImage *pImage,double fGamma,bool bColorCorrect,int nContrast )
def MVImageBayerToBGREx(hCam, fGamma, bColorCorrect, nContrast):
    MVGigE.MVImageBayerToBGREx.argtype = (c_uint64, c_void_p, c_void_p, c_double, c_bool, c_int)
    MVGigE.MVImageBayerToBGREx.restype = c_int
    pInfo = c_uint()
    pImage = c_uint()
    res = MVGigE.MVImageBayerToBGREx(c_uint64(hCam), byref(pInfo), byref(pImage), c_double(fGamma),
                                     c_bool(bColorCorrect), c_int(nContrast))
    return res, pInfo.value, pImage.value


# MVZoomImageBGR(HANDLE hCam, unsigned char* pSrc, int srcWidth, int srcHeight, unsigned char* pDst, double fFactorX, double fFactorY)
def MVZoomImageBGR(hCam, pSrc, srcWidth, srcHeight, fFactorX, fFactorY):
    MVGigE.MVZoomImageBGR.argtype = (c_uint64, c_void_p, c_int, c_int, c_void_p, c_double, c_double)
    MVGigE.MVZoomImageBGR.restype = c_int
    zoomX = int(srcWidth * 3 * fFactorX)
    zoomY = int(srcHeight * fFactorY)
    pDst = (c_ubyte * zoomX * zoomY)()
    res = MVGigE.MVZoomImageBGR(c_uint64(hCam), pSrc, c_int(srcWidth), c_int(srcHeight), byref(pDst),
                                c_double(fFactorX), c_double(fFactorY))
    return res, pDst


# MVGetStreamStatistic(HANDLE hCam, MVStreamStatistic* pStatistic)
def MVGetStreamStatistic(hCam):
    MVGigE.MVGetStreamStatistic.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetStreamStatistic.restype = c_int
    pStatistic = MVStreamStatistic()
    res = MVGigE.MVGetStreamStatistic(c_uint64(hCam), byref(pStatistic))
    return res, pStatistic


# MVSetDebugParam(HANDLE hCam, int i, unsigned int v)
def MVSetDebugParam(hCam, i, v):
    MVGigE.MVSetDebugParam.argtype = (c_uint64, c_int, c_uint)
    MVGigE.MVSetDebugParam.restype = c_int
    res = MVGigE.MVSetDebugParam(c_uint64(hCam), c_int(i), c_uint(v))
    return res


# MVGetDebugParam(HANDLE hCam, int i, unsigned int* pV)
def MVGetDebugParam(hCam, i):
    MVGigE.MVGetDebugParam.argtype = (c_uint64, c_int, c_void_p)
    MVGigE.MVGetDebugParam.restype = c_int
    pV = c_uint()
    res = MVGigE.MVGetDebugParam(c_uint64(hCam), c_int(i), byref(pV))
    return res, pV.value


# MVWriteMemory( HANDLE hCam, unsigned long aAddr, unsigned int aSize, void * aIn )
def MVWriteMemory(hCam, aAddr, aSize):
    MVGigE.MVWriteMemory.argtype = (c_uint64, c_ulong, c_uint, c_void_p)
    MVGigE.MVWriteMemory.restype = c_int
    aIn = c_uint()
    res = MVGigE.MVWriteMemory(c_uint64(hCam), c_ulong(aAddr), c_uint(aSize), byref(aIn))
    return res, aIn.value


# MVReadMemory( HANDLE hCam, unsigned long aAddr, unsigned int aSize, void * aOut )
def MVReadMemory(hCam, aAddr, aSize):
    MVGigE.MVReadMemory.argtype = (c_uint64, c_ulong, c_uint, c_void_p)
    MVGigE.MVReadMemory.restype = c_int
    aOut = c_uint()
    res = MVGigE.MVReadMemory(c_uint64(hCam), c_ulong(aAddr), c_uint(aSize), byref(aOut))
    return res, aOut.value


# MVUpgradeElf(HANDLE hCam,char *fname)
def MVUpgradeElf(hCam):
    MVGigE.MVUpgradeElf.argtype = (c_uint64, c_char_p)
    MVGigE.MVUpgradeElf.restype = c_int
    fname = c_char()
    res = MVGigE.MVUpgradeElf(c_uint64(hCam), byref(fname))
    return res, fname.value


# MVUpgradeFir(HANDLE hCam,char *fname)
def MVUpgradeFir(hCam):
    MVGigE.MVUpgradeFir.argtype = (c_uint64, c_char_p)
    MVGigE.MVUpgradeFir.restype = c_int
    fname = c_char()
    res = MVGigE.MVUpgradeFir(c_uint64(hCam), byref(fname))
    return res, fname.value


# MVGetCustomError(HANDLE hCam)
def MVGetCustomError(hCam):
    MVGigE.MVGetCustomError.argtype = (c_uint64)
    MVGigE.MVGetCustomError.restype = c_int
    res = MVGigE.MVGetCustomError(c_uint64(hCam))
    return res


# MVLoadUserSet(HANDLE hCam, UserSetSelectorEnums userset)
def MVLoadUserSet(hCam, userset):
    MVGigE.MVLoadUserSet.argtype = (c_uint64, c_uint)
    MVGigE.MVLoadUserSet.restype = c_int
    res = MVGigE.MVLoadUserSet(c_uint64(hCam), userset)
    return res


# MVSaveUserSet(HANDLE hCam, UserSetSelectorEnums userset)
def MVSaveUserSet(hCam, userset):
    MVGigE.MVSaveUserSet.argtype = (c_uint64, c_uint)
    MVGigE.MVSaveUserSet.restype = c_int
    res = MVGigE.MVSaveUserSet(c_uint64(hCam), userset)
    return res


# MVSetDefaultUserSet(HANDLE hCam, UserSetSelectorEnums userset)
def MVSetDefaultUserSet(hCam, userset):
    MVGigE.MVSetDefaultUserSet.argtype = (c_uint64, c_uint)
    MVGigE.MVSetDefaultUserSet.restype = c_int
    res = MVGigE.MVSetDefaultUserSet(c_uint64(hCam), userset)
    return res


# MVGetDefaultUserSet(HANDLE hCam, UserSetSelectorEnums* pUserset)
def MVGetDefaultUserSet(hCam):
    MVGigE.MVGetDefaultUserSet.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetDefaultUserSet.restype = c_int
    pUserset = c_uint()
    res = MVGigE.MVGetDefaultUserSet(c_uint64(hCam), byref(pUserset))
    return res, pUserset.value


# MVImageFlip(HANDLE hCam, MVImage* pSrcImage, MVImage* pDstImage, ImageFlipType flipType)
def MVImageFlip(hCam, flipType):
    MVGigE.MVImageFlip.argtype = (c_uint64, c_void_p, c_void_p, ImageFlipType)
    MVGigE.MVImageFlip.restype = c_int
    pSrcImage = c_uint()
    pDstImage = c_uint()
    res = MVGigE.MVImageFlip(c_uint64(hCam), byref(pSrcImage), byref(pDstImage), flipType)
    return res, pSrcImage.value, pDstImage.value


# MVImageRotate(HANDLE hCam, MVImage* pSrcImage, MVImage* pDstImage, ImageRotateType roateType)
def MVImageRotate(hCam, roateType):
    MVGigE.MVImageRotate.argtype = (c_uint64, c_void_p, c_void_p, ImageRotateType)
    MVGigE.MVImageRotate.restype = c_int
    pSrcImage = c_uint()
    pDstImage = c_uint()
    res = MVGigE.MVImageRotate(c_uint64(hCam), byref(pSrcImage), byref(pDstImage), roateType)
    return res, pSrcImage.value, pDstImage.value


# MVBGRToGray(HANDLE hCam, unsigned char* psrc, unsigned char* pdst, unsigned int width, unsigned int height)
def MVBGRToGray(hCam, psrc, width, height):
    MVGigE.MVBGRToGray.argtype = (c_uint64, c_void_p, c_void_p, c_uint, c_uint)
    MVGigE.MVBGRToGray.restype = c_int
    pdst = (c_ubyte * width * height)()
    res = MVGigE.MVBGRToGray(c_uint64(hCam), psrc, byref(pdst), c_uint(width), c_uint(height))
    return res, pdst


# MVImageBGRToGray(HANDLE hCam, MVImage* pSrcImage, MVImage* pDstImage)
def MVImageBGRToGray(hCam):
    MVGigE.MVImageBGRToGray.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVImageBGRToGray.restype = c_int
    pSrcImage = c_uint()
    pDstImage = c_uint()
    res = MVGigE.MVImageBGRToGray(c_uint64(hCam), byref(pSrcImage), byref(pDstImage))
    return res, pSrcImage.value, pDstImage.value


# MVImageBGRToYUV(HANDLE hCam, MVImage* pSrcImage, unsigned char* pDst)
def MVImageBGRToYUV(hCam):
    MVGigE.MVImageBGRToYUV.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVImageBGRToYUV.restype = c_int
    pSrcImage = c_uint()
    pDst = c_ubyte()
    res = MVGigE.MVImageBGRToYUV(c_uint64(hCam), byref(pSrcImage), byref(pDst))
    return res, pSrcImage.value, pDst.value


# MVGrayToBGR(HANDLE hCam, unsigned char* pSrc, unsigned char* pDst, int width, int height)
def MVGrayToBGR(hCam, pSrc, width, height):
    MVGigE.MVGrayToBGR.argtype = (c_uint64, c_void_p, c_void_p, c_int, c_int)
    MVGigE.MVGrayToBGR.restype = c_int
    pDst = (c_ubyte * width * height * 3)()
    res = MVGigE.MVGrayToBGR(c_uint64(hCam), pSrc, byref(pDst), c_int(width), c_int(height))
    return res, pDst


# MVGetExposureAuto(HANDLE hCam, ExposureAutoEnums* pExposureAuto)
def MVGetExposureAuto(hCam):
    MVGigE.MVGetExposureAuto.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetExposureAuto.restype = c_int
    pExposureAuto = c_uint()
    res = MVGigE.MVGetExposureAuto(c_uint64(hCam), byref(pExposureAuto))
    return res, pExposureAuto.value


# MVSetExposureAuto(HANDLE hCam, ExposureAutoEnums ExposureAuto)
def MVSetExposureAuto(hCam, ExposureAuto):
    MVGigE.MVSetExposureAuto.argtype = (c_uint64, c_uint)
    MVGigE.MVSetExposureAuto.restype = c_int
    res = MVGigE.MVSetExposureAuto(c_uint64(hCam), ExposureAuto)
    return res


# MVGetGainAuto(HANDLE hCam, GainAutoEnums* pGainAuto)
def MVGetGainAuto(hCam):
    MVGigE.MVGetGainAuto.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetGainAuto.restype = c_int
    pGainAuto = c_uint()
    res = MVGigE.MVGetGainAuto(c_uint64(hCam), byref(pGainAuto))
    return res, pGainAuto.value


# MVSetGainAuto(HANDLE hCam, GainAutoEnums GainAuto)
def MVSetGainAuto(hCam, GainAuto):
    MVGigE.MVSetGainAuto.argtype = (c_uint64, c_uint)
    MVGigE.MVSetGainAuto.restype = c_int
    res = MVGigE.MVSetGainAuto(c_uint64(hCam), GainAuto)
    return res


# MVGetBalanceWhiteAuto(HANDLE hCam, BalanceWhiteAutoEnums* pBalanceWhiteAuto)
def MVGetBalanceWhiteAuto(hCam):
    MVGigE.MVGetBalanceWhiteAuto.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetBalanceWhiteAuto.restype = c_int
    pBalanceWhiteAuto = c_uint()
    res = MVGigE.MVGetBalanceWhiteAuto(c_uint64(hCam), byref(pBalanceWhiteAuto))
    return res, pBalanceWhiteAuto.value


# MVSetBalanceWhiteAuto(HANDLE hCam, BalanceWhiteAutoEnums BalanceWhiteAuto)
def MVSetBalanceWhiteAuto(hCam, BalanceWhiteAuto):
    MVGigE.MVSetBalanceWhiteAuto.argtype = (c_uint64, c_uint)
    MVGigE.MVSetBalanceWhiteAuto.restype = c_int
    res = MVGigE.MVSetBalanceWhiteAuto(c_uint64(hCam), BalanceWhiteAuto)
    return res


# MVGetAutoGainLowerLimit(HANDLE hCam, double* pAutoGainLowerLimit)
def MVGetAutoGainLowerLimit(hCam):
    MVGigE.MVGetAutoGainLowerLimit.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetAutoGainLowerLimit.restype = c_int
    pAutoGainLowerLimit = c_double()
    res = MVGigE.MVGetAutoGainLowerLimit(c_uint64(hCam), byref(pAutoGainLowerLimit))
    return res, pAutoGainLowerLimit.value


# MVSetAutoGainLowerLimit(HANDLE hCam, double fAutoGainLowerLimit)
def MVSetAutoGainLowerLimit(hCam, fAutoGainLowerLimit):
    MVGigE.MVSetAutoGainLowerLimit.argtype = (c_uint64, c_double)
    MVGigE.MVSetAutoGainLowerLimit.restype = c_int
    res = MVGigE.MVSetAutoGainLowerLimit(c_uint64(hCam), c_double(fAutoGainLowerLimit))
    return res


# MVGetAutoGainUpperLimit(HANDLE hCam, double* pAutoGainUpperLimit)
def MVGetAutoGainUpperLimit(hCam):
    MVGigE.MVGetAutoGainUpperLimit.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetAutoGainUpperLimit.restype = c_int
    pAutoGainUpperLimit = c_double()
    res = MVGigE.MVGetAutoGainUpperLimit(c_uint64(hCam), byref(pAutoGainUpperLimit))
    return res, pAutoGainUpperLimit.value


# MVSetAutoGainUpperLimit(HANDLE hCam, double fAutoGainUpperLimit)
def MVSetAutoGainUpperLimit(hCam, fAutoGainUpperLimit):
    MVGigE.MVSetAutoGainUpperLimit.argtype = (c_uint64, c_double)
    MVGigE.MVSetAutoGainUpperLimit.restype = c_int
    res = MVGigE.MVSetAutoGainUpperLimit(c_uint64(hCam), c_double(fAutoGainUpperLimit))
    return res


# MVGetAutoExposureTimeLowerLimit(HANDLE hCam, double* pAutoExposureTimeLowerLimit)
def MVGetAutoExposureTimeLowerLimit(hCam):
    MVGigE.MVGetAutoExposureTimeLowerLimit.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetAutoExposureTimeLowerLimit.restype = c_int
    pAutoExposureTimeLowerLimit = c_double()
    res = MVGigE.MVGetAutoExposureTimeLowerLimit(c_uint64(hCam), byref(pAutoExposureTimeLowerLimit))
    return res, pAutoExposureTimeLowerLimit.value


# MVSetAutoExposureTimeLowerLimit(HANDLE hCam, double fAutoExposureTimeLowerLimit)
def MVSetAutoExposureTimeLowerLimit(hCam, fAutoExposureTimeLowerLimit):
    MVGigE.MVSetAutoExposureTimeLowerLimit.argtype = (c_uint64, c_double)
    MVGigE.MVSetAutoExposureTimeLowerLimit.restype = c_int
    res = MVGigE.MVSetAutoExposureTimeLowerLimit(c_uint64(hCam), c_double(fAutoExposureTimeLowerLimit))
    return res


# MVGetAutoExposureTimeUpperLimit(HANDLE hCam, double* pAutoExposureTimeUpperLimit)
def MVGetAutoExposureTimeUpperLimit(hCam):
    MVGigE.MVGetAutoExposureTimeUpperLimit.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetAutoExposureTimeUpperLimit.restype = c_int
    pAutoExposureTimeUpperLimit = c_double()
    res = MVGigE.MVGetAutoExposureTimeUpperLimit(c_uint64(hCam), byref(pAutoExposureTimeUpperLimit))
    return res, pAutoExposureTimeUpperLimit.value


# MVSetAutoExposureTimeUpperLimit(HANDLE hCam, double fAutoExposureTimeUpperLimit)
def MVSetAutoExposureTimeUpperLimit(hCam, fAutoExposureTimeUpperLimit):
    MVGigE.MVSetAutoExposureTimeUpperLimit.argtype = (c_uint64, c_double)
    MVGigE.MVSetAutoExposureTimeUpperLimit.restype = c_int
    res = MVGigE.MVSetAutoExposureTimeUpperLimit(c_uint64(hCam), c_double(fAutoExposureTimeUpperLimit))
    return res


# MVGetAutoTargetValue(HANDLE hCam, int* pAutoTargetValue)
def MVGetAutoTargetValue(hCam):
    MVGigE.MVGetAutoTargetValue.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetAutoTargetValue.restype = c_int
    pAutoTargetValue = c_int()
    res = MVGigE.MVGetAutoTargetValue(c_uint64(hCam), byref(pAutoTargetValue))
    return res, pAutoTargetValue.value


# MVSetAutoTargetValue(HANDLE hCam, int nAutoTargetValue)
def MVSetAutoTargetValue(hCam, nAutoTargetValue):
    MVGigE.MVSetAutoTargetValue.argtype = (c_uint64, c_int)
    MVGigE.MVSetAutoTargetValue.restype = c_int
    res = MVGigE.MVSetAutoTargetValue(c_uint64(hCam), c_int(nAutoTargetValue))
    return res


# MVGetAutoFunctionProfile(HANDLE hCam, AutoFunctionProfileEnums* pAutoFunctionProfile)
def MVGetAutoFunctionProfile(hCam):
    MVGigE.MVGetAutoFunctionProfile.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetAutoFunctionProfile.restype = c_int
    pAutoFunctionProfile = c_uint()
    res = MVGigE.MVGetAutoFunctionProfile(c_uint64(hCam), byref(pAutoFunctionProfile))
    return res, pAutoFunctionProfile.value


# MVSetAutoFunctionProfile(HANDLE hCam, AutoFunctionProfileEnums AutoFunctionProfile)
def MVSetAutoFunctionProfile(hCam, AutoFunctionProfile):
    MVGigE.MVSetAutoFunctionProfile.argtype = (c_uint64, c_uint)
    MVGigE.MVSetAutoFunctionProfile.restype = c_int
    res = MVGigE.MVSetAutoFunctionProfile(c_uint64(hCam), AutoFunctionProfile)
    return res


# MVGetAutoThreshold(HANDLE hCam, int* pAutoThreshold)
def MVGetAutoThreshold(hCam):
    MVGigE.MVGetAutoThreshold.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetAutoThreshold.restype = c_int
    pAutoThreshold = c_int()
    res = MVGigE.MVGetAutoThreshold(c_uint64(hCam), byref(pAutoThreshold))
    return res, pAutoThreshold.value


# MVSetAutoThreshold(HANDLE hCam, int nAutoThreshold)
def MVSetAutoThreshold(hCam, nAutoThreshold):
    MVGigE.MVSetAutoThreshold.argtype = (c_uint64, c_int)
    MVGigE.MVSetAutoThreshold.restype = c_int
    res = MVGigE.MVSetAutoThreshold(c_uint64(hCam), c_int(nAutoThreshold))
    return res


# MVGetGamma(HANDLE hCam, double* pGamma)
def MVGetGamma(hCam):
    MVGigE.MVGetGamma.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetGamma.restype = c_int
    pGamma = c_double()
    res = MVGigE.MVGetGamma(c_uint64(hCam), byref(pGamma))
    return res, pGamma.value


# MVGetGammaRange(HANDLE hCam, double* pGammaMin, double* pGammaMax)
def MVGetGammaRange(hCam):
    MVGigE.MVGetGammaRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetGammaRange.restype = c_int
    pGammaMin = c_double()
    pGammaMax = c_double()
    res = MVGigE.MVGetGammaRange(c_uint64(hCam), byref(pGammaMin), byref(pGammaMax))
    return res, pGammaMin.value, pGammaMax.value


# MVSetGamma(HANDLE hCam, double fGamma)
def MVSetGamma(hCam, fGamma):
    MVGigE.MVSetGamma.argtype = (c_uint64, c_double)
    MVGigE.MVSetGamma.restype = c_int
    res = MVGigE.MVSetGamma(c_uint64(hCam), c_double(fGamma))
    return res


# MVSetLUT(HANDLE hCam, unsigned long* pLUT, int nCnt)
def MVSetLUT(hCam, nCnt):
    MVGigE.MVSetLUT.argtype = (c_uint64, c_void_p, c_int)
    MVGigE.MVSetLUT.restype = c_int
    pLUT = c_ulong()
    res = MVGigE.MVSetLUT(c_uint64(hCam), byref(pLUT), c_int(nCnt))
    return res, pLUT.value


# MVSetEnableLUT(HANDLE hCam, bool bEnable)
def MVSetEnableLUT(hCam, bEnable):
    MVGigE.MVSetEnableLUT.argtype = (c_uint64, c_bool)
    MVGigE.MVSetEnableLUT.restype = c_int
    res = MVGigE.MVSetEnableLUT(c_uint64(hCam), c_bool(bEnable))
    return res


# MVGetEnableLUT(HANDLE hCam, bool* bEnable)
def MVGetEnableLUT(hCam):
    MVGigE.MVGetEnableLUT.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetEnableLUT.restype = c_int
    bEnable = c_bool()
    res = MVGigE.MVGetEnableLUT(c_uint64(hCam), byref(bEnable))
    return res, bEnable.value


# MVSingleGrab(HANDLE hCam, HANDLE hImage, unsigned long nWaitMs)
def MVSingleGrab(hCam, hImage, nWaitMs):
    MVGigE.MVSingleGrab.argtype = (c_uint64, c_uint64, c_ulong)
    MVGigE.MVSingleGrab.restype = c_int
    res = MVGigE.MVSingleGrab(c_uint64(hCam), c_uint64(hImage), c_ulong(nWaitMs))
    return res


# MVStartGrabWindow(HANDLE hCam, HWND hWnd, HWND hWndMsg)
def MVStartGrabWindow(hCam, hWnd=0, hWndMsg=0):
    MVGigE.MVStartGrabWindow.argtype = (c_uint64, c_uint64, c_uint64)
    MVGigE.MVStartGrabWindow.restype = c_int
    res = MVGigE.MVStartGrabWindow(c_uint64(hCam), c_uint64(hWnd), c_uint64(hWndMsg))
    return res


# MVStopGrabWindow(HANDLE hCam)
def MVStopGrabWindow(hCam):
    MVGigE.MVStopGrabWindow.argtype = (c_uint64)
    MVGigE.MVStopGrabWindow.restype = c_int
    res = MVGigE.MVStopGrabWindow(c_uint64(hCam))
    return res


# MVFreezeGrabWindow(HANDLE hCam, bool bFreeze)
def MVFreezeGrabWindow(hCam, bFreeze):
    MVGigE.MVFreezeGrabWindow.argtype = (c_uint64, c_bool)
    MVGigE.MVFreezeGrabWindow.restype = c_int
    res = MVGigE.MVFreezeGrabWindow(c_uint64(hCam), c_bool(bFreeze))
    return res


# MVSetGrabWindow(HANDLE hCam, int xDest, int yDest, int wDest, int hDest, int xSrc, int ySrc, int wSrc, int hSrc)
def MVSetGrabWindow(hCam, xDest, yDest, wDest, hDest, xSrc, ySrc, wSrc, hSrc):
    MVGigE.MVSetGrabWindow.argtype = (c_uint64, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int)
    MVGigE.MVSetGrabWindow.restype = c_int
    res = MVGigE.MVSetGrabWindow(c_uint64(hCam), c_int(xDest), c_int(yDest), c_int(wDest), c_int(hDest), c_int(xSrc),
                                 c_int(ySrc), c_int(wSrc), c_int(hSrc))
    return res


# MVGetSampleGrab(HANDLE hCam, MVImage* image, int* nFrameID, int msTimeout)
def MVGetSampleGrab(hCam, msTimeout):
    MVGigE.MVGetSampleGrab.argtype = (c_uint64, c_void_p, c_void_p, c_int)
    MVGigE.MVGetSampleGrab.restype = c_int
    image = c_uint()
    nFrameID = c_int()
    res = MVGigE.MVGetSampleGrab(c_uint64(hCam), byref(image), byref(nFrameID), c_int(msTimeout))
    return res, image.value, nFrameID.value


# MVGetSampleGrabBuf(HANDLE hCam, unsigned char *pImgBuf, unsigned long szBuf,int *pFrameID, int msTimeout)
def MVGetSampleGrabBuf(hCam, pImg, msTimeout):
    MVGigE.MVGetSampleGrab.argtype = (c_uint64, POINTER(c_ubyte), c_int, c_void_p, c_int)
    MVGigE.MVGetSampleGrab.restype = c_int
    pImgBuf = cast(pImg.ctypes.data, POINTER(c_ubyte))
    szBuf = pImg.nbytes
    nFrameID = c_int()
    res = MVGigE.MVGetSampleGrabBuf(c_uint64(hCam), pImgBuf, szBuf, byref(nFrameID), c_int(msTimeout))
    return res, nFrameID.value


# MVGetDroppedFrame(HANDLE hCam,unsigned long *pDroppedFrames)
def MVGetDroppedFrame(hCam):
    MVGigE.MVGetDroppedFrame.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetDroppedFrame.restype = c_int
    pDroppedFrames = c_ulong()
    res = MVGigE.MVGetDroppedFrame(c_uint64(hCam), byref(pDroppedFrames))
    return res, pDroppedFrames.value


# MVGetDeviceVendorName(HANDLE hCam,char *pBuf,int *szBuf)
def MVGetDeviceVendorName(hCam):
    MVGigE.MVGetDeviceVendorName.argtype = (c_uint64, c_char_p, c_void_p)
    MVGigE.MVGetDeviceVendorName.restype = c_int
    pBuf = (c_char * 32)()
    szBuf = c_int(32)
    res = MVGigE.MVGetDeviceVendorName(c_uint64(hCam), pBuf, byref(szBuf))
    return res, pBuf.value, szBuf.value


# MVGetDeviceModelName(HANDLE hCam,char *pBuf,int *szBuf)
def MVGetDeviceModelName(hCam):
    MVGigE.MVGetDeviceModelName.argtype = (c_uint64, c_char_p, c_void_p)
    MVGigE.MVGetDeviceModelName.restype = c_int
    pBuf = (c_char * 32)()
    szBuf = c_int(32)
    res = MVGigE.MVGetDeviceModelName(c_uint64(hCam), pBuf, byref(szBuf))
    return res, pBuf.value, szBuf.value


# MVGetDeviceDeviceID(HANDLE hCam,char *pBuf,int *szBuf)
def MVGetDeviceDeviceID(hCam):
    MVGigE.MVGetDeviceDeviceID.argtype = (c_uint64, c_char_p, c_void_p)
    MVGigE.MVGetDeviceDeviceID.restype = c_int
    pBuf = (c_char * 16)()
    szBuf = c_int(16)
    res = MVGigE.MVGetDeviceDeviceID(c_uint64(hCam), pBuf, byref(szBuf))
    return res, pBuf.value, szBuf.value


# MVIsRunning(HANDLE hCam)
def MVIsRunning(hCam):
    MVGigE.MVIsRunning.argtype = (c_uint64)
    MVGigE.MVIsRunning.restype = c_bool
    res = MVGigE.MVIsRunning(c_uint64(hCam))
    return res


# MVConvertImage( HANDLE hCam, MVImage* pImageSrc,MVImage* pImageDst )
def MVConvertImage(hCam):
    MVGigE.MVConvertImage.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVConvertImage.restype = c_int
    pImageSrc = c_uint()
    pImageDst = c_uint()
    res = MVGigE.MVConvertImage(c_uint64(hCam), byref(pImageSrc), byref(pImageDst))
    return res, pImageSrc.value, pImageDst.value


# MVCopyImageInfoROI( HANDLE hCam, MV_IMAGE_INFO* pInfoSrc, MV_IMAGE_INFO* pInfoDst, RECT roi )
def MVCopyImageInfoROI(hCam, pInfoSrc, x, y, w, h):
    MVGigE.MVCopyImageInfoROI.argtype = (c_uint64, POINTER(MV_IMAGE_INFO), POINTER(MV_IMAGE_INFO), RECT)
    MVGigE.MVCopyImageInfoROI.restype = c_int
    pInfoDst = MV_IMAGE_INFO()
    data = (c_ubyte * w * h)()
    pInfoDst.nSizeX = w
    pInfoDst.nSizeY = h
    pInfoDst.nPixelType = pInfoSrc.nPixelType
    pInfoDst.pImageBuffer = POINTER(c_ubyte)(data[0])
    res = MVGigE.MVCopyImageInfoROI(c_uint64(hCam), pInfoSrc, byref(pInfoDst), RECT(x, y, x + w, y + h))
    return res, pInfoDst


# MVSetSaturation(HANDLE hCam, int nSaturation )
def MVSetSaturation(hCam, nSaturation):
    MVGigE.MVSetSaturation.argtype = (c_uint64, c_int)
    MVGigE.MVSetSaturation.restype = c_int
    res = MVGigE.MVSetSaturation(c_uint64(hCam), c_int(nSaturation))
    return res


# MVGetSaturation(HANDLE hCam, int *nSaturation )
def MVGetSaturation(hCam):
    MVGigE.MVGetSaturation.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetSaturation.restype = c_int
    nSaturation = c_int()
    res = MVGigE.MVGetSaturation(c_uint64(hCam), byref(nSaturation))
    return res, nSaturation.value


# MVSetColorCorrect(HANDLE hCam, int nColorCorrect )
def MVSetColorCorrect(hCam, nColorCorrect):
    MVGigE.MVSetColorCorrect.argtype = (c_uint64, c_int)
    MVGigE.MVSetColorCorrect.restype = c_int
    res = MVGigE.MVSetColorCorrect(c_uint64(hCam), c_int(nColorCorrect))
    return res


# MVGetColorCorrect(HANDLE hCam, int *nColorCorrect )
def MVGetColorCorrect(hCam):
    MVGigE.MVGetColorCorrect.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetColorCorrect.restype = c_int
    nColorCorrect = c_int()
    res = MVGigE.MVGetColorCorrect(c_uint64(hCam), byref(nColorCorrect))
    return res, nColorCorrect.value


# MVRegisterMessage(HANDLE hCam, HWND hWnd, UINT nMsg)
def MVRegisterMessage(hCam, hWnd, nMsg):
    MVGigE.MVRegisterMessage.argtype = (c_uint64, HWND, UINT)
    MVGigE.MVRegisterMessage.restype = c_int
    res = MVGigE.MVRegisterMessage(c_uint64(hCam), hWnd, nMsg)
    return res


# MVEnableMessage(HANDLE hCam, int nMessageType, bool bEnable)
def MVEnableMessage(hCam, nMessageType, bEnable):
    MVGigE.MVEnableMessage.argtype = (c_uint64, c_int, c_bool)
    MVGigE.MVEnableMessage.restype = c_int
    res = MVGigE.MVEnableMessage(c_uint64(hCam), c_int(nMessageType), c_bool(bEnable))
    return res


# MVGetUserDefinedName(HANDLE hCam, char *pBuf,int *szBuf )
def MVGetUserDefinedName(hCam):
    MVGigE.MVGetUserDefinedName.argtype = (c_uint64, c_char_p, POINTER(c_int))
    MVGigE.MVGetUserDefinedName.restype = c_int
    pBuf = (c_char * 16)()
    szBuf = c_int(16)
    res = MVGigE.MVGetUserDefinedName(c_uint64(hCam), pBuf, byref(szBuf))
    return res, pBuf.value, szBuf.value


# MVSetUserDefinedName(HANDLE hCam, char *pBuf,int szBuf )
def MVSetUserDefinedName(hCam, pBuf, szBuf):
    MVGigE.MVSetUserDefinedName.argtype = (c_uint64, c_char_p, c_int)
    MVGigE.MVSetUserDefinedName.restype = c_int
    # pBuf = c_char()
    res = MVGigE.MVSetUserDefinedName(c_uint64(hCam), pBuf, c_int(szBuf))
    return res


# MVOpenCamByIndexReadOnly(unsigned char idx,HANDLE *hCam)
def MVOpenCamByIndexReadOnly(idx):
    MVGigE.MVOpenCamByIndexReadOnly.argtype = (c_ubyte, c_void_p)
    MVGigE.MVOpenCamByIndexReadOnly.restype = c_int
    hCam = c_uint64()
    res = MVGigE.MVOpenCamByIndexReadOnly(c_ubyte(idx), byref(hCam))
    return res, hCam.value


# MVEnumerateAllDevices( int *pDevCnt)
def MVEnumerateAllDevices():
    MVGigE.MVEnumerateAllDevices.argtype = (c_void_p)
    MVGigE.MVEnumerateAllDevices.restype = c_int
    pDevCnt = c_int()
    res = MVGigE.MVEnumerateAllDevices(byref(pDevCnt))
    return res, pDevCnt.value


# MVForceIp( const char* pMacAddress, const char* pIpAddress, const char* pSubnetMask, const char* pDefaultGateway)
def MVForceIp(pMacAddress, pIpAddress, pSubnetMask, pDefaultGateway):
    MVGigE.MVForceIp.argtype = (POINTER(c_ubyte), c_char_p, c_char_p, c_char_p)
    MVGigE.MVForceIp.restype = c_int
    res = MVGigE.MVForceIp(pMacAddress, pIpAddress, pSubnetMask, pDefaultGateway)
    return res


# MVGetDevInfo(unsigned char idx,MVCamInfo *pCamInfo)
def MVGetDevInfo(idx):
    MVGigE.MVGetDevInfo.argtype = (c_ubyte, c_void_p)
    MVGigE.MVGetDevInfo.restype = c_int
    pCamInfo = MVCamInfo()
    res = MVGigE.MVGetDevInfo(c_ubyte(idx), byref(pCamInfo))
    return res, pCamInfo


# MVSetPersistentIpAddress( HANDLE hCam, const char* pIpAddress, const char* pSubnetMask, const char* pDefaultGateway)
def MVSetPersistentIpAddress(hCam, pIpAddress, pSubnetMask, pDefaultGateway):
    MVGigE.MVSetPersistentIpAddress.argtype = (c_uint64, c_void_p, c_void_p, c_void_p)
    MVGigE.MVSetPersistentIpAddress.restype = c_int
    res = MVGigE.MVSetPersistentIpAddress(c_uint64(hCam), pIpAddress, pSubnetMask, pDefaultGateway)
    return res


# MVGetPersistentIpAddress( HANDLE hCam, char* pIpAddress, size_t* pIpAddressLen, char* pSubnetMask, size_t* pSubnetMaskLen, char* pDefaultGateway, size_t* pDefaultGatewayLen)
def MVGetPersistentIpAddress(hCam):
    MVGigE.MVGetPersistentIpAddress.argtype = (c_uint64, c_char_p, c_void_p, c_char_p, c_void_p, c_char_p, c_void_p)
    MVGigE.MVGetPersistentIpAddress.restype = c_int
    pIpAddress = (c_ubyte * 4)()
    pIpAddressLen = c_uint(4)
    pSubnetMask = (c_ubyte * 4)()
    pSubnetMaskLen = c_uint(4)
    pDefaultGateway = (c_ubyte * 4)()
    pDefaultGatewayLen = c_uint(4)
    res = MVGigE.MVGetPersistentIpAddress(c_uint64(hCam), pIpAddress, byref(pIpAddressLen), pSubnetMask,
                                          byref(pSubnetMaskLen), pDefaultGateway, byref(pDefaultGatewayLen))
    return res, pIpAddress, pSubnetMask, pDefaultGateway


# MVSetTransferControlMode(HANDLE hCam,TransferControlModeEnums mode)
def MVSetTransferControlMode(hCam, mode):
    MVGigE.MVSetTransferControlMode.argtype = (c_uint64, c_uint)
    MVGigE.MVSetTransferControlMode.restype = c_int
    res = MVGigE.MVSetTransferControlMode(c_uint64(hCam), mode)
    return res


# MVSetTransferBlockCount(HANDLE hCam,unsigned long cnt)
def MVSetTransferBlockCount(hCam, cnt):
    MVGigE.MVSetTransferBlockCount.argtype = (c_uint64, c_ulong)
    MVGigE.MVSetTransferBlockCount.restype = c_int
    res = MVGigE.MVSetTransferBlockCount(c_uint64(hCam), c_ulong(cnt))
    return res


# MVTransferStart(HANDLE hCam)
def MVTransferStart(hCam):
    MVGigE.MVTransferStart.argtype = (c_uint64)
    MVGigE.MVTransferStart.restype = c_int
    res = MVGigE.MVTransferStart(c_uint64(hCam))
    return res


# MVTransferStop(HANDLE hCam)
def MVTransferStop(hCam):
    MVGigE.MVTransferStop.argtype = (c_uint64)
    MVGigE.MVTransferStop.restype = c_int
    res = MVGigE.MVTransferStop(c_uint64(hCam))
    return res


# MVSetRegionSelector(HANDLE hCam, int nRegionSelector )
def MVSetRegionSelector(hCam, nRegionSelector):
    MVGigE.MVSetRegionSelector.argtype = (c_uint64, c_int)
    MVGigE.MVSetRegionSelector.restype = c_int
    res = MVGigE.MVSetRegionSelector(c_uint64(hCam), c_int(nRegionSelector))
    return res


# MVGetRegionSelector(HANDLE hCam, int *pRegionSelector )
def MVGetRegionSelector(hCam):
    MVGigE.MVGetRegionSelector.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetRegionSelector.restype = c_int
    pRegionSelector = c_int()
    res = MVGigE.MVGetRegionSelector(c_uint64(hCam), byref(pRegionSelector))
    return res, pRegionSelector.value


# MVSetBlackLevelTaps( HANDLE hCam, double fBlackLevel, int nTap )
def MVSetBlackLevelTaps(hCam, fBlackLevel, nTap):
    MVGigE.MVSetBlackLevelTaps.argtype = (c_uint64, c_double, c_int)
    MVGigE.MVSetBlackLevelTaps.restype = c_int
    res = MVGigE.MVSetBlackLevelTaps(c_uint64(hCam), c_double(fBlackLevel), c_int(nTap))
    return res


# MVGetBlackLevelTaps( HANDLE hCam, double *pBlackLevel,int nTap )
def MVGetBlackLevelTaps(hCam, nTap):
    MVGigE.MVGetBlackLevelTaps.argtype = (c_uint64, c_void_p, c_int)
    MVGigE.MVGetBlackLevelTaps.restype = c_int
    pBlackLevel = c_double()
    res = MVGigE.MVGetBlackLevelTaps(c_uint64(hCam), byref(pBlackLevel), c_int(nTap))
    return res, pBlackLevel.value


# MVSetBlackLevel( HANDLE hCam, double fBlackLevel)
def MVSetBlackLevel(hCam, fBlackLevel):
    MVGigE.MVSetBlackLevel.argtype = (c_uint64, c_double)
    MVGigE.MVSetBlackLevel.restype = c_int
    res = MVGigE.MVSetBlackLevel(c_uint64(hCam), c_double(fBlackLevel))
    return res


# MVGetBlackLevel( HANDLE hCam, double *pBlackLevel)
def MVGetBlackLevel(hCam):
    MVGigE.MVGetBlackLevel.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetBlackLevel.restype = c_int
    pBlackLevel = c_double()
    res = MVGigE.MVGetBlackLevel(c_uint64(hCam), byref(pBlackLevel))
    return res, pBlackLevel.value


def MVSetPixelFormat(hCam, pixel_format):
    MVGigE.MVSetPixelFormat.argtype = (c_uint64, c_uint)
    MVGigE.MVSetPixelFormat.restype = c_int
    res = MVGigE.MVSetPixelFormat(c_uint64(hCam), pixel_format)
    return res
