from enum import Enum
from ctypes import *

# 
INT64_MAX = 0x7fffffffffffffff  #/*maximum signed __int64 value */
INT64_MIN = 0x8000000000000000  #/*minimum signed __int64 value */
UINT64_MAX = 0xffffffffffffffff  #/*maximum unsigned __int64 value */

INT32_MAX = 0x000000007fffffff  #/*maximum signed __int32 value */
INT32_MIN = 0xffffffff80000000  #/*minimum signed __int32 value */
UINT32_MAX = 0x00000000ffffffff  #/*maximum unsigned __int32 value */

INT8_MAX = 0x000000000000007f  #/*maximum signed __int8 value */
INT8_MIN = 0xffffffffffffff80  #/*minimum signed __int8 value */
UINT8_MAX = 0x00000000000000ff  #/*maximum unsigned __int8 value */

FALSE = 0
TRUE = 1

# Bayer颜色模式
# MV_BAYER_MODE
BayerRG = 0	#< 颜色模式RGGB
BayerBG = 1	#< 颜色模式BGGR
BayerGR = 2	#< 颜色模式GRBG
BayerGB = 3	#< 颜色模式GBRG
BayerGRW = 4	#< 颜色模式GRW
BayerInvalid = 5
 
# 图像像素格式
# MV_PixelFormatEnums
PixelFormat_Mono8 = 0x01080001	#!<8Bit灰度
PixelFormat_BayerBG8=0x0108000B	#!<8Bit Bayer图,颜色模式为BGGR
PixelFormat_BayerRG8=0x01080009	#!<8Bit Bayer图,颜色模式为RGGB
PixelFormat_BayerGB8=0x0108000A	#!<8Bit Bayer图,颜色模式为GBRG
PixelFormat_BayerGR8=0x01080008	#!<8Bit Bayer图,颜色模式为GRBG
PixelFormat_BayerGRW8=0x0108000C	#!<8Bit Bayer图,颜色模式为GRW8
PixelFormat_Mono16=0x01100007	#!<16Bit灰度
PixelFormat_BayerGR16=0x0110002E	#!<16Bit Bayer图,颜色模式为GR
PixelFormat_BayerRG16=0x0110002F	#!<16Bit Bayer图,颜色模式为RG
PixelFormat_BayerGB16=0x01100030	#!<16Bit Bayer图,颜色模式为GB
PixelFormat_BayerBG16=0x01100031	#!<16Bit Bayer图,颜色模式为BG

 
# 错误返回值类型
# MVSTATUS_CODES 
MVST_SUCCESS                = 0      #/< 没有错误      
MVST_ERROR                  = -1001  #/< 一般错误
MVST_ERR_NOT_INITIALIZED    = -1002  #!< 没有初始化
MVST_ERR_NOT_IMPLEMENTED    = -1003  #!< 没有实现
MVST_ERR_RESOURCE_IN_USE    = -1004  #!< 资源被占用
MVST_ACCESS_DENIED          = -1005  #/< 无法访问
MVST_INVALID_HANDLE         = -1006  #/< 错误句柄
MVST_INVALID_ID             = -1007  #/< 错误ID
MVST_NO_DATA                = -1008  #/< 没有数据
MVST_INVALID_PARAMETER      = -1009  #/< 错误参数
MVST_FILE_IO                = -1010  #/< IO错误
MVST_TIMEOUT                = -1011  #/< 超时
MVST_ERR_ABORT              = -1012  #/< 退出
MVST_INVALID_BUFFER_SIZE    = -1013  #/< 缓冲区尺寸错误
MVST_ERR_NOT_AVAILABLE      = -1014  #/< 无法访问
MVST_INVALID_ADDRESS        = -1015  #/< 地址错误

# TriggerSourceEnums
TriggerSource_Software=0#!<触发模式下，由软触发(软件指令)来触发采集
TriggerSource_Line1=2 #!<触发模式下，有外触发信号来触发采集

# TriggerModeEnums
TriggerMode_Off = 0  #!<触发模式关，即FreeRun模式，相机连续采集
TriggerMode_On = 1	#!<触发模式开，相机等待软触发或外触发信号再采集图像

# TriggerActivationEnums 
TriggerActivation_RisingEdge = 0  #!<上升沿触发
TriggerActivation_FallingEdge = 1 #!<下降沿触发

# LineSourceEnums
LineSource_Off=0  #!<关闭
LineSource_ExposureActive=5  #!<和曝光同时
LineSource_Timer1Active=6	#!<由定时器控制
LineSource_UserOutput0=12	#!<直接由软件控制

# UserSetSelectorEnums
UserSetSelector_Default = 0  #!<出厂设置
UserSetSelector_UserSet1 = 1  #!<用户设置1
UserSetSelector_UserSet2 = 2   #!<用户设置2

# SensorTapsEnums
SensorTaps_One = 0  #!<单通道
SensorTaps_Two = 1  #!<双通道
SensorTaps_Three = 2  #!<三通道
SensorTaps_Four = 3  #!<四通道

# AutoFunctionProfileEnums
AutoFunctionProfile_GainMinimum = 0  #!<Keep gain at minimum
AutoFunctionProfile_ExposureMinimum = 1  #!<Exposure Time at minimum


# GainAutoEnums
GainAuto_Off = 0  #!<Disables the Gain Auto function.
GainAuto_Once = 1  #!<Sets operation mode to 'once'.
GainAuto_Continuous = 2   #!<Sets operation mode to 'continuous'.


#! ExposureAutoEnums
ExposureAuto_Off = 0 #!<Disables the Exposure Auto function.
ExposureAuto_Once = 1  #!<Sets operation mode to 'once'.
ExposureAuto_Continuous = 2   #!<Sets operation mode to 'continuous'.


# BalanceWhiteAutoEnums
BalanceWhiteAuto_Off = 0  #!<Disables the Balance White Auto function.
BalanceWhiteAuto_Once = 1  #!<Sets operation mode to 'once'.
BalanceWhiteAuto_Continuous = 2   #!<Sets operation mode to 'continuous'.


#####################################
# ImageFlipType
FlipHorizontal = 0  #!< 左右翻转
FlipVertical = 1 #!< 上下翻转
FlipBoth = 2 #!< 旋转180度


# ImageRotateType
Rotate90DegCw = 0       #!< 顺时针旋转90度
Rotate90DegCcw = 1       #!< 逆时针旋转90度


# TransferControlModeEnums
TransferControlMode_Basic = 0,
TransferControlMode_UserControlled = 2

#! EventIdEnums
EVID_LOST = 0,	#!< 事件ID，相机断开
EVID_RECONNECT = 1	#! 事件ID，相机重新连上了


###################################################################

class RECT(Structure):
	_fields_ = [('left', c_long),          
                ('top', c_long),  
                ('right', c_long),          
                ('bottom', c_long)]
				
#回调函数得到的数据的结构
class MV_IMAGE_INFO(Structure):
    _fields_ = [('nTimeStamp', c_ulonglong),         # 时间戳，采集到图像的时刻，精度为0.01us
                ('nBlockId', c_ushort),              # 帧号，从开始采集开始计数
                ('pImageBuffer', POINTER(c_ubyte)),  # 图像指针，即指向(0,0)像素所在内存位置的指针，通过该指针可以访问整个图
                ('nImageSizeAcq', c_ulong),          # 采集到的图像大小[字节]
                ('nMissingPackets', c_ubyte),        # 传输过程中丢掉的包数量
                ('nPixelType', c_ulonglong),         # 图像格式
                ('nSizeX', c_uint),                  # 图像宽度
                ('nSizeY', c_uint),                  # 图像高度
                ('nOffsetX', c_uint),                # 图像水平坐标
                ('nOffsetY', c_uint)]                # 图像垂直坐标

# 相机的信息
class MVCamInfo(Structure):
    _fields_ = [('mIpAddr', c_ubyte*4),           # 相机的IP地址
                ('mEthernetAddr', c_ubyte*6),     # 相机的MAC地址
                ('mMfgName', c_char*32),          # 相机的厂商名称
                ('mModelName', c_char*32),        # 相机型号
                ('mSerialNumber', c_char*16),     # 相机序列号
				('mUserDefinedName', c_char*16),  # 用户设置相机名称
                ('m_IfIP', c_ubyte*4),            # 计算机和相机连接网卡IP地址
                ('m_IfMAC', c_ubyte*6)]           # 计算机和相机连接的网卡MAC地址

			
class MVStreamStatistic(Structure):
    _fields_ = [('m_nTotalBuf', c_ulong),         # 从开始采集，总计成功收到的完成图像帧数
                ('m_nFailedBuf', c_ulong),        # 从开始采集，总计收到的不完成图像帧数
                ('m_nTotalPacket', c_ulong),      # 从开始采集，总计收到的图像数据包数
                ('m_nFailedPacket', c_ulong),     # 从开始采集，总计丢失的图像包数
                ('m_nResendPacketReq', c_ulong),  # 从开始采集，总计重发请求的图像数据包数
                ('m_nResendPacket', c_ulong)]     # 从开始采集，总计重发成功的图像数据包数

MVStreamCB = WINFUNCTYPE(c_int, POINTER(MV_IMAGE_INFO), POINTER(c_longlong))
	