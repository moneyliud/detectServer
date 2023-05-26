import json

from detect.djangomodels import ImgStore, ImgCompareResult, ImgCompareResultV, SysDict, SysDictItem
from django.http import JsonResponse, StreamingHttpResponse
from detect.utils.convertor import model_obj_to_dict
from detect.utils.filter import get_filter_by_request
from detect.core.imageCompre import compare_image
import cv2
import numpy as np
from io import BytesIO
from django.core.files import File
# from detect.camera.cameraFactory import CameraFactory

# Create your views here.
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage


def test(request):
    return JsonResponse({"msg": "成功123123！"})


def get_image_list(request):
    param_filter = get_filter_by_request(request, ImgStore)
    page_num = request.GET.get("page_num")
    page_size = request.GET.get("page_size")
    img_list = param_filter.values("img_id", "product_name", "part_no", "batch_no",
                                   "plane_no", "img_content", "create_time", "update_time", "is_basic_img",
                                   "compare_status").order_by("img_id")
    paginator = Paginator(img_list, page_size)
    try:
        page_info = paginator.page(page_num)
    except PageNotAnInteger:
        page_info = paginator.page(1)
    except EmptyPage:
        page_info = paginator.page(paginator.num_pages)
    return JsonResponse({"data": [i for i in page_info], "total": paginator.count})


def get_image_list_all(request):
    param_filter = get_filter_by_request(request, ImgStore)
    img_list = param_filter.values("img_id", "product_name", "part_no", "batch_no",
                                   "plane_no", "img_content", "create_time", "update_time", "is_basic_img",
                                   "compare_status")
    return JsonResponse({"data": list(img_list)})


def get_compare_result(request):
    param_filter = get_filter_by_request(request, ImgCompareResultV)
    img_list = param_filter.values("img_compare_id", "img_src_id", "img_dst_id", "product_name", "part_no",
                                   "batch_no_src", "batch_no_dst",
                                   "plane_no_src", "plane_no_dst", "result_img", "create_time", "update_time",
                                   "diff_count")
    return JsonResponse({"data": list(img_list)})


def set_basic_img(request):
    img_id = json.loads(request.body).get('img_id')
    if img_id is not None:
        image = ImgStore.objects.get(img_id=img_id)
        compare_result = ImgCompareResult.objects.get(img_dst_id=img_id)
        image_src = ImgStore.objects.get(img_id=compare_result.img_src_id)
        image_src.is_basic_img = False
        image_src.save()
        image.is_basic_img = True
        image.save()
        ret = model_obj_to_dict(image)
        ret["img_content"] = str(ret["img_content"])
        ret.pop("img_feature")
        return JsonResponse({"data": ret})
    else:
        raise RuntimeError("参数错误！")


def image_upload(request):
    image = request.FILES.get("file")
    if image is None:
        return JsonResponse({"msg": "请选择上传文件！"})

    # 缩小图片
    resize_image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    min_width = 1800
    resize_image = cv2.resize(resize_image, dsize=(
        min_width, int(min_width * resize_image.shape[0] / resize_image.shape[1])))
    _, enc_img = cv2.imencode('.jpg', resize_image)
    io = BytesIO(enc_img.tobytes())
    img_file = File(io)
    img_file.name = image.name.split(".")[0] + ".jpg"

    img = ImgStore(product_name=request.POST["product_name"],
                   part_no=image.name.split(".")[0],
                   batch_no=request.POST["batch_no"],
                   plane_no=request.POST["plane_no"],
                   img_content=img_file)

    compare_image(img)
    ret = model_obj_to_dict(img)
    ret["img_content"] = str(ret["img_content"])
    ret.pop("img_feature")
    return JsonResponse({"data": ret})


def delete_image(request):
    img_id = json.loads(request.body).get('img_id')
    if img_id is not None:
        image = ImgStore.objects.get(img_id=img_id)
        image.img_content.delete(False)
        image.delete()
        pass
    else:
        return JsonResponse({"msg": "删除失败！"})
    return JsonResponse({"msg": "删除成功！"})


def get_dict_item(request):
    dict_name_en = request.GET.get("dict_name_en")
    ret = list(SysDict.objects.filter(dict_name_en=dict_name_en))
    data = []
    if len(ret) > 0:
        data = list(SysDictItem.objects.filter(dict_id=ret[0].dict_id).order_by('dict_index'))
    return JsonResponse({"data": [model_obj_to_dict(i) for i in data]})


# def get_camera_stream(request):
#     factory = CameraFactory()
#     msg, camera = factory.get_camera()
#     # 初始化成功
#     if msg == 0:
#         try:
#             return StreamingHttpResponse(gen_display(camera), content_type='multipart/x-mixed-replace; boundary=frame')
#         except Exception as e:
#             camera.close()
#             print(e)
#     else:
#         print(msg)
#         return JsonResponse({"msg": "摄像头打开失败！"})


def gen_display(camera):
    """
    视频流生成器功能。
    """
    while True:
        # 读取图片
        frame = camera.get_image()
        if frame is not None:
            frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # 将图片进行解码
            ret, frame = cv2.imencode('.jpeg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
