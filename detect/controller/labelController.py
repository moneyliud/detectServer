import json

import django.core.management.base
from detect.djangomodels import ImgLabel, ImgLabelMsg, ImgCompareResultV, SysDict, SysDictItem
from django.http import JsonResponse, StreamingHttpResponse
from detect.utils.convertor import model_obj_to_dict
from detect.utils.filter import get_filter_by_request
from detect.core.imageCompre import compare_image
import cv2
import time
import numpy as np
from io import BytesIO
from django.core.files import File
# from detect.camera.cameraFactory import CameraFactory

# Create your views here.
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

from detect.projector.laserProjector import LaserProjector


def get_label_list(request):
    label_list = ImgLabel.objects.all()
    return JsonResponse({"data": [model_obj_to_dict(i) for i in label_list]})


def save_label_msg(request):
    info = json.loads(request.body)
    img_id = json.loads(info['labelPictureStr'])['pictureId']
    label_msg_list = json.loads(info['labelMsgStr'])
    print(img_id, label_msg_list)
    remove_keys = ["color", "galleryId", "pictureId", "index"]
    item_list = []
    for i in label_msg_list:
        i["img_id"] = img_id
        for key in remove_keys:
            if key in i:
                i.pop(key)
        item = ImgLabelMsg(**i)
        item_list.append(item)

    ImgLabelMsg.objects.filter(img_id=img_id).delete()
    ImgLabelMsg.objects.bulk_create(item_list)
    label_msg_list = ImgLabelMsg.objects.filter(img_id=img_id)
    return JsonResponse({"data": {"rows": [model_obj_to_dict(i) for i in label_msg_list], "message": "保存成功！"}})


def get_label_msg_list(request):
    img_id = request.GET.get("img_id")
    label_msg_list = ImgLabelMsg.objects.filter(img_id=img_id)
    return JsonResponse({"data": [model_obj_to_dict(i) for i in label_msg_list]})
