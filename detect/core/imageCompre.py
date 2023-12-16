from concurrent.futures.thread import ThreadPoolExecutor

from django.core.files import File

from detect.djangomodels import ImgStore, ImgCompareResult, IMG_COMPARE_STATUS, ImgLabelMsg
from detect.core.diffdetect.missDetect import MissDetect
from detect.core.diffdetect.deepDetect import DeepDetect
from detect.utils.general import xyxy2xywhlt, xywhlt2xyxy
from pickle import dumps
from typing import List
import cv2
import numpy as np
from io import BytesIO


class ImageCompareThreadPool(object):
    def __init__(self):
        self.executor = ThreadPoolExecutor(2)
        self.future_dict = {}

    def is_project_thread_running(self, project_id):
        future = self.future_dict.get(project_id, None)
        if future and future.running():
            return True
        return False

    def check_future(self):
        data = {}
        for project_id, future in self.future_dict.items():
            data[project_id] = future.running()
        return data

    def __del__(self):
        self.executor.shutdown()


image_thread_pool = ImageCompareThreadPool()


def compare_image(image: ImgStore):
    stored_img = ImgStore.objects.filter(product_name=image.product_name, batch_no=image.batch_no,
                                         plane_no=image.plane_no,
                                         part_no=image.part_no)
    if len(stored_img) == 0:
        image.compare_status = IMG_COMPARE_STATUS.COMPARING.value
        image.save()
    else:
        image = stored_img[0]
        if image.compare_status == IMG_COMPARE_STATUS.COMPARING.value:
            raise RuntimeError("图片已在对比中！")
    # compare_image_func(image)
    future = image_thread_pool.executor.submit(compare_image_func, image)
    pass


def label_to_model(label_list, img_id):
    ret_list = []
    for item in label_list:
        label_msg = ImgLabelMsg()
        label_msg.conf = item[4]
        label_msg.label_id = int(item[5])
        label_msg.x, label_msg.y, label_msg.w, label_msg.h = xyxy2xywhlt(item[0:4])
        label_msg.enable = 1
        label_msg.auto_detect = 1
        label_msg.img_id = img_id
        ret_list.append(label_msg)
    return ret_list


def model_to_label(model_list: List[ImgLabelMsg]):
    ret = np.zeros((len(model_list), 8))
    for idx, item in enumerate(model_list):
        x1, y1, x2, y2 = xywhlt2xyxy(np.array([item.x, item.y, item.w, item.h]))
        ret[idx] = [x1, y1, x2, y2, item.conf, item.label_id, item.auto_detect, item.enable]
    return np.array(ret)


def save_image_detect_info(image: ImgStore):
    image_cv = cv2.imdecode(np.frombuffer(image.img_content.file.read(), np.uint8), cv2.IMREAD_COLOR)
    detector = DeepDetect()
    labels = detector.detect_one(image_cv)
    label_msg_list = label_to_model(labels, image.img_id)
    ImgLabelMsg.objects.bulk_create(label_msg_list)


def compare_image_func(image: ImgStore):
    basic_image = ImgStore.objects.filter(product_name=image.product_name, part_no=image.part_no,
                                          is_basic_img=True).exclude(img_id=image.img_id)
    if len(basic_image) == 0:
        image.is_basic_img = True
        image.compare_status = IMG_COMPARE_STATUS.NO_DIFFERENCE.value
        save_image_detect_info(image)
        # image.img_content.save(image.part_no + ".jpg", img_file)
        image.save()
        image.img_content.file.close()
    else:
        src_image = basic_image[0]
        dst_image = image
        dst_image.is_basic_img = False
        dst_image.save()
        # detector = MissDetect()
        detector = DeepDetect()
        src_image_cv = cv2.imdecode(np.frombuffer(src_image.img_content.file.read(), np.uint8), cv2.IMREAD_COLOR)
        dst_image_cv = cv2.imdecode(np.frombuffer(dst_image.img_content.file.read(), np.uint8), cv2.IMREAD_COLOR)
        try:
            tmp = ImgLabelMsg.objects.filter(img_id=src_image.img_id)
            base_label = None
            if tmp is not None and len(tmp) > 0:
                base_label = model_to_label(tmp)
            # detector.save_img = True
            out_img, all_result = detector.detect(src_image_cv, dst_image_cv, base_label)
            if len(all_result) > 0:
                dst_image.compare_status = IMG_COMPARE_STATUS.HAS_DIFFERENCE.value
            else:
                dst_image.compare_status = IMG_COMPARE_STATUS.NO_DIFFERENCE.value
            dst_image.save()
            _, enc_img = cv2.imencode('.jpg', out_img)
            io = BytesIO(enc_img.tobytes())
            img_file = File(io)
            result = ImgCompareResult(img_src_id=src_image.img_id, img_dst_id=dst_image.img_id,
                                      diff_count=len(all_result), compare_result=dumps(all_result))
            result.result_img.save(image.part_no + ".jpg", img_file)
            result.save()
            print("save end")
            src_image.img_content.file.close()
            dst_image.img_content.file.close()
            print("file closed")
        except Exception as e:
            print(e)
            print(e.__traceback__.tb_frame.f_globals["__file__"])
            print(e.__traceback__.tb_lineno)
            src_image.img_content.file.close()
            dst_image.img_content.file.close()
            dst_image.compare_status = IMG_COMPARE_STATUS.COMPARE_ERROR.value
            dst_image.save()
        pass
