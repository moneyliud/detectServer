from concurrent.futures.thread import ThreadPoolExecutor

from django.core.files import File

from detect.djangomodels import ImgStore, ImgCompareResult, IMG_COMPARE_STATUS
from detect.core.diffdetect.missDetect import MissDetect
from detect.core.diffdetect.deepDetect import DeepDetect
from pickle import dumps
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


def compare_image_func(image: ImgStore):
    basic_image = ImgStore.objects.filter(product_name=image.product_name, part_no=image.part_no,
                                          is_basic_img=True).exclude(img_id=image.img_id)
    if len(basic_image) == 0:
        image.is_basic_img = True
        image.compare_status = IMG_COMPARE_STATUS.NO_DIFFERENCE.value
        # image.img_content.save(image.part_no + ".jpg", img_file)
        image.save()
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
            out_img, all_result = detector.detect(src_image_cv, dst_image_cv)
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
