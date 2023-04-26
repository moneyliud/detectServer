import time

from detect.core.diffdetect import missDetect
import os
import cv2
import numpy as np

if __name__ == '__main__':
    path = 'C:/Users/Administrator/Pictures'
    path1 = path + '/微信图片_20230426141142.jpg'
    path2 = path + '/微信图片_20230426141150.jpg'

    # 载入图像
    img1 = cv2.imdecode(np.fromfile(path1, dtype=np.uint8), -1)
    img2 = cv2.imdecode(np.fromfile(path2, dtype=np.uint8), -1)

    detector = missDetect.MissDetect()
    detector.save_img = True
    detector.dir = "test"
    img, diff_left = detector.detect(img1, img2)
    cv2.imwrite('./test/' + os.path.basename(path1).split(".")[0] + '-' +
                os.path.basename(path2).split(".")[0] + '.jpg', img)

    # start_time = time.time()
    # for i in range(10):
    #     sub_start_time = time.time()
    #     img, diff_left = detector.detect(img1, img2)
    #     cv2.imwrite('./test/result' + str(i) + '.jpg', img)
    #     sub_end_time = time.time()
    #     print(str(i) + " time:" + str(sub_end_time - sub_start_time))
    # end_time = time.time()
    # print("total:" + str(end_time - start_time))
