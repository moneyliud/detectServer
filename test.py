import time

from detect.core.diffdetect import missDetect
import os
import cv2
import numpy as np

if __name__ == '__main__':
    path = './'
    path1 = path + '1.jpg'
    path2 = path + '2.jpg'

    # 载入图像
    img1 = cv2.imdecode(np.fromfile(path1, dtype=np.uint8), -1)
    img2 = cv2.imdecode(np.fromfile(path2, dtype=np.uint8), -1)

    detector = missDetect.MissDetect()
    detector.save_img = True
    detector.dir = "test"
    img, diff_left = detector.detect(img1, img2)
    cv2.imwrite('./test/' + os.path.basename(path1).split(".")[0] + '-' +
                os.path.basename(path2).split(".")[0] + '.jpg', img)
