from abc import abstractmethod
import cv2
from channels.layers import get_channel_layer
from threading import Thread, Event
from asgiref.sync import async_to_sync
import time


class CameraInterface:
    def __init__(self):
        self.run_flag = True
        self.event = Event()
        self.camera_thread = Thread(target=self.start_get_image, args=(self.event,))
        self.channel_layer = get_channel_layer()

    def start(self):
        self.camera_thread.start()

    def start_get_image(self, event):
        while True:
            if event.is_set():
                self.close()
                print("camera closed!")
                break
            # 读取图片
            frame = self.get_image()
            if frame is not None:
                frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # 将图片进行解码
                ret, frame = cv2.imencode('.jpeg', frame)
                print(time.time())
                async_to_sync(self.channel_layer.group_send)(
                    "camera",
                    {
                        'type': 'send_image',
                        'message': frame.tobytes(),
                        'valid': True
                    }
                )
                # for i in range(4):
                #     self.send_none()

    def send_none(self):
        async_to_sync(self.channel_layer.group_send)(
            "camera",
            {
                'type': 'send_image',
                'message': None,
                'valid': False
            }
        )

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def get_image(self):
        pass

    @abstractmethod
    def close(self):
        pass
