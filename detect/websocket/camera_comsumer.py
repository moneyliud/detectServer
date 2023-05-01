import json
import time
from threading import Thread, Event
from channels.generic.websocket import AsyncWebsocketConsumer, WebsocketConsumer
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from detect.camera.cameraFactory import CameraFactory


class CameraWebsocketConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_name = "camera"
        self.channel_name = "camera"
        self.disconnect_flag = False
        factory = CameraFactory()
        msg, self.camera = factory.get_camera()

    async def connect(self):
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()
        print("websocket connect")
        self.disconnect_flag = False
        self.camera.start()

    async def disconnect(self, close_code):
        print("websocket disconnect")
        self.disconnect_flag = True
        self.camera.event.set()
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        print("receive ", text_data)
        # await self.send(text_data=text_data)

    async def send_image(self, event):
        print("send", time.time())
        await self.send(text_data=None, bytes_data=event["message"])
