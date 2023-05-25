from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import re_path
from detect.websocket import camera_comsumer

websocket_urlpatterns = [
    # re_path(r"ws/camera/", camera_comsumer.CameraWebsocketConsumer.as_asgi()),
]
