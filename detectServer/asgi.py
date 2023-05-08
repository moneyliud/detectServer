"""
ASGI config for detectServer project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/asgi/
"""

import os
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

import detectServer.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'detectServer.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    'websocket': URLRouter(
        detectServer.routing.websocket_urlpatterns
    )
})

print("asgi")
