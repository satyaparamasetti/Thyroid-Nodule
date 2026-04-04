"""
ASGI config for Thyroid_Nodules_Detection_Deep_Reinforcement_learning project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Thyroid_Nodules_Detection_Deep_Reinforcement_learning.settings')

application = get_asgi_application()
