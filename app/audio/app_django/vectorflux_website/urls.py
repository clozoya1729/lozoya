"""vectorflux_website URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import include
from django.contrib import admin
from django.urls import path
from . import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('contact/', include('contact.urls', namespace='contact')),
    path('', include('home.urls')),
    path('info/', include('info.urls', namespace='info')),
    path('legal/', include('legal.urls', namespace='legal')),
    path('order/', include('order.urls', namespace='order')),
]
if settings.DEBUG == True:
    pass  # urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
