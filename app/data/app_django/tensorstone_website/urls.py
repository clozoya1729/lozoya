"""tensorstone_website URL Configuration

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

urlpatterns = [
    # path(
    #     '',
    #     include('data.urls')
    # ),
    path(
        'admin/',
        admin.site.urls
    ),
    path(
        '',
        include(
            'ts_info.urls',
            namespace='ts_info'
        )
    ),
    # path(
    #     'ts_data/',
    #     include('ts_data.urls', namespace='ts_data')
    # ),
    # path(
    #     'ts_math/',
    #     include('ts_math.urls', namespace='ts_math')
    # ),
    # path(
    #     'ts_ml/',
    #     include('ts_ml.urls', namespace='ts_ml')
    # ),
    # path(
    #     'ts_plot/',
    #     include('ts_plot.urls', namespace='ts_plot')
    # ),
]
# urlpatterns = urlpatterns + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
