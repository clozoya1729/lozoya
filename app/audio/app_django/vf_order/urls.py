"""django_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.8/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add an import:  from blog import urls as blog_urls
    2. Add a URL to urlpatterns:  url(r'^blog/', include(blog_urls))
"""
from django.conf.urls import url

from . import views

urlpatterns = [url(r'^ajax/validate-coupon/$', views.ajax_validate_coupon, name='ajax-validate-coupon'),
               url(r'^(?P<job_id>[0-9a-f-]+)/wav$', views.results, name='results'),
               url(r'^(?P<job_id>[0-9a-f-]+)/$', views.TSViewSuccessOrder.as_view(), name='success'),
               url(r'', views.TSViewFormOrder.as_view(), name='form'), ]

"""django_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.8/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add an import:  from blog import urls as blog_urls
    2. Add a URL to urlpatterns:  url(r'^blog/', include(blog_urls))
"""
from django.urls import re_path, path

from . import views

app_name = 'order'
urlpatterns = [re_path(r'order/ajax/validate-coupon/$', views.ajax_validate_coupon),
               re_path(r'order/(?P<job_id>[0-9a-f-]+)/wav$', views.results),
               re_path(r'order/(?P<job_id>[0-9a-f-]+)/$', views.TSViewSuccessOrder.as_view()),
               path('', views.TSViewFormOrder.as_view()), ]

"""django_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.8/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add an import:  from blog import urls as blog_urls
    2. Add a URL to urlpatterns:  url(r'^blog/', include(blog_urls))
"""
from django.conf.urls import url

from . import views

urlpatterns = [  # url(r'^$', views.upload_file, name='order')
    url(r'^$', views.TSViewOrder.as_view(), name='order'),
    url(r'success/', views.TSViewSuccess.as_view(), name='success')]
"""django_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.8/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add an import:  from blog import urls as blog_urls
    2. Add a URL to urlpatterns:  url(r'^blog/', include(blog_urls))
"""
from django.conf.urls import url

from . import views

urlpatterns = [url(r'^$', views.TSViewServiceSelection.as_view(), name='service-selection'),
               url(r'mixing/', views.TSViewFormMixing.as_view(), name='mixing-form'),
               url(r'mastering/', views.TSViewFormMastering.as_view(), name='mastering-form'),
               url(r'^(?P<job_id>[0-9a-f-]+)/$', views.TSViewSuccess.as_view(), name='success')]

from django.urls import path

from . import views

app_name = 'order'
urlpatterns = [path('ajax/validate-coupon/', views.ajax_validate_coupon, name='ajax-validate-coupon'),
               path('<uuid:job_id>/wav', views.results, name='success'),
               path('<uuid:job_id>/', views.TSViewSuccessOrder.as_view()),
               path('', views.TSViewFormOrder.as_view(), name='form'), ]
