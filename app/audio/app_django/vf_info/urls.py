from django.conf.urls import url
from django.urls import path, re_path

from . import views

app_name = 'info'
urlpatterns = [
    path('contact-us/', views.TSViewContactUs.as_view(), name='contact-us'),
    path('contact/(<uuid:message_id>/', views.TSViewSuccess.as_view()),
    path('credits/', views.TSViewCredits.as_view(), name='credits'),
    path('faq/', views.TSViewFaq.as_view(), name='faq'),
    path('', views.TSViewHome.as_view(), name='home'),
    path('portfolio/', views.TSViewPortfolio.as_view(), name='portfolio'),
    path('privacy/', views.TSViewPrivacy.as_view(), name='privacy'),
    path('services/', views.TSViewServices.as_view(), name='services'),
    path('terms/', views.TSViewTerms.as_view(), name='terms'),
    re_path(r'contact/(?P<message_id>[0-9a-f-]+)/$', views.TSViewSuccess.as_view()),
    re_path(r'^$', views.TSViewHome.as_view(), name='home'),
    url(r'^(?P<message_id>[0-9a-f-]+)/$', views.TSViewSuccess.as_view(), name='success'),
]
