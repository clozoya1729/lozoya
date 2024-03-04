import django.urls
from . import views

app_name = 'ts_info'
urlpatterns = [
    django.urls.path('about/', views.AboutView.as_view(), name='about', ),
    django.urls.path('credits/', views.CreditsView.as_view(), name='credits', ),
    django.urls.path('faq/', views.FaqView.as_view(), name='faq', ),
    django.urls.path('', views.HomeView.as_view(), name='', ),
    django.urls.path('privacy/', views.PrivacyView.as_view(), name='privacy', ),
    django.urls.path('services/', views.ServicesView.as_view(), name='services', ),
    django.urls.path('terms/', views.TermsView.as_view(), name='terms', ),
    django.urls.path('tutorials/', views.TutorialsView.as_view(), name='tutorials', ),
]
