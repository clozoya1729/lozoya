from django.urls import path

import views

app_name = 'ts_math'
urlpatterns = [
    path('ajax/create_object/', views.create_object, name='ajax_create_math', ),
    path('ajax/delete_object/', views.delete_object, name='ajax_delete_math', ),
    path('ajax/update_input/', views.update_input, name='ajax_update_input_math', ),
    path('ajax/update_object/', views.update_object, name='ajax_update_math', ),
    path('ajax/refresh_columns/', views.refresh_columns, name='ajax_refresh_columns'),
    path('ajax/refresh_plot/', views.refresh_plot, name='ajax_refresh_plot'),
    path('ajax/refresh_controls/', views.refresh_controls, name='ajax_refresh_controls'),
    path('<uuid:id>', views.results.as_view(), name='success'),
    path('', views.StatsAnalysisView.as_view(), name='stats_analysis'),
]
