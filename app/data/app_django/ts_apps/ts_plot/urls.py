import views
from django.urls import path

app_name = 'ts_plot'
urlpatterns = [
    path('ajax/add_plot/', views.add_plot, name='ajax_add_plot'),
    path('ajax/create_object/', views.create_object, name='ajax_create_math', ),
    path('ajax/delete_object/', views.delete_object, name='ajax_delete_math', ),
    path('ajax/update_input/', views.update_input, name='ajax_update_input_math', ),
    path('ajax/update_object/', views.update_object, name='ajax_update_math', ),
    path('ajax/refresh_plot/', views.refresh_plot, name='ajax_refresh_plot', ),
    path('ajax/update_axes/', views.update_axes, name='ajax_update_axes', ),
    path('ajax/update_dimensions/', views.update_dimensions, name='ajax_update_dimensions', ),
]
