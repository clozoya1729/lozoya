from django.urls import path

import views

app_name = 'dataprep'
urlpatterns = [path(
    'ajax/download_model/',
    views.download_model,
    name='ajax_download_model'
),
    path(
        'ajax/refresh_columns/',
        views.refresh_columns,
        name='ajax_refresh_columns'
    ),
    path(
        'ajax/refresh_controls/',
        views.refresh_controls,
        name='ajax_refresh_controls'
    ),
    path(
        'ajax/refresh_plot/',
        views.refresh_plot,
        name='ajax_refresh_plot'
    ),
    path(
        'ajax/add_column/',
        views.add_column,
        name='ajax_add_column'
    ),
    path(
        'ajax/update_model/',
        views.update_model,
        name='ajax_update_model'
    ),
    path(
        'ajax/update_criteria/',
        views.update_criteria,
        name='ajax_update_model'
    ),
    path(
        '<uuid:id>',
        views.DataPrepAppView.as_view(),
        name='success'
    ),
    path(
        '',
        views.DataPrepFormView.as_view(),
        name='dataprep'
    ),
]
