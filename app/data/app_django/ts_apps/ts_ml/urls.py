import _deprecated.ts_ml.model_actions
from django.urls import path
from django.urls import path
from django.urls import path

from . import views
import ml.util.model_actions
from django.urls import path
import app.data.app_django.ts_apps.ts_plot.views as ts_plot_views
import views
import ts_ml.regression.regression

app_name = 'ts_ml'
urlpatterns = [
    path(
        'ajax/update_controls/',
        _deprecated.ts_ml.model_actions.update_controls,
        name='ajax_update_controls',
    ),
    path(
        'ajax/update_model/',
        _deprecated.ts_ml.model_actions.update_model,
        name='ajax_update_model',
    ),
    path(
        'ajax/refresh_columns/',
        _deprecated.ts_ml.model_actions.refresh_columns,
        name='ajax_refresh_columns',
    ),
    path(
        'ajax/fit_model/',
        _deprecated.ts_ml.model_actions.fit_model,
        name='ajax_fit_model',
    ),
    path(
        'regression/ajax/populate_sidebar/',
        ts_ml.regression.regression.populate_sidebar,
        name='ajax_populate_sidebar',
    ),
    path(
        'regression/ajax/change_sidebar_menu/',
        ts_ml.regression.regression.change_sidebar_menu,
        name='ajax_change_sidebar_menu',
    ),

]

app_name = 'ml'
urlpatterns = [
    path('ajax/download_model/', views.download_model, name='ajax_download_model'),
    path('ajax/refresh_controls/', views.refresh_controls, name='ajax_refresh_controls'),
    path('ajax/update_model/', views.update_model, name='ajax_update_model'),
    path('ajax/refresh_columns/', views.refresh_columns, name='ajax_refresh_columns'),
]

app_name = 'ml'
urlpatterns = [
    # path('ajax/download_model/', ml.util.model_actions.download_model, name='ajax_download_model'),
    path(
        'ajax/update_controls/',
        ml.util.model_actions.update_controls,
        name='ajax_update_controls',
    ),
    path(
        'ajax/update_model/',
        ml.util.model_actions.update_model,
        name='ajax_update_model',
    ),
    path(
        'ajax/refresh_columns/',
        ml.util.model_actions.refresh_columns,
        name='ajax_refresh_columns',
    ),
    path(
        'ajax/fit_model/',
        ml.util.model_actions.fit_model,
        name='ajax_fit_model',
    ),
    path(
        'regression/ajax/populate_sidebar/',
        ml.regression.regression.populate_sidebar,
        name='ajax_populate_sidebar',
    ),
    path(
        'regression/ajax/change_sidebar_menu/',
        ml.regression.regression.change_sidebar_menu,
        name='ajax_change_sidebar_menu',
    ),

]

from django.urls import path

app_name = 'ml'
urlpatterns = [
    path('ajax/download_model/', views.download_model, name='ajax_download_model'),
    path('ajax/refresh_controls/', views.refresh_controls, name='ajax_refresh_controls'),
    path('ajax/update_model/', views.update_model, name='ajax_update_model'),
    path('ajax/refresh_columns/', views.refresh_columns, name='ajax_refresh_columns'),
]
import ts_ml.util.model_actions
from django.urls import path

app_name = 'ts_ml'
urlpatterns = [
    path(
        'ajax/update_controls/',
        ts_ml.util.model_actions.update_controls,
        name='ajax_update_controls',
    ),
    path(
        'ajax/update_model/',
        ts_ml.util.model_actions.update_model,
        name='ajax_update_model',
    ),
    path(
        'ajax/refresh_columns/',
        ts_ml.util.model_actions.refresh_columns,
        name='ajax_refresh_columns',
    ),
    path(
        'ajax/fit_model/',
        ts_ml.util.model_actions.fit_model,
        name='ajax_fit_model',
    ),
    path(
        'regression/ajax/populate_sidebar/',
        ts_ml.regression.regression.populate_sidebar,
        name='ajax_populate_sidebar',
    ),
    path(
        'regression/ajax/change_sidebar_menu/',
        ts_ml.regression.regression.change_sidebar_menu,
        name='ajax_change_sidebar_menu',
    ),

]

app_name = 'regression'
urlpatterns = [
    path('ajax/fit_model/', views.fit_model, name='ajax_fit_model'),
    path('ajax/add_plot/', ts_plot_views.add_plot, name='ajax_add_plot'),
    path('<uuid:id>', views.RegressionAppView.as_view(), name='success'),
    path('', views.RegressionFormView.as_view(), name='regression'),
    # ?
    path('ajax/refresh_plot/', ts_plot_views.refresh_plot, name='ajax_refresh_plot'),
    path('ajax/update_dimensions/', ts_plot_views.update_dimensions, name='ajax_update_dimensions'),
]
