import django.contrib.admin
from django.contrib import admin

import app.data.app_django.ts_apps.ts_math.models as ts_math_models
import app.data.app_django.ts_apps.ts_plot.models as ts_plot_models
import models


@django.contrib.admin.register(
    models.TSModelUser,
)
class TSUserAdmin(django.contrib.admin.ModelAdmin):
    list_filter = (
        'email',
        'admin',
    )


@django.contrib.admin.register(
    models.TSModelServer,
)
class TSAdminServer(django.contrib.admin.ModelAdmin):
    list_filter = (
        'tsPrivate',
        'tsServerID',
        'users',
    )


@django.contrib.admin.register(
    models.TSModelDashboard,
)
class TSAdminDashboard(django.contrib.admin.ModelAdmin):
    list_filter = (
        'tsDashboardID',
        'tsServer',
        'users',
    )


@django.contrib.admin.register(
    models.TSModelUser,
)
@django.contrib.admin.register(models.TSModelDashboard)
class TSAdminDashboard(django.contrib.admin.ModelAdmin):
    def get_readonly_fields(self, request, obj=None):
        return [f.name for f in self.opts.local_fields]

    list_display = (
        'dashboardID',
        'email',
        'private',
        'serverID',
        'terms',
    )
    list_filter = (
        'dashboardID',
        'email',
        'private',
        'serverID',
        'terms',
    )

    fieldsets = (
        ('User', {
            'fields': (
                'dashboardID',
                'email',
                'private',
                'serverID',
                'terms',
            )
        }),
        ('Files', {
            'fields': (
                'testDataFile',
                'trainDataFile'
            )
        }),
        ('Apps', {
            'fields': (
                'appIDs',
                'appTypes'
            )
        }),
    )


@django.contrib.admin.register(
    models.TSModelObject,
    models.TSModelAppSingleInput,
    models.TSModelAppMultipleInput,
)
class TSAdminAppBase(django.contrib.admin.ModelAdmin):
    pass


@django.contrib.admin.register(
    ts_math_models.TSModelMathBaseGenerator,
    ts_math_models.TSModelMathRange,
    ts_math_models.TSModelMathRandomNormal,
    ts_math_models.TSModelMathRandomUniform,
)
class TSAdminGenerator(django.contrib.admin.ModelAdmin):
    pass


@django.contrib.admin.register(
    ts_math_models.TSModelMathBase,
    ts_math_models.TSModelMathSlopIntercept,
    ts_math_models.TSModelMathSine,
)
class TSAdminMath(django.contrib.admin.ModelAdmin):
    pass


@django.contrib.admin.register(
    ts_plot_models.TSPlotBase,
    ts_plot_models.TSPlotScatter2D,
)
class TSAdminPlot(django.contrib.admin.ModelAdmin):
    pass


# @django.contrib.admin.register(
#     ts_apps.ts_ml.models.TSMLBase,
#     ts_apps.ts_ml.models.TSRegressionBase,
#     ts_apps.ts_ml.regression.models.TSRegressionLinear,
#     ts_apps.ts_ml.regression.models.TSRegressionGP,
#     ts_apps.ts_ml.regression.models.TSRegressionKNN,
#     ts_apps.ts_ml.regression.models.TSRegressionRF,
#     ts_apps.ts_ml.regression.models.TSRegressionSVM,
# )
# class MLAdmin(django.contrib.admin.ModelAdmin):
#     def get_readonly_fields(self, request, obj=None):
#         return [f.name for f in self.opts.local_fields]
#
#     list_filter = (
#         'objectID',
#         'objectType',
#         'dashboard',
#     )
#
#     list_display = (
#         'objectID',
#         'objectType',
#         'dashboard',
#     )
# @django.contrib.admin.register(
#     ts_apps.ts_data.models.TSFileBase,
#     ts_apps.ts_data.models.TSFileRead,
#     ts_apps.ts_data.models.TSFileWrite,
# )
# class TSFileAdmin(django.contrib.admin.ModelAdmin):
#     pass
#
#
# @django.contrib.admin.register(
#     ts_data.models.TSDataFrameConcat,
# )
# class TSDataFrameAdmin(django.contrib.admin.ModelAdmin):
#     pass
# @django.contrib.admin.register(
#     ts_apps.ts_ml.models.TSMLBase,
#     ts_apps.ts_ml.models.TSRegressionBase,
#     ts_apps.ts_ml.regression.models.TSRegressionLinear,
#     ts_apps.ts_ml.regression.models.TSRegressionGP,
#     ts_apps.ts_ml.regression.models.TSRegressionKNN,
#     ts_apps.ts_ml.regression.models.TSRegressionRF,
#     ts_apps.ts_ml.regression.models.TSRegressionSVM,
# )
# class MLAdmin(django.contrib.admin.ModelAdmin):
#     def get_readonly_fields(self, request, obj=None):
#         return [f.name for f in self.opts.local_fields]
#
#     list_filter = (
#         'objectID',
#         'objectType',
#         'dashboard',
#     )
#
#     list_display = (
#         'objectID',
#         'objectType',
#         'dashboard',
#     )
# @django.contrib.admin.register(
#     ts_apps.ts_data.models.TSFileBase,
#     ts_apps.ts_data.models.TSFileRead,
#     ts_apps.ts_data.models.TSFileWrite,
# )
# class TSFileAdmin(django.contrib.admin.ModelAdmin):
#     pass
#
#
# @django.contrib.admin.register(
#     ts_data.models.TSDataFrameConcat,
# )
# class TSDataFrameAdmin(django.contrib.admin.ModelAdmin):
#     pass
# @django.contrib.admin.register(TSSessionModel)
# class TSSessionAdmin(django.contrib.admin.ModelAdmin):
#     pass
# def get_readonly_fields(self, request, obj=None):
#     return [f.name for f in self.opts.local_fields]
#
# list_display = ('id',
#                 'files',
#                 'appNames',
#                 'appTypes',
#                 'flowchart')
# list_filter = ('id',
#                'files',
#                 'appNames',
#                'appTypes',
#                'flowchart')
#
# fieldsets = (
#     ('Files', {
#         'fields': ('files',)
#     }),
#     ('Apps', {
#         'fields': ('appNames',
#                    'appTypes',)
#     }),
#     ('Flowchart', {
#         'fields': ('flowchart',)
#     }),
#
# )


@admin.register(models.TSModelSession)
class TSSessionAdmin(admin.ModelAdmin):
    pass
    # def get_readonly_fields(self, request, obj=None):
    #     return [f.name for f in self.opts.local_fields]
    #
    # list_display = ('id',
    #                 'files',
    #                 'appNames',
    #                 'appTypes',
    #                 'flowchart')
    # list_filter = ('id',
    #                'files',
    #                 'appNames',
    #                'appTypes',
    #                'flowchart')
    #
    # fieldsets = (
    #     ('Files', {
    #         'fields': ('files',)
    #     }),
    #     ('Apps', {
    #         'fields': ('appNames',
    #                    'appTypes',)
    #     }),
    #     ('Flowchart', {
    #         'fields': ('flowchart',)
    #     }),
    #
    # )
