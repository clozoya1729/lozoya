import django.contrib.admin

from .regression import models as regression_models
from . import models


@django.contrib.admin.register(
    models.TSBaseML,
    regression_models.TSBaseRegression,
    regression_models.TSRegressionLinear,
    regression_models.TSRegressionGP,
    regression_models.TSRegressionKNN,
    regression_models.TSRegressionRF,
    regression_models.TSRegressionSVM,
)
class MLAdmin(django.contrib.admin.ModelAdmin):
    def get_readonly_fields(self, request, obj=None):
        return [f.name for f in self.opts.local_fields]

    list_filter = (
        'objectID',
        'objectType',
        'dashboard',
    )
    list_display = (
        'objectID',
        'objectType',
        'dashboard',
    )


@django.contrib.admin.register(regression_models.TSRegressionModel)
class RegressionAdmin(django.contrib.admin.ModelAdmin):
    def get_readonly_fields(self, request, obj=None):
        return [f.name for f in self.opts.local_fields]

    list_display = (
        'modelID',
        'dashboardID',
        # 'trainDataFile',
        # 'testDataFile'
    )
    list_filter = (
        'modelID',
        'dashboardID',
        # 'trainDataFile',
        # 'testDataFile'
    )

    fieldsets = (
        ('User', {
            'fields': (
                'modelID',
                'dashboardID',
                'n',
            ),
        }),
        ('Files', {
            'fields': (
                'testDataFile',
                'trainDataFile',
            )
        }),
        ('Regression', {
            'fields': (
                'x',
                'y',
                'scaler',
                'algorithms',
                'fit'
            )
        }),
        ('Plot', {
            'fields': (
                'plotDim',
                'plotx',
                'ploty',
                'plotz',
            )
        }),
        ('Linear', {
            'fields': (
                'lrFitIntercept',
                'lrNormalize',
            ),
        }),
        ('Gaussian Process', {
            'fields': (
                'gpAlpha',
                'gpNRestartsOptimizer',
                'gpNormalizeY',
                'gpRandomState',
            )
        }),
        ('K Nearest Neighbors', {
            'fields': (
                'knnNNeighbors',
                'knnWeights',
                'knnAlgorithm',
                'knnLeafSize',
                'knnP',
                'knnDistanceMetric',
            )
        }),
        ('Random Forest', {
            'fields': (
                'rfNEstimators',
                'rfCriterion',
                'rfMaxDepth',
                'rfMinSamplesSplit',
                'rfMinSamplesLeaf',
                'rfMinWeightFractionLeaf',
                'rfMaxFeatures',
                'rfMaxLeafNodes',
                'rfMinImpurityDecrease',
                'rfBootstrap',
                'rfOutOfBag',
                'rfRandomState',
                'rfWarmStart',
            )
        }),
        ('Support Vector Machine', {
            'fields': (
                'svmKernel',
                'svmDegree',
                'svmGamma',
                'svmCoef0',
                'svmTolerance',
                'svmC',
                'svmEpsilon',
                'svmShrinking',
                'svmMaximumIterations',
            )
        })
    )


from django.contrib import admin

from models import TSRegressionModel


@admin.register(TSRegressionModel)
class TSRegressionAdmin(admin.ModelAdmin):
    def get_readonly_fields(self, request, obj=None):
        return [f.name for f in self.opts.local_fields]

    list_display = ('id', 'trainDataFile', 'testDataFile')
    list_filter = ('id', 'trainDataFile', 'testDataFile')

    fieldsets = (('Files', {'fields': ('testDataFile', 'trainDataFile')}),
                 ('Regression', {
                     'fields': ('x', 'y', 'scaler', 'algorithms')
                 }),
                 ('Linear', {
                     'fields': ('lrFitIntercept', 'lrNormalize')
                 }),
                 ('Gaussian Process', {
                     'fields': ('gpAlpha', 'gpNRestartsOptimizer', 'gpNormalizeY', 'gpRandomState')
                 }),
                 ('K Nearest Neighbors', {
                     'fields': (
                         'knnNNeighbors', 'knnWeights', 'knnAlgorithm', 'knnLeafSize', 'knnP', 'knnDistanceMetric')
                 }),
                 ('Random Forest', {
                     'fields': (
                         'rfNEstimators', 'rfCriterion', 'rfMaxDepth', 'rfMinSamplesSplit', 'rfMinSamplesLeaf',
                         'rfMinWeightFractionLeaf', 'rfMaxFeatures', 'rfMaxLeafNodes', 'rfMinImpurityDecrease',
                         'rfBootstrap', 'rfOutOfBag', 'rfRandomState', 'rfWarmStart')
                 }),
                 ('Support Vector Machine', {
                     'fields': (
                         'svmKernel', 'svmDegree', 'svmGamma', 'svmCoef0', 'svmTolerance', 'svmC', 'svmEpsilon',
                         'svmShrinking',
                         'svmMaximumIterations')
                 }))
