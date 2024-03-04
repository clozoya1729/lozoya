from django.contrib import admin

import models


@admin.register(
    models.TSModelMathBase,
    models.TSModelMathSlopIntercept,
    models.TSModelMathLinear,
    models.TSModelMathSine,
    models.TSModelMathBaseGenerator,
    models.TSModelMathRange,
    models.TSModelMathRandomNormal,
    models.TSModelMathRandomUniform,
)
class TSMathAdmin(admin.ModelAdmin):
    pass
