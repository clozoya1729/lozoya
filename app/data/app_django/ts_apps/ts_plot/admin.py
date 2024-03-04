import models
from django.contrib import admin


@admin.register(
    models.TSPlotBase,
    models.TSPlotHistogram,
    models.TSPlotScatter,
)
class BasePlotAdmin(admin.ModelAdmin):
    pass
