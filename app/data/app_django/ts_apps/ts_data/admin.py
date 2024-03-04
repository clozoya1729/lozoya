from django.contrib import admin

import models


@admin.register(models.TSModelFileBase, models.TSModelFileRead, models.TSModelFileWrite, )
class TSFileAdmin(admin.ModelAdmin):
    pass


@admin.register(models.TSModelDataFrameConcat, )
class TSDataFrameAdmin(admin.ModelAdmin):
    pass


@admin.register(models.TSModelDataPrep)
class TSDataPrepAdmin(admin.ModelAdmin):
    def get_readonly_fields(self, request, obj=None):
        return [f.name for f in self.opts.local_fields]

    list_display = ('id', 'dataFile',)
    list_filter = ('id', 'dataFile',)
    fieldsets = (
        ('Files', {'fields': ('dataFile',)}),
        ('Columns', {'fields': ('columns',)}),
        ('Criteria', {'fields': ('criteria',)}),
    )
