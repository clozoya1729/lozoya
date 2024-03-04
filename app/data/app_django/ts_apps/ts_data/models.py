import os
import uuid

import pandas as pd
from django.db import models

import app.data.app_django.tensorstone.configuration as ts_general_configuration
import app.data.app_django.tensorstone.models as ts_general_models


class TSModelFileBase(ts_general_models.TSModelObject):
    filepath = models.FileField(null=True, blank=True, )

    def __init__(self, *args, **kwargs):
        super(TSModelFileBase, self).__init__(*args, **kwargs)
        self.objectType = 'file'

    @property
    def get_dirname(self):
        return os.path.join(ts_general_configuration.uploadsPath, self.dashboard.dashboardID)

    class Meta:
        verbose_name = 'Base'
        verbose_name_plural = 'Base'


class TSModelFileRead(TSModelFileBase):
    class Meta:
        verbose_name = 'Read'
        verbose_name_plural = 'Read'


class TSModelFileWrite(TSModelFileBase):
    class Meta:
        verbose_name = 'Write'
        verbose_name_plural = 'Write'


class TSModelDataFrameConcat(ts_general_models.TSModelAppMultipleInput):
    df = pd.DataFrame

    class Meta:
        verbose_name = 'Concatenate'
        verbose_name_plural = 'Concatenate'


class TSModelDataPrep(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataFile = models.CharField(max_length=100)
    columns = models.TextField()
    criteria = models.TextField()

    class Meta:
        verbose_name = 'DataPrep Model'
        verbose_name_plural = 'DataPrep Models'


objectTypes = {

}
propertyDict = {

}
