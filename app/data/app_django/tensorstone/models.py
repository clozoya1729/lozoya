import os
import pickle
import uuid
import zlib

import django.db.models
import pandas as pd
from django.core.exceptions import ValidationError

from app.data.app_django.tensorstone.configuration import appTypes
from util import ic, ik
from . import configuration, forms, util

objectTypes = [
    ('file', 'File'),
    ('generator', 'Generator'),
    ('mathfunction', 'Math Function'),
    ('regression', 'Regression'),
    ('plot', 'Plot'),
]


class TSModelUser(django.db.models.Model):
    email = django.db.models.EmailField(
        max_length=100,
        primary_key=True,
    )
    admin = django.db.models.BooleanField(
        default=False,
    )

    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'User'


class TSModelServer(django.db.models.Model):
    serverID = django.db.models.UUIDField(
        default=uuid.uuid4,
        blank=True,
        primary_key=True,
    )
    private = django.db.models.BooleanField(
        editable=True,
        default=False,
    )
    users = django.db.models.ManyToManyField(TSModelUser, blank=True, )
    tsServerID = django.db.models.UUIDField(
        default=uuid.uuid4,
        blank=True,
        primary_key=True,
    )
    tsPrivate = django.db.models.BooleanField(
        editable=True,
        default=False,
    )
    tsUsers = django.db.models.ManyToManyField(
        TSModelUser,
        blank=True,
    )

    class Meta:
        verbose_name = 'Server'
        verbose_name_plural = 'Server'


class TSModelDashboard(django.db.models.Model):
    tsServer = django.db.models.ForeignKey(TSModelServer, editable=True, on_delete=django.db.models.CASCADE, )
    tsDashboardID = django.db.models.UUIDField(default=uuid.uuid4, primary_key=True, )
    dashboardName = django.db.models.CharField(max_length=20, blank=False, null=False, )
    users = django.db.models.ManyToManyField(TSModelUser, blank=True, )

    class Meta:
        verbose_name = 'Dashboard'
        verbose_name_plural = 'Dashboard'


class TSModelDashboard(django.db.models.Model):
    dashboardID = django.db.models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = django.db.models.EmailField(blank=True, max_length=100, null=False, )
    testDataFile = django.db.models.TextField(blank=True, max_length=100, null=False, )
    trainDataFile = django.db.models.TextField(blank=True, max_length=100, null=False, )
    appIDs = django.db.models.TextField(blank=True, null=False, )
    appTypes = django.db.models.TextField(blank=True, null=False)
    terms = django.db.models.BooleanField(default=True)
    private = django.db.models.BooleanField(default=False)
    serverID = django.db.models.UUIDField(blank=True, null=True, )

    @property
    def next_n(self):
        return str(len(util.str_to_list(self.appIDs)) + 1)

    def insert_app(self, model):
        setattr(self, 'appIDs', self.appIDs + util.append_list_as_str(self.appIDs, str(model.modelID)), )
        setattr(self, 'appTypes', self.appTypes + util.append_list_as_str(self.appTypes, model.appType), )
        self.save()

    def get_all_app_id(self):
        return util.str_to_list(self.appIDs)

    def get_app_id(self, n):
        return util.str_to_list(self.appIDs)[int(n) - 1]

    def get_app_type(self, n):
        return util.str_to_list(self.appTypes)[int(n) - 1]

    class Meta:
        verbose_name = 'Session'
        verbose_name_plural = 'Sessions'


class PublicDashboardModel(django.db.models.Model):
    def delete_app(self, n):
        appType = self.get_app_type(n)
        appTypes[appType].objects.filter(parent=(self.id), name=n).delete()
        aN, naN = ic(self.appIDs, n)
        aT = ik(self.appTypes, n)
        setattr(self, 'appIDs', naN)
        setattr(self, 'appTypes', aT)
        self.save()
        aN2 = util.str_to_list(aN)[n - 1:]
        aT2 = util.str_to_list(aT)[n - 1:]
        oldIDList = util.get_all_ids(aN2, aT2)
        for ge, ga in zip(aN2, aT2):
            app = appTypes[ga].objects.get(parent=str(self.id), name=ge)
            setattr(app, 'name', int(ge) - 1)
            app.save()
        return oldIDList, aN2

    def replace_app_type(self, n):
        # TODO
        appTypes = self.appTypes
        setattr(self, 'appTypes', appTypes)

    def remove_file(self, filename):
        files = self.files
        if '{},'.format(filename) in files:
            setattr(self, 'files', self.files.replace('{},'.format(filename), ''))
        elif ',{}'.format(filename) in files:
            setattr(self, 'files', self.files.replace(',{}'.format(filename), ''))
        elif '{}'.format(filename) in files:
            setattr(self, 'files', self.files.replace('{}'.format(filename), ''))
        self.save()
        os.remove(os.path.join(configuration.uploadsPath, str(self.id), filename))

    def add_files(self, f):
        new = '' if self.files == '' else ','
        for i, _ in enumerate(f):
            new += _
            if i != len(f) - 1:
                new += ','
        setattr(self, 'files', self.files + new)
        self.save()

    def get_files(self):
        u = self.files.split(',')
        if u == None:
            u = [self.files]
        return u


class TSModelSession(django.db.models.Model):
    """
    files stores all filenames as a comma separated string
    apps stores all app types as a comma separated string
    flowchart stores the flowchart generating instructions as a string
    """
    id = django.db.models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    files = django.db.models.TextField(blank=True)
    appNames = django.db.models.TextField(blank=True)
    appTypes = django.db.models.TextField(blank=True)
    flowchart = django.db.models.TextField(blank=True)

    def replace_app_type(self, n):
        # TODO
        appTypes = self.appTypes
        setattr(self, 'appTypes', appTypes)

    def insert_app(self, n, t):
        setattr(self, 'appNames', self.appNames + util.langer(self.appNames, n))
        setattr(self, 'appTypes', self.appTypes + util.langer(self.appTypes, t))
        self.save()

    def delete_app(self, n):
        appType = self.get_app_type(n)
        appTypes[appType].objects.filter(parent=(self.id), name=n).delete()
        aN, naN = ic(self.appNames, n)
        aT = ik(self.appTypes, n)
        setattr(self, 'appNames', naN)
        setattr(self, 'appTypes', aT)
        self.save()
        aN2 = util.listify(aN)[n - 1:]
        aT2 = util.listify(aT)[n - 1:]
        oldIDList = util.get_all_ids(aN2, aT2)
        for ge, ga in zip(aN2, aT2):
            app = appTypes[ga].objects.get(parent=str(self.id), name=ge)
            setattr(app, 'name', int(ge) - 1)
            app.save()
        return oldIDList, aN2

    def remove_file(self, filename):
        files = self.files
        if '{},'.format(filename) in files:
            setattr(self, 'files', self.files.replace('{},'.format(filename), ''))
        elif ',{}'.format(filename) in files:
            setattr(self, 'files', self.files.replace(',{}'.format(filename), ''))
        elif '{}'.format(filename) in files:
            setattr(self, 'files', self.files.replace('{}'.format(filename), ''))
        self.save()
        os.remove(os.path.join(configuration.uploadsPath, str(self.id), filename))

    def add_files(self, f):
        new = '' if self.files == '' else ','
        for i, _ in enumerate(f):
            new += _
            if i != len(f) - 1:
                new += ','
        setattr(self, 'files', self.files + new)
        self.save()

    def get_files(self):
        u = self.files.split(',')
        if u == None:
            u = [self.files]
        return u

    def get_app_type(self, n):
        appTypes = util.listify(self.appTypes)
        return appTypes[int(n) - 1]

    @property
    def next_n(self):
        return str(len(util.listify(self.appNames)) + 1)


def get_dashboard(dashboardID):
    try:
        return TSModelDashboard.objects.get(dashboardID=dashboardID)
    except Exception as e:
        return e


def get_model(dashboardID):
    try:
        return TSModelDashboard.objects.get(dashboardID=dashboardID)
    except Exception as e:
        return e


class TSModelApp(django.db.models.Model):
    id = django.db.models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )


class TSModelObject(django.db.models.Model):
    tsObjectID = django.db.models.UUIDField(default=uuid.uuid4, primary_key=True, )
    tsObjectType = django.db.models.TextField(choices=configuration.objectTypes, default='file', )
    tsDashboard = django.db.models.ForeignKey(TSModelDashboard, on_delete=django.db.models.CASCADE, )
    tsOutput = django.db.models.BinaryField(blank=True, null=True, )
    _output = {}

    def get_inputs_form(self):
        return forms.TSFormInputs(self)

    def get_output(self):
        if self.tsOutput:
            return pickle.loads(zlib.decompress(self.tsOutput))
        return 'No input'

    def get_settings_form(self):
        return forms.TSFormSettings(self)

    def set_output(self):
        self.tsOutput = zlib.compress(pickle.dumps(self.tsOutput), 9)
        # self.output = json.dumps(self._output)

    def set_output(self, val):
        self.tsOutput = zlib.compress(pickle.dumps(val), 9)

    def set_output(self):
        self.tsOutput = zlib.compress(pickle.dumps(self.tsOutput), 9)

    def save(self, *args, **kwargs):
        super(TSModelObject, self).save()
        util.refresh_inputs_recursively(TSModelAppSingleInput, self.tsObjectID)

    class Meta:
        verbose_name = 'Base/Object'
        verbose_name_plural = 'Base/Object'


class TSModelAppSingleInput(TSModelObject):
    defaultColumn = 'use entire dataframe'
    tsInput = django.db.models.ForeignKey(
        TSModelObject, blank=True, null=True, on_delete=django.db.models.SET_NULL, related_name='+', )
    inputColumns = django.db.models.TextField(blank=False, null=False, default=defaultColumn, )
    outputColumns = django.db.models.TextField(blank=False, null=False, default=defaultColumn, )

    def clean(self, *args, **kwargs):
        if self.tsInput and (self.tsObjectID == self.tsInput.tsObjectID):
            raise ValidationError('An app cannot be its own input.')
        super(TSModelAppSingleInput, self).clean()

    def get_columns_input(self):
        return str(self.inputColumns).split(',')

    def get_columns_output(self):
        return str(self.outputColumns).split(',')

    def get_input(self):
        # return self.input.get_output()
        # return pickle.loads(zlib.decompress(self.input.output))
        # return json.loads(self.input.output)
        if self.tsInput:
            return self.tsInput.get_output()
        return None

    def refresh_input(self):
        print('TSAppSingleInput.refresh_input() not yet implemented')

    def set_output(self, val):
        if isinstance(val, pd.DataFrame):
            columns = self.get_columns_output()
            if not (self.defaultColumn in columns):
                val = val[columns]
        super(TSModelAppSingleInput, self).set_output(val)

    def set_output(self):
        if (type(self.output) == pd.DataFrame):  # == type(pd.DataFrame()) ??
            collies = self.get_columns_output()
            if not (self.defaultColumn in collies):
                self.output = self.output[collies]
        super(TSModelAppSingleInput, self).set_output()

    def save(self, *args, **kwargs):
        super(TSModelAppSingleInput, self).save()

    class Meta:
        verbose_name = 'Base/App/Single-Input'
        verbose_name_plural = 'Base/App/Single-Input'


class TSModelAppMultipleInput(TSModelObject):
    tsInput = django.db.models.ManyToManyField(
        TSModelObject,
        blank=True,
        related_name='+',
    )

    def get_input(self):
        return pickle.loads(self.tsInput.tsOutput)
        # return json.loads(self.input.output)

    class Meta:
        verbose_name = 'App Multiple-Input'
        verbose_name_plural = 'App Multiple-Input'
        verbose_name = 'Base/App/Multiple-Input'
        verbose_name_plural = 'Base/App/Multiple-Input'
