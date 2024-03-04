import json
import uuid

import django.db.models
import django.http
import django.shortcuts
import django.urls
import django.views.generic
import pandas as pd
import os

import tools.ids
from ml.model_settings import modelClasses
from ml.util.util import get_form_a, get_form_s, get_form_x, get_form_y
from tools import path, templates
from tools.generators.button import make_regression_button
from tools.generators.menu import make_options_sidebar, make_settings_sidebar
from tools.generators.util import replacement
from tools.processor import get_xy_choices

import app.data.app_django.ts_apps.ts_ml.forms as ml_forms
import app.data.app_django.ts_apps.ts_plot.forms as plot_forms

import app.data.app_django.tensorstone.configuration
import app.data.app_django.ts_apps.ts_data.models as ts_data_models
import app.data.app_django.ts_apps.ts_math.models as ts_math_models
import app.data.app_django.ts_apps.ts_ml.models as ts_ml_models
import app.data.app_django.ts_apps.ts_plot.models as ts_plot_models
import lozoya.data
from . import configuration, forms, models, util

appMakers = {
    'regression': (make_regression_app, ['n', 'modelChoices', 'project'])
}

baseObjects = models.TSModelObject
dashboards = models.TSModelDashboard
objectTypes = {
    'Data': ts_data_models.objectTypes,
    'Math': ts_math_models.objectTypes,
    'ML':   ts_ml_models.objectTypes,
    'Plot': ts_plot_models.objectTypes,
}
propertyDict = {
    'Data': ts_data_models.propertyDict,
    'Math': ts_math_models.propertyDict,
    'ML':   ts_ml_models.propertyDict,
    'Plot': ts_plot_models.propertyDict,
}
publicServerIDString = 'c7e26f6a-a920-4b22-a8e7-61a06c18f121'
publicServerID = uuid.UUID(publicServerIDString)


def add_session_file(request):
    project = util.get_from('session', request, ('project',))
    file = util.get_from('ajax', request, ('file',))
    obj = models.TSModelSession.objects.get(id=project)
    setattr(obj, 'files', obj.files + file)


def create_object(request):
    ajax = util.get_from(source='ajax', request=request)
    node = json.loads(ajax['node'])
    objTypes = objectTypes[node['family']]
    obj = instantiate_object(node, ajax, dashboards, objTypes)
    return django.http.JsonResponse({'objectID': obj.objectID, })


def create_object(request):
    ajax, node, objTypes = util.get_ajax_data(request)
    obj = instantiate_object(node, ajax, objTypes)
    return django.http.JsonResponse({'tsObjectID': obj.tsObjectID, })


def delete_object(request):
    ajax = get_from(source='ajax', request=request)
    node = json.loads(ajax['node'])
    objTypes = objectTypes[node['family']]
    obj = get_object(node, ajax, dashboards, objTypes)
    obj.delete()
    return django.http.JsonResponse({})


def delete_object(request):
    ajax, node, objTypes = util.get_ajax_data(request)
    obj = get_object(node, ajax, objTypes)
    obj.delete()
    return django.http.JsonResponse({})


def get_ajax_data(request):
    ajax = get_from(source='ajax', request=request)
    node = json.loads(ajax['node'])
    objTypes = objectTypes[node['family']]
    return ajax, node, objTypes


def get_object(node, ajax, objectTypes):
    return objectTypes[node['objType']].objects.get(
        tsDashboard=dashboards.objects.get(tsDashboardID=ajax['dashboardID']), tsObjectID=node['tsObjectID'], )


def get_base_object(ajax):
    return baseObjects.objects.get(tsDashboard=ajax['dashboardID'], tsObjectID=ajax['value'], )


def get_dashboard(tsDashboardID):
    return models.TSModelDashboard.objects.get(tsDashboardID=tsDashboardID)


def get_insert_app_modal(request):
    footerContext = dict()
    footer = util.render_html(
        request=request,
        template=configuration.insertAppFooter,
        context=footerContext, )
    bodyContext = dict(serverID='serverID', insertAppForm=forms.TSFormInsertApp(), )
    body = util.render_html(
        request=request, template=configuration.insertAppBody,
        context=bodyContext, )
    insertAppModalContext = dict(body=body, footer=footer, )
    insertAppModal = util.render_html(
        request=request, template=configuration.insertAppModal,
        context=insertAppModalContext, )
    return insertAppModal


def get_insert_app_modal(request):
    footerContext = dict()
    footer = app.tensorstone.app_django.ts_general.util.render_html(
        request=request,
        template=configuration.insertAppFooter,
        context=footerContext,
    )
    bodyContext = dict(
        serverID='serverID',
        insertAppForm=forms.InsertAppForm(),
    )
    body = render_html(
        request=request,
        template=configuration.insertAppBody,
        context=bodyContext,
    )
    insertAppModalContext = dict(
        body=body,
        footer=footer,
    )
    insertAppModal = render_html(
        request=request,
        template=configuration.insertAppModal,
        context=insertAppModalContext,
    )
    return insertAppModal


def instantiate_object(node, ajax, objectTypes):
    return objectTypes[node['objType']].objects.create(
        tsDashboard=dashboards.objects.get(tsDashboardID=ajax['dashboardID'])
    )


def make_dashboard(request, form, server):
    dashboardModel = form.save(commit=False)
    post = get_from(
        source='post',
        request=request,
    )
    dashboardModel.email = str(post['email'])
    dashboardModel.name = str(post['name'])
    dashboardModel.server = server
    configuration.templateTerms = True if post['terms'] == 'on' else False
    dashboardModel.save()
    return dashboardModel


def make_dashboard(request, form, server):
    dashboardModel = form.save(commit=False)
    email, terms, name = get_from(
        source='post',
        request=request,
        argnames=('email', 'terms', 'name'),
    )
    dashboardModel.email = str(email)
    dashboardModel.name = str(name)
    dashboardModel.server = server
    configuration.templateTerms = True if terms == 'on' else False
    dashboardModel.save()
    return dashboardModel


def populate_sidebar(request, navbar, body):
    objectType, n = get_from(
        source='ajax',
        request=request,
        argnames=('objectType', 'n'),
    )
    return django.http.JsonResponse(
        dict(
            navbar=navbar,
            header='{} {} Settings'.format(objectType.capitalize(), n),
            body=body,
        )
    )


def update_object_input(request):
    ajax = get_from(source='ajax', request=request)
    node = json.loads(ajax['node'])
    objTypes = objectTypes[node['family']]
    obj = ts_util.dashboard.get_object(node, ajax, dashboards, objTypes)
    if ajax['value']:
        other = baseObjects.objects.get(dashboard=ajax['dashboardID'], objectID=ajax['value'])
        obj.input = other
    else:
        obj.input = None
    obj.save()
    return django.http.JsonResponse({'output': str(obj.get_output()), })


def update_object_input(request):
    # TODO CLEANUP
    ajax, node, objTypes = ts_apps.app_util.get_ajax_data(request)
    obj = ts_apps.app_util.get_object(node, ajax, objTypes)
    if ajax['value']:
        other = ts_apps.app_util.get_base_object(ajax)
        obj.tsInput = other
    else:
        obj.tsInput = None
    obj.save()
    return django.http.JsonResponse({'output': str(obj.get_output()), })


def update_object_property(request):
    ajax = get_from(source='ajax', request=request)
    node = json.loads(ajax['node'])
    objTypes = objectTypes[node['family']]
    propDict = propertyDict[node['family']]
    obj = ts_util.dashboard.get_object(node, ajax, dashboards, objTypes)
    if ajax['property'] in propDict.keys():
        setattr(obj, propDict[ajax['property']], ajax['value'])
    else:
        setattr(obj, str(ajax['property']).lower(), ajax['value'])
    obj.save()
    return django.http.JsonResponse({'output': str(obj.get_output()), })


def update_object_property(request):
    # TODO CLEANUP
    corrector = {django.db.models.IntegerField: int, django.db.models.FloatField: float, }
    ajax, node, objTypes = ts_apps.app_util.get_ajax_data(request)
    obj = ts_apps.app_util.get_object(node, ajax, objTypes)
    prop = str(ajax['property'].lower())
    field = obj._meta.get_field(prop)
    if type(field) in corrector.keys():
        val = corrector[type(field)](ajax['value'])
    else:
        val = ajax['value']
    print(field, type(field))
    print(val, type(val))
    setattr(obj, prop, val)
    obj.save()
    return django.http.JsonResponse({'output': str(obj.get_output()), })


def update_sidebars(request):
    ajax = get_from(source='ajax', request=request)
    node = json.loads(ajax['node'])
    objTypes = objectTypes[node['family']]
    obj = ts_util.dashboard.get_object(node, ajax, dashboards, objTypes)
    return django.http.JsonResponse({'output': str(obj.get_output()), })


def update_sidebars(request):
    # TODO CLEANUP
    ajax, node, objTypes = ts_apps.app_util.get_ajax_data(request)
    obj = ts_apps.app_util.get_object(node, ajax, objTypes)
    obj.compute()  # TODO is this necessary?
    out = obj.get_output()
    if isinstance(out, pd.DataFrame):
        # TODO make interactive table
        out = out.to_html()
    return django.http.JsonResponse(
        {
            'headingLeft': '{} Settings'.format(node['objType']), 'headingRight': '{} Output'.format(node['objType']),
            'leftUpper':   str(obj.get_inputs_form()), 'leftLower': str(obj.get_settings_form()),
            'rightUpper':  str(out),
            'rightLower':  '',
        }
    )


def update_session(request):
    project = get_from('session', request, ('project',))
    field, value = get_from('ajax', request, ('field', 'value'))
    obj = models.TSModelSession.objects.get(id=project)
    setattr(obj, field, value)
    return django.http.JsonResponse({})


class TSViewDashboard(django.views.generic.TemplateView):
    template_name = configuration.dashboardBody

    def get_file(self, dirname):
        return os.path.join(configuration.uploadsPath, dirname, next(os.walk(dirname))[2][0])

    def get_context_data(self, **kwargs):
        context = super(TSViewDashboard, self).get_context_data(**kwargs)
        dashboardID = str(self.kwargs['dashboardID'])
        dirname = os.path.join(configuration.uploadsPath, dashboardID)
        trainDataFile, testDataFile = get_from(
            source='session',
            request=self.request,
            argnames=('trainDataFile', 'testDataFile'),
        )
        dirname = os.path.join(configuration.uploadsPath, dashboardID)
        trainDataFile = self.request.session['trainDataFile']
        trainPath = os.path.join(dirname, trainDataFile)
        x, y = lozoya.data_api.get_xy_cols(trainPath)
        self.request.session['columns'] = [col[0] for col in x]
        self.request.session['filename'] = os.path.basename(os.path.normpath(trainDataFile))
        self.request.session['dashboardID'] = dashboardID
        publicDashboardModel = models.get_model(dashboardID=dashboardID)
        apps = publicDashboardModel.get_all_app_id()
        appsList = []
        if len(apps) == 0:
            regressionApp, regressionModel = get_ml_app(
                request=self.request,
                dashboardID=dashboardID,
                trainDataFile=trainDataFile,
                testDataFile=testDataFile,
                n=0,
                appType='regression',
                trainPath=os.path.join(dirname, trainDataFile)
            )
            publicDashboardModel.insert_app(regressionModel)
            appsList.append(regressionApp)
        else:
            for i, app in enumerate(apps):
                regressionApp, regressionModel = get_ml_app(
                    request=self.request,
                    appID=app,
                    dashboardID=dashboardID,
                    trainDataFile=trainDataFile,
                    testDataFile=testDataFile,
                    n=i,
                    appType='regression',
                    trainPath=os.path.join(dirname, trainDataFile)
                )
                appsList.append(regressionApp)
        insertAppModal = get_insert_app_modal(self.request)
        navbarContext = dict(
            serverID='serverID',
            insertAppModal=insertAppModal,
        )
        navbar = render_html(
            request=self.request,
            template=configuration.dashboardNavbar,
            context=navbarContext,
        )
        context.update(
            dict(
                apps=appsList,
                filename=os.path.basename(os.path.normpath(trainDataFile)).split('.')[0],
                navbar=navbar,
            )
        )
        return context

    def get_context_data(self, **kwargs):
        context = super(TSViewDashboard, self).get_context_data(**kwargs)
        dashboardID = str(self.kwargs['dashboardID'])
        self.request.session['dashboardID'] = dashboardID
        publicDashboardModel = get_dashboard(tsDashboardID=dashboardID)
        navbarContext = dict(dashboardName=publicDashboardModel.dashboardName, )
        navbar = ts_util.general.render_html(
            request=self.request, template=configuration.dashboardNavbar,
            context=navbarContext, )
        context.update(dict(navbar=navbar, name=publicDashboardModel.dashboardName, ))
        return context

    def get_context_data(self, **kwargs):
        context = super(TSViewDashboard, self).get_context_data(**kwargs)
        dashboardID = str(self.kwargs['dashboardID'])
        self.request.session['dashboardID'] = dashboardID
        publicDashboardModel = get_dashboard(dashboardID=dashboardID)
        appsList = []
        navbarContext = dict(serverID='serverID', name=publicDashboardModel.name, )
        navbar = util.render_html(
            request=self.request, template=configuration.dashboardNavbar,
            context=navbarContext, )
        context.update(dict(apps=appsList, navbar=navbar, name=publicDashboardModel.name, ))
        return context

    def get_context_data(self, **kwargs):
        context = super(TSViewDashboard, self).get_context_data(**kwargs)
        dashboardID = str(self.kwargs['dashboardID'])
        self.request.session['dashboardID'] = dashboardID
        publicDashboardModel = get_dashboard(dashboardID=dashboardID)
        appsList = []
        insertAppModal = get_insert_app_modal(self.request)
        navbarContext = dict(serverID='serverID', insertAppModal=insertAppModal, )
        navbar = util.render_html(
            request=self.request, template=configuration.dashboardNavbar,
            context=navbarContext, )
        context.update(dict(apps=appsList, navbar=navbar, name=publicDashboardModel.name, ))
        return context


class TSViewFormCreateDashboardPrivate(django.views.generic.FormView):
    form_class = forms.CreateSessionForm
    template_name = configuration.createPublic

    def get_context_data(self, **kwargs):
        context = super(django.views.generic.FormView, self).get_context_data(**kwargs)
        return context

    def get_success_url(self, **kwargs):
        kwargs = dict(
            dashboardID=self.publicDashboardModel.dashboardID
        )
        return django.urls.reverse('general:dashboard', kwargs=kwargs)

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        self.form = self.get_form(form_class)
        self.publicDashboardModel = self.form.save(commit=False)
        email, terms = get_from(
            source='post',
            request=request,
            argnames=('email', 'terms'),
        )
        trainFile, testFile = get_from(
            source='files',
            request=request,
            argnames=('trainDataFile', 'testDataFile',),
        )
        self.files = [trainFile, testFile]
        self.request.session['email'] = str(email)
        self.request.session['trainDataFile'] = str(trainFile)
        self.request.session['testDataFile'] = str(testFile)
        self.publicDashboardModel.email = str(email)
        self.publicDashboardModel.trainDataFile = str(trainFile)
        self.publicDashboardModel.testDataFile = str(testFile)
        configuration.templateTerms = True if terms == 'on' else False
        self.publicDashboardModel.private = True
        self.publicDashboardModel.serverID = ''
        self.publicDashboardModel.save()
        app.tensorstone.app_django.ts_general.util.upload_files(self.files, str(self.publicDashboardModel.dashboardID))
        return super(django.views.generic.FormView, self).form_valid(self.form)


class TSViewFormCreateServerPrivate(django.views.generic.FormView):
    form_class = forms.CreateSessionForm
    template_name = configuration.createPrivate

    def get_context_data(self, **kwargs):
        context = super(django.views.generic.FormView, self).get_context_data(**kwargs)
        return context

    def get_success_url(self, **kwargs):
        kwargs = dict(
            dashboardID=self.privateDashboardModel.dashboardID
        )
        return django.urls.reverse('general:server', kwargs=kwargs)

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        self.form = self.get_form(form_class)
        self.privateDashboardModel = self.form.save(commit=False)
        email, terms = get_from(
            source='post',
            request=request,
            argnames=('email', 'terms'),
        )
        self.request.session['email'] = str(email)
        self.privateDashboardModel.email = str(email)
        configuration.templateTerms = True if terms == 'on' else False
        self.privateDashboardModel.save()
        return super(django.views.generic.FormView, self).form_valid(self.form)

    def form_is_valid(self):
        return super(django.views.generic.FormView, self).form_valid(self.form)

    def form_is_invalid(self, request, errs):
        return django.shortcuts.render(
            request, configuration.createPublic, {'form': self.form, 'errs': errs}
        )


class TSViewFormCreateDashboardPublic(django.views.generic.FormView):
    form_class = forms.CreateSessionForm
    template_name = configuration.createPublic

    def get_context_data(self, **kwargs):
        context = super(django.views.generic.FormView, self).get_context_data(**kwargs)
        return context

    def get_success_url(self, **kwargs):
        kwargs = dict(
            dashboardID=self.publicDashboardModel.dashboardID
        )
        return django.urls.reverse('general:dashboard', kwargs=kwargs)

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        self.form = self.get_form(form_class)
        self.publicDashboardModel = self.form.save(commit=False)
        email, terms = get_from(
            source='post',
            request=request,
            argnames=('email', 'terms'),
        )
        trainFile, testFile = get_from(
            source='files',
            request=request,
            argnames=('trainDataFile', 'testDataFile',),
        )
        self.files = [trainFile, testFile]
        self.request.session['email'] = str(email)
        self.request.session['trainDataFile'] = str(trainFile)
        self.request.session['testDataFile'] = str(testFile)
        self.publicDashboardModel.email = str(email)
        self.publicDashboardModel.trainDataFile = str(trainFile)
        self.publicDashboardModel.testDataFile = str(testFile)
        configuration.templateTerms = True if terms == 'on' else False
        self.publicDashboardModel.save()
        configuration.upload_files(self.files, str(self.publicDashboardModel.dashboardID))
        return super(django.views.generic.FormView, self).form_valid(self.form)


class TSViewFormCreatePublic(django.views.generic.FormView):
    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        self.form = self.get_form(form_class)
        self.publicDashboardModel = self.form.save(commit=False)
        email, terms = get_post_data(request, ('email', 'terms'))
        trainFile, testFile = get_session_files(request, ('trainDataFile', 'testDataFile',))
        self.files = [trainFile, testFile]
        self.request.session['email'] = str(email)
        self.request.session['trainDataFile'] = str(trainFile)
        self.request.session['testDataFile'] = str(testFile)
        self.publicDashboardModel.email = str(email)
        self.publicDashboardModel.trainDataFile = str(trainFile)
        self.publicDashboardModel.testDataFile = str(testFile)
        configuration.templateTerms = True if terms == 'on' else False
        errs = app.tensorstone.app_django.ts_general.util.validate_order_form(self.files)
        if self.form.is_valid():
            if errs:
                return self.form_is_invalid(request, errs)
            self.publicDashboardModel.save()
            app.tensorstone.app_django.ts_general.util.upload_files(self.files, str(self.publicDashboardModel.id))
            return self.form_is_valid()
        else:
            return self.form_is_invalid(request, errs)

    def form_is_valid(self):
        return super(django.views.generic.FormView, self).form_valid(self.form)

    def form_is_invalid(self, request, errs):
        return django.shortcuts.render(
            request, configuration.createPublic, {'form': self.form, 'errs': errs}
        )


class TSViewLogin(django.views.generic.TemplateView):
    template_name = configuration.login


class TSViewResetPasswordRequest(django.views.generic.TemplateView):
    template_name = configuration.resetPasswordRequest


class TSViewFormCreateDashboard(django.views.generic.FormView):
    """
    Base form for creating a dashboard.
    Contains default values for creating a public dashboard.
    Override form_class, server, and template_name to creates private dashboard.
    """
    dashboardModel = None
    form_class = None
    server = None
    template_name = None

    def get_context_data(self, **kwargs):
        context = super(TSViewFormCreateDashboard, self).get_context_data(**kwargs)
        return context

    def get_success_url(self, **kwargs):
        kwargs = dict(tsDashboardID=self.dashboardModel.dashboardID)
        return django.urls.reverse('data:dashboard', kwargs=kwargs)

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        self.form = self.get_form(form_class)
        self.dashboardModel = make_dashboard(request=request, form=self.form, server=self.server, )
        return self.form_valid(self.form)


class TSViewFormCreateDashboard(django.views.generic.FormView):
    """
    Base form for creating a dashboard.
    Contains default values for creating a public dashboard.
    Override form_class, server, and template_name to creates private dashboard.
    """
    try:
        dashboardModel = None
        form_class = ts_general_forms.TSFormCreateSession
        server = ts_server_models.TSModelServer.objects.get(
            serverID=publicServerID
        )  # THIS CAUSES AN ERROR BECAUSE DJANGO IS STUPID
        template_name = configuration.createPublic
    except:
        pass

    def get_context_data(self, **kwargs):
        context = super(TSViewFormCreateDashboard, self).get_context_data(**kwargs)
        return context

    def get_success_url(self, **kwargs):
        kwargs = dict(dashboardID=self.dashboardModel.dashboardID)
        return django.urls.reverse('general:dashboard', kwargs=kwargs)

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        self.form = self.get_form(form_class)
        self.dashboardModel = make_dashboard(request=request, form=self.form, server=self.server, )
        return super(self.__class__, self).form_valid(self.form)


class TSViewFormCreateDashboardPublic(TSViewFormCreateDashboard):
    form_class = ts_general_forms.TSFormCreateSession
    template_name = configuration.createPublic

    def post(self, request, *args, **kwargs):
        self.server = ts_server_models.TSModelServer.objects.get(serverID=publicServerID)
        return super(self.__class__, self).post(request, *args, **kwargs)


class TSViewFormCreateDashboardPublic(TSViewFormCreateDashboard):
    """
    Form for creating public dashboard.
    Parent class contains default values.
    """
    pass


class TSViewFormCreateDashboardPrivate(django.views.generic.FormView):
    """
    Form for creating private dashboard.
    Overrides form_class, server, and template_name.
    """
    form_class = ts_general_forms.TSFormCreateSession  # todo
    template_name = configuration.createPublic  # todo

    def post(self, request, *args, **kwargs):
        self.server = ''  # todo
        return super(self.__class__, self).post(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        self.server = ''  # todo
        super(self.__class__, self).post(self, request, *args, **kwargs)


class TSViewFormCreateServerPrivate(django.views.generic.FormView):  # TODO

    """
    Class for creating a private server.
    """
    pass


class TSViewLogIn(django.views.generic.TemplateView):
    template_name = configuration.login


class TSViewPasswordResetRequest(django.views.generic.TemplateView):
    template_name = configuration.resetPasswordRequest


class TSViewServer(django.views.generic.TemplateView):
    template_name = configuration.dashboardBody

    def get_file(self, dirname):
        return os.path.join(
            configuration.uploadsPath, dirname, next(os.walk(dirname))[2][0]
        )

    # def get_context_data(self, **kwargs):
    #     context = super(ServerView, self).get_context_data(**kwargs)
    #     dashboardID = str(self.kwargs['dashboardID'])
    #     dirname = os.path.join(util.configuration.uploadsPath, dashboardID)
    #     trainDataFile = self.request.session['trainDataFile']
    #     trainPath = os.path.join(dirname, trainDataFile)
    #     x, y = lozoya.data_api.get_xy_cols(trainPath)
    #     self.request.session['columns'] = [col[0] for col in x]
    #     self.request.session['filename'] = os.path.basename(os.path.normpath(trainDataFile))
    #     self.request.session['dashboardID'] = dashboardID
    #     regressionApp = ml.util.makeapp.get_ml_app(
    #         self.request,
    #         1,
    #         'regression',
    #         x,
    #         y,
    #     )
    #     # scatterApp = pmake.get_plot_app_body(self.request, 2, 'scatter', x, y)
    #     # histogramApp = pmake.get_plot_app_body(self.request, 3, 'histogram', x, y)
    #     context['app'] = [regressionApp]  # , scatterApp, histogramApp]
    #     context['filename'] = os.path.basename(os.path.normpath(trainDataFile)).split('.')[0]
    #     return context


class TSViewFormSession(django.views.generic.FormView):
    form_class = forms.TSFormSession
    template_name = configuration.analysisForm

    def get_context_data(self, **kwargs):
        context = super(django.views.generic.FormView, self).get_context_data(**kwargs)
        return context

    def get_success_url(self, **kwargs):
        return django.urls.reverse('ts_session:success', kwargs={'id': self.modelo.id})

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        self.form = self.get_form(form_class)
        self.modelo = self.form.save(commit=False)
        self.modelo.save()
        return self.form_valid(self.form)


class TSViewSession(django.views.generic.TemplateView):
    template_name = configuration.analysisApp

    def get_context_data(self, **kwargs):
        context = super(TSViewSession, self).get_context_data(**kwargs)
        project = str(self.kwargs['id'])
        self.request.session['project'] = project
        context['projectID'] = project
        obj = models.TSModelSession.objects.get(id=project)
        appNames = obj.appNames
        if appNames != '':
            aT = obj.appTypes
            a = []
            for n, t in zip(appNames.split(','), aT.split(',')):
                a.append(
                    ts.make_app(
                        obj, project, n, t, **configuration.appTypes[t].creation_kwargs()
                    )[0]
                )
            context['sessionApps'] = a
        context['flowchartURL'] = os.path.join(configuration.uploadsPath, project, 'flowchart.jpg')
        context['fileSidebar'] = ts.make_sidebar(
            'Files', 'file-title', 'file-sidebar', 'file-body',
            'file-sidebar-button', 'Files',
            "btn files-button open-sidebar settings",
            "icon glyphicon glyphicon-file",
            'close-file-sidebar',
            ts.make_file_menu(
                buttons.format_files(obj.get_files())
            )
        )
        return context


class TSViewSignUp(django.views.generic.TemplateView):
    template_name = configuration.signup


def make_plot_app(obj, f, project, n, **kwargs):
    appCode = replacement(
        os.path.join(path.templatesPath, templates.plotApp),
        {
            '%n%':          str(n),
            '%plot%':       str(kwargs['plot']),
            '%axes%':       str(plot_forms.TSFormAxesSelect(n, 2, kwargs['columns'])),
            '%dimensions%': str(plot_forms.TSFormDimensionSelect(n)),
            '%type%':       str(plot_forms.PlotTypeSelectForm(n)),
            '%axArea%':     tools.ids.axArea.format(n),
            '%dimArea%':    tools.ids.dimArea.format(n),
            '%plotArea%':   tools.ids.plot.format(n),
            '%typeArea%':   tools.ids.plotTypeArea.format(n),
        }
    )
    return make_new_app(obj, n, 'plot', '', appCode, **{})


def make_regression_app(obj, f, project, n, **kwargs):
    modelObject, created = modelClasses['regression'].objects.get_or_create(parent=project, name=n)
    ch = get_xy_choices(f)
    appCode = replacement(
        os.path.join(path.templatesPath, templates.regressionApp),
        {
            '%n%':               str(n),
            '%modelX%':          str(
                ml_forms.TSFormIndependentVariableSelect(n, ch, ch[0][0], initial=get_form_x(modelObject))
            ),
            '%modelY%':          str(
                ml_forms.TSFormDependentVariableSelect(
                    n,
                    [v for v in ch if
                     v[0] not in modelObject.x.split(',')],
                    ch[0][0],
                    initial=get_form_y(modelObject)
                )
            ),
            '%modelScaler%':     str(ml_forms.TSFormScalerSelect(n, initial=get_form_s(modelObject))),
            '%modelAlgorithms%': str(ml_forms.TSFormAlgorithmSelect(n, initial=get_form_a(modelObject))),
            '%yContainer%':      tools.ids.yContainer.format(n),
        }
    )
    buttons = make_regression_button(n)
    return make_new_app(obj, n, 'regression', buttons, appCode, **kwargs)


def make_new_app(obj, n, appType, buttons='', appCode='', **kwargs):
    options = make_options_sidebar(obj, n, appType)
    settings = make_settings_sidebar(obj, n, appType, **kwargs)
    return replacement(
        os.path.join(path.templatesPath, templates.appContainer),
        {
            '%n%':           n,
            '%appType%':     appType,
            '%appButtons%':  buttons,
            '%appOptions%':  options,
            '%appSettings%': settings,
            '%appCode%':     appCode,
            '%appName%':     tools.ids.appName.format(n),
            '%appDistrict%': tools.ids.appDistrict.format(n),
            '%appHeading%':  tools.ids.appHeading.format(n),
            '%accordionID%': tools.ids.appAccordion.format(n),
            '%appZone%':     tools.ids.appZone.format(n),
        }
    )


def make_app(obj, project, n, appType, **kwargs):
    dirname = os.path.join(path.uploadsPath, project)
    f = obj.get_files()
    if len(f) > 0 and f[0] != '':
        a = appMakers[appType]
        return a[0](obj, os.path.join(dirname, f[0]), project, n, **kwargs), ''
    return None, 'You must upload at least one file before using an app.'


def get_app_container(request, n, appType, body):
    template = tool.general.path.appContainer
    context = dict(
        accordionID='{}-accordion-{}'.format(appType, n),
        appID='{}-area-{}'.format(appType, n),
        appHeading='{}-accordion-heading-{}'.format(appType, n),
        appType='{} {}'.format(appType.capitalize(), n),
        body=body,
        href='{}-accordion-{}'.format(appType, n), n=n, )
    return tool.general.util.render_html(request, template, context)


def instantiate_object(node, ajax, dashboards, objectTypes):
    return objectTypes[node['objType']].objects.create(
        dashboard=dashboards.objects.get(dashboardID=ajax['dashboardID'])
    )


def get_object(node, ajax, dashboards, objectTypes):
    return objectTypes[node['objType']].objects.get(
        dashboard=dashboards.objects.get(dashboardID=ajax['dashboardID']),
        objectID=node['objectID'], )


def populate_sidebar(request, navbar, body):
    ajax = ts_util.general.get_from(source='ajax', request=request, )
    return django.http.JsonResponse(
        dict(navbar=navbar, header='{} {} Settings'.format(ajax['objectType'].capitalize(), ajax['n']), body=body, )
    )


def populate_sidebar(request, navbar, body):
    objectType, n = ts_util.general.get_from(source='ajax', request=request, argnames=('objectType', 'n'), )
    return django.http.JsonResponse(
        dict(navbar=navbar, header='{} {} Settings'.format(objectType.capitalize(), n), body=body, )
    )


def action_hero(obj, id):
    subbestclass = list(ts_util.general.get_all_subclasses(obj.__class__))
    for subclass in subbestclass:
        if not ('Base' in subclass.__name__):
            results = subclass.objects.filter(objectID=id)
            if results:
                return results


def get_app_container(request, n, objectType, body):
    template = ts_util.general.path.appContainer
    context = dict(
        accordionID='{}-accordion-{}'.format(objectType, n), objectID='{}-area-{}'.format(objectType, n),
        appHeading='{}-accordion-heading-{}'.format(objectType, n),
        objectType='{} {}'.format(objectType.capitalize(), n), body=body,
        href='{}-accordion-{}'.format(objectType, n), n=n, )
    return ts_util.general.render_html(request, template, context)


def make_dashboard(request, form, server):
    dashboardModel = form.save(commit=False)
    email, terms, name = ts_util.general.get_from(source='post', request=request, argnames=('email', 'terms', 'name'), )
    dashboardModel.email = str(email)
    dashboardModel.name = str(name)
    dashboardModel.server = server
    app.tensorstone.app_django.ts_general.configuration.templateTerms = True if terms == 'on' else False
    dashboardModel.save()
    return dashboardModel


def get_insert_app_modal(request):
    footerContext = dict()
    footer = ts_util.general.render_html(
        request=request, template=ts_util.path.insertAppFooter, context=footerContext, )
    bodyContext = dict(serverID='serverID', insertAppForm=ts_general.forms.TSFormInsertApp(), )
    body = ts_util.general.render_html(request=request, template=ts_util.path.insertAppBody, context=bodyContext, )
    insertAppModalContext = dict(body=body, footer=footer, )
    insertAppModal = ts_util.general.render_html(
        request=request, template=ts_util.path.insertAppModal,
        context=insertAppModalContext, )
    return insertAppModal


################################################################################
def get_ajax_data(request, argnames):
    q = []
    for arg in argnames:
        q.append(lozoya.data.smart(str(request.GET.get(arg, None))))
    return q if len(q) > 1 else q[0]


def get_ajax_data(request):
    return dict(request.GET)


def get_from(source, request, argnames):
    return sources[source](request, argnames)


def get_ajax_data(request):
    return dict(request.GET)


def get_from(source, request, argnames, asList=False):
    q = []
    for arg in argnames:
        if source == 'ajax':
            q.append(lozoya.data.smart(str(request.GET.get(arg, None))))
        elif source == 'session':
            q.append(request.session[arg])
        elif source == 'post':
            q.append(request.POST.get(arg))
        elif source == 'files':
            q.append(request.FILES.get(arg))
    if asList:
        return q
    return q if len(q) > 1 else q[0]


def get_from(source, request, argnames):
    if source == 'ajax':
        return get_ajax_data(request, argnames)
    elif source == 'session':
        return get_session_data(request, argnames)
    elif source == 'post':
        return get_post_data(request, argnames)
    elif source == 'files':
        return get_session_files(request, argnames)


def get_from(source, request):
    _ = sources[source](request)
    return {k: _[k][0] for k in _}


def get_from(source, request):
    sources = dict(
        ajax=dict(request.GET), files=dict(request.FILES), post=dict(request.POST),
        session=dict(request.session), )
    _ = sources[source]
    return {k: _[k][0] for k in _}


def get_post_data(request):
    return dict(request.POST)


def get_post_data(request, argnames):
    q = []
    for arg in argnames:
        q.append(request.POST[arg])
    return q if len(q) > 1 else q[0]


def get_request_data(request, argnames):
    q = []
    for arg in argnames:
        q.append(request.session[arg])
    return q if len(q) > 1 else q[0]


def get_session_data(request, argnames):
    q = []
    for arg in argnames:
        q.append(request.session[arg])
    return q if len(q) > 1 else q[0]


def get_session_data(request):
    return dict(request.session)


def get_session_files(request):
    return dict(request.FILES)


def get_session_files(request, filenames):
    q = []
    for f in filenames:
        q.append(request.FILES.get(f))
    return q if len(q) > 1 else q[0]


def render_html(request, template, context):
    return render(request, template, context).content.decode('utf-8')
