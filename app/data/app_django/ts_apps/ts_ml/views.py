import copy
import os
import uuid

import app.data.app_django.tensorstone.ts_dashboard.views as ts_dashboard_views
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.static import serve
from ml.util.general_ml import mlModels
from ml.util.ids import histogramArea, scatterArea
from ml.util.model_actions import fit_loop, get_fit_parameters, save_model, validate_fit
from ml.views import MLAppView, MLFormView
from plotter.util import plotter
from tensorstone.classification.forms import *
from tools import path
from tools.processor import get_xy

import app.data.app_django.tensorstone.configuration
import app.data.app_django.tensorstone.configuration as ts_general_configuration
import app.data.app_django.tensorstone.forms as ts_general_forms
import app.data.app_django.tensorstone.util as ts_general_util
import app.data.app_django.ts_apps.ts_ml.forms as ts_ml_forms
import app.data.app_django.ts_apps.ts_ml.views as ts_ml_views
import lozoya.data
import lozoya.data
import lozoya.plot
from app.data.app_django.ts_apps.ts_ml.util import fit_loop, get_fit_parameters, save_model, validate_fit
from .forms import *
from .models import ClassificationModel, TSRegressionModel
from .util import choices

# LIST OF ALL EXTENSIONS ASSOCIATED WITH EACH FILE TYPE
EXTENSIONS = {
    'csv': ['csv', 'txt'], 'excel': ['xls', 'xlsx', 'xlsm', 'xltx', 'xltm'], 'hdf5': ['hdf5'], 'mat': ['mat'],
    'wav': ['wav']
}
supportedExt = ['wav', 'mp3', 'txt', 'csv']
sources = dict(
    ajax=ts_general_util.get_ajax_data,
    files=ts_general_util.get_session_files,
    post=ts_general_util.get_post_data,
    session=ts_general_util.get_session_data,
)
fileButtonStyle = "cursor: pointer; padding: 0rem;"
colors = ["#00AFBB", "#ff9f00", "#CC79A7", "#009E73", "#66ccff",  # Not as distinguishable with #00AFBB for colorblind
          "#F0E442"]


class RegressionAppView(ts_ml_views.TSViewMLApp):
    template_name = ts_general_configuration.analysisApp
    modelChoices = configuration.modelChoices


class RegressionFormView(ts_ml_views.TSViewFormML):
    form_class = RegressionForm
    template_name = ts_general_configuration.analysisForm
    analysisType = 'Regression'


def change_sidebar_menu(request):
    objectType, menu, n = ts_ml_views.get_from(
        source='ajax',
        request=request,
        argnames=('objectType', 'menu', 'n'),
    )
    body = menus[menu](request, objectType, n)
    return JsonResponse({'body': body, })


def fit_model(request):
    n = ts_ml_views.get_ajax_data(request, ('n',))
    project = request.session['project']
    modelObject = TSRegressionModel.objects.get(id=project)
    datapath = os.path.join(ts_general_configuration.uploadsPath, project, modelObject.trainDataFile)
    x, y, scaler, a = get_fit_parameters(modelObject)
    errors = validate_fit(a, x, y)
    if errors:
        return JsonResponse({'errors': errors})
    x, y, algorithms = x.split(','), y.split(',')[0], a.split(',')
    xData, yData = lozoya.data_api.get_xy(datapath, x, y)
    yFits = fit_loop(algorithms, project, xData, yData, scaler, mlModels, TSRegressionModel)
    plot = lozoya.plot_api.plot2d(xData, yData, yFits, x, y)  # TODO
    save_model(request.session['project'], x, xData, yData, y, yFits)  # TODO
    histogramData = ''  # get_histogram_data(path.fitPath.format(project), x, y, yData)#TODO
    histogram = ''  # plotter.histogram(histogramData, y)
    return JsonResponse(
        {
            'plot':        plot, 'histogram': histogram, 'errors': errors, 'plotID': pids.plot.format(n),
            'histogramID': pids.plot.format(int(n) + 1),
        }
    )


def get_app_menu(request, objectType, n):
    template = ts_general_configuration.regressionAppMenu
    objectType, n = ts_ml_views.general.get_from(source='ajax', request=request, argnames=('objectType', 'n'), )
    dashboardID = ts_ml_views.get_from(source='session', request=request, argnames=('dashboardID',), )
    modelillo = 'GaussianProcessRegression'
    model = ts_ml_views.get_model(dashboardID, n)
    kwargs = ts_ml_views.get_model_kwargs(modelName=modelillo, model=model, )
    context = dict(
        objectType=objectType, n=n,
        appInputForm=ts_general_forms.AppInputForm(n, choices=[('a', 'a'), ('b', 'b')], initial=kwargs, ),
        appOutputForm=ts_general_forms.AppOutputForm(n, choices=[('a', 'a'), ('b', 'b')], initial=kwargs, ),
        replaceAppForm=ts_general_forms.AppReplaceForm(n, choices=[('a', 'a'), ('b', 'b')], initial=kwargs, ), )
    return ts_ml_views.render_html(request, template, context)


def get_hp_menu(request, objectType, n):
    template = ts_general_configuration.regressionHPMenu
    objectType, n = ts_ml_views.get_from(source='ajax', request=request, argnames=('objectType', 'n'), )
    dashboardID = ts_ml_views.get_from(source='session', request=request, argnames=('dashboardID',), )
    modelName = 'GaussianProcessRegression'
    model = ts_ml_views.get_model(dashboardID, n)
    kwargs = ts_ml_views.get_form_kwargs(model=model, modelName=modelName)
    context = dict(
        model='Gaussian Process', modelSettingsSelect=ts_ml_forms.TSFormModelSettingsSelect(objectType, n),
        modelSettingsForm=ts_ml_forms.modelSettingsForms[modelName](objectType, n, initial=kwargs), )
    return ts_ml_views.render_html(request, template, context)


def get_settings_menu(request, objectType, n):
    template = ts_general_configuration.regressionSettingsMenu
    dashboardID, trainDataFile = ts_ml_views.get_from(
        source='session', request=request, argnames=('dashboardID', 'trainDataFile'), )
    dirname = os.path.join(ts_general_configuration.uploadsPath, dashboardID)
    trainPath = os.path.join(dirname, trainDataFile)
    x, y = lozoya.data_api.get_xy_cols(trainPath)
    model = ts_ml_views.get_model(dashboardID, n)
    parameters = ts_ml_views.get_fit_parameters(model)
    context = dict(
        modelX=ts_ml_forms.TSFormXSelect(objectType=objectType, n=n, choices=x, initial=parameters['x']),
        modelY=ts_ml_forms.TSFormYSelect(objectType=objectType, n=n, choices=y, initial=parameters['y']),
        modelScaler=ts_ml_forms.TSFormScalerSelect(objectType=objectType, n=n, initial=parameters['scaler']),
        modelSelect=ts_ml_forms.TSFormModelSelect(objectType=objectType, n=n, initial=parameters['algorithms']),
    )
    context = dict(
        modelX=ts_ml_forms.TSFormIndependentVariableSelect(
            objectType=objectType, n=n, choices=x, initial=parameters['x']
        ),
        modelY=ts_ml_forms.TSFormDependentVariableSelect(appType=objectType, n=n, choices=y, initial=parameters['y']),
        modelScaler=ts_ml_forms.TSFormScalerSelect(objectType=objectType, n=n, initial=parameters['scaler']),
        modelSelect=ts_ml_forms.TSFormModelSelect(appType=objectType, n=n, initial=parameters['algorithms']),
    )
    return ts_ml_views.render_html(request, template, context)


def populate_sidebar(request):
    objectType, n = ts_ml_views.get_from(source='ajax', request=request, argnames=('objectType', 'n'), )
    navbarContext = dict(
        options=[dict(
            text='Regression',
            onclick="change_sidebar_menu('ts_ml', 'regression', 'regressionSettingsMenu', {});".format(n), ), dict(
            text='HP', onclick="change_sidebar_menu('ts_ml', 'regression', 'regressionHPMenu', {});".format(n), ), dict(
            text='General',
            onclick="change_sidebar_menu('ts_ml', 'regression', 'regressionAppMenu', {});".format(n), ), ],
        groupID='sidebar_navbar_option',
        n=n,
    )
    navbar = ts_ml_views.render_html(request, ts_general_configuration.sidebarNavbar, navbarContext)
    body = get_settings_menu(request, objectType, n)
    return ts_dashboard_views.populate_sidebar(request, navbar, body)


menus = {
    'regressionAppMenu':      get_app_menu,
    'regressionHPMenu':       get_hp_menu,
    'regressionSettingsMenu': get_settings_menu,
}


def fit_model(request):
    n = get_from('ajax', request, ('n',))
    project = request.session['project']
    modelObject = ClassificationModel.objects.get(id=project)
    datapath = os.path.join(path.uploadsPath, project, modelObject.trainDataFile)
    x, y, scaler, a = get_fit_parameters(modelObject)
    errors = validate_fit(a, x, y)
    if errors:
        return JsonResponse({'errors': errors})
    x, y, algorithms = x.split(','), y.split(',')[0], a.split(',')
    xData, yData = get_xy(datapath, x, y)
    yFits = fit_loop(algorithms, project, xData, yData, scaler, mlModels, ClassificationModel)
    save_model(request.session['project'], xData, yData, x, y, copy.copy(yFits))  # TODO
    plot = plotter.get_scatter(x, y, xData, yData, yFits)
    histogram = plotter.get_histogram(x, y, yData, project)
    return JsonResponse(
        {
            'errors':        errors,
            'scatterArea':   scatterArea.format(n),
            'scatter':       plot,
            'histogramArea': histogramArea.format(n),
            'histogram':     histogram,
        }
    )


class RegressionAppView(MLAppView):
    template_name = app.tensorstone.app_django.ts_general.configuration.analysisApp
    modelChoices = choices.modelChoicesClassification
    analysisType = 'Classification'


class RegressionFormView(MLFormView):
    form_class = ClassificationForm
    template_name = app.tensorstone.app_django.ts_general.configuration.analysisForm
    analysisType = 'Classification'


#################################################################################
def action_hero(obj, id):
    subbestclass = list(get_all_subclasses(obj.__class__))
    for subclass in subbestclass:
        if not ('Base' in subclass.__name__):
            results = subclass.objects.filter(tsObjectID=id)
            if results:
                return results


def clean_model_field(field):
    fieldList = field.split('-')
    f = fieldList[-2]
    n = fieldList[-1]
    return f, n


def download_model(request):
    project = request.session['project']
    filepath = os.path.join(path.uploadsPath, project, 'fit.csv')
    return serve(request, os.path.basename(filepath), os.path.dirname(filepath))


def fit_loop(algorithms, dashboardID, modelID, X, y, scaler, models, MLModel):
    X = X.values
    y = y.values
    scaler = scalers[scaler]
    if scaler:
        scaler = scaler()
        X = scaler.fit_transform(X)
    yFits = {}
    for model in algorithms:
        kwargs = get_model_kwargs(dashboardID, modelID, smart(model), MLModel)
        regressor = models[model](**kwargs)
        regressor.fit(X, y)
        yFits[modelNames[smart(model)]] = regressor.predict(X)
    return yFits


def fit_loop(parameters, model, data, models):
    scaler = ml.util.general.scalers[parameters['scaler']]
    X, y = data['x'].values, data['y'].values
    if scaler:
        scaler = scaler()
        X = scaler.fit_transform(X)
    yFits = {}

    for modelName in parameters['algorithms']:
        kwargs = ml.util.general.get_model_kwargs(modelName=tool.general.util.smart(modelName), model=model, )
        learner = models[modelName](**kwargs)
        learner.fit(X, y)
        yFits[ml.regression.choices.modelNames[tool.general.util.smart(modelName)]] = learner.predict(X)
    return yFits


def fit_model(request):
    n = tool.general.util.get_from(source='ajax', request=request, argnames=('n',), )
    dashboardID = tool.general.util.get_from(source='session', request=request, argnames=('dashboardID',), )
    model = get_model(dashboardID, n)
    datapath = os.path.join(tool.general.path.uploadsPath, dashboardID, model['modelObject'].trainDataFile)
    parameters = ml.util.general.get_fit_parameters(model)
    errors = ml.util.general.validate_fit(parameters)
    if errors:
        return JsonResponse({'errors': errors})
    data = tool.app.processor.get_xy(datapath, parameters)
    yFits = fit_loop(parameters=parameters, model=model, data=data, models=ml.util.general.mlModels, )
    model['modelObject'].fit = True
    model['modelObject'].save()
    plot = plotter.util.plotter.plot2d(data=data, parameters=parameters, yFits=yFits, )  # TODO
    ml.util.general.save_model(dashboardID=dashboardID, parameters=parameters, data=data, yFits=yFits, )  # TODO
    return JsonResponse({'errors': errors, 'plot': plot, 'plotID': tool.app.id.plotContainer.format(n), })


def fit_model(request):
    n = get_from(
        source='ajax',
        request=request,
        argnames=('n',),
    )
    dashboardID = request.session['dashboardID']
    publicDashboardModel = DashboardModel.objects.get(dashboardID=dashboardID)
    modelID = publicDashboardModel.get_app_id(n)
    modelType = publicDashboardModel.get_app_type(n)
    modelClass = modelClasses[modelType]
    modelObject = modelClass.objects.get(modelID=modelID, dashboardID=dashboardID)
    datapath = os.path.join(path.uploadsPath, dashboardID, modelObject.trainDataFile)
    x, y, scaler, a = get_fit_parameters(modelObject)
    errors = validate_fit(a, x, y)
    if errors:
        return JsonResponse({'errors': errors})
    x, y, algorithms = x.split(','), y.split(',')[0], a.split(',')
    xData, yData = get_xy(datapath, x, y)
    yFits = fit_loop(algorithms, dashboardID, modelID, xData, yData, scaler, mlModels, modelClass)
    modelObject.fit = True
    modelObject.save()
    plot = plotter.plot2d(xData, yData, yFits, x, y)  # TODO
    save_model(request.session['dashboardID'], x, xData, yData, y, yFits)  # TODO
    return JsonResponse(
        {
            'errors': errors,
            'plot':   plot,
            'plotID': ids.plotContainer.format(n),
        }
    )


def fit_model(request):
    n = tool.general.util.get_from(source='ajax', request=request, argnames=('n',), )
    dashboardID = tool.general.util.get_from(source='session', request=request, argnames=('dashboardID',), )
    model = get_model(dashboardID, n)
    datapath = os.path.join(tool.general.path.uploadsPath, dashboardID, model['modelObject'].trainDataFile)
    parameters = ts_ml.util.general.get_fit_parameters(model)
    errors = ts_ml.util.general.validate_fit(parameters)
    if errors:
        return JsonResponse({'errors': errors})
    data = tool.app.processor.get_xy(datapath, parameters)
    model['modelObject'].fit_model()
    plot = ts_plot.util.plotter.plot2d(data=data, parameters=parameters, yFits=model['modelObject'].yFits, )  # TODO
    ts_ml.util.general.save_model(
        dashboardID=dashboardID, parameters=parameters, data=data,
        yFits=model['modelObject'].yFits, )  # TODO
    return JsonResponse({'errors': errors, 'plot': plot, 'plotID': tool.app.id.plotContainer.format(n), })


def fit_model(request):
    n = tool.general.get_from(source='ajax', request=request, argnames=('n',), )
    dashboardID = tool.general.get_from(source='session', request=request, argnames=('dashboardID',), )
    model = get_model(dashboardID, n)
    datapath = os.path.join(tool.path.uploadsPath, dashboardID, model['modelObject'].trainDataFile)
    parameters = _deprecated.ts_ml.general.get_fit_parameters(model)
    errors = _deprecated.ts_ml.general.validate_fit(parameters)
    if errors:
        return JsonResponse({'errors': errors})
    data = tool.processor.get_xy(datapath, parameters)
    model['modelObject'].fit_model()
    plot = _deprecated.ts_plot._plotter.plot2d(
        data=data, parameters=parameters,
        yFits=model['modelObject'].yFits, )  # TODO
    _deprecated.ts_ml.general.save_model(
        dashboardID=dashboardID, parameters=parameters, data=data,
        yFits=model['modelObject'].yFits, )  # TODO
    return JsonResponse({'errors': errors, 'plot': plot, 'plotID': _deprecated.id.plotContainer.format(n), })


def fit_model(request):
    n = tool.general.get_from(source='ajax', request=request, argnames=('n',), )
    dashboardID = tool.general.get_from(source='session', request=request, argnames=('dashboardID',), )
    model = get_model(dashboardID, n)
    datapath = os.path.join(tool.path.uploadsPath, dashboardID, model['modelObject'].trainDataFile)
    parameters = _deprecated.py.ml_general.get_fit_parameters(model)
    errors = _deprecated.py.ml_general.validate_fit(parameters)
    if errors:
        return JsonResponse({'errors': errors})
    data = tool.processor.get_xy(datapath, parameters)
    model['modelObject'].fit_model()
    plot = _deprecated.py.plot_plotter.plot2d(
        data=data, parameters=parameters,
        yFits=model['modelObject'].yFits, )  # TODO
    _deprecated.py.ml_general.save_model(
        dashboardID=dashboardID, parameters=parameters, data=data,
        yFits=model['modelObject'].yFits, )  # TODO
    return JsonResponse({'errors': errors, 'plot': plot, 'plotID': _deprecated.py.id.plotContainer.format(n), })


def get_ml_app(request, dashboardID, trainDataFile, testDataFile, objectType, n, trainPath, objectID=None):
    mlModel, created = _deprecated.ts_ml.general.modelClasses[objectType].objects.get_or_create(
        objectID=uuid.uuid4() if objectID == None else objectID, dashboardID=dashboardID, n=n,
        testDataFile=testDataFile, trainDataFile=trainDataFile, )
    dashboardID = request.session['dashboardID']
    trainDataFile = request.session['trainDataFile']
    dirname = os.path.join(tool.path.uploadsPath, dashboardID)
    trainPath = os.path.join(dirname, trainDataFile)
    x, y = lozoya.data_api.get_xy_cols(trainPath)
    initialAxes = dict(x=mlModel.plotx, y=mlModel.ploty, z=mlModel.plotz, )
    initialDim = dict(dimensions=mlModel.plotDim, )
    mlModel.save()
    plot = _deprecated.ts_plot._makeapp.get_plot_body_and_settings(
        request=request, objectType=objectType, n=n,
        plotType='scatter', x=x, initialAxes=initialAxes,
        initialDim=initialDim, )
    context = dict(buttons=get_ml_buttons(request=request, objectType=objectType, n=n, ), body=plot, )
    app = app.tensorstone.app_django.ts_general.util.app.get_app_container(
        request=request, n=n, objectType=objectType,
        body=tool.general.render_html(
            request, template=tool.path.appBody,
            context=context, ), )
    return app, mlModel


def get_ml_app(request, dashboardID, trainDataFile, testDataFile, objectType, n, trainPath, objectID=None):
    mlModel, created = _deprecated.py.ml_general.modelClasses[objectType].objects.get_or_create(
        objectID=uuid.uuid4() if objectID == None else objectID, dashboardID=dashboardID, n=n,
        testDataFile=testDataFile, trainDataFile=trainDataFile, )
    dashboardID = request.session['dashboardID']
    trainDataFile = request.session['trainDataFile']
    dirname = os.path.join(tool.path.uploadsPath, dashboardID)
    trainPath = os.path.join(dirname, trainDataFile)
    x, y = lozoya.data_api.get_xy_cols(trainPath)
    initialAxes = dict(x=mlModel.plotx, y=mlModel.ploty, z=mlModel.plotz, )
    initialDim = dict(dimensions=mlModel.plotDim, )
    mlModel.save()
    plot = _deprecated.py.plot_makeapp.get_plot_body_and_settings(
        request=request, objectType=objectType, n=n,
        plotType='scatter', x=x, initialAxes=initialAxes,
        initialDim=initialDim, )
    context = dict(buttons=get_ml_buttons(request=request, objectType=objectType, n=n, ), body=plot, )
    return mlModel


def get_ml_app(request, dashboardID, trainDataFile, testDataFile, appType, n, trainPath, appID=None):
    mlModel, created = ml.util.general.modelClasses[appType].objects.get_or_create(
        modelID=uuid.uuid4() if appID == None else appID,
        dashboardID=dashboardID,
        n=n,
        testDataFile=testDataFile,
        trainDataFile=trainDataFile,
    )
    dashboardID = request.session['dashboardID']
    trainDataFile = request.session['trainDataFile']
    dirname = os.path.join(tool.general.path.uploadsPath, dashboardID)
    trainPath = os.path.join(dirname, trainDataFile)
    x, y = tool.app.processor.get_xy_cols(trainPath)
    initialAxes = dict(
        x=mlModel.plotx,
        y=mlModel.ploty,
        z=mlModel.plotz,
    )
    initialDim = dict(
        dimensions=mlModel.plotDim,
    )
    mlModel.save()
    plot = plotter.util.makeapp.get_plot_body_and_settings(
        request=request,
        appType=appType,
        n=n,
        plotType='scatter',
        x=x,
        initialAxes=initialAxes,
        initialDim=initialDim,
    )
    context = dict(
        buttons=get_ml_buttons(
            request=request,
            appType=appType,
            n=n,
        ),
        body=plot,
    )
    app = app.tensorstone.app_django.ts_general.util.app.get_app_container(
        request=request,
        n=n,
        appType=appType,
        body=tool.general.util.render_html(
            request,
            template=tool.general.path.appBody,
            context=context,
        ),
    )
    return app, mlModel


def get_ml_app(request, dashboardID, trainDataFile, testDataFile, objectType, n, trainPath, objectID=None):
    mlModel, created = ts_ml.util.general.modelClasses[objectType].objects.get_or_create(
        objectID=uuid.uuid4() if objectID == None else objectID,
        dashboardID=dashboardID,
        n=n,
        testDataFile=testDataFile,
        trainDataFile=trainDataFile,
    )
    dashboardID = request.session['dashboardID']
    trainDataFile = request.session['trainDataFile']
    dirname = os.path.join(tool.general.path.uploadsPath, dashboardID)
    trainPath = os.path.join(dirname, trainDataFile)
    x, y = tool.app.processor.get_xy_cols(trainPath)
    initialAxes = dict(
        x=mlModel.plotx,
        y=mlModel.ploty,
        z=mlModel.plotz,
    )
    initialDim = dict(
        dimensions=mlModel.plotDim,
    )
    mlModel.save()
    plot = ts_plot.util.makeapp.get_plot_body_and_settings(
        request=request,
        objectType=objectType,
        n=n,
        plotType='scatter',
        x=x,
        initialAxes=initialAxes,
        initialDim=initialDim,
    )
    context = dict(
        buttons=get_ml_buttons(
            request=request,
            objectType=objectType,
            n=n,
        ),
        body=plot,
    )
    app = app.tensorstone.app_django.ts_general.util.app.get_app_container(
        request=request,
        n=n,
        objectType=objectType,
        body=tool.general.util.render_html(
            request,
            template=tool.general.path.appBody,
            context=context,
        ),
    )
    return app, mlModel


def get_ml_buttons(request, objectType, n):
    template = tool.path.mlButtons
    context = dict(objectType=objectType, n=n, )
    return tool.general.render_html(request, template, context)


def get_ml_buttons(request, appType, n):
    template = tool.general.path.mlButtons
    context = dict(
        appType=appType,
        n=n,
    )
    return tool.general.util.render_html(request, template, context)


def get_ml_buttons(request, objectType, n):
    template = tool.general.path.mlButtons
    context = dict(
        objectType=objectType,
        n=n,
    )
    return tool.general.util.render_html(request, template, context)


def get_fit_parameters(modelObject):
    return modelObject.x, modelObject.y, modelObject.scaler, modelObject.algorithms


def get_form_a(mlModel):
    return {'algorithms': [v for v in getattr(mlModel, 'algorithms').split(',')]}


def get_form_attr(dashboardID, modelID, MLModel, attr):
    mlModel = MLModel.objects.get(modelID=modelID, dashboardID=dashboardID)
    a = getattr(mlModel, attr)
    if len(a.split(',')) != 1:
        a = a.split(',')
    kwargs = {attr: a}
    return kwargs


def get_form_s(mlModel):
    return {'scaler': [v for v in getattr(mlModel, 'scaler').split(',')]}


def get_form_y(mlModel):
    kwargs = {
        'y': [v for v in getattr(mlModel, 'y').split(',')
              if v not in getattr(mlModel, 'x').split(',')]
    }
    mlModel.y = smart(str(kwargs['y']))
    mlModel.save()
    return kwargs


def get_form_kwargs(dashboardID, modelID, model, MLModel):
    mlModel = MLModel.objects.get(modelID=modelID, dashboardID=dashboardID)
    kwargs = {}
    abv = modelAbbreviations[model]
    lenAbv = len(abv)
    for k in mlModel._meta.fields:
        c = str(k).split('.')[-1]
        if c[:lenAbv] == abv:
            kwargs[c] = getattr(mlModel, c)
    return kwargs


def get_model_kwargs(dashboardID, modelID, model, MLModel):
    mlModel = MLModel.objects.get(modelID=modelID, dashboardID=dashboardID)
    kwargs = {}
    abv = modelAbbreviations[model]
    lenAbv = len(abv)
    for k in mlModel._meta.fields:
        c = str(k).split('.')[-1]
        if c[:lenAbv] == abv:
            kwargs[kwargTranslation[c]] = getattr(mlModel, c)
    return kwargs


def populate_sidebar(request, navbar, body):
    ajax = get_from(source='ajax', request=request, )
    return JsonResponse(
        dict(navbar=navbar, header='{} {} Settings'.format(ajax['objectType'].capitalize(), ajax['n']), body=body, )
    )


def refresh_columns(request):
    x, y, n = get_ajax_data(request, ('x', 'y', 'n'))
    x = x.split(',')
    y = y.split(',')
    newColumns = [(col, col) for col in request.session['columns'] if col not in x]
    newForm = str(DependentVariableSelectForm(n, newColumns, y))
    return JsonResponse({'newForm': newForm, 'field': ids.yContainer.format(n), })


def refresh_columns(request):
    objectType, x, y, n = tool.general.get_from(
        source='ajax', request=request,
        argnames=('objectType', 'x', 'y', 'n'), )
    newColumns = [(col, col) for col in request.session['columns'] if col not in x.split(',')]
    return JsonResponse(
        {
            'newForm': str(
                ts_ml.forms.TSFormYSelect(objectType=objectType, n=n, choices=newColumns, initial=y.split(','), )
            ),
            'field':   _deprecated.id.yContainer.format(n),
        }
    )


def refresh_columns(request):
    objectType, x, y, n = tool.general.get_from(
        source='ajax', request=request,
        argnames=('objectType', 'x', 'y', 'n'), )
    newColumns = [(col, col) for col in request.session['columns'] if col not in x.split(',')]
    return JsonResponse(
        {
            'newForm': str(
                ts_apps.ts_ml.forms.TSFormYSelect(
                    objectType=objectType, n=n, choices=newColumns, initial=y.split(','), )
            ),
            'field':   _deprecated.py.id.yContainer.format(n),
        }
    )


def refresh_controls(request):
    model = get_ajax_data(request, ('model',))
    project, analysisType = get_request_data(request, ('project', 'analysisType'))
    modelClass = modelClasses[analysisType]
    kwargs = get_form_kwargs(project, model, modelClass)
    form = modelSettingsForms[model](1, initial=kwargs)
    return JsonResponse({'settings': str(form), })


def refresh_columns(request):
    appType, x, y, n = tool.general.util.get_from(source='ajax', request=request, argnames=('appType', 'x', 'y', 'n'), )
    newColumns = [(col, col) for col in request.session['columns'] if col not in x.split(',')]
    return JsonResponse(
        {
            'newForm': str(
                ml.forms.TSFormDependentVariableSelect(appType=appType, n=n, choices=newColumns, initial=y.split(','), )
            ),
            'field':   tool.app.id.yContainer.format(n),
        }
    )


def refresh_columns(request):
    objectType, x, y, n = tool.general.util.get_from(
        source='ajax', request=request,
        argnames=('objectType', 'x', 'y', 'n'), )
    newColumns = [(col, col) for col in request.session['columns'] if col not in x.split(',')]
    return JsonResponse(
        {
            'newForm': str(
                ts_ml.forms.TSFormYSelect(objectType=objectType, n=n, choices=newColumns, initial=y.split(','), )
            ),
            'field':   tool.app.id.yContainer.format(n),
        }
    )


def refresh_columns(request):
    appType, x, y, n = get_from(
        source='ajax',
        request=request,
        argnames=('appType', 'x', 'y', 'n'),
    )
    x = x.split(',')
    y = y.split(',')
    newColumns = [(col, col) for col in request.session['columns'] if col not in x]
    newForm = str(DependentVariableSelectForm(appType, n, newColumns, y))
    return JsonResponse(
        {
            'newForm': newForm,
            'field':   ids.yContainer.format(n),
        }
    )


def save_model(dashboardID, xNames, xData, yData, yCol, datas):
    for xName in xNames:
        datas[xName] = xData.loc[:, xName]
    datas[yCol] = yData
    df = pd.DataFrame(datas).set_index(xNames)
    filePath = os.path.join(path.uploadsPath, dashboardID, 'fit')
    df.to_csv('{}.csv'.format(filePath))


def text_formatter(text, width=100):
    """
    Iterates through each character in text
    and store the number of characters that have been
    iterated over so far in a counter variable.
    If the number of characters exceeds the width,
    a new line character is inserted and the counter resets.
    text: str
    width: int
    return: str
    """
    line = 1
    p = ''
    for e, char in enumerate(text):
        if e != len(text) + 1:
            p += (text[e])
            if (e > line * width and text[e] == ' '):
                line += 1
                p += '\n'
    return p


def update_controls(request):
    objectType, modelName, n = tool.general.get_from(
        source='ajax', request=request,
        argnames=('objectType', 'model', 'n'), )
    dashboardID = tool.general.get_from(source='session', request=request, argnames=('dashboardID',), )
    if modelName == 'None':  # TODO is this needed?
        modelName = 'GaussianProcessRegression'
    model = _deprecated.py.ml_general.get_model(dashboardID, n)
    form = _deprecated.py.ml_general.modelSettingsForms[modelName](
        objectType=objectType, n=n,
        initial=_deprecated.py.ml_general.get_form_kwargs(
            model=model, modelName=modelName
        ), )
    return JsonResponse({'settings': str(form), })


def update_controls(request):
    objectType, modelName, n = tool.general.get_from(
        source='ajax', request=request,
        argnames=('objectType', 'model', 'n'), )
    dashboardID = tool.general.get_from(source='session', request=request, argnames=('dashboardID',), )
    if modelName == 'None':  # TODO is this needed?
        modelName = 'GaussianProcessRegression'
    model = _deprecated.ts_ml.general.get_model(dashboardID, n)
    form = _deprecated.ts_ml.general.modelSettingsForms[modelName](
        objectType=objectType, n=n,
        initial=_deprecated.ts_ml.general.get_form_kwargs(
            model=model, modelName=modelName
        ), )
    return JsonResponse({'settings': str(form), })


def update_controls(request):
    appType, modelName, n = tool.general.util.get_from(
        source='ajax', request=request,
        argnames=('appType', 'model', 'n'), )
    dashboardID = tool.general.util.get_from(source='session', request=request, argnames=('dashboardID',), )
    if modelName == 'None':  # TODO is this needed?
        modelName = 'GaussianProcessRegression'
    model = ml.util.general.get_model(dashboardID, n)
    form = ml.util.general.modelSettingsForms[modelName](
        appType=appType, n=n,
        initial=ml.util.general.get_form_kwargs(
            model=model,
            modelName=modelName
        ), )
    return JsonResponse({'settings': str(form), })


def update_controls(request):
    objectType, modelName, n = tool.general.util.get_from(
        source='ajax', request=request,
        argnames=('objectType', 'model', 'n'), )
    dashboardID = tool.general.util.get_from(source='session', request=request, argnames=('dashboardID',), )
    if modelName == 'None':  # TODO is this needed?
        modelName = 'GaussianProcessRegression'
    model = ts_ml.util.general.get_model(dashboardID, n)
    form = ts_ml.util.general.modelSettingsForms[modelName](
        objectType=objectType, n=n,
        initial=ts_ml.util.general.get_form_kwargs(
            model=model,
            modelName=modelName
        ), )
    return JsonResponse({'settings': str(form), })


def update_controls(request):
    appType, model, n = get_from(
        source='ajax',
        request=request,
        argnames=('appType', 'model', 'n'),
    )
    dashboardID = get_from(
        source='session',
        request=request,
        argnames=('dashboardID',),
    )
    modelClass = modelClasses[appType]
    mlID = DashboardModel.objects.get(dashboardID=dashboardID).get_app_id(n)
    if model == 'None':
        model = 'GaussianProcessRegression'
    kwargs = get_form_kwargs(dashboardID, mlID, model, modelClass)
    form = modelSettingsForms[model](appType, 1, initial=kwargs)
    return JsonResponse(
        {
            'settings': str(form),
        }
    )


def update_model(request):
    appType, field, value = get_from(
        source='ajax',
        request=request,
        argnames=('appType', 'field', 'value'),
    )
    field, n = clean_model_field(field)
    dashboardID = get_from(
        source='session',
        request=request,
        argnames=('dashboardID',),
    )
    modelID = DashboardModel.objects.get(dashboardID=dashboardID).get_app_id(n)
    modelObject = modelClasses[appType].objects.get(modelID=modelID, dashboardID=dashboardID)
    value = js_to_python(value)
    setattr(modelObject, field, value)
    if field not in ['plotDim', 'plotx', 'ploty', 'plotz']:
        modelObject.fit = False
    modelObject.save()
    return JsonResponse({'field': value})


def update_model(request):
    objectType, field, value = tool.general.util.get_from(
        source='ajax', request=request,
        argnames=('objectType', 'field', 'value'), )
    field, n = tool.general.util.clean_model_field(field)
    dashboardID = tool.general.util.get_from(source='session', request=request, argnames=('dashboardID',), )
    model = get_model(dashboardID, n)
    value = tool.general.util.js_to_python(value)
    setattr(model['modelObject'], field, value)
    if field not in ['plotDim', 'plotx', 'ploty', 'plotz']:
        model['modelObject'].set_outdated()
    model['modelObject'].save()
    return JsonResponse({'field': value})


def update_model(request):
    appType, field, value = tool.general.util.get_from(
        source='ajax', request=request,
        argnames=('appType', 'field', 'value'), )
    field, n = tool.general.util.clean_model_field(field)
    dashboardID = tool.general.util.get_from(source='session', request=request, argnames=('dashboardID',), )
    model = get_model(dashboardID, n)
    value = tool.general.util.js_to_python(value)
    setattr(model['modelObject'], field, value)
    if field not in ['plotDim', 'plotx', 'ploty', 'plotz']:
        model['modelObject'].set_outdated()
    model['modelObject'].save()
    return JsonResponse({'field': value})


def update_model(request):
    field, value = get_ajax_data(request, ('field', 'value'))
    field, n = clean_model_field(field)
    project, analysisType = get_request_data(request, ('project', 'analysisType'))
    modelObject = modelClasses[analysisType].objects.get(id=project)
    value = js_to_python(value)
    setattr(modelObject, field, value)
    modelObject.save()
    return JsonResponse({'field': value})


def update_model(request):
    objectType, field, value = tool.general.get_from(
        source='ajax', request=request,
        argnames=('objectType', 'field', 'value'), )
    field, n = tool.general.clean_model_field(field)
    dashboardID = tool.general.get_from(source='session', request=request, argnames=('dashboardID',), )
    model = get_model(dashboardID, n)
    value = tool.general.js_to_python(value)
    setattr(model['modelObject'], field, value)
    if field not in ['plotDim', 'plotx', 'ploty', 'plotz']:
        model['modelObject'].set_outdated()
    model['modelObject'].save()
    return JsonResponse({'field': value})


def validate_fit(a, x, y):
    g = ('undefined', 'null', ' ', '', [])
    errors = ''
    if (a in g):
        errors += 'No model is selected.\n'
    if (x in g):
        errors += 'No x variable is selected.\n'
    if (y in g):
        errors += 'No y variable is selected.\n'
    return errors


class TSViewFormML(django.views.generic.FormView):
    analysisType = ''

    def get_context_data(self, **kwargs):
        context = super(TSViewFormML, self).get_context_data(**kwargs)
        context['analysisType'] = self.analysisType
        self.request.session['analysisType'] = self.analysisType
        return context

    def get_success_url(self, **kwargs):
        return reverse('{}:success'.format(self.analysisType.lower()), kwargs={'id': self.mlModel.id})

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        self.form = self.get_form(form_class)
        self.mlModel = self.form.save(commit=False)
        trainFile, testFile = get_session_files(request, ('trainDataFile', 'testDataFile'))
        self.request.session['trainDataFile'] = str(trainFile)
        self.request.session['testDataFile'] = str(testFile)
        self.files = [trainFile, testFile]
        self.mlModel.trainDataFile = str(trainFile)
        self.mlModel.testDataFile = str(testFile)
        errs = form_util.validate_order_form(self.files)

        if self.form.is_valid():
            if errs:
                return self.form_invalid(request, errs)
            self.mlModel.save()
            form_util.upload_files(self.files, str(self.mlModel.id))
            return self.form_valid()
        else:
            return self.form_invalid(request, errs)

    def form_valid(self):
        # self.request.session['mlModel'] = model_to_dict(self.mlModel)
        return super(TSViewFormML, self).form_valid(self.form)

    def form_invalid(self, request, errs):
        return render(
            request, app.tensorstone.app_django.ts_general.configuration.analysisForm,
            {'form': self.form, 'errs': errs}
        )


class TSViewMLApp(django.views.generic.TemplateView):
    modelChoices = None
    analysisType = ''

    def get_file(self, dirname):
        return os.path.join(path.uploadsPath, dirname, next(os.walk(dirname))[2][0])

    def get_context_data(self, **kwargs):
        context = super(TSViewMLApp, self).get_context_data(**kwargs)
        project = str(self.kwargs['id'])
        dirname = os.path.join(path.uploadsPath, project)
        trainDataFile = self.request.session['trainDataFile']
        context['analysisType'] = self.analysisType
        self.request.session['filename'] = os.path.basename(os.path.normpath(trainDataFile))
        self.request.session['project'] = project
        trainPath = os.path.join(dirname, trainDataFile)
        x, y = get_xy_cols(trainPath)
        plot, histogram = plotter.new_plot(project, trainDataFile)
        context['filename'] = os.path.basename(os.path.normpath(trainDataFile)).split('.')[0]
        context['modelX'] = IndependentVariableSelectForm(1, x, x[0][0])
        context['modelY'] = DependentVariableSelectForm(1, y, y[0][0])
        context['modelScaler'] = ScalerSelectForm(1)
        context['modelAlgorithms'] = AlgorithmSelectForm(1)
        context['modelSettingsSelectForm'] = TSFormModelSettingsSelect(1, self.modelChoices)
        context['modelSettingsorm'] = AlgorithmSelectForm(1)
        context['plot'] = plot
        context['plotDimensions'] = DimensionSelectForm(1)
        context['plotAxes'] = AxesSelectForm(1, 2, x)
        context['histogram'] = histogram
        context['histogramDimensions'] = DimensionSelectForm(2)
        context['histogramAxes'] = AxesSelectForm(2, 2, x)
        self.request.session['columns'] = [col[0] for col in x]
        return context
