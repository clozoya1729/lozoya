import os

import app.data.app_django.tensorstone.dashboard.views as ts_general_views
import pandas as pd
import ts_general.dashboard.models
import ts_util.general
import ts_util.path
from ml.general_ml import modelAbbreviations, modelNames, scalers
from regression.util.dicts import kwargTranslation
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, QuantileTransformer, RobustScaler, \
    StandardScaler
from sklearn.svm import SVR
from util.general_util import smart

import regression.forms
import regression.models

import lozoya.text
import util

modelAbbreviations = {
    'LinearRegression':               'lr',
    'GaussianProcessRegression':      'gp',
    'KNearestNeighborsRegression':    'knn',
    'RandomForestRegression':         'rf',
    'SupportVectorMachineRegression': 'svm'
}
modelClasses = {
    'regression': regression.models.TSRegressionModel,
}
modelSettingsForms = {
    'LinearRegression':               regression.forms.RegressionLinearForm,
    'GaussianProcessRegression':      regression.forms.RegressionGPForm,
    'KNearestNeighborsRegression':    regression.forms.RegressionKNNForm,
    'RandomForestRegression':         regression.forms.RegressionRFForm,
    'SupportVectorMachineRegression': regression.forms.RegressionSVMForm,
}
modelSettingsForms = {
    'LinearRegression':               regression.forms.RegressionLinearForm,
    'GaussianProcessRegression':      regression.forms.RegressionGPForm,
    'KNearestNeighborsRegression':    regression.forms.RegressionKNNForm,
    'RandomForestRegression':         regression.forms.RegressionRFForm,
    'SupportVectorMachineRegression': regression.forms.RegressionSVMForm,
}
mlModels = {
    'LinearRegression':               LinearRegression,
    'GaussianProcessRegression':      GaussianProcessRegressor,
    'KNearestNeighborsRegression':    KNeighborsRegressor,
    'RandomForestRegression':         RandomForestRegressor,
    'SupportVectorMachineRegression': SVR,
}
scalers = {
    'None':                None,
    'MaxAbsScaler':        MaxAbsScaler,
    'MinMaxScaler':        MinMaxScaler,
    'Normalizer':          Normalizer,
    'QuantileTransformer': QuantileTransformer,
    'RobustScaler':        RobustScaler,
    'StandardScaler':      StandardScaler,
    # 'PowerTransformer': PowerTransformer,
}


def validate_fit(parameters):
    g = ('undefined', 'null', ' ', '', [])
    errors = ''
    if (parameters['algorithms'] in g):
        errors += 'No model is selected.\n'
    if (parameters['x'] in g):
        errors += 'No x variable is selected.\n'
    if (parameters['y'] in g):
        errors += 'No y variable is selected.\n'
    return errors


def get_form_attr(model, attr):  # todo this is not used
    a = getattr(model, attr)
    if len(a.split(',')) != 1:
        a = a.split(',')
    kwargs = {attr: a}
    return kwargs


def get_form_kwargs(model, modelName):
    kwargs = {}
    abv = modelAbbreviations[modelName]
    lenAbv = len(abv)
    for k in model['modelObject']._meta.fields:
        c = str(k).split('.')[-1]
        if c[:lenAbv] == abv:
            kwargs[c] = getattr(model['modelObject'], c)
    return kwargs


def save_model(dashboardID, parameters, data, yFits):
    for xName in parameters['x']:
        yFits[xName] = data['x'].loc[:, xName]
    yFits[parameters['y']] = data['y']
    df = pd.DataFrame(yFits).set_index(parameters['x'])
    filePath = os.path.join(tool.general.path.uploadsPath, dashboardID, 'fit')
    df.to_csv('{}.csv'.format(filePath))


def get_model(dashboardID, n):
    dashboardObject = util.get_model(dashboardID=dashboardID)
    modelID = dashboardObject.get_app_id(n)
    modelType = dashboardObject.get_app_type(n)
    modelClass = util.general.modelClasses[modelType]
    modelObject = modelClass.objects.get(modelID=modelID, dashboardID=dashboardID)
    return dict(
        dashboardObject=dashboardObject, modelObject=modelObject, modelType=modelType, modelID=modelID,
        modelClass=modelClass, )


def get_fit_parameters(model):
    return dict(
        x=lozoya.text.str_to_list(model['modelObject'].x), y=model['modelObject'].y,
        scaler=model['modelObject'].scaler,
        algorithms=lozoya.text.str_to_list(model['modelObject'].algorithms), )


def get_model_kwargs(modelName, model):
    kwargs = {}
    abv = modelAbbreviations[modelName]
    lenAbv = len(abv)
    for k in model['modelObject']._meta.fields:
        c = str(k).split('.')[-1]
        if c[:lenAbv] == abv:
            kwargs[kwargTranslation[c]] = getattr(model['modelObject'], c)
    return kwargs


def validate_fit(parameters):
    g = ('undefined', 'null', ' ', '', [])
    errors = ''
    if (parameters['x'] in g):
        errors += 'No x variable is selected.\n'
    if (parameters['y'] in g):
        errors += 'No y variable is selected.\n'
    return errors


def get_form_kwargs(model, modelName):
    kwargs = {}
    abv = modelAbbreviations[modelName]
    lenAbv = len(abv)
    for k in model['modelObject']._meta.fields:
        c = str(k).split('.')[-1]
        if c[:lenAbv] == abv:
            kwargs[c] = getattr(model['modelObject'], c)
    return kwargs


def get_model(dashboardID, n):
    dashboardObject = ts_general.models.get_dashboard(dashboardID=dashboardID)
    objectID = dashboardObject.get_app_id(n)
    modelType = dashboardObject.get_app_type(n)
    modelClass = ts_util.general.modelClasses[modelType]
    modelObject = modelClass.objects.get(objectID=objectID, dashboardID=dashboardID)
    return dict(
        dashboardObject=dashboardObject, modelObject=modelObject, modelType=modelType, objectID=objectID,
        modelClass=modelClass, )


def get_fit_parameters(model):
    return dict(
        x=lozoya.text.str_to_list(model['modelObject'].x), y=model['modelObject'].y,
        scaler=model['modelObject'].scaler,
        algorithms=lozoya.text.str_to_list(model['modelObject'].algorithms), )


def get_model_kwargs(modelName, model):
    kwargs = {}
    abv = modelAbbreviations[modelName]
    lenAbv = len(abv)
    for k in model['modelObject']._meta.fields:
        c = str(k).split('.')[-1]
        if c[:lenAbv] == abv:
            kwargs[kwargTranslation[c]] = getattr(model['modelObject'], c)
    return kwargs


def validate_fit(parameters):
    g = ('undefined', 'null', ' ', '', [])
    errors = ''
    if (parameters['x'] in g):
        errors += 'No x variable is selected.\n'
    if (parameters['y'] in g):
        errors += 'No y variable is selected.\n'
    return errors


def get_form_kwargs(model, modelName):
    kwargs = {}
    abv = modelAbbreviations[modelName]
    lenAbv = len(abv)
    for k in model['modelObject']._meta.fields:
        c = str(k).split('.')[-1]
        if c[:lenAbv] == abv:
            kwargs[c] = getattr(model['modelObject'], c)
    return kwargs


def save_model(dashboardID, parameters, data, yFits):
    for xName in parameters['x']:
        yFits[xName] = data['x'].loc[:, xName]
    yFits[parameters['y']] = data['y']
    df = pd.DataFrame(yFits).set_index(parameters['x'])
    filePath = os.path.join(tool.path.uploadsPath, dashboardID, 'fit')
    df.to_csv('{}.csv'.format(filePath))


def get_model(dashboardID, n):
    dashboardObject = ts_general.models.get_dashboard(dashboardID=dashboardID)
    objectID = dashboardObject.get_app_id(n)
    modelType = dashboardObject.get_app_type(n)
    modelClass = _deprecated.ts_plot.general.modelClasses[modelType]
    modelObject = modelClass.objects.get(objectID=objectID, dashboardID=dashboardID)
    return dict(
        dashboardObject=dashboardObject, modelObject=modelObject, modelType=modelType, objectID=objectID,
        modelClass=modelClass, )


def get_fit_parameters(model):
    return dict(
        x=lozoya.text.str_to_list(model['modelObject'].x), y=model['modelObject'].y,
        scaler=model['modelObject'].scaler,
        algorithms=lozoya.text.str_to_list(model['modelObject'].algorithms), )


def get_model_kwargs(modelName, model):
    kwargs = {}
    abv = modelAbbreviations[modelName]
    lenAbv = len(abv)
    for k in model['modelObject']._meta.fields:
        c = str(k).split('.')[-1]
        if c[:lenAbv] == abv:
            kwargs[kwargTranslation[c]] = getattr(model['modelObject'], c)
    return kwargs


def validate_fit(parameters):
    g = ('undefined', 'null', ' ', '', [])
    errors = ''
    if (parameters['x'] in g):
        errors += 'No x variable is selected.\n'
    if (parameters['y'] in g):
        errors += 'No y variable is selected.\n'
    return errors


def get_form_kwargs(model, modelName):
    kwargs = {}
    abv = modelAbbreviations[modelName]
    lenAbv = len(abv)
    for k in model['modelObject']._meta.fields:
        c = str(k).split('.')[-1]
        if c[:lenAbv] == abv:
            kwargs[c] = getattr(model['modelObject'], c)
    return kwargs


def save_model(dashboardID, parameters, data, yFits):
    for xName in parameters['x']:
        yFits[xName] = data['x'].loc[:, xName]
    yFits[parameters['y']] = data['y']
    df = pd.DataFrame(yFits).set_index(parameters['x'])
    filePath = os.path.join(ts_util.path.uploadsPath, dashboardID, 'fit')
    df.to_csv('{}.csv'.format(filePath))


def get_model(dashboardID, n):
    dashboardObject = ts_apps.SklearnModels.get_dashboard(dashboardID=dashboardID)
    objectID = dashboardObject.get_app_id(n)
    modelType = dashboardObject.get_app_type(n)
    modelClass = _deprecated.ts_plot.general.modelClasses[modelType]
    modelObject = modelClass.objects.get(objectID=objectID, dashboardID=dashboardID)
    return dict(
        dashboardObject=dashboardObject, modelObject=modelObject, modelType=modelType, objectID=objectID,
        modelClass=modelClass, )


def get_fit_parameters(model):
    return dict(
        x=lozoya.text.str_to_list(model['modelObject'].x), y=model['modelObject'].y,
        scaler=model['modelObject'].scaler,
        algorithms=lozoya.text.str_to_list(model['modelObject'].algorithms), )


def get_model_kwargs(modelName, model):
    kwargs = {}
    abv = modelAbbreviations[modelName]
    lenAbv = len(abv)
    for k in model['modelObject']._meta.fields:
        c = str(k).split('.')[-1]
        if c[:lenAbv] == abv:
            kwargs[kwargTranslation[c]] = getattr(model['modelObject'], c)
    return kwargs


def validate_fit(parameters):
    g = ('undefined', 'null', ' ', '', [])
    errors = ''
    if (parameters['x'] in g):
        errors += 'No x variable is selected.\n'
    if (parameters['y'] in g):
        errors += 'No y variable is selected.\n'
    return errors


def get_form_kwargs(model, modelName):
    kwargs = {}
    abv = modelAbbreviations[modelName]
    lenAbv = len(abv)
    for k in model['modelObject']._meta.fields:
        c = str(k).split('.')[-1]
        if c[:lenAbv] == abv:
            kwargs[c] = getattr(model['modelObject'], c)
    return kwargs


def get_model(dashboardID, n):
    dashboardObject = ts_general_views.get_dashboard(dashboardID=dashboardID)
    objectID = dashboardObject.get_app_id(n)
    modelType = dashboardObject.get_app_type(n)
    modelClass = ts_plot.general.modelClasses[modelType]
    modelObject = modelClass.objects.get(objectID=objectID, dashboardID=dashboardID)
    return dict(
        dashboardObject=dashboardObject, modelObject=modelObject, modelType=modelType, objectID=objectID,
        modelClass=modelClass, )


def get_fit_parameters(model):
    return dict(
        x=lozoya.text.str_to_list(model['modelObject'].x), y=model['modelObject'].y,
        scaler=model['modelObject'].scaler, algorithms=lozoya.text.str_to_list(model['modelObject'].algorithms), )


def get_model_kwargs(modelName, model):
    kwargs = {}
    abv = modelAbbreviations[modelName]
    lenAbv = len(abv)
    for k in model['modelObject']._meta.fields:
        c = str(k).split('.')[-1]
        if c[:lenAbv] == abv:
            kwargs[kwargTranslation[c]] = getattr(model['modelObject'], c)
    return kwargs


def fit_loop(algorithms, project, X, y, scaler, models, MLModel):
    X = X.values
    y = y.values
    scaler = scalers[scaler]
    if scaler:
        scaler = scaler()
        scaler.fit(X)
    yFits = {}
    for model in algorithms:
        kwargs = get_model_kwargs(project, smart(model), MLModel)
        regressor = models[model](**kwargs)
        regressor.fit(X, y)
        yFits[modelNames[smart(model)]] = regressor.predict(X)
    return yFits


def get_form_kwargs(project, model, MLModel):
    mlModel = MLModel.objects.get(id=project)
    kwargs = {}
    abv = modelAbbreviations[model]
    lenAbv = len(abv)
    for k in mlModel._meta.fields:
        c = str(k).split('.')[-1]
        if c[:lenAbv] == abv:
            kwargs[c] = getattr(mlModel, c)
    return kwargs


def get_model_kwargs(project, model, MLModel):
    mlModel = MLModel.objects.get(id=project)
    kwargs = {}
    abv = modelAbbreviations[model]
    lenAbv = len(abv)
    for k in mlModel._meta.fields:
        c = str(k).split('.')[-1]
        if c[:lenAbv] == abv:
            kwargs[kwargTranslation[c]] = getattr(mlModel, c)
    return kwargs


def save_model(project, xNames, xData, yData, yCol, datas):
    for xName in xNames:
        datas[xName] = xData.loc[:, xName]
    datas[yCol] = yData
    df = pd.DataFrame(datas).set_index(xNames)
    filePath = os.path.join(path.uploadsPath, project, 'fit')
    df.to_csv('{}.csv'.format(filePath))


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


def get_fit_parameters(modelObject):
    return modelObject.x, modelObject.y, modelObject.scaler, modelObject.algorithms
