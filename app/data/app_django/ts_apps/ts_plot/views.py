import os

import pandas as pd
import plotly.graph_objs as go
from django.http import JsonResponse
from plotly import offline as po

import app.data.app_django.ts_apps.ts_ml.regression.models as ts_ml_models
import app.data.app_django.tensorstone.configuration as ts_general_configuration
import app.data.app_django.tensorstone.views as ts_general_views
import configuration
import forms
import lozoya.data
import lozoya.plot
import util
from app.data.app_django.tensorstone import app

plottableClasses = {'regression': ts_ml_models.TSRegressionModel, }
plotDimensionRanges = dict(scatter=(1, 2, 3), histogram=(1, 2), )


def add_plot(request):
    trainDataFile, columns, project = ts_general_views.get_request_data(
        request, ('trainDataFile', 'columns', 'project')
    )
    columns = [(col, col) for col in columns]
    appN, plotN = ts_general_views.get_ajax_data(request, ('appN', 'plotN',))
    plot, histogram = lozoya.plot_api.new_plot(project, trainDataFile)
    newPlot = util.get_plot_template(plotN, columns, plot)
    return JsonResponse({'appArea': configuration.appArea.format(appN), 'newPlot': newPlot, })


def get_plot(request, n, plotType, x, y):
    template = configuration.plotBody
    plot = make_scatter(request, x, y)
    context = dict(plotContainerID=configuration.plotContainer.format(n), n=n, plot=plot, )
    return ts_general_views.render_html(request, template, context)


def get_plot_app(request, objectType, n, plotType, x, initialAxes=None, initialDim=None, xCol=None, yCol=None):
    template = ts_general_configuration.appBody
    buttons = get_plot_buttons(request, n)
    filePath = os.path.join(
        ts_general_configuration.uploadsPath, request.session['dashboardID'], request.session['filename'], )
    columns = pd.read_csv(filePath, nrows=1).columns
    if not xCol:
        xCol = columns[0]
    if not yCol:
        yCol = columns[1]
    plotSettingsForm = get_plot_settings(request, objectType, n, plotType, x, initialAxes, initialDim)
    body = get_plot(request, n, objectType, xCol, yCol)
    context = dict(
        buttons=buttons,
        settings=plotSettingsForm,
        body=body,
    )
    body = ts_general_views.render_html(request, template, context)
    return ts_general_views.get_app_container(
        request=request,
        n=n,
        objectType=objectType,
        body=body
    )


def get_plot_app(
    request, objectType, n, plotType, x, initialAxes=None, initialDim=None, xCol=None,
    yCol=None
):
    template = ts_general_configuration.appBody
    buttons = get_plot_buttons(request, n)
    filePath = os.path.join(
        ts_general_configuration.uploadsPath, request.session['dashboardID'], request.session['filename'], )
    columns = pd.read_csv(filePath, nrows=1).columns
    if not xCol:
        xCol = columns[0]
    if not yCol:
        yCol = columns[1]
    plotSettingsForm = get_plot_settings(request, objectType, n, plotType, x, initialAxes, initialDim, )
    body = get_plot(request, n, plotType, xCol, yCol)
    context = dict(buttons=buttons, settings=plotSettingsForm, body=body, )
    body = ts_general_views.render_html(request, template, context)
    return app.tensorstone.app_django.app.get_app_container(
        request=request, n=n, objectType=plotType, body=body
    )


def get_plot_buttons(request, n):
    template = configuration.plotButtons
    navbarContext = dict(
        options=[dict(
            text='Graph', onclick="change_sidebar_menu('ts_ml', 'regression', 'regressionHPMenu', {});".format(n), ),
            dict(
                text='Table',
                onclick="change_sidebar_menu('ts_ml', 'regression', 'regressionAppMenu', {});".format(n), ), ],
        groupID='graph_table_toggle_option', n=n,
    )
    graphTableToggle = ts_general_views.render_html(
        request, configuration.sidebarNavbar, navbarContext
    )
    context = dict(graphTableToggle=graphTableToggle, n=n, )
    return ts_general_views.render_html(request, template, context)


def get_plot_body(request, objectType, n, plotType, x, y, xCol=None, yCol=None):
    template = configuration.appBody
    buttons = get_plot_buttons(request, n)
    filePath = os.path.join(
        ts_general_configuration.uploadsPath, request.session['dashboardID'], request.session['filename'], )
    columns = pd.read_csv(filePath, nrows=1).columns
    if not xCol:
        xCol = columns[0]
    if not yCol:
        yCol = columns[1]
    body = get_plot(request, n, plotType, xCol, yCol)
    context = dict(body=body, )
    return ts_general_views.render_html(request, template, context)


def get_plot_body_and_settings(
    request, objectType, n, plotType, x, initialAxes=None, initialDim=None, xCol=None,
    yCol=None
):
    template = ts_general_configuration.appBody
    buttons = get_plot_buttons(request, n)
    filePath = os.path.join(
        ts_general_configuration.uploadsPath, request.session['dashboardID'], request.session['filename'], )
    columns = pd.read_csv(filePath, nrows=1).columns
    if not xCol:
        xCol = columns[0]
    if not yCol:
        yCol = columns[1]
    plotSettingsForm = get_plot_settings(request, objectType, n, plotType, x, initialAxes, initialDim, )
    body = get_plot(request, n, plotType, xCol, yCol)
    context = dict(buttons=buttons, settings=plotSettingsForm, body=body, )
    return ts_general_views.render_html(request, template, context)


def get_plot_settings(request, objectType, n, plotType, x, initialAxes=None, initialDim=None):
    template = ts_general_configuration.plotSettings
    context = dict(
        axes=forms.TSFormAxesSelect(
            objectType=objectType, n=n, dims=initialDim['dimensions'] if initialDim else 2, cols=x, initial=initialAxes,
            onchange='update_model("{}", "{}");', ), dimensions=forms.TSFormDimensionSelect(
            objectType=objectType, n=n, dimRange=plotDimensionRanges[plotType], initial=initialDim,
            onchange='update_model("{}", "{}");', ), n=n, )
    return ts_general_views.render_html(request, template, context)


def make_histogram(request, xCol, yCol):
    filePath = os.path.join(
        ts_general_configuration.uploadsPath, request.session['project'],
        request.session['filename'], )
    columns = pd.read_csv(filePath, nrows=1).columns
    config = {'displaylogo': False, }
    if xCol and yCol:
        data = pd.read_csv(filePath, usecols=[xCol, yCol])
    else:
        data = pd.read_csv(filePath, usecols=[columns[0], columns[1]])
    if not yCol:
        yCol = columns[1]
    yData = data.loc[:, yCol]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=yData.values, histnorm='percent', name=yCol, opacity=0.75, ))
    fig = lozoya.plot_api.format(fig, xCol, yCol)
    fig.update_layout(
        barmode='overlay', bargap=0.2, bargroupgap=0.1, xaxis_title_text=yCol,
        yaxis_title_text='Percent (%)'
    )
    div = po.plot(fig, auto_open=False, output_type='div', config=config)
    return div


def make_scatter(request, xCol=None, yCol=None):
    filePath = os.path.join(
        ts_general_configuration.uploadsPath, request.session['dashboardID'], request.session['filename'], )
    columns = pd.read_csv(filePath, nrows=1).columns
    if not xCol:
        xCol = columns[0]
    if not yCol:
        yCol = columns[1]
    if xCol and yCol:
        data = pd.read_csv(filePath, usecols=[xCol, yCol], )
    else:
        data = pd.read_csv(filePath, usecols=[columns[0], columns[1]], )
    xData = data.loc[:, xCol]
    yData = data.loc[:, yCol]
    trace = go.Scatter(x=xData, y=yData, mode='markers', name=yCol, )
    figure = lozoya.plot_api.format(go.Figure(data=trace), xCol, yCol)
    div = po.plot(figure, auto_open=False, output_type='div', config=lozoya.plot_api.config, )
    return div


def refresh_plot(request):
    dashboardID, filename = ts_general_views.get_from(
        source='session', request=request, argnames=('dashboardID', 'filename'), )
    x, y = ts_general_views.get_from(source='ajax', request=request, argnames=('x', 'y'), )
    x = x.split(',')[0]
    plot, histogram = lozoya.plot_api.new_plot(
        dashboardID=dashboardID, filename=filename, xCol=x, yCol=y, )
    return JsonResponse(dict(plot=plot, histogram=histogram, x=x, y=y, ))


def refresh_plot(request):
    project, filename = ts_general_views.get_request_data(request, ('project', 'filename'))
    x, y = ts_general_views.get_ajax_data(request, ('x', 'y'))
    x = x.split(',')[0]
    plot, histogram = lozoya.plot_api.new_plot(project, filename, x, y)
    return JsonResponse({'plot': plot, 'histogram': histogram, 'x': x, 'y': y, })


def update_axes(request):
    objectType, n, xCol, yCol = ts_general_views.general.get_from(
        source='ajax', request=request,
        argnames=('objectType', 'n', 'xCol', 'yCol'), )
    dashboardID, trainDataFile = ts_general_views.general.get_from(
        source='session', request=request,
        argnames=('dashboardID', 'trainDataFile'), )
    dirname = os.path.join(ts_general_configuration.uploadsPath, dashboardID)
    trainPath = os.path.join(dirname, trainDataFile)
    x, y = lozoya.data_api.get_xy_cols(trainPath)
    plot = get_plot_body(
        request=request, objectType=objectType, n=n, plotType='scatter',
        x=x, y=y, xCol=xCol, yCol=yCol, )
    return JsonResponse(dict(field=configuration.plotContainer.format(n), plot=plot, ))


def update_dimensions(request):
    newDim, oldDim, selected, n = ts_general_views.get_ajax_data(
        request, ('newDim', 'oldDim', 'selected', 'n')
    )
    selected = selected.split(',')
    columns = request.session['columns']
    columns = [(col, col) for col in columns]
    newForm = str(forms.TSFormAxesSelect(n, newDim, columns, selected))
    return JsonResponse({'newForm': newForm, 'axArea': configuration.axArea.format(n), })


def update_dimensions(request):
    objectType, n, newDim = ts_general_views.get_from(
        source='ajax', request=request,
        argnames=('objectType', 'n', 'newDim'), )
    columns = request.session['columns']
    columns = [(col, col) for col in columns]
    dashboardID = ts_general_views.get_from(
        source='session', request=request, argnames=('dashboardID',), )
    dashboardModel = ts_general_views.models.get_model(dashboardID=dashboardID)
    objectID = dashboardModel.get_app_id(n)
    mlModel = plottableClasses[objectType].objects.get(objectID=objectID, dashboardID=dashboardID)
    initial = dict(x=mlModel.plotx, y=mlModel.ploty, z=mlModel.plotz, )
    newForm = str(
        forms.TSFormAxesSelect(
            objectType=objectType, n=n, dims=newDim, cols=columns, initial=initial,
            onchange='update_model("{}", "{}");', )
    )
    return JsonResponse(dict(newForm=newForm, axArea=configuration.axArea.format(n), ))
