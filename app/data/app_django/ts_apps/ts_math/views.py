import json
import os

import pandas as pd
import plotly.express as px
import plotly.offline as po
from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.generic import FormView, TemplateView

import app.data.app_django.tensorstone.configuration as ts_general_configuration
import app.data.app_django.tensorstone.models as ts_general_models
import app.data.app_django.tensorstone.util as ts_general_util
import app.data.app_django.ts_apps.ts_plot.models as ts_plot_models
import models
import forms


def action_hero(obj, id):
    subbestclass = list(ts_general_util.get_all_subclasses(obj.__class__))
    for subclass in subbestclass:
        if not ('Base' in subclass.__name__):
            results = subclass.objects.filter(objectID=id)
            if results:
                return results


def create_object(request):
    dashboardID, objectType = ts_general_util.get_from(
        source='ajax',
        request=request,
        argnames=(
            'dashboardID',
            'objectType',
        ),
    )
    obj = objectTypes[objectType].objects.create(
        dashboard=dashboards.objects.get(  # data.models.DashboardModel.objects.get( ?
            dashboardID=dashboardID
        )
    )
    return JsonResponse({'id': obj.objectID, })


def delete_object(request):
    dashboardID, objectType, objectID = ts_general_util.get_from(
        source='ajax',
        request=request,
        argnames=(
            'dashboardID',
            'objectType',
            'objectID',
        ),
    )
    obj = objectTypes[objectType].objects.get(
        dashboard=dashboards.objects.get(
            dashboardID=dashboardID
        ),
        objectID=objectID,
    )
    obj.delete()
    return JsonResponse({})


def refresh_columns(request):
    x = ts_general_util.smart(str(request.GET.get('x', None)))
    y = ts_general_util.smart(str(request.GET.get('y', None)))
    columns = ts_general_util.smart(str(request.GET.get('columns', None)))
    columns = columns.split(',')
    newColumns = []
    for col in columns:
        if col != x:
            newColumns.append(col)
    if (x == y):
        y = newColumns[0]
    data = {
        'x':       x,
        'y':       y,
        'columns': json.dumps(newColumns),
    }
    return JsonResponse(data)


def refresh_controls(request):
    model = ts_general_util.smart(str(request.GET.get('model', None)))
    data = {'model': model, 'controls': controls[model], }
    return JsonResponse(data)


def refresh_plot(request):
    x = ts_general_util.smart(str(request.GET.get('x', None)))
    y = ts_general_util.smart(str(request.GET.get('y', None)))
    project = ts_general_util.smart(str(request.GET.get('project', None)))
    filename = ts_general_util.smart(str(request.GET.get('filename', None)))
    p = os.path.join(ts_general_configuration.uploadsPath, project, filename)
    dataF = pd.read_csv(p).reset_index()
    plot = px.scatter(dataF, x=x, y=y)
    data = {
        'x':        x,
        'y':        y,
        'project':  project,
        'filename': filename,
        'plot':     po.plot(
            plot,
            auto_open=False,
            output_type='div'
        ),
    }
    return JsonResponse(data)


def update_input(request):
    dashboardID, objectType, objectID, value = ts_general_util.get_from(
        source='ajax',
        request=request,
        argnames=(
            'dashboardID',
            'objectType',
            'objectID',
            'value'
        ),
    )
    obj = objectTypes[objectType].objects.get(
        dashboard=dashboards.objects.get(
            dashboardID=dashboardID
        ),
        objectID=objectID,
    )
    if value:
        other = baseObjects.objects.get(
            dashboard=dashboardID,
            objectID=value
        )
        obj.input = other
    else:
        obj.input = None
    obj.save()
    return JsonResponse({'success': True, })


def update_input(request):
    return JsonResponse({})


def update_object(request):
    dashboardID, objectType, objectID, property, value = ts_general_util.get_from(
        source='ajax',
        request=request,
        argnames=(
            'dashboardID',
            'objectType',
            'objectID',
            'property',
            'value'
        ),
    )
    obj = objectTypes[objectType].objects.get(
        dashboard=dashboards.objects.get(
            dashboardID=dashboardID
        ),
        objectID=objectID,
    )
    propertyDict = {
        'Amplitude':        'a',
        'Frequency':        'f',
        'HorizontalOffset': 'h',
        'VerticalOffset':   'v',
    }
    if property in propertyDict.keys():
        setattr(obj, propertyDict[property], value)
    else:
        setattr(obj, property.lower(), value)
    obj.save()
    return JsonResponse({})


def update_object(request):
    dashboardID, objectType, objectID, property, value = ts_general_util.get_from(
        source='ajax',
        request=request,
        argnames=(
            'dashboardID',
            'objectType',
            'objectID',
            'property',
            'value'
        ),
    )
    obj = objectTypes[objectType].objects.get(
        dashboard=dashboards.objects.get(
            dashboardID=dashboardID
        ),
        objectID=objectID,
    )
    propertyDict = {
        'Minimum':           'a',
        'Maximum':           'b',
        'Mean':              'a',
        'StandardDeviation': 'b',
    }
    if property in propertyDict.keys():
        setattr(obj, propertyDict[property], value)
    else:
        setattr(obj, property.lower(), value)
    obj.save()
    return JsonResponse({})


class results(TemplateView):
    template_name = ts_general_configuration.analysisApp

    def get_file(self, dirname):
        return os.path.join(ts_general_configuration.uploadsPath, dirname, next(os.walk(dirname))[2][0])

    def get_context_data(self, **kwargs):
        context = super(results, self).get_context_data(**kwargs)
        id = str(self.kwargs['id'])
        dirname = os.path.join(ts_general_configuration.uploadsPath, id)
        filename = self.get_file(dirname)
        context['analysisType'] = 'Statistical'
        context['filename'] = os.path.basename(os.path.normpath(filename))
        context['project'] = id
        data = pd.read_csv(filename).reset_index()
        plot = px.scatter(data, x='index', y=data.columns[1])
        context['plot'] = po.plot(
            plot,
            auto_open=False,
            output_type='div'
        )
        x = [(i, i) for i in data.columns]
        y = x[1:]
        context['plotForm'] = forms.TSFormPlot(x=x, y=y)
        context['columns'] = list(data.columns)
        return context


class StatsAnalysisView(FormView):
    form_class = forms.TSFormStatsAnalysis
    template_name = ts_general_configuration.analysisForm

    def get_context_data(self, **kwargs):
        context = super(StatsAnalysisView, self).get_context_data(**kwargs)
        context['analysisType'] = 'Machine Learning'
        return context

    def get_success_url(self, **kwargs):
        return reverse('stats_analysis:success', kwargs={'id': self.statsModel.id})

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        self.form = self.get_form(form_class)
        self.statsModel = self.form.save(commit=False)
        self.files = request.FILES.getlist('file')
        self.statsModel.tracks = self.files
        errs = ts_general_util.validate_order_form(self.statsModel.tracks)

        if self.form.is_valid():
            if errs:
                return self.form_invalid(request, errs)
            self.statsModel.save()
            ts_general_util.upload_files(self.files, str(self.statsModel.id))
            return self.form_valid()
        else:
            return self.form_invalid(request, errs)

    def form_valid(self):
        # self.request.session['statsModel'] = model_to_dict(self.statsModel)
        return super(StatsAnalysisView, self).form_valid(self.form)

    def form_invalid(self, request, errs):
        return render(
            request, ts_general_configuration.analysisForm,
            context={'form': self.form, 'errs': errs}
        )


objectTypes = ts_plot_models.objectTypes
dashboards = ts_general_models.TSModelDashboard
baseObjects = ts_general_models.TSModelObject
dashboards = ts_general_models.TSModelDashboard
objectTypes = models.objectTypes
controls = {
    'Regression':       forms.TSFormRegression,
    'Fit Distribution': forms.TSFormFitDistribution,
}
