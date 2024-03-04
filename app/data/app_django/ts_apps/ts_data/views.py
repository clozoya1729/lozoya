import os

import pandas as pd
from django.http import JsonResponse
from django.urls import reverse
from django.views.generic import FormView, TemplateView

import app.data.app_django.tensorstone.configuration as ts_general_configuration
import app.data.app_django.tensorstone.util as ts_general_util
import configuration
import forms
import lozoya.data
from app.data.app_django.tensorstone.views import get_ajax_data


def add_column(request):
    cols = request.session['columns']
    selected = get_ajax_data(request, ('selected',))
    selected = selected.split(',')
    columns = [(c, c) for c in cols]
    n = len(selected) + 1
    values = request.session['values'].split(',')
    dtypes = request.session['dtypes'].split(',')
    colForm = forms.TSFormColumnSelect(n, columns=columns)
    if dtypes[0] == 'Numerical':
        criterion = 'LessThan'
    criteriaForm = forms.TSFormCriteriaSelect(n, dtypes[0])
    criteriaParamsForm = forms.TSFormCriteriaParameters(n, criterion, values)
    return JsonResponse({'colForm': str(colForm) + str(criteriaForm) + str(criteriaParamsForm), 'n': n, })


def download_model():
    pass


def optimize_series(v):
    unique = v.unique()
    percentUnique = len(unique) / len(v)
    if any(not lozoya.data.is_number(x) for x in v) and percentUnique < 0.5:
        dtype = 'category'
    else:
        dtype = 'float32'
    try:
        d = v.astype(dtype)
    except Exception as e:
        d = v
    return d, dtype


def get_histogram_data(path, x, y, yData):
    histogramData = pd.read_csv(path)
    histogramData.loc[:, x[0]] = yData  # x[0] or x?
    histogramData.rename(columns={x[0]: y}, inplace=True)
    return histogramData


def get_xy(path, parameters):
    x, y = parameters['x'], parameters['y']
    data = pd.read_csv(path, usecols=x + [y]).sort_values(x)
    return dict(x=data[x], y=data.loc[:, y], )


def get_xy_choices(path):
    data = pd.read_csv(path, nrows=1)
    ch = [(i, i) for i in data.columns]
    return ch


def get_xy_cols(path):
    data = pd.read_csv(path, nrows=1)
    x = [(i, i) for i in data.columns]
    y = x[1:]
    return x, y


def refresh_columns():
    pass


def refresh_controls():
    pass


def refresh_plot():
    pass


def validate_order_form(files):
    errs = []
    for f in files:
        name = str(f)
        ext = name.split('.')[-1]
        if name.count('.') > 1:
            errs.append('{0} has multiple extensions. Files must have only one extension.'.format(name))
        if ext.lower() not in ts_general_configuration.supportedExt:
            errs.append('{0} files are not currently supported.'.format(ext))
    return errs if errs.__len__() > 0 else None


def update_criteria(request):
    project = request.session['project']
    dataFile = request.session['dataFile']
    cols = request.session['columns']
    dataFilePath = os.path.join(ts_general_configuration.uploadsPath, project, dataFile)
    col, criterion, n = get_ajax_data(request, ('col', 'criterion', 'n'))
    columns = [(c, c) for c in cols]
    dtype = lozoya.data.get_dtypes(dataFilePath, (col,))
    values = list(pd.read_csv(dataFilePath, usecols=[col]).iloc[:, 0].unique())
    criteria = [c[0] for c in configuration.criteriaTypes[dtype]]
    newCriterion = criteria[0]
    for k in configuration.criteriaTypes[dtype]:
        if criterion == k[0]:
            newCriterion = criterion
            break
    colForm = forms.TSFormColumnSelect(n, columns=columns)
    criteriaForm = forms.TSFormCriteriaSelect(n, dtype)
    criteriaParamsForm = forms.TSFormCriteriaParameters(n, newCriterion, values)
    return JsonResponse(
        {'new': str(colForm) + str(criteriaForm) + str(criteriaParamsForm), 'col': col, 'newCriterion': newCriterion, }
    )


def update_model():
    pass


def upload_files(files, directory):
    dashboardDir = os.path.join(ts_general_configuration.uploadsPath, directory)
    try:
        os.mkdir(dashboardDir)
    except:
        pass
    for f in files:
        with open(os.path.join(dashboardDir, str(f)), 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)


class DataPrepAppView(TemplateView):
    template_name = ts_general_configuration.dataprepApp

    def get_context_data(self, **kwargs):
        context = super(DataPrepAppView, self).get_context_data(**kwargs)
        project = str(self.kwargs['id'])
        dirname = os.path.join(ts_general_configuration.uploadsPath, project)
        dataFile = self.request.session['dataFile']
        self.request.session['filename'] = os.path.basename(os.path.normpath(dataFile))
        self.request.session['project'] = project
        dataFilePath = os.path.join(dirname, dataFile)
        context['filename'] = os.path.basename(os.path.normpath(dataFile)).split('.')[0]
        x, y = get_xy_cols(dataFilePath)
        dtypes = lozoya.data.get_dtypes(dataFilePath)
        self.request.session['dtypes'] = ts_general_util.smart(str(dtypes))
        values = pd.read_csv(dataFilePath, usecols=[x[0][0]]).iloc[:, 0].unique()
        self.request.session['values'] = ts_general_util.smart(str(list(values)))
        if dtypes[0] == 'Numerical':
            criteria = 'LessThan'
        context['columnSelect1'] = forms.TSFormColumnSelect(1, columns=x)
        context['criteriaSelect1'] = forms.TSFormCriteriaSelect(1, dtypes[0])
        context['criteriaParams1'] = forms.TSFormCriteriaParameters(1, criteria, values)
        self.request.session['columns'] = [col[0] for col in x]
        return context


class DataPrepFormView(FormView):
    form_class = forms.TSFormDataPrep
    template_name = ts_general_configuration.dataprepForm

    def get_context_data(self, **kwargs):
        context = super(DataPrepFormView, self).get_context_data(**kwargs)
        return context

    def get_success_url(self, **kwargs):
        return reverse('dataprep:success', kwargs={'id': self.dataprepModel.id})

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        self.form = self.get_form(form_class)
        self.dataprepModel = self.form.save(commit=False)
        dataFile = ts_general_util.get_session_files(request, ('dataFile',))
        self.request.session['dataFile'] = str(dataFile)
        self.files = (dataFile,)
        self.dataprepModel.dataFile = str(dataFile)
        # errs = form_util.validate_order_form(self.files)

        if self.form.is_valid():
            # if errs:
            #     return self.form_invalid(request, errs)
            self.dataprepModel.save()
            ts_general_util.upload_files(self.files, str(self.dataprepModel.id))
            return self.form_valid()  # else:  #     return self.form_invalid(request, errs)

    def form_valid(self):
        # self.request.session['mlModel'] = model_to_dict(self.mlModel)
        return super(DataPrepFormView, self).form_valid(
            self.form
        )  # # def form_invalid(self, request, errs):  #     return render(request, templates.analysisForm, {'form': self.form, 'errs': errs})
