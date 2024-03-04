import django.forms

import configuration
import models
import app.data.app_django.tensorstone.forms as ts_general_forms


class TSFormColumnSelect(django.forms.Form):
    def __init__(self, n, columns, *args, **kwargs):
        super(TSFormColumnSelect, self).__init__(*args, **kwargs)
        id = 'column_selector{}'.format(n)
        self.fields['column'] = django.forms.ChoiceField(choices=columns, required=False)
        self.fields['column'].widget.attrs.update(
            {'id': id, 'class': 'column_selector inline', 'onchange': 'update_criteria("{}");'.format(id)}
        )


class TSFormCriteriaParameters(django.forms.Form):
    def __init__(self, n, criterion, values, *args, **kwargs):
        super(TSFormCriteriaParameters, self).__init__(*args, **kwargs)
        if criterion in ['LessThan', 'GreaterThan', 'EqualTo', 'Between']:
            if len(values) <= 100:
                self.fields['value'] = django.forms.ChoiceField(choices=[(v, v) for v in sorted(values)])
            else:
                self.fields['value'] = django.forms.FloatField()
        elif criterion == 'Contains':
            self.fields['value'] = django.forms.ChoiceField(choices=[(v, v) for v in sorted(values)])
        self.fields['value'].widget.attrs.update(
            {'id': 'criteria_parameter{}'.format(n), 'class': 'criteria_parameter inline'}
        )


class TSFormCriteriaSelect(django.forms.Form):
    def __init__(self, n, dtype, *args, **kwargs):
        super(TSFormCriteriaSelect, self).__init__(*args, **kwargs)
        id = 'criteria_selector{}'.format(n)
        self.fields['criteria'] = django.forms.ChoiceField(choices=configuration.criteriaTypes[dtype])
        self.fields['criteria'].widget.attrs.update(
            {'id': id, 'class': 'criteria_selector inline', 'onchange': 'update_criteria_params("{}")', }
        )


class TSFormDataPrep(django.forms.ModelForm):
    dataFile = django.forms.forms.FileField(required=True, label='File')
    terms = django.forms.BooleanField(
        required=True, label='I agree to the terms of service', widget=ts_general_forms.termsWidget
    )

    class Meta:
        model = models.TSModelDataPrep
        fields = ('dataFile',)


class TSFormFileObject(django.forms.Form):
    file = django.forms.FileField(required=True, widget=ts_general_forms.fileUploadWidget, )
    # tool.server.file_util.upload_files(self.files, str(self.publicDashboardModel.dashboardID))
