import datetime

from django import forms

import models


class TSFormStatsAnalysis(forms.ModelForm):
    today = datetime.datetime.today()
    nextWeek = (today + datetime.timedelta(days=7)).strftime('%m/%d/%Y')

    terms = forms.BooleanField(required=True, label='', widget=forms.CheckboxInput(attrs={'required': True, }))
    file = forms.FileField(
        required=True, widget=forms.ClearableFileInput(
            attrs={
                'class':    'form-control',
                'multiple': False,
                'accept':   '*',
                # 'onchange': 'validate_files();',
                'required': True,
            }
        )
    )

    class Meta:
        model = models.TSModelStatsAnalysis
        fields = (
            'file',
            'terms',
        )


class TSFormControl(forms.Form):
    model = forms.ChoiceField(
        choices=(
            ('Regression', 'Regression'),
            ('Fit Distribution', 'Fit Distribution'),
        )
    )


class TSFormRegression(TSFormControl):
    model = forms.ChoiceField(
        choices=(
            ('Linear', 'Linear'),
            ('Polynomial', 'Polynomial'),
            ('Exponential', 'Exponential'),
            ('Logarithmic', 'Logarithmic'),
            ('Sinusoidal', 'Sinusoidal'))
    )


class TSFormFitDistribution(TSFormControl):
    topN = forms.IntegerField(initial=1)


class TSFormPlot(forms.Form):
    def __init__(self, x, y, *args, **kwargs):
        super(TSFormPlot, self).__init__(*args, **kwargs)
        self.fields['x'] = forms.ChoiceField(choices=x)
        self.fields['y'] = forms.ChoiceField(choices=y)

    class Meta:
        fields = ('x,', 'y')
