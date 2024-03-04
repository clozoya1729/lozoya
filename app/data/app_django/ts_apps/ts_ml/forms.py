import uuid

import django.db.models
import django.forms

import app.data.app_django.tensorstone.forms as ts_forms
import app.data.app_django.tensorstone.forms as ts_general_forms
import app.data.app_django.tensorstone.models as ts_models
import configuration

updateFunction = 'update_model("{}");'
updateFunction = "update_model('{}', '{}');"
hyperparameterUpdateFunction = "update_controls('{}', {});"
analysisAlgorithmsWidget = django.forms.SelectMultiple(
    attrs={'size': len(configuration.modelChoicesClassification), 'style': 'vertical-align:top; overflow: hidden;'}
)
scalerSelectWidget = django.forms.Select(
    attrs={'size': len(configuration.scalerChoices), 'style': 'vertical-align:top; overflow:hidden;'}
)
trainDataWidget = django.forms.ClearableFileInput(
    attrs={
        'class':    'form-control',
        'multiple': False,
        'accept':   '*',
        # 'onchange': 'validate_files("{}");',
        'required': True,
    }
)
testDataWidget = django.forms.ClearableFileInput(
    attrs={
        'class':    'form-control',
        'multiple': False,
        'accept':   '*',
        # 'onchange': 'validate_files("{}");',
        'required': True,
    }
)
modelSettingsSelectWidget = django.forms.Select(
    attrs={'size': len(configuration.modelChoicesClassification), 'style': 'vertical-align:top; overflow:hidden;'}
)

# GAUSSIAN PROCESS
gpAlphaWidget = django.forms.NumberInput(attrs={'step': 1e-10, 'min': 1e-10, 'max': 'none', })
gpNRestartsOptimizerWidget = django.forms.NumberInput(attrs={'step': 1, 'min': 0, 'max': 'none', })
gpRandomStateWidget = django.forms.NumberInput(attrs={'step': 1, 'min': 'none', 'max': 'none', })
# K NEAREST NEIGHBORS
knnNNeighborsWidget = django.forms.NumberInput(attrs={'step': 1, 'min': 1, 'max': 'none', })
knnLeafSizeWidget = django.forms.NumberInput(attrs={'step': 1, 'min': 1, 'max': 'none', })
knnPWidget = django.forms.NumberInput(attrs={'step': 1, 'min': 2, 'max': 'none', })
knnWeightsWidget = django.forms.Select(attrs={'size': 2, 'style': 'vertical-align: top;overflow: hidden;'})
knnDistanceMetricWidget = django.forms.Select(attrs={'onchange': 'knn_metric("{}");', })
# RANDOM FOREST
rfNEstimatorsWidget = django.forms.NumberInput(attrs={'step': 1, 'min': 1, 'max': 'none', })
rfCriterionWidget = django.forms.Select(
    attrs={'size': len(configuration.rfCriterionChoices), 'style': 'vertical-align: top;overflow: hidden;'}
)
rfMaxDepthWidget = django.forms.NumberInput(attrs={'step': 1, 'min': 1, 'max': 'none', })
rfMinSamplesSplitWidget = django.forms.NumberInput(attrs={'step': 1e-3, 'min': 1e-3, 'max': 1, })
rfMinSamplesLeafWidget = django.forms.NumberInput(attrs={'step': 1e-3, 'min': 1e-3, 'max': 0.5, })
rfMinWeightFractionLeafWidget = django.forms.NumberInput(attrs={'step': 1e-3, 'min': 0, 'max': 1, })
rfMaxFeaturesWidget = django.forms.NumberInput(attrs={'step': 1e-3, 'min': 1, 'max': 'none', })
rfMaxLeafNodesWidget = django.forms.NumberInput(attrs={'step': 1, 'min': 1, 'max': 'none', })
rfMinImpurityDecreaseWidget = django.forms.NumberInput(attrs={'step': 1e-3, 'min': 0, 'max': 'none', })
rfRandomStateWidget = django.forms.NumberInput(attrs={'step': 1, 'min': 1, 'max': 'none', })
# SUPPORT VECTOR MACHINE
svmKernelWidget = django.forms.Select(attrs={'onchange': 'svm_kernel("{}");', })
svmDegreeWidget = django.forms.NumberInput(attrs={'step': 1, 'min': 0, 'max': 'none', })
svmGammaWidget = django.forms.NumberInput(attrs={'step': 1e-3, 'min': 1e-3, 'max': 'none', })
svmCoef0Widget = django.forms.NumberInput(attrs={'step': 1e-3, 'min': 1e-3, 'max': 'none', })
svmCWidget = django.forms.NumberInput(attrs={'step': 1e-3, 'min': 1e-3, 'max': 'none', })
svmEpsilonWidget = django.forms.NumberInput(attrs={'step': 1e-3, 'min': 0, 'max': 'none', })
svmMaximumIterationsWidget = django.forms.NumberInput(attrs={'step': 1, 'min': 1, 'max': 'none', })


class TSFormAppInput(django.forms.Form):
    def __init__(self, n, choices, *args, **kwargs):
        super(TSFormAppInput, self).__init__(*args, **kwargs)
        self.fields['Input'] = django.forms.ChoiceField(choices=choices, widget=ts_forms.appInputWidget, )
        ts_forms.set_field_attr(form=self, field='Input', attr='id', val=configuration.appInput.format(n), )


class TSFormAppOutput(django.forms.Form):
    def __init__(self, n, choices, initial=None, *args, **kwargs):
        super(TSFormAppOutput, self).__init__(*args, **kwargs)
        self.fields['Output'] = django.forms.CharField(
            initial=initial, max_length=32,
            widget=ts_forms.appOutputWidget, )
        ts_forms.set_field_attr(form=self, field='Output', attr='id', val=configuration.appOutput.format(n), )


class TSFormAppReplace(django.forms.Form):
    def __init__(self, n, choices, *args, **kwargs):
        super(TSFormAppReplace, self).__init__(*args, **kwargs)
        self.fields['Replace'] = django.forms.ChoiceField(choices=choices, widget=ts_forms.replaceAppWidget, )
        ts_forms.set_field_attr(form=self, field='Replace', attr='id', val=configuration.appReplace.format(n), )


class TSFormAlgorithmSelect(django.forms.Form):
    def __init__(self, n, *args, **kwargs):
        super(TSFormAlgorithmSelect, self).__init__(*args, **kwargs)
        self.fields['algorithms'] = django.forms.MultipleChoiceField(
            choices=configuration.modelChoicesClassification, widget=analysisAlgorithmsWidget
        )
        fieldID = configuration.algorithms.format(n)
        onchange = updateFunction.format(fieldID)
        ts_forms.set_field_attr(self, 'algorithms', 'id', fieldID)
        ts_forms.set_field_attr(self, 'algorithms', 'onchange', onchange)


class TSFormCreateSession(django.forms.ModelForm):
    email = django.forms.EmailField(required=True, widget=ts_forms.emailWidget, )
    name = django.forms.CharField(required=True, max_length=20, widget=ts_forms.nameWidget, )
    terms = django.forms.BooleanField(required=True, widget=ts_forms.termsOfServiceWidget, )
    trainDataFile = django.forms.FileField(required=True, widget=ts_forms.trainDataWidget, )
    testDataFile = django.forms.FileField(required=True, widget=ts_forms.testDataWidget, )

    class Meta:
        model = ts_models.TSModelDashboard
        fields = ('email', 'name', 'terms', 'trainDataFile', 'testDataFile', 'terms',)


class TSFormDependentVariableSelect(django.forms.Form):
    def __init__(self, n, choices, initial=None, *args, **kwargs):
        super(TSFormDependentVariableSelect, self).__init__(*args, **kwargs)
        self.fields['y'] = django.forms.ChoiceField(choices=choices, initial=initial, widget=analysisAlgorithmsWidget)
        fieldID = configuration.y.format(n)
        onchange = updateFunction.format(fieldID)
        ts_forms.set_field_attr(self, 'y', 'id', fieldID)
        ts_forms.set_field_attr(self, 'y', 'onchange', onchange)

    def __init__(self, appType, n, choices, initial=None, *args, **kwargs):
        super(TSFormDependentVariableSelect, self).__init__(*args, **kwargs)
        self.fields['y'] = django.forms.ChoiceField(
            choices=choices, initial=initial,
            widget=ts_forms.get_var_widget(choices), )
        if self.fields['y'] and initial != None:
            self.fields['y'].initial = initial
        fieldID = configuration.y.format(n)
        onchange = updateFunction.format(appType, fieldID)
        ts_forms.set_field_attr(form=self, field='y', attr='id', val=fieldID, )
        ts_forms.set_field_attr(form=self, field='y', attr='onchange', val=onchange, )


class TSFormHyperparameter(django.forms.Form):
    def __init__(self, n, *args, **kwargs):
        super(TSFormHyperparameter, self).__init__(*args, **kwargs)
        for field in self.fields:
            fieldID = '{}-{}'.format(field, n)
            self.fields[field].required = False
            if 'onchange' in self.fields[field].widget.attrs:
                uf = updateFunction.format(fieldID)
                onchange = '{}{}'.format(self.fields[field].widget.attrs['onchange'].format(fieldID), uf)
            else:
                onchange = updateFunction.format(fieldID)
            ts_forms.set_field_attr(self, field, 'id', fieldID)
            ts_forms.set_field_attr(self, field, 'onchange', onchange)

    def __init__(self, objectType, n, *args, **kwargs):
        super(TSFormHyperparameter, self).__init__(*args, **kwargs)
        for field in self.fields:
            fieldID = '{}-{}'.format(field, n)
            self.fields[field].required = False
            if 'onchange' in self.fields[field].widget.attrs:
                uf = updateFunction.format(objectType, fieldID)
                onchange = '{}{}'.format(self.fields[field].widget.attrs['onchange'].format(fieldID), uf)
            else:
                onchange = updateFunction.format(objectType, fieldID)
            ts_forms.set_field_attr(form=self, field=field, attr='id', val=fieldID, )
            ts_forms.set_field_attr(form=self, field=field, attr='onchange', val=onchange, )


class TSFormInsertApp(django.forms.Form):
    dropdown = django.forms.ChoiceField(choices=(('ts_ml', 'Machine Learning'), ('plot', 'Plot')))


class TSFormIndependentVariableSelect(django.forms.Form):
    def __init__(self, objectType, n, choices, initial=None, *args, **kwargs):
        super(TSFormIndependentVariableSelect, self).__init__(*args, **kwargs)
        self.fields['x'] = django.forms.MultipleChoiceField(
            choices=choices,
            initial=initial,
            widget=ts_forms.get_var_widget(choices), )
        fieldID = configuration.x.format(n)
        if self.fields['x'] and initial != None:
            self.fields['x'].initial = initial
        ts_forms.set_field_attr(form=self, field='x', attr='id', val=fieldID, )
        ts_forms.set_field_attr(form=self, field='x', attr='multiple', val='true', )
        ts_forms.set_field_attr(form=self, field='x', attr='objectType', val=objectType, )

    def __init__(self, n, choices, initial=None, *args, **kwargs):
        super(TSFormIndependentVariableSelect, self).__init__(*args, **kwargs)
        self.fields['x'] = django.forms.MultipleChoiceField(
            choices=choices,
            initial=initial,
            widget=ts_forms.get_var_widget(choices), )
        fieldID = configuration.x.format(n)
        onchange = 'refresh_columns("{}");update_model("{}");'.format(fieldID, fieldID)
        ts_forms.set_field_attr(form=self, field='x', attr='id', val=fieldID, )
        ts_forms.set_field_attr(form=self, field='x', attr='multiple', val='true', )
        ts_forms.set_field_attr(form=self, field='x', attr='onchange', val=onchange, )


class TSFormModelSettingsSelect(django.forms.Form):
    def __init__(self, appType, n, initial={'model': 'GaussianProcessRegression'}, *args, **kwargs):
        super(TSFormModelSettingsSelect, self).__init__(*args, **kwargs)
        self.fields['model'] = django.forms.ChoiceField(
            choices=configuration.modelChoicesClassification, initial=initial,
            widget=modelSettingsSelectWidget, )
        if self.fields['model']:
            self.fields['model'].initial = initial['model']
        ts_forms.set_field_attr(form=self, field='model', attr='id', val=configuration.settings.format(n), )
        ts_forms.set_field_attr(
            form=self, field='model', attr='onchange', val=hyperparameterUpdateFunction.format(appType, n), )

    def __init__(self, n, choices, *args, **kwargs):
        super(TSFormModelSettingsSelect, self).__init__(*args, **kwargs)
        self.fields['model'] = django.forms.ChoiceField(
            choices=choices, widget=modelSettingsSelectWidget
        )
        ts_forms.set_field_attr(self, 'model', 'id', configuration.settings.format(n))


class TSFormModelSelect(django.forms.Form):
    def __init__(self, appType, n, initial=None, *args, **kwargs):
        super(TSFormModelSelect, self).__init__(*args, **kwargs)
        self.fields['algorithms'] = django.forms.MultipleChoiceField(
            choices=configuration.modelChoicesClassification,
            initial=initial,
            widget=ts_forms.analysisAlgorithmsWidget, )
        fieldID = configuration.algorithms.format(n)
        if self.fields['algorithms'] and initial != None:
            self.fields['algorithms'].initial = initial
        ts_forms.set_field_attr(form=self, field='algorithms', attr='id', val=fieldID, )
        ts_forms.set_field_attr(form=self, field='algorithms', attr='appType', val=appType, )


class TSFormScalerSelect(django.forms.Form):
    def __init__(self, objectType, n, initial={'scaler': 'None'}, *args, **kwargs):
        super(TSFormScalerSelect, self).__init__(*args, **kwargs)
        scalerId = configuration.scaler.format(n)
        self.fields['scaler'] = django.forms.ChoiceField(
            choices=configuration.scalerChoices,
            initial=initial,
            widget=ts_forms.scalerSelectWidget, )
        if self.fields['scaler']:
            self.fields['scaler'].initial = initial
        ts_forms.set_field_attr(form=self, field='scaler', attr='id', val=scalerId, )
        ts_forms.set_field_attr(
            form=self, field='scaler', attr='onchange', val=updateFunction.format(objectType, scalerId), )

    def __init__(self, objectType, n, choices, initial={'scaler': 'None'}, *args, **kwargs):
        super(TSFormScalerSelect, self).__init__(*args, **kwargs)
        scalerId = configuration.scaler.format(n)
        self.fields['scaler'] = django.forms.ChoiceField(
            choices=choices,
            initial=initial,
            widget=ts_forms.get_select_widget(size=len(choices), scroll=False), )
        if self.fields['scaler']:
            self.fields['scaler'].initial = initial
        ts_forms.set_field_attr(form=self, field='scaler', attr='id', val=scalerId, )
        ts_forms.set_field_attr(
            form=self, field='scaler', attr='onchange', val=updateFunction.format(objectType, scalerId), )

    def __init__(self, n, *args, **kwargs):
        super(TSFormScalerSelect, self).__init__(*args, **kwargs)
        scalerId = configuration.scaler.format(n)
        self.fields['scaler'] = django.forms.ChoiceField(
            choices=configuration.scalerChoices, initial='None', widget=scalerSelectWidget
        )
        ts_forms.set_field_attr(self, 'scaler', 'id', scalerId)
        ts_forms.set_field_attr(self, 'scaler', 'onchange', updateFunction.format(scalerId))


class TSFormXSelect(django.forms.Form):
    def __init__(self, objectType, n, choices, initial=None, *args, **kwargs):
        super(TSFormXSelect, self).__init__(*args, **kwargs)
        self.fields['x'] = django.forms.MultipleChoiceField(
            choices=choices,
            initial=initial,
            widget=ts_forms.get_select_widget(size=min((5, len(choices))), multiple=True), )
        fieldID = configuration.x.format(n)
        if self.fields['x'] and initial != None:
            self.fields['x'].initial = initial
        ts_forms.set_field_attr(form=self, field='x', attr='id', val=fieldID, )
        ts_forms.set_field_attr(form=self, field='x', attr='multiple', val='true', )
        ts_forms.set_field_attr(form=self, field='x', attr='objectType', val=objectType, )


class TSFormYSelect(django.forms.Form):
    def __init__(self, objectType, n, choices, initial=None, *args, **kwargs):
        super(TSFormYSelect, self).__init__(*args, **kwargs)
        self.fields['y'] = django.forms.ChoiceField(
            choices=choices, initial=initial,
            widget=ts_forms.get_select_widget(min(5, len(choices)), multiple=True), )
        if self.fields['y'] and initial != None:
            self.fields['y'].initial = initial
        fieldID = configuration.y.format(n)
        onchange = updateFunction.format(objectType, fieldID)
        ts_forms.set_field_attr(form=self, field='y', attr='id', val=fieldID, )
        ts_forms.set_field_attr(form=self, field='y', attr='onchange', val=onchange, )


class TSFormClassification():
    pass


class RegressionLinearForm(TSFormHyperparameter):
    lrFitIntercept = django.forms.BooleanField(initial=True, label='Fit Intercept', )
    lrNormalize = django.forms.BooleanField(initial=False, label='Normalize', )


class RegressionGPForm(TSFormHyperparameter):
    gpAlpha = django.forms.FloatField(
        initial=1e-10, label='Alpha', widget=ts_general_forms.get_number_input_widget(step=1e-10, min=1e-10), )
    gpAlpha = django.forms.FloatField(initial=1e-10, label='Alpha', widget=gpAlphaWidget, )
    gpNRestartsOptimizer = django.forms.IntegerField(
        initial=0, label='Optimizer Restarts', widget=ts_general_forms.get_number_input_widget(step=1, min=0), )
    gpNRestartsOptimizer = django.forms.IntegerField(
        initial=0, label='Optimizer Restarts', widget=gpNRestartsOptimizerWidget, )
    gpNormalizeY = django.forms.BooleanField(initial=False, label='Normalize Y', )
    gpRandomState = django.forms.IntegerField(
        label='Random State', widget=ts_general_forms.get_number_input_widget(step=1), )
    gpRandomState = django.forms.IntegerField(label='Random State', widget=gpRandomStateWidget, )


class RegressionKNNForm(TSFormHyperparameter):
    knnNNeighbors = django.forms.IntegerField(
        initial=5, label='Neighbors',
        widget=ts_general_forms.get_number_input_widget(step=1, min=1), )
    knnWeights = django.forms.ChoiceField(
        choices=configuration.knnWeightsChoices, initial='uniform',
        label='Weights', widget=knnWeightsWidget, )
    knnAlgorithm = django.forms.ChoiceField(
        choices=configuration.knnAlgorithmChoices,
        initial='auto',
        label='Algorithm',
    )
    knnLeafSize = django.forms.IntegerField(
        initial=30, label='Leaf Size',
        widget=ts_general_forms.get_number_input_widget(step=1, min=1), )
    knnDistanceMetric = django.forms.ChoiceField(
        choices=configuration.knnDistanceMetricChoices,
        initial='minkowski', label='Distance Metric',
        widget=knnDistanceMetricWidget, )
    knnP = django.forms.IntegerField(
        initial=2, label='Minkowski Power',
        widget=ts_general_forms.get_number_input_widget(step=1, min=2), )
    knnNNeighbors = django.forms.IntegerField(
        initial=5,
        label='Neighbors',
        widget=knnNNeighborsWidget,
    )
    knnWeights = django.forms.ChoiceField(
        choices=configuration.knnWeightsChoices,
        initial='uniform',
        label='Weights',
        widget=knnWeightsWidget,
    )
    knnLeafSize = django.forms.IntegerField(
        initial=30,
        label='Leaf Size',
        widget=knnLeafSizeWidget
    )
    knnDistanceMetric = django.forms.ChoiceField(
        choices=configuration.knnDistanceMetricChoices,
        initial='minkowski',
        label='Distance Metric',
        widget=knnDistanceMetricWidget,
    )
    knnP = django.forms.IntegerField(
        initial=2,
        label='Minkowski Power',
        widget=knnPWidget,
    )


class RegressionRFForm(TSFormHyperparameter):
    rfNEstimators = django.forms.IntegerField(
        initial=10, label='Estimators',
        widget=ts_general_forms.get_number_input_widget(step=1, min=1), )
    rfNEstimators = django.forms.IntegerField(
        initial=10,
        label='Estimators',
        widget=rfNEstimatorsWidget,
    )
    rfCriterion = django.forms.ChoiceField(
        choices=configuration.rfCriterionChoices,
        initial='mse',
        label='Criterion',
        widget=configuration.rfCriterionWidget,
    )
    rfCriterion = django.forms.ChoiceField(
        choices=configuration.rfCriterionChoices,
        initial='mse',
        label='Criterion',
        widget=rfCriterionWidget,
    )
    rfMaxDepth = django.forms.IntegerField(
        label='Maximum Depth',
        widget=ts_general_forms.get_number_input_widget(step=1, min=1),
    )
    rfMaxDepth = django.forms.IntegerField(
        label='Maximum Depth',
        widget=rfMaxDepthWidget,
    )
    rfMinSamplesSplit = django.forms.FloatField(
        initial=1e-3, label='Minimum Samples Split',
        widget=ts_general_forms.get_number_input_widget(step=1e-3, min=1e-3, max=1),
    )
    rfMinSamplesSplit = django.forms.FloatField(
        initial=1e-3,
        label='Minimum Samples Split',
        widget=rfMinSamplesSplitWidget,
    )
    rfMinSamplesLeaf = django.forms.FloatField(
        initial=1e-3, label='Minimum Samples Leaf',
        widget=ts_general_forms.get_number_input_widget(step=1e-3, min=1e-3, max=0.5),
    )
    rfMinSamplesLeaf = django.forms.FloatField(
        initial=1e-3,
        label='Minimum Samples Leaf',
        widget=rfMinSamplesLeafWidget,
    )
    rfMinWeightFractionLeaf = django.forms.FloatField(
        initial=0.0, label='Minimum Weight Fraction Leaf',
        widget=ts_general_forms.get_number_input_widget(step=1e-3, min=0, max=1),
    )
    rfMinWeightFractionLeaf = django.forms.FloatField(
        initial=0.0,
        label='Minimum Weight Fraction Leaf',
        widget=rfMinWeightFractionLeafWidget,
    )
    rfMaxFeatures = django.forms.FloatField(
        label='Maximum Features',
        widget=ts_general_forms.get_number_input_widget(step=1e-3, min=1),
    )
    rfMaxFeatures = django.forms.FloatField(
        label='Maximum Features',
        widget=rfMaxFeaturesWidget,
    )
    rfMaxLeafNodes = django.forms.IntegerField(
        label='Maximum Leaf Nodes',
        widget=ts_general_forms.get_number_input_widget(step=1, min=1),
    )
    rfMaxLeafNodes = django.forms.IntegerField(
        label='Maximum Leaf Nodes',
        widget=rfMaxLeafNodesWidget,
    )
    rfMinImpurityDecrease = django.forms.FloatField(
        initial=0.0,
        label='Minimum Impurity Decrease',
        widget=ts_general_forms.get_number_input_widget(step=1e-3, min=0),
    )
    rfMinImpurityDecrease = django.forms.FloatField(
        initial=0.0,
        label='Minimum Impurity Decrease',
        widget=rfMinImpurityDecreaseWidget,
    )
    rfBootstrap = django.forms.BooleanField(
        initial=True,
        label='Bootstrap',
    )
    rfOutOfBag = django.forms.BooleanField(
        initial=False,
        label='Out of Bag Samples',
    )
    rfRandomState = django.forms.IntegerField(
        label='Random State',
        widget=ts_general_forms.get_number_input_widget(step=1, min=1),
    )
    rfRandomState = django.forms.IntegerField(
        label='Random State',
        widget=rfRandomStateWidget,
    )
    rfWarmStart = django.forms.BooleanField(
        initial=False,
        label='Warm Start',
    )


class RegressionSVMForm(TSFormHyperparameter):
    svmKernel = django.forms.ChoiceField(
        choices=configuration.svmKernelChoices,
        initial='rbf',
        label='Kernel',
        widget=svmKernelWidget,
    )
    svmDegree = django.forms.IntegerField(
        initial=3,
        label='Degree',
        widget=ts_general_forms.get_number_input_widget(step=1, min=0),
    )
    svmDegree = django.forms.IntegerField(
        initial=3,
        label='Degree',
        widget=svmDegreeWidget,
    )
    svmGamma = django.forms.FloatField(
        label='Gamma',
        initial=1,
        widget=ts_general_forms.get_number_input_widget(step=1e-3, min=1e-3),
    )
    svmGamma = django.forms.FloatField(
        label='Gamma',
        initial=1,
        widget=svmGammaWidget,
    )
    svmCoef0 = django.forms.FloatField(
        label='Coefficient 0',
        initial=1,
        widget=ts_general_forms.get_number_input_widget(step=1e-3, min=1e-3),
    )
    svmCoef0 = django.forms.FloatField(
        label='Coefficient 0',
        initial=1,
        widget=svmCoef0Widget,
    )
    svmTolerance = django.forms.FloatField(
        initial=1e-3,
        label='Tolerance'
    )
    svmC = django.forms.FloatField(
        initial=1.0,
        label='C',
        widget=ts_general_forms.get_number_input_widget(step=1e-3, min=1e-3),
    )
    svmC = django.forms.FloatField(
        initial=1.0,
        label='C',
        widget=svmCWidget,
    )
    svmEpsilon = django.forms.FloatField(
        initial=0.1,
        label='Epsilon',
        widget=ts_general_forms.get_number_input_widget(step=1e-3, min=0),
    )
    svmEpsilon = django.forms.FloatField(
        initial=0.1,
        label='Epsilon',
        widget=svmEpsilonWidget,
    )
    svmShrinking = django.forms.BooleanField(
        initial=True,
        label='Shrinking'
    )
    svmMaximumIterations = django.forms.IntegerField(
        initial=100, label='Maximum Iterations',
        widget=ts_general_forms.get_number_input_widget(step=1, min=1),
    )
    svmMaximumIterations = django.forms.IntegerField(
        initial=100,
        label='Maximum Iterations',
        widget=svmMaximumIterationsWidget,
    )
