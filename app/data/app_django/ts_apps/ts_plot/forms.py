import django.forms

import configuration

minDims = {'scatter': 2, 'histogram': 1, }
maxDims = {'scatter': 4, 'histogram': 2, }
dimensionNames = ('x', 'y', 'z', 'color')
dimensionChoices = (('2', '2'), ('3', '3'), ('4', '4'))


class TSFormDimensionSelect(django.forms.Form):
    def __init__(self, n, *args, **kwargs):
        super(TSFormDimensionSelect, self).__init__(*args, **kwargs)
        id = configuration.dimensions.format(n)
        self.fields['dimensions'] = django.forms.ChoiceField(choices=dimensionChoices, required=False)
        self.fields['dimensions'].widget.attrs.update(
            {
                'id':       id,
                'class':    'dimension_selector inline',
                'onchange': 'update_dimensions("{}");'.format(id),
            }
        )


class TSFormDimensionSelect(django.forms.Form):
    def __init__(self, objectType, n, dimRange, initial=None, onchange=None, *args, **kwargs):
        super(TSFormDimensionSelect, self).__init__(*args, **kwargs)
        id = configuration.dimensions.format(n)
        dimChoices = dimensionChoices[dimRange[0] - 1:dimRange[1]]
        self.fields['dimensions'] = django.forms.ChoiceField(choices=dimChoices, initial=initial, required=False, )
        if self.fields['dimensions']:
            self.fields['dimensions'].initial = initial['dimensions']
        if onchange:
            oc = onchange.format(objectType, id)
        else:
            oc = ''
        self.fields['dimensions'].widget.attrs.update(
            {
                'id':       id,
                'class':    'dimension_selector inline',
                'onchange': 'update_dimensions("{}", "{}");{}'.format(objectType, id, oc),
            }
        )


class TSFormAxesSelect(django.forms.Form):
    def __init__(self, n, dims, cols, inits=[], *args, **kwargs):
        super(TSFormAxesSelect, self).__init__(*args, **kwargs)
        for i in range(int(dims)):
            id = configuration.axes.format(dimensionNames[i], n)
            fieldName = dimensionNames[i]
            self.fields[fieldName] = django.forms.ChoiceField(
                choices=cols, initial=inits[i] if i < len(inits) else cols[i][0]
            )
            self.fields[fieldName].widget.attrs.update(
                {
                    'id':       id,
                    'class':    'axes_selector inline',
                    'onchange': 'update_axes("{}");'.format(id),
                }
            )


class TSFormAxesSelect(django.forms.Form):
    def __init__(self, objectType, n, dims, cols, initial={}, onchange=None, *args, **kwargs):
        super(TSFormAxesSelect, self).__init__(*args, **kwargs)
        for i in range(int(dims)):
            id = configuration.axes.format(dimensionNames[i], n)
            fieldName = dimensionNames[i]
            condition = (fieldName in initial.keys() and initial[fieldName] != '')
            self.fields[fieldName] = django.forms.ChoiceField(
                choices=cols,
                initial=initial[fieldName] if condition else cols[i][0], )
            if self.fields[fieldName]:
                self.fields[fieldName].initial = initial[fieldName] if condition else cols[i][0]
            if onchange:
                oc = onchange.format(objectType, id)
            else:
                oc = ''
            self.fields[fieldName].widget.attrs.update(
                {
                    'id':       id,
                    'class':    'axes_selector inline',
                    'onchange': 'update_axes("{}", "{}");{}'.format(objectType, id, oc, ),
                }
            )
