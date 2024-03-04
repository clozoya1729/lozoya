import django.db.models
import django.forms

from . import models


def set_field_attr(form, field, attr, val):
    form.fields[field].widget.attrs.update({attr: val, })


def get_number_input_widget(step=1, min='none', max='none'):
    return django.forms.NumberInput(attrs={'step': step, 'min': min, 'max': max, })


def get_select_widget(size, multiple=False, scroll=True):
    style = 'vertical-align:top;'
    if not scroll:
        style += 'overflow:hidden;'
    if multiple:
        return django.forms.SelectMultiple(attrs={'size': size, 'style': style, })
    return django.forms.Select(attrs={'size': size, 'style': style, })


def get_var_widget(choices):
    xWidget = django.forms.SelectMultiple(
        attrs={'size': 5 if len(choices) > 5 else len(choices), 'style': 'vertical-align:top;'}
    )
    return xWidget


class TSFormSidebar(django.forms.Form):
    suffix = ''

    def __init__(self, model, *args, **kwargs):
        super(TSFormSidebar, self).__init__(*args, **kwargs)
        self.model = model
        self.create_all_fields()

    @staticmethod
    def is_select_field(field):
        if isinstance(field, django.db.models.fields.CharField) or isinstance(field, django.db.models.fields.TextField):
            return True
        if field in ['inputColumns', 'outputColumns']:
            return True
        return False

    def create_all_fields(self):
        pass  # override this function

    def add_field(self, field):
        if self.is_select_field(field):
            self.create_select_field(field)
        else:
            self.create_number_field(field)

    def create_select_field(self, field):
        try:
            choices = [('use entire dataframe', 'Use Entire Dataframe')]
            choices += [(c, c) for c in self.model.get_input().columns]
            size = min(5, len(choices))
            self.fields[field.name] = fieldTypes[type(field)](
                label=field.verbose_name.title(),
                initial=getattr(self.model, field.name),
                choices=choices,
                widget=get_select_widget(
                    size=size, multiple=True, scroll=True
                ),
            )
        except Exception as e:
            print(e)

    def create_number_field(self, field):
        self.fields[field.name] = fieldTypes[type(field)](
            label=field.verbose_name.title(),
            initial=getattr(self.model, field.name),
            widget=get_number_input_widget(step=1, min='none', max='none')
        )

    def field_setup(self, field, fieldID):
        # TODO USE WIDGETS FOR FIELDS
        # TODO field there's a bug regarding widgets
        self.set_field_attr(
            field=field.name,
            attr='id',
            val=fieldID,
        )
        self.set_field_attr(
            field=field.name,
            attr='onchange',
            val=onchangeFunction.format(
                fieldID,
                field.verbose_name.title()
            ),
        )

    def set_field_attr(self, field, attr, val):
        self.fields[field].widget.attrs.update(
            {
                attr: val,
            }
        )

    def generate_field_id(self, i):
        return '{}-{}-{}'.format(self.model.tsObjectID, self.suffix, i)


class TSFormInputs(TSFormSidebar):
    suffix = 'inputs-field'

    def create_all_fields(self):
        # TODO
        i = 0
        for field in self.model._meta.fields:
            if (field.name in inputFormFields):
                fieldID = self.generate_field_id(i)
                print(fieldID)
                self.add_field(field)
                self.field_setup(field, fieldID)
                i += 1


class TSFormSession(django.forms.ModelForm):
    terms = django.forms.BooleanField(
        required=True,
        label='I agree to the terms of service',
    )

    class Meta:
        model = models.TSModelSession
        fields = (
            'terms',
        )


class TSFormSettings(TSFormSidebar):
    suffix = 'settings-field'

    def create_all_fields(self):
        i = 0
        for field in self.model._meta.fields:
            if not ('ts' in field.name) and not (field.name in inputFormFields):
                fieldID = self.generate_field_id(i)
                self.add_field(field)
                self.field_setup(field, fieldID)
                i += 1


fieldTypes = {
    django.db.models.FloatField:   django.forms.FloatField,
    django.db.models.IntegerField: django.forms.IntegerField,
    django.db.models.TextField:    django.forms.ChoiceField,
    django.db.models.CharField:    django.forms.ChoiceField,
}
fieldWidgets = {
    django.db.models.FloatField:   get_number_input_widget,
    django.db.models.IntegerField: get_number_input_widget,
    django.db.models.TextField:    get_select_widget,
    django.db.models.CharField:    get_select_widget,
}
onchangeFunction = 'ts_update_focused_node("{}","{}");'
inputFormFields = ['inputcolumns', 'outputcolumns']
appInputWidget = django.forms.Select(
    attrs={
        'class': 'btn btn-light dropdown-toggle app_input_selector inline',
    }
)
appOutputWidget = django.forms.SelectMultiple(
    attrs={
        'class': 'btn btn-light text-left dropdown-toggle app_output_selector inline',
        'style': 'text-align: left !important;'
    }
)
fileUploadWidget = django.forms.ClearableFileInput(
    attrs={
        'class':    'form-control',
        'multiple': False,
        'accept':   '*',
        # 'onchange': 'validate_files("{}");', 'required': True
    }
)
insertAppWidget = django.forms.Select(attrs={})
nameWidget = django.forms.TextInput(
    attrs={'class': 'form-control', 'name': 'name', 'placeholder': 'Dashboard Name', 'required': True, }
)
replaceAppWidget = django.forms.Select(attrs={'class': 'btn btn-light dropdown-toggle app_type_selector inline', })
dataFileWidget = django.forms.ClearableFileInput(
    attrs={
        'class':    'form-control',
        'multiple': False,
        'accept':   '*',
        # 'onchange': 'validate_files();',
        'required': True,
    }
)
