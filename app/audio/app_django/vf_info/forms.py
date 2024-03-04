import django.forms
from . import models
import lozoya.django


class MessageForm(django.forms.ModelForm):
    message = lozoya.django.widgetCharField
    email = lozoya.django.widgetEmailField

    class Meta:
        model = models.VFModelMessage
        fields = ('message', 'email',)
