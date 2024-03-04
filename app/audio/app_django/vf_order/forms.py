import datetime

import django.forms
from . import models


class _OrderForm(django.forms.ModelForm):
    project_name = django.forms.CharField(
        required=False, max_length=100, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'e.g. Artist Name - Song Title', 'required': False, }
        )
    )

    details = django.forms.CharField(
        required=False, max_length=500, widget=django.forms.Textarea(
            attrs={
                'class':       'form-control',
                'placeholder': 'Request specific procedures for your song. (500 character limit)',
                'required':    False,
            }
        )
    )

    reference = django.forms.CharField(
        required=False, max_length=100, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'List artists/songs to reference.', 'required': False, }
        )
    )

    email = django.forms.EmailField(
        required=True, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'example@email.com', 'required': True, }
        )
    )

    today = datetime.datetime.today()
    nextWeek = (today + datetime.timedelta(days=7)).strftime('%m/%d/%Y')
    date_due = django.forms.CharField(
        widget=django.forms.DateInput(
            attrs={
                'class':    'form-control datepicker', 'default': nextWeek, 'placeholder': nextWeek,
                'onchange': 'calculate_price();', 'readonly': True
            }
        )
    )

    price = django.forms.CharField(
        max_length=10, label='', widget=django.forms.TextInput(
            attrs={
                'class': 'form-control', 'default': '$0.00', 'placeholder': '$0.00', 'readonly': True,
                'type':  'hidden',
            }
        )
    )

    coupon_redeemed = django.forms.CharField(
        label='Coupon', max_length=10, required=False,
        widget=django.forms.TextInput(attrs={'class': 'form-control', 'required': False, })
    )

    discount_redeemed = django.forms.CharField(
        max_length=4, label='', required=False, widget=django.forms.TextInput(
            attrs={'hidden': True, 'readonly': True, 'placeholder': 0, 'default': 0, 'required': False}
        )
    )

    coupon_generated = django.forms.CharField(
        label='', max_length=10, required=False, widget=django.forms.TextInput(
            attrs={'hidden': True, 'readonly': True, 'required': False, }
        )
    )

    discount_generated = django.forms.CharField(
        max_length=4, label='', required=False, widget=django.forms.TextInput(
            attrs={'hidden': True, 'readonly': True, 'placeholder': 0, 'default': 0, 'required': False}
        )
    )

    terms = django.forms.BooleanField(
        required=True, label='', widget=django.forms.CheckboxInput(attrs={'required': True, })
    )

    duration = django.forms.CharField(
        max_length=7, label='', widget=django.forms.TextInput(
            attrs={'default': '0.00', 'placeholder': '0.00', 'hidden': True, 'readonly': True, }
        )
    )


class OrderForm(_OrderForm):
    files = django.forms.FileField(
        required=True, widget=django.forms.ClearableFileInput(
            attrs={
                'class':    'form-control', 'multiple': True, 'accept': 'audio/*',
                'onchange': 'validate_files(); calculate_price();', 'required': True,
            }
        )
    )

    mix = django.forms.BooleanField(
        initial=False, required=False, widget=django.forms.CheckboxInput(attrs={'required': False, })
    )

    performance_correction = django.forms.BooleanField(
        initial=False, required=False,
        widget=django.forms.CheckboxInput(attrs={'required': False, })
    )

    repair = django.forms.BooleanField(
        initial=False, required=False, widget=django.forms.CheckboxInput(attrs={'required': False, })
    )

    master = django.forms.BooleanField(
        initial=False, required=False, widget=django.forms.CheckboxInput(attrs={'required': False, })
    )

    class Meta:
        model = models.VFModelOrder
        fields = ('project_name', 'files', 'mix', 'performance_correction', 'repair', 'master', 'date_due', 'details',
                  'reference', 'email', 'coupon_redeemed', 'discount_redeemed', 'price', 'terms')


class OrderForm(django.forms.ModelForm):
    files = django.forms.FileField(
        required=True, widget=django.forms.ClearableFileInput(
            attrs={
                'class':    'form-control', 'multiple': True, 'accept': 'audio/*',
                'onchange': 'validate_files(); calculate_price();', 'required': True,
            }
        )
    )

    project_name = django.forms.CharField(
        required=False, max_length=100, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'e.g. Artist Name - Song Title', 'required': False, }
        )
    )

    details = django.forms.CharField(
        required=False, max_length=500, widget=django.forms.Textarea(
            attrs={
                'class':       'form-control',
                'placeholder': 'Request specific procedures for your song. (500 character limit)',
                'required':    False,
            }
        )
    )

    reference = django.forms.CharField(
        required=False, max_length=100, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'List artists/songs to reference.', 'required': False, }
        )
    )

    email = django.forms.EmailField(
        required=True, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'example@email.com', 'required': True, }
        )
    )

    today = datetime.datetime.today()
    nextWeek = (today + datetime.timedelta(days=7)).strftime('%m/%d/%Y')
    date_due = django.forms.CharField(
        widget=django.forms.DateInput(
            attrs={
                'class':    'form-control datepicker', 'default': nextWeek, 'placeholder': nextWeek,
                'onchange': 'calculate_price();', 'readonly': True
            }
        )
    )

    price = django.forms.CharField(
        max_length=10, label='', widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'default': '$0.00', 'placeholder': '$0.00', 'readonly': True}
        )
    )

    terms = django.forms.BooleanField(
        label='',
        widget=django.forms.CheckboxInput(attrs={'class': 'form-control', 'required': True, })
    )

    class Meta:
        model = models.VFModelOrder
        fields = ('files', 'project_name', 'details', 'reference', 'email', 'date_due', 'price', 'terms')


class OrderForm(django.forms.ModelForm):
    file = django.forms.FileField(
        widget=django.forms.ClearableFileInput(attrs={'class': 'form-control', 'multiple': True})
    )

    artist_name = django.forms.CharField(
        max_length=100,
        widget=django.forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Artist Name'})
    )

    song_title = django.forms.CharField(
        max_length=100, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'Reference Song Title'}
        )
    )

    details = django.forms.CharField(
        max_length=500,
        widget=django.forms.Textarea(attrs={'class': 'form-control', 'placeholder': 'Artist Name'})
    )

    reference_artist = django.forms.CharField(
        max_length=100, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'Artist Name'}
        )
    )

    reference_song = django.forms.CharField(
        max_length=100, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'Reference Artist Name'}
        )
    )

    email = django.forms.EmailField(
        widget=django.forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Reference Artist Name'})
    )

    date_due = django.forms.DateField(widget=django.forms.DateInput(attrs={'class': 'datepicker'}))

    class Meta:
        model = models.VFModelOrder
        fields = (
            'file', 'artist_name', 'song_title', 'details', 'reference_artist', 'reference_song', 'email', 'date_due')


class OrderForm(django.forms.ModelForm):
    file = django.forms.FileField(
        widget=django.forms.ClearableFileInput(attrs={'class': 'form-control', 'multiple': True})
    )

    artist_name = django.forms.CharField(
        max_length=100,
        widget=django.forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Artist Name'})
    )

    song_title = django.forms.CharField(
        max_length=100, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'Reference Song Title'}
        )
    )

    details = django.forms.CharField(
        max_length=500,
        widget=django.forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Artist Name'})
    )

    reference_artist = django.forms.CharField(
        max_length=100, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'Artist Name'}
        )
    )

    reference_song = django.forms.CharField(
        max_length=100, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'Reference Artist Name'}
        )
    )

    email = django.forms.EmailField(
        widget=django.forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Reference Artist Name'})
    )

    date_due = django.forms.DateField(widget=django.forms.DateInput(attrs={'class': 'datepicker'}))

    class Meta:
        model = models.VFModelOrder
        fields = (
            'file', 'artist_name', 'song_title', 'details', 'reference_artist', 'reference_song', 'email', 'date_due')


class OrderForm(django.forms.ModelForm):
    project_name = django.forms.CharField(
        required=False, max_length=100, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'e.g. Artist Name - Song Title', 'required': False, }
        )
    )

    details = django.forms.CharField(
        required=False, max_length=500, widget=django.forms.Textarea(
            attrs={
                'class':       'form-control',
                'placeholder': 'Request specific procedures for your song. (500 character limit)',
                'required':    False,
            }
        )
    )

    reference = django.forms.CharField(
        required=False, max_length=100, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'List artists/songs to reference.', 'required': False, }
        )
    )

    email = django.forms.EmailField(
        required=True,
        widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'test@email.com', 'required': True, }
        )
    )

    today = datetime.datetime.today()
    nextWeek = (today + datetime.timedelta(days=7)).strftime('%m/%d/%Y')
    date_due = django.forms.CharField(
        widget=django.forms.DateInput(
            attrs={
                'class':    'form-control datepicker', 'default': nextWeek, 'placeholder': nextWeek,
                'onchange': 'calculate_price();', 'readonly': True
            }
        )
    )

    price = django.forms.CharField(
        max_length=10, label='', widget=django.forms.TextInput(
            attrs={
                'class': 'form-control', 'default': '$0.00', 'placeholder': '$0.00', 'readonly': True,
                'type':  'hidden',
            }
        )
    )

    coupon_redeemed = django.forms.CharField(
        label='Coupon', max_length=10, required=False,
        widget=django.forms.TextInput(attrs={'class': 'form-control', 'required': False, })
    )

    discount_redeemed = django.forms.CharField(
        max_length=4, label='', required=False, widget=django.forms.TextInput(
            attrs={'hidden': True, 'readonly': True, 'placeholder': 0, 'default': 0, 'required': False}
        )
    )

    coupon_generated = django.forms.CharField(
        label='', max_length=10, required=False,
        widget=django.forms.TextInput(attrs={'hidden': True, 'readonly': True, 'required': False, })
    )

    discount_generated = django.forms.CharField(
        max_length=4, label='', required=False, widget=django.forms.TextInput(
            attrs={'hidden': True, 'readonly': True, 'placeholder': 0, 'default': 0, 'required': False}
        )
    )

    terms = django.forms.BooleanField(
        required=True, label='', widget=django.forms.CheckboxInput(attrs={'required': True, })
    )

    duration = django.forms.CharField(
        max_length=7, label='',
        widget=django.forms.TextInput(
            attrs={'default': '0.00', 'placeholder': '0.00', 'hidden': True, 'readonly': True, }
        )
    )

    files = django.forms.FileField(
        required=True, widget=django.forms.ClearableFileInput(
            attrs={
                'class':    'form-control', 'multiple': True, 'accept': 'audio/*',
                'onchange': 'validate_files(); calculate_price();', 'required': True,
            }
        )
    )

    mix = django.forms.BooleanField(
        initial=False, required=False, widget=django.forms.CheckboxInput(attrs={'required': False, })
    )

    performance_correction = django.forms.BooleanField(
        initial=False, required=False,
        widget=django.forms.CheckboxInput(attrs={'required': False, })
    )

    repair = django.forms.BooleanField(
        initial=False, required=False, widget=django.forms.CheckboxInput(attrs={'required': False, })
    )

    master = django.forms.BooleanField(
        initial=False, required=False, widget=django.forms.CheckboxInput(attrs={'required': False, })
    )

    class Meta:
        model = models.VFModelOrder
        fields = ('project_name', 'files', 'mix', 'performance_correction', 'repair', 'master', 'date_due', 'details',
                  'reference', 'email', 'coupon_redeemed', 'discount_redeemed', 'price', 'terms')


class OrderForm(django.forms.ModelForm):
    project_name = django.forms.CharField(
        required=False, max_length=100, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'e.g. Artist Name - Song Title', 'required': False, }
        )
    )

    details = django.forms.CharField(
        required=False, max_length=500, widget=django.forms.Textarea(
            attrs={
                'class':       'form-control',
                'placeholder': 'Request specific procedures for your song. (500 character limit)',
                'required':    False,
            }
        )
    )

    reference = django.forms.CharField(
        required=False, max_length=100, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'List artists/songs to reference.', 'required': False, }
        )
    )

    email = django.forms.EmailField(
        required=True,
        widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'example@email.com', 'required': True, }
        )
    )

    today = datetime.datetime.today()
    nextWeek = (today + datetime.timedelta(days=7)).strftime('%m/%d/%Y')
    date_due = django.forms.CharField(
        widget=django.forms.DateInput(
            attrs={
                'class':    'form-control datepicker', 'default': nextWeek, 'placeholder': nextWeek,
                'onchange': 'calculate_price();', 'readonly': True
            }
        )
    )

    price = django.forms.CharField(
        max_length=10, label='', widget=django.forms.TextInput(
            attrs={
                'class': 'form-control', 'default': '$0.00', 'placeholder': '$0.00', 'readonly': True,
                'type':  'hidden',
            }
        )
    )

    coupon_redeemed = django.forms.CharField(
        label='Coupon', max_length=10, required=False,
        widget=django.forms.TextInput(attrs={'class': 'form-control', 'required': False, })
    )

    discount_redeemed = django.forms.CharField(
        max_length=4, label='', required=False, widget=django.forms.TextInput(
            attrs={'hidden': True, 'readonly': True, 'placeholder': 0, 'default': 0, 'required': False}
        )
    )

    coupon_generated = django.forms.CharField(
        label='', max_length=10, required=False,
        widget=django.forms.TextInput(attrs={'hidden': True, 'readonly': True, 'required': False, })
    )

    discount_generated = django.forms.CharField(
        max_length=4, label='', required=False, widget=django.forms.TextInput(
            attrs={'hidden': True, 'readonly': True, 'placeholder': 0, 'default': 0, 'required': False}
        )
    )

    terms = django.forms.BooleanField(
        required=True, label='', widget=django.forms.CheckboxInput(attrs={'required': True, })
    )

    duration = django.forms.CharField(
        max_length=7, label='',
        widget=django.forms.TextInput(
            attrs={'default': '0.00', 'placeholder': '0.00', 'hidden': True, 'readonly': True, }
        )
    )

    files = django.forms.FileField(
        required=True, widget=django.forms.ClearableFileInput(
            attrs={
                'class':    'form-control', 'multiple': True, 'accept': 'audio/*',
                'onchange': 'validate_files(); calculate_price();', 'required': True,
            }
        )
    )

    mix = django.forms.BooleanField(
        initial=False, required=False, widget=django.forms.CheckboxInput(attrs={'required': False, })
    )

    performance_correction = django.forms.BooleanField(
        initial=False, required=False,
        widget=django.forms.CheckboxInput(attrs={'required': False, })
    )

    repair = django.forms.BooleanField(
        initial=False, required=False, widget=django.forms.CheckboxInput(attrs={'required': False, })
    )

    master = django.forms.BooleanField(
        initial=False, required=False, widget=django.forms.CheckboxInput(attrs={'required': False, })
    )

    class Meta:
        model = models.VFModelOrder
        fields = ('project_name', 'files', 'mix', 'performance_correction', 'repair', 'master', 'date_due', 'details',
                  'reference', 'email', 'coupon_redeemed', 'discount_redeemed', 'price', 'terms')


class OrderForm(django.forms.ModelForm):
    project_name = django.forms.CharField(
        required=False, max_length=100, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'e.g. Artist Name - Song Title', 'required': False, }
        )
    )

    details = django.forms.CharField(
        required=False, max_length=500, widget=django.forms.Textarea(
            attrs={
                'class':       'form-control',
                'placeholder': 'Request specific procedures for your song. (500 character limit)',
                'required':    False,
            }
        )
    )

    reference = django.forms.CharField(
        required=False, max_length=100, widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'List artists/songs to reference.', 'required': False, }
        )
    )

    email = django.forms.EmailField(
        required=True,
        widget=django.forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': 'example@email.com', 'required': True, }
        )
    )

    today = datetime.datetime.today()
    nextWeek = (today + datetime.timedelta(days=7)).strftime('%m/%d/%Y')
    date_due = django.forms.CharField(
        widget=django.forms.DateInput(
            attrs={
                'class':    'form-control datepicker', 'default': nextWeek, 'placeholder': nextWeek,
                'onchange': 'calculate_price();', 'readonly': True
            }
        )
    )

    price = django.forms.CharField(
        max_length=10, label='', widget=django.forms.TextInput(
            attrs={
                'class': 'form-control', 'default': '$0.00', 'placeholder': '$0.00', 'readonly': True,
                'type':  'hidden',
            }
        )
    )

    coupon_redeemed = django.forms.CharField(
        label='Coupon', max_length=10, required=False,
        widget=django.forms.TextInput(attrs={'class': 'form-control', 'required': False, })
    )

    discount_redeemed = django.forms.CharField(
        max_length=4, label='', required=False, widget=django.forms.TextInput(
            attrs={'hidden': True, 'readonly': True, 'placeholder': 0, 'default': 0, 'required': False}
        )
    )

    coupon_generated = django.forms.CharField(
        label='', max_length=10, required=False,
        widget=django.forms.TextInput(attrs={'hidden': True, 'readonly': True, 'required': False, })
    )

    discount_generated = django.forms.CharField(
        max_length=4, label='', required=False, widget=django.forms.TextInput(
            attrs={'hidden': True, 'readonly': True, 'placeholder': 0, 'default': 0, 'required': False}
        )
    )

    terms = django.forms.BooleanField(
        required=True, label='', widget=django.forms.CheckboxInput(attrs={'required': True, })
    )

    duration = django.forms.CharField(
        max_length=7, label='',
        widget=django.forms.TextInput(
            attrs={'default': '0.00', 'placeholder': '0.00', 'hidden': True, 'readonly': True, }
        )
    )

    files = django.forms.FileField(
        required=True, widget=django.forms.ClearableFileInput(
            attrs={
                'class':    'form-control', 'multiple': True, 'accept': 'audio/*',
                'onchange': 'validate_files(); calculate_price();', 'required': True,
            }
        )
    )

    mix = django.forms.BooleanField(
        initial=False, required=False, widget=django.forms.CheckboxInput(attrs={'required': False, })
    )

    performance_correction = django.forms.BooleanField(
        initial=False, required=False,
        widget=django.forms.CheckboxInput(attrs={'required': False, })
    )

    repair = django.forms.BooleanField(
        initial=False, required=False, widget=django.forms.CheckboxInput(attrs={'required': False, })
    )

    master = django.forms.BooleanField(
        initial=False, required=False, widget=django.forms.CheckboxInput(attrs={'required': False, })
    )

    class Meta:
        model = models.VFModelOrder
        fields = ('project_name', 'files', 'mix', 'performance_correction', 'repair', 'master', 'date_due', 'details',
                  'reference', 'email', 'coupon_redeemed', 'discount_redeemed', 'price', 'terms')


class PersonalInfoForm(django.forms.ModelForm):
    email = django.forms.EmailField()

    class Meta:
        model = models.VFModelOrder
        fields = ('email',)


class DetailsForm(django.forms.ModelForm):
    details = django.forms.CharField(widget=django.forms.Textarea, max_length=500)

    class Meta:
        model = models.VFModelOrder
        fields = ('details',)


class UploadForm(django.forms.ModelForm):
    file = django.forms.FileField(widget=django.forms.ClearableFileInput(attrs={'multiple': True}))

    class Meta:
        model = models.VFModelOrder
        fields = ('file',)


'''
class MasteringForm(OrderForm):
    files = django.forms.FileField(
        label='File',
        required=True,
        widget=django.forms.ClearableFileInput(
            attrs={'class': 'form-control',
                   'multiple': False,
                   'accept': 'audio/*',
                   'onchange': 'validate_files(); calculate_price();',
                   'required': True,
                   }))

    class Meta:
        model = Mastering
        fields = ('files',
                  'project_name',
                  'details',
                  'reference',
                  'email',
                  'date_due',
                  'price',
                  'terms')
'''
'''
class MasteringForm(OrderForm):
    files = django.forms.FileField(
        label='File',
        required=True,
        widget=django.forms.ClearableFileInput(
            attrs={'class': 'form-control',
                   'multiple': False,
                   'accept': 'audio/*',
                   'onchange': 'validate_files(); calculate_price();',
                   'required': True,
                   }))

    class Meta:
        model = Mastering
        fields = ('files',
                  'project_name',
                  'details',
                  'reference',
                  'email',
                  'date_due',
                  'price',
                  'terms')
'''
'''
class MasteringForm(OrderForm):
    files = django.forms.FileField(
        label='File',
        required=True,
        widget=django.forms.ClearableFileInput(
            attrs={'class': 'form-control',
                   'multiple': False,
                   'accept': 'audio/*',
                   'onchange': 'validate_files(); calculate_price();',
                   'required': True,
                   }))

    class Meta:
        model = Mastering
        fields = ('files',
                  'project_name',
                  'details',
                  'reference',
                  'email',
                  'date_due',
                  'price',
                  'terms')
'''
'''
class MasteringForm(OrderForm):
    files = django.forms.FileField(
        label='File',
        required=True,
        widget=django.forms.ClearableFileInput(
            attrs={'class': 'form-control',
                   'multiple': False,
                   'accept': 'audio/*',
                   'onchange': 'validate_files(); calculate_price();',
                   'required': True,
                   }))

    class Meta:
        model = Mastering
        fields = ('files',
                  'project_name',
                  'details',
                  'reference',
                  'email',
                  'date_due',
                  'price',
                  'terms')
'''
'''
class MasteringForm(OrderForm):
    files = django.forms.FileField(
        label='File',
        required=True,
        widget=django.forms.ClearableFileInput(
            attrs={'class': 'form-control',
                   'multiple': False,
                   'accept': 'audio/*',
                   'onchange': 'validate_files(); calculate_price();',
                   'required': True,
                   }))

    class Meta:
        model = Mastering
        fields = ('files',
                  'project_name',
                  'details',
                  'reference',
                  'email',
                  'date_due',
                  'price',
                  'terms')
'''
