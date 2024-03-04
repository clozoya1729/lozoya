import os

import app.data.app_django.tensorstone.configuration as ts_general_configuration
import app.data.app_django.tensorstone.forms as ts_general_forms
import lozoya.data
from app.data.app_django.tensorstone import configuration
from app.data.app_django.ts_apps.ts_ml.forms import TSFormHyperparameter, TSFormModelSettingsSelect

regressionFields = {
    'gpAlpha':              None,
    'gpNRestartsOptimizer': None,
    'gpNormalizeY':         None,
    'gpRandomState':        None,
}

mc = {
    'regression': modelChoices
}


def get_all_ids(ns, aT):
    r = []
    for n, a in zip(ns, aT):
        ids = allIDs[a] + generalIDs + sidebarIDs
        for i in ids:
            r.append(i.format(n))
    return r


def js_to_python(value):
    if value == 'true':
        return True
    elif value == 'false':
        return False
    return value


def js_to_python(value):
    return value == 'true'


def refresh_inputs_recursively(objectClass, id):
    receivers = objectClass.objects.filter(tsInput__tsObjectID=id)
    for receiver in receivers:
        subbestclass = list(get_all_subclasses(receiver.__class__))
        for subclass in subbestclass:
            if not ('Base' in subclass.__name__):
                results = subclass.objects.filter(tsInput__tsObjectID=id)
                if results:
                    results[0].refresh_input()


def replacement(template, dict):
    with open(template, 'r') as f:
        results = lozoya.data.clean_file_contents(f.read())
    for key in dict:
        results = results.replace(key, dict[key])
    return results


def validate_order_form(files, supportedExtensions):
    errs = []
    for f in files:
        name = str(f)
        ext = name.split('.')[-1]
        if name.count('.') > 1:
            errs.append('{0} has multiple extensions. Files must have only one extension.'.format(name))
        if ext.lower() not in supportedExtensions:
            errs.append('{0} files are not currently supported.'.format(ext))
    return errs if errs.__len__() > 0 else None


def upload_files(files, directory):
    projectDir = os.path.join(uploadsPath, directory)
    try:
        os.mkdir(projectDir)
    except:
        pass
    for f in files:
        with open(os.path.join(projectDir, str(f)), 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)


def upload_files(files, destination):
    """
    destination = os.path.join(ts_util.path.uploadsPath, directory)
    """
    try:
        os.mkdir(destination)
    except:
        pass
    for f in files:
        with open(os.path.join(destination, str(f)), 'wb+') as destinationPath:
            for chunk in f.chunks():
                destinationPath.write(chunk)


def upload_files(files, directory):
    projectDir = os.path.join(uploadsPath, directory)
    os.mkdir(projectDir)
    for f in files:
        with open(os.path.join(projectDir, str(f)), 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)


def format_files(files):
    s = ''
    if len(files) > 0 and files[0] != '':
        for f in files:
            _ = lozoya.data.lozoya.data.smart(f)
            s += '<div style="color: white;">{} {} {}</div>'.format(
                make_edit_file_button(_), _,
                make_remove_file_button(_)
            )
    return s


def make_button(templatePath, *args, **kwargs):
    return lozoya.data.replacement(templatePath, kwargs)


def make_file_menu(filesList):
    return (os.path.join(configuration.templatesPath, configuration.fileSidebar), {'%filesList%': filesList, })


def make_remove_app_button(n):
    return (os.path.join(configuration.templatesPath, configuration.removeAppButton),
            {'%selectorID%': configuration.appReplace.format(n), '%buttonID%': configuration.removeButton.format(n)})


def make_remove_app_button(n):
    return make_button(
        templatePath=os.path.join(configuration.templatesPath, configuration.removeAppButton),
        **{
            '%selectorID%': configuration.appReplace.format(n),
            '%buttonID%':   configuration.removeButton.format(n)
        }, )


def make_remove_app_button(n):
    return lozoya.data.replacement(
        os.path.join(configuration.templatesPath, configuration.removeAppButton),
        {
            '%selectorID%': configuration.appReplace.format(n),
            '%buttonID%':   configuration.removeButton.format(n)
        }
    )


def make_remove_file_button(file):
    return (os.path.join(configuration.templatesPath, configuration.removeFileButton),
            {'%style%': configuration.fileButtonStyle, '%file%': file.strip()})


def make_remove_file_button(file):
    return lozoya.data.replacement(
        os.path.join(configuration.templatesPath, configuration.removeFileButton),
        {'%style%': configuration.fileButtonStyle, '%file%': file.strip()}
    )


def make_edit_file_button(file):
    return (os.path.join(configuration.templatesPath, configuration.editFileButton),
            {'%style%': configuration.fileButtonStyle, '%file%': file.strip()})


def make_edit_file_button(file):
    return lozoya.data.replacement(
        os.path.join(configuration.templatesPath, configuration.editFileButton),
        {'%style%': configuration.fileButtonStyle, '%file%': file.strip()}
    )


def make_regression_button(n):
    return (
        os.path.join(configuration.templatesPath, configuration.regressionButton), {'%n%': n}
    )


def make_regression_button(n):
    return lozoya.data.replacement(
        os.path.join(configuration.templatesPath, configuration.regressionButton), {'%n%': n}
    )


########################################################################
def make_sidebar(title, titleID, sidebarID, bodyID, buttonID, buttonText, buttonClass, icon, closeButtonID, code):
    return lozoya.data.replacement(
        os.path.join(ts_general_configuration.templatesPath, ts_general_configuration.sidebar),
        {
            '%sidebarID%':       sidebarID,
            '%sidebarTitle%':    title,
            '%sidebarBodyID%':   bodyID,
            '%sidebarButtonID%': buttonID,
            '%buttonText%':      buttonText,
            '%buttonClass%':     buttonClass,
            '%iconClass%':       icon,
            '%sidebarCode%':     code,
            '%sidebarTitleID%':  titleID,
            '%closeButtonID%':   closeButtonID,
        }
    )


def make_options_menu(n, files, ty):
    return lozoya.data.replacement(
        os.path.join(ts_general_configuration.templatesPath, ts_general_configuration.appOptions),
        {
            '%n%':                n,
            '%appInputID%':       'app-input-{}'.format(n),
            '%appOutputID%':      'app-output-{}'.format(n),
            '%appSidebarID%':     'app-sidebar-{}'.format(n),
            '%appInputSelect%':   str(ts_general_forms.AppInputForm(n, [(f, f) for f in files])),
            '%appOutputSelect%':  str(ts_general_forms.AppOutputForm(n)),
            '%replaceAppSelect%': str(ts_general_forms.AppReplaceForm(n, ty)),
            '%removeAppButton%':  make_remove_app_button(n),
        }
    )


def make_settings_menu(n, ty, **kwargs):
    if ty == 'regression':
        return lozoya.data.replacement(
            os.path.join(ts_general_configuration.templatesPath, ts_general_configuration.regressionSettings),
            {
                '%n%':                       str(n),
                '%modelSettingsSelectForm%': str(TSFormModelSettingsSelect(n, kwargs['modelChoices'])),
                '%modelSettingsForm%':       str(TSFormHyperparameter(n, regressionFields)),
            }
        )
    return ''


def make_file_menu(filesList):
    return lozoya.data.replacement(
        os.path.join(ts_general_configuration.templatesPath, ts_general_configuration.fileSidebar),
        {'%filesList%': filesList, }
    )


def make_options_sidebar(obj, n, appType):
    return make_sidebar(
        'App {}: {}\nOptions'.format(n, appType),
        ts_general_configuration.optionsTitle.format(n),
        'options-{}'.format(n),
        'options-body-{}'.format(n),
        'options-button-{}'.format(n),
        '',
        'btn btn-info open-sidebar options',
        'icon glyphicon glyphicon-cog open-sidebar',
        ts_general_configuration.closeOptions.format(n),
        make_options_menu(n, obj.get_files(), appType.lower())
    )


def make_settings_sidebar(obj, n, appType, **kwargs):
    return make_sidebar(
        'App {}: {}\nSettings'.format(n, appType),
        ts_general_configuration.settingsTitle.format(n),
        'settings-{}'.format(n),
        'settings-body-{}'.format(n),
        'settings-button-{}'.format(n),
        '',
        'btn btn-warning open-sidebar settings',
        'icon ion-ios-settings-strong open-sidebar',
        ts_general_configuration.closeSettings.format(n),
        make_settings_menu(n, appType.lower(), **kwargs)
    )
