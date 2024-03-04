import django.forms

widgetCharField = django.forms.CharField(
    required=True,
    max_length=500,
    widget=django.forms.Textarea(
        attrs={
            'class':       'form-control',
            'placeholder': 'Type Here. (500 character limit)',
            'required':    True,
        }
    )
)
widgetEmailField = django.forms.EmailField(
    required=True,
    widget=django.forms.TextInput(
        attrs={
            'class':       'form-control',
            'placeholder': 'example@email.com',
            'required':    True,
        }
    )
)
widgetEmailinput = django.forms.EmailInput(
    attrs={
        'class':       'form-control',
        'name':        'email',
        'placeholder': 'Email',
        'required':    True,
        'type':        'email',
    }
)
widgetTermsOfService = django.forms.CheckboxInput(
    attrs={
        'required': True,
    }
)

# DJANGOFY BSS ############################################################################################
# !C:\Users\keren\AppData\Local\Enthought\Canopy\edm\envs\User\
import datetime
import os
import re
import shutil
import sys

try:
    projectRoot = sys.argv[1]
except:
    projectRoot = '.'
print(projectRoot)
assetsDir = 'assets'
components = 'bss-components'
subpages = ['paypal-button.html', ]
urls = ['https:', 'http:', '.com', '.net', '.io', '.org', '.gov']

rename = {'home:home': 'home'}

images = [

]


def get_apps():
    m = next(os.walk(projectRoot))[1]
    return [i for i in m if i != assetsDir]


def is_url(string):
    for u in urls:
        if u in string:
            return True
    return False


def get_images():
    images = []
    imgPath = os.path.join('assets', 'img')
    for root, dirs, files in os.walk(imgPath):
        for file in files:
            images.append(file)
    return images


def get_files():
    files = []
    for root, dirs, fs in os.walk(projectRoot):
        for file in fs:
            if '.html' in file:
                root = ''
                for app in get_apps():
                    if app in root:
                        root = app
                        break
                files.append(os.path.join(root, file))
    return files


def process_images(content, root):
    for image in images:
        i1 = "{% static 'assets/img/" + image + "' %}"
        i0 = root + 'assets/img/' + image
        content = content.replace(i0, i1)
    return content


def process_css(string, root):
    if '<link rel="stylesheet" href="' + root in string and not is_url(string):
        result = string
        L0 = '<link rel="stylesheet" href="' + root
        L1 = "<link rel=\"stylesheet\" href=\"{% static '"
        result = result.replace(L0, L1)
        L0 = '.css">'
        L1 = ".css' %}\">"
        result = result.replace(L0, L1)
        return result
    return string


def process_scripts(string, root):
    if '<script src="' + root in string and not is_url(string):
        result = string
        L0 = '<script src="' + root
        L1 = "<script src=\"{% static '"
        result = result.replace(L0, L1)
        L0 = '.js">'
        L1 = ".js' %}\">"
        result = result.replace(L0, L1)
        return result
    return string


def process_links(string):
    if bool(re.search(r"<a href=[\'\"]\.*/*[\w+/*]*[\w+].html[\'\"]", string)) and not is_url(string):
        result = string
        matches = re.finditer(r"[\'\"]\.*/*[\w+/*]*[\w+].html[\'\"]", result)
        for d in matches:
            match = d.group(0)
            x = match.replace('"', '').replace("'", '').split('/')
            folder = x[-2]
            file = x[-1].split('.')[0]
            y = '{}:{}'.format(folder, file)
            if y in rename:
                y = rename[y]
            y = "\"{% url '" + y + "' %}\""
            result = result.replace(match, y)
        return result
    return string


def load_static(m):
    return '{% load static %}\n' + m


def fix(filename):
    with open(filename, 'r+', encoding='latin-1') as f:
        if os.sep in filename:
            root = '../'
        else:
            root = ''
        print(filename)
        content = f.read()
        content = process_images(content, root)
        m = ''
        for string in content.split('\n'):
            string = process_css(string, root)
            string = process_scripts(string, root)
            string = process_links(string)
            m += string + '\n'
        content = load_static(m)
        if os.sep in filename and filename not in subpages:
            content += '\n{% endblock %}'
            f.seek(0, 0)
            line = "{% extends 'base.html' %}\n{% block body %}"
            f.write(line.rstrip('\r\n') + '\n' + content)
        else:
            f.seek(0, 0)
            f.write(content)


def run():
    print(get_apps())
    root = os.getcwd()
    for app in get_apps():
        path = os.path.join(root, app, components)
        try:
            shutil.rm(path)
        except Exception as e:
            print(e)
    images = get_images()
    print(images)
    files = get_files()
    for file in files:
        print(file)
    for file in files:
        fix(file)
    print(datetime.datetime.now())
