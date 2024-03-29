"""
Django settings for vectorflux_website project.
Generated by 'django-admin startproject' using Django 2.2.5.
For more information on this file, see
https://docs.djangoproject.com/en/2.2/topics/settings/
For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.2/ref/settings/
"""
import os
import netifaces

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'xb9%bt2fvynj8n79(vd)fg0fnsm42bukio6z(#@((r(gk1dg6o'
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
ALLOWED_HOSTS = []
# Application definition
INSTALLED_APPS = ['django.contrib.admin', 'django.contrib.auth', 'django.contrib.contenttypes',
                  'django.contrib.sessions', 'django.contrib.messages', 'django.contrib.staticfiles',
                  'home.apps.HomeConfig', 'contact.apps.ContactConfig', 'info.apps.InfoConfig',
                  'legal.apps.LegalConfig', 'order.apps.OrderConfig', 'vectorflux.apps.VFConfigGeneral', ]
MIDDLEWARE = ['django.middleware.security.SecurityMiddleware', 'django.contrib.sessions.middleware.SessionMiddleware',
              'django.middleware.common.CommonMiddleware', 'django.middleware.csrf.CsrfViewMiddleware',
              'django.contrib.auth.middleware.AuthenticationMiddleware',
              'django.contrib.messages.middleware.MessageMiddleware',
              'django.middleware.clickjacking.XFrameOptionsMiddleware', ]
ROOT_URLCONF = 'vectorflux_website.urls'
TEMPLATES = [
    {
        'BACKEND':  'django.template.backends.django.DjangoTemplates', 'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True, 'OPTIONS': {
        'context_processors': ['django.template.context_processors.debug', 'django.template.context_processors.request',
                               'django.contrib.auth.context_processors.auth',
                               'django.contrib.messages.context_processors.messages', ],
    },
    }, ]
WSGI_APPLICATION = 'vectorflux_website.wsgi.application'
# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases
DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': os.path.join(BASE_DIR, 'db.sqlite3'), }}
# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators
AUTH_PASSWORD_VALIDATORS = [{'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator', },
                            {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', },
                            {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator', },
                            {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator', }, ]
# Internationalization
# https://docs.djangoproject.com/en/1.8/topics/i18n/
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'America/Denver'
USE_I18N = True
USE_L10N = True
USE_TZ = True
# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/
STATIC_URL = '/static/'
STATIC_ROOT = "/static/"
STATICFILES_DIRS = [os.path.join(BASE_DIR, "static"), ]
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_PROJECT = os.path.realpath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.8/howto/deployment/checklist/
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'bej@4gkj87rx=-1o^!5j-y!a%%*_#-c-*_^#*l98eqqwy0#$)l'
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
ALLOWED_HOSTS = ['audio.com', 'www.audio.com', '142.93.19.36']
# Application definition
INSTALLED_APPS = (
    'django.contrib.admin', 'django.contrib.auth', 'django.contrib.contenttypes', 'django.contrib.sessions',
    'django.contrib.messages', 'django.contrib.staticfiles', 'home.apps.HomeConfig', 'order.apps.OrderConfig',
    'info.apps.InfoConfig',)
MIDDLEWARE_CLASSES = (
    'django.contrib.sessions.middleware.SessionMiddleware', 'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware', 'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.auth.middleware.SessionAuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware', 'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django.middleware.security.SecurityMiddleware',)
ROOT_URLCONF = 'django_project.urls'
WSGI_APPLICATION = 'django_project.wsgi.application'


# STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
# Allow Django from all hosts. This snippet is installed from
# /var/lozoya/digitalocean/allow_hosts.py
# Find out what the IP addresses are at run time
# This is necessary because otherwise Gunicorn will reject the connections
def ip_addresses():
    ip_list = []
    for interface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(interface)
        for x in (netifaces.AF_INET, netifaces.AF_INET6):
            if x in addrs:
                ip_list.append(addrs[x][0]['addr'])
    return ip_list


# Discover our IP address
# ALLOWED_HOSTS += ip_addresses()
'''TEMPLATE_DIRS = (
    # Put strings here, like "/home/html/django_templates" or "C:/www/django/templates".
    # Always use forward slashes, even on Windows.
    # Don't forget to use absolute paths, not relative paths.
    os.path.join(PROJECT_ROOT, 'templates').replace('\\','/'),
)
# List of callables that know how to import templates from various sources.
TEMPLATE_LOADERS = (
    'django.template.loaders.filesystem.Loader',
    'django.template.loaders.app_directories.Loader',
#     'django.template.loaders.eggs.Loader',
)'''
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_PROJECT = os.path.realpath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.8/howto/deployment/checklist/
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'bej@4gkj87rx=-1o^!5j-y!a%%*_#-c-*_^#*l98eqqwy0#$)l'
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
ALLOWED_HOSTS = ['audio.com', 'www.audio.com', '142.93.19.36']
# Application definition
INSTALLED_APPS = (
    'django.contrib.admin', 'django.contrib.auth', 'django.contrib.contenttypes', 'django.contrib.sessions',
    'django.contrib.messages', 'django.contrib.staticfiles', 'home.apps.HomeConfig', 'order.apps.OrderConfig',
    'info.apps.InfoConfig',)
ROOT_URLCONF = 'django_project.urls'
TEMPLATES = [
    {
        'BACKEND':  'django.template.backends.django.DjangoTemplates', 'DIRS': [os.path.join(BASE_DIR, 'templates'), ],
        'APP_DIRS': True, 'OPTIONS': {
        'context_processors': ['django.template.context_processors.debug', 'django.template.context_processors.request',
                               'django.contrib.auth.context_processors.auth',
                               'django.contrib.messages.context_processors.messages', ],
    },
    }, ]
WSGI_APPLICATION = 'django_project.wsgi.application'
# Database
# https://docs.djangoproject.com/en/1.8/ref/settings/#databases
DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': os.path.join(BASE_DIR, 'db.sqlite3'), }}
# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.8/howto/static-files/
STATIC_URL = '/static/'
# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
# Allow Django from all hosts. This snippet is installed from
# /var/lozoya/digitalocean/allow_hosts.py
# Discover our IP address
# ALLOWED_HOSTS += ip_addresses()
'''TEMPLATE_DIRS = (
    # Put strings here, like "/home/html/django_templates" or "C:/www/django/templates".
    # Always use forward slashes, even on Windows.
    # Don't forget to use absolute paths, not relative paths.
    os.path.join(PROJECT_ROOT, 'templates').replace('\\','/'),
)
# List of callables that know how to import templates from various sources.
TEMPLATE_LOADERS = (
    'django.template.loaders.filesystem.Loader',
    'django.template.loaders.app_directories.Loader',
#     'django.template.loaders.eggs.Loader',
)'''
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_PROJECT = os.path.realpath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.8/howto/deployment/checklist/
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'bej@4gkj87rx=-1o^!5j-y!a%%*_#-c-*_^#*l98eqqwy0#$)l'
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
ALLOWED_HOSTS = ['audio.com', 'www.audio.com', '142.93.19.36']
# Application definition
INSTALLED_APPS = (
    'django.contrib.admin', 'django.contrib.auth', 'django.contrib.contenttypes', 'django.contrib.sessions',
    'django.contrib.messages', 'django.contrib.staticfiles', 'home.apps.HomeConfig', 'order.apps.OrderConfig',
    'info.apps.InfoConfig',)
MIDDLEWARE_CLASSES = (
    'django.contrib.sessions.middleware.SessionMiddleware', 'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware', 'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.auth.middleware.SessionAuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware', 'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django.middleware.security.SecurityMiddleware',)
ROOT_URLCONF = 'django_project.urls'
WSGI_APPLICATION = 'django_project.wsgi.application'
# Database
# https://docs.djangoproject.com/en/1.8/ref/settings/#databases
DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': os.path.join(BASE_DIR, 'db.sqlite3'), }}

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.8/howto/static-files/
STATIC_URL = '/static/'
# Allow Django from all hosts. This snippet is installed from
# /var/lozoya/digitalocean/allow_hosts.py
import netifaces

# Discover our IP address
# ALLOWED_HOSTS += ip_addresses()
'''TEMPLATE_DIRS = (
    # Put strings here, like "/home/html/django_templates" or "C:/www/django/templates".
    # Always use forward slashes, even on Windows.
    # Don't forget to use absolute paths, not relative paths.
    os.path.join(PROJECT_ROOT, 'templates').replace('\\','/'),
)
# List of callables that know how to import templates from various sources.
TEMPLATE_LOADERS = (
    'django.template.loaders.filesystem.Loader',
    'django.template.loaders.app_directories.Loader',
#     'django.template.loaders.eggs.Loader',
)'''
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'xb9%bt2fvynj8n79(vd)fg0fnsm42bukio6z(#@((r(gk1dg6o'
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
ALLOWED_HOSTS = []
# Application definition
INSTALLED_APPS = ['django.contrib.admin', 'django.contrib.auth', 'django.contrib.contenttypes',
                  'django.contrib.sessions', 'django.contrib.messages', 'django.contrib.staticfiles',
                  'home.apps.HomeConfig', 'contact.apps.ContactConfig', 'info.apps.InfoConfig',
                  'legal.apps.LegalConfig', 'order.apps.OrderConfig', 'vectorflux0.apps.VectorfluxConfig', ]
MIDDLEWARE = ['django.middleware.security.SecurityMiddleware', 'django.contrib.sessions.middleware.SessionMiddleware',
              'django.middleware.common.CommonMiddleware', 'django.middleware.csrf.CsrfViewMiddleware',
              'django.contrib.auth.middleware.AuthenticationMiddleware',
              'django.contrib.messages.middleware.MessageMiddleware',
              'django.middleware.clickjacking.XFrameOptionsMiddleware', ]
ROOT_URLCONF = 'vectorflux_website.urls'
WSGI_APPLICATION = 'vectorflux_website.wsgi.application'
# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases
DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': os.path.join(BASE_DIR, 'db.sqlite3'), }}
# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators
AUTH_PASSWORD_VALIDATORS = [{'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator', },
                            {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', },
                            {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator', },
                            {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator', }, ]

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/
STATIC_URL = '/static/'
STATIC_ROOT = "/static/"
STATICFILES_DIRS = [os.path.join(BASE_DIR, "static"), ]
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'
# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'xb9%bt2fvynj8n79(vd)fg0fnsm42bukio6z(#@((r(gk1dg6o'
# SECURITY WARNING: don't run with debug turned on in production!
# Application definition
INSTALLED_APPS = ['django.contrib.admin', 'django.contrib.auth', 'django.contrib.contenttypes',
                  'django.contrib.sessions', 'django.contrib.messages', 'django.contrib.staticfiles',
                  'home.apps.HomeConfig', 'contact.apps.ContactConfig', 'info.apps.InfoConfig',
                  'legal.apps.LegalConfig',  # 'order.apps.OrderConfig',
                  'vectorflux.apps.VFConfigGeneral', ]
ROOT_URLCONF = 'vectorflux_website.urls'
WSGI_APPLICATION = 'vectorflux_website.wsgi.application'
# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases
DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': os.path.join(BASE_DIR, 'db.sqlite3'), }}
# import netifaces
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_PROJECT = os.path.realpath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.8/howto/deployment/checklist/
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'bej@4gkj87rx=-1o^!5j-y!a%%*_#-c-*_^#*l98eqqwy0#$)l'
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False
ALLOWED_HOSTS = ['audio.com', 'www.audio.com', '142.93.19.36']
# Application definition
INSTALLED_APPS = (
    'django.contrib.admin', 'django.contrib.auth', 'django.contrib.contenttypes', 'django.contrib.sessions',
    'django.contrib.messages', 'django.contrib.staticfiles', 'home.apps.HomeConfig', 'contact.apps.ContactConfig',
    'legal.apps.LegalConfig', 'order.apps.OrderConfig', 'info.apps.InfoConfig',)
MIDDLEWARE_CLASSES = (
    'django.contrib.sessions.middleware.SessionMiddleware', 'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware', 'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.auth.middleware.SessionAuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware', 'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django.middleware.security.SecurityMiddleware',)
ROOT_URLCONF = 'django_project.urls'
WSGI_APPLICATION = 'django_project.wsgi.application'
# Database
# https://docs.djangoproject.com/en/1.8/ref/settings/#databases
DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': os.path.join(BASE_DIR, 'db.sqlite3'), }}
# STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
# Allow Django from all hosts. This snippet is installed from
# /var/lozoya/digitalocean/allow_hosts.py
# Find out what the IP addresses are at run time
# This is necessary because otherwise Gunicorn will reject the connections
'''def ip_addresses():
    ip_list = []
    for interface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(interface)
        for x in (netifaces.AF_INET, netifaces.AF_INET6):
            if x in addrs:
                ip_list.append(addrs[x][0]['addr'])
    return ip_list'''
# Discover our IP address
# ALLOWED_HOSTS += ip_addresses()
MEDIA_URL = '/media/'
MEDIA_ROOT = 'media'  # os.path.join(BASE_DIR, 'media')
'''TEMPLATE_DIRS = (
    # Put strings here, like "/home/html/django_templates" or "C:/www/django/templates".
    # Always use forward slashes, even on Windows.
    # Don't forget to use absolute paths, not relative paths.
    os.path.join(PROJECT_ROOT, 'templates').replace('\\','/'),
)
# List of callables that know how to import templates from various sources.
TEMPLATE_LOADERS = (
    'django.template.loaders.filesystem.Loader',
    'django.template.loaders.app_directories.Loader',
#     'django.template.loaders.eggs.Loader',
)'''
