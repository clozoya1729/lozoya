import django.views.generic

from . import configuration


class AboutView(django.views.generic.TemplateView):
    template_name = configuration.templateAbout


class CreditsView(django.views.generic.TemplateView):
    template_name = configuration.templateCredits


class FaqView(django.views.generic.TemplateView):
    template_name = configuration.templateFAQ


class HomeView(django.views.generic.TemplateView):
    template_name = configuration.templateHome


class PrivacyView(django.views.generic.TemplateView):
    template_name = configuration.templatePrivacy


class ServicesView(django.views.generic.TemplateView):
    template_name = configuration.templateServices


class TermsView(django.views.generic.TemplateView):
    template_name = configuration.templateTerms


class TutorialsView(django.views.generic.TemplateView):
    template_name = configuration.templateTutorial
