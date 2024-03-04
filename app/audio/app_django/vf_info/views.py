from django.core.urlresolvers import reverse
from django.views.generic import FormView, TemplateView
from util import templates_util

import app.data.app_django.tensorstone.configuration
from . import initializer
from .forms import MessageForm


class TSViewContactUs(FormView):
    form_class = MessageForm
    template_name = 'template/info/contact-us.html'

    def get_success_url(self, **kwargs):
        return reverse('info:success', kwargs={'message_id': self.ferm.message_id})

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        self.ferm = form.save(commit=False)
        self.ferm.message_id = str(initializer.generate_job_id())
        self.ferm.save()
        if form.is_valid():
            return self.form_valid(form)
        else:
            return self.form_invalid(form)


class TSViewCredits(TemplateView):
    template_name = templates_util.creditsTemplate  # 'legal/credits1.html'


class TSViewFaq(TemplateView):
    template_name = templates_util.faqTemplate  # 'info/faq.html'


class TSViewHome(TemplateView):
    template_name = app.tensorstone.app_django.ts_general.configuration.homeTemplate  # 'home/home2.html'


class TSViewPortfolio(TemplateView):
    template_name = templates_util.portfolioTemplate  # 'info/portfolio.html'


class TSViewPrivacy(TemplateView):
    template_name = templates_util.privacyTemplate  # 'legal/privacy3.html'


class TSViewServices(TemplateView):
    template_name = templates_util.servicesTemplate  # 'info/services.html'


class TSViewSuccess(TemplateView):
    template_name = 'template/info/success.html'


class TSViewTerms(TemplateView):
    template_name = templates_util.termsTemplate  # 'legal/terms3.html'
