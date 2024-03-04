import datetime

import initializer
from django.core.urlresolvers import reverse, reverse_lazy
from django.db import transaction
from django.forms.models import model_to_dict
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.utils.encoding import smart_str
from django.views.generic import TemplateView
from django.views.generic.edit import FormView
import util
from . import configuration
from . import forms
from . import models


class TSViewFormMixing(FormView):
    form_class = forms.OrderForm
    template_name = 'order/mixing-form.html'

    def get_success_url(self, **kwargs):
        return reverse('order:success', kwargs={'job_id': self.ferm.job_id})

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        self.form = self.get_form(form_class)
        self.ferm = self.form.save(commit=False)
        if self.ferm.date_due == "":
            nextWeek = (datetime.now().date() + datetime.timedelta(days=7))  # .strftime('%m/%d/%Y')
            self.ferm.due_date = nextWeek
        if self.form.is_valid():  # and validate_inputs(ferm):
            self.ferm.job_id = str(initializer.generate_job_id())
            self.ferm.save()
            files = request.FILES.getlist('files')
            initializer.handle(files, str(self.ferm.job_id))
            return self.form_valid()
        else:
            return self.form_invalid()

    def form_valid(self):
        self.request.session['farm'] = model_to_dict(self.ferm)
        return super(TSViewFormMixing, self).form_valid(self.form)

    def validate_inputs(self, ferm):
        darts = [ferm.files, ferm.email, ferm.date_due, ferm.price

                 ]
        for dart in darts:
            if not dart:
                self.success_url = reverse('info:success')
                return False
        return True


class TSViewFormMastering(FormView):
    form_class = forms.OrderForm
    template_name = 'order/mastering-form.html'

    def get_success_url(self, **kwargs):
        return reverse('order:success', kwargs={'job_id': self.ferm.job_id})

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        self.form = self.get_form(form_class)
        self.ferm = self.form.save(commit=False)
        if self.ferm.date_due == "":
            nextWeek = (datetime.now().date() + datetime.timedelta(days=7))  # .strftime('%m/%d/%Y')
            self.ferm.due_date = nextWeek
        if self.form.is_valid():  # and validate_inputs(ferm):
            self.ferm.job_id = str(initializer.generate_job_id())
            self.ferm.save()
            files = request.FILES.getlist('files')
            initializer.handle(files, str(self.ferm.job_id))
            return self.form_valid()
        else:
            return self.form_invalid()

    def form_valid(self):
        self.request.session['farm'] = model_to_dict(self.ferm)
        return super(TSViewFormMastering, self).form_valid(self.form)

    def validate_inputs(self, ferm):
        darts = [ferm.files, ferm.email, ferm.date_due, ferm.price

                 ]
        for dart in darts:
            if not dart:
                self.success_url = reverse('info:success')
                return False
        return True


class TSViewFormOrder(FormView):
    form_class = forms.OrderForm
    template_name = configuration.orderFormTemplate

    def get_success_url(self, **kwargs):
        return reverse('order:success', kwargs={'job_id': self.orderModel.job_id})

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        self.form = self.get_form(form_class)
        self.orderModel = self.form.save(commit=False)
        self.orderModel.tracks = request.FILES.getlist('files')
        errs = util.validate_order_form(self.orderModel.tracks, self.orderModel.email)

        if self.form.is_valid():
            if self.orderModel.coupon_redeemed:
                self.orderModel, self.newCode = apply_coupon(self.orderModel)
            else:
                self.reset_discount()
            self.validate_date_due()
            if errs:
                return render(request, 'order/form.html', {'form': self.form, 'errs': errs})
            self.orderModel.number_of_files = self.orderModel.tracks.__len__()
            self.orderModel.save()
            util.upload_files(self.orderModel.tracks, str(self.orderModel.job_id))
            return self.form_valid()
        else:
            return self.form_invalid(request, errs)

    def reset_discount(self):
        self.newCode = ''
        self.orderModel.discount_redeemed = '$0'
        self.orderModel.discount_generated = '$0'

    def validate_date_due(self):
        if self.orderModel.date_due == "":
            nextWeek = (datetime.now().date() + datetime.timedelta(days=7))  # .strftime('%m/%d/%Y')
            self.orderModel.due_date = nextWeek

    def form_valid(self):
        self.request.session['orderModel'] = model_to_dict(self.orderModel)
        self.request.session['newCode'] = self.newCode
        self.request.session['tracks'] = self.orderModel.tracks
        return super(TSViewFormOrder, self).form_valid(self.form)

    def form_invalid(self, request, errs):
        return render(request, 'order/form.html', {'form': self.form, 'errs': errs})

    def get_success_url(self, **kwargs):
        return reverse('order:success', kwargs={'job_id': self.orderModel.job_id})

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        self.form = self.get_form(form_class)
        self.orderModel = self.form.save(commit=False)
        self.orderModel.tracks = request.FILES.getlist('files')
        errs = util.validate_order_form(self.orderModel.tracks, self.orderModel.email)

        if self.form.is_valid():
            if self.orderModel.coupon_redeemed:
                self.orderModel, self.newCode = apply_coupon(self.orderModel)
            else:
                self.reset_discount()
            self.validate_date_due()
            if errs:
                return render(request, 'order/form0.html', {'form': self.form, 'errs': errs})
            self.orderModel.number_of_files = self.orderModel.tracks.__len__()
            self.orderModel.save()
            util.upload_files(self.orderModel.tracks, str(self.orderModel.job_id))
            return self.form_valid()
        else:
            return self.form_invalid(request, errs)

    def reset_discount(self):
        self.newCode = ''
        self.orderModel.discount_redeemed = '$0'
        self.orderModel.discount_generated = '$0'

    def validate_date_due(self):
        if self.orderModel.date_due == "":
            nextWeek = (datetime.now().date() + datetime.timedelta(days=7))  # .strftime('%m/%d/%Y')
            self.orderModel.due_date = nextWeek

    def form_valid(self):
        self.request.session['orderModel'] = model_to_dict(self.orderModel)
        self.request.session['newCode'] = self.newCode
        self.request.session['tracks'] = self.orderModel.tracks
        return super(TSViewFormOrder, self).form_valid(self.form)

    def form_invalid(self, request, errs):
        return render(request, 'order/form0.html', {'form': self.form, 'errs': errs})

    def get_success_url(self, **kwargs):
        return reverse('order:success', kwargs={'job_id': self.ferm.job_id})

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        self.form = self.get_form(form_class)
        self.ferm = self.form.save(commit=False)

        if self.form.is_valid():
            if self.ferm.coupon_redeemed:
                self.ferm, self.newCode = apply_coupon(self.ferm)
            else:
                self.newCode = ''
                self.ferm.discount_redeemed = '$0'
                self.ferm.discount_generated = '$0'
            if self.ferm.date_due == "":
                nextWeek = (datetime.now().date() + datetime.timedelta(days=7))  # .strftime('%m/%d/%Y')
                self.ferm.due_date = nextWeek
            self.ferm.job_id = str(util.generate_job_id())
            files = request.FILES.getlist('files')
            errs = util.validate_order_form(files, self.ferm.email)
            if errs:
                return render(request, 'order/form.html', {'form': self.form, 'errs': errs})
            self.ferm.number_of_files = files.__len__()
            self.ferm.save()
            util.upload_files(files, str(self.ferm.job_id))
            return self.form_valid()
        else:
            files = request.FILES.getlist('files')
            errs = util.validate_order_form(files, self.ferm.email)
            return self.form_invalid(request, errs)

    def form_valid(self):
        self.request.session['farm'] = model_to_dict(self.ferm)
        self.request.session['newCode'] = self.newCode
        return super(TSViewFormOrder, self).form_valid(self.form)

    def form_invalid(self, request, errs):
        return render(request, 'order/form.html', {'form': self.form, 'errs': errs})

    def get_success_url(self, **kwargs):
        return reverse('order:success', kwargs={'job_id': self.ferm.job_id})

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        self.form = self.get_form(form_class)
        self.ferm = self.form.save(commit=False)

        if self.form.is_valid():
            if self.ferm.coupon_redeemed:
                self.ferm, self.newCode = apply_coupon(self.ferm)
            else:
                self.newCode = ''
                self.ferm.discount_redeemed = '$0'
                self.ferm.discount_generated = '$0'
            if self.ferm.date_due == "":
                nextWeek = (datetime.now().date() + datetime.timedelta(days=7))  # .strftime('%m/%d/%Y')
                self.ferm.due_date = nextWeek
            self.ferm.job_id = str(util.generate_job_id())
            files = request.FILES.getlist('files')
            errs = util.validate_order_form(files, self.ferm.email)
            if errs:
                return render(request, 'order/form.html', {'form': self.form, 'errs': errs})
            self.ferm.number_of_files = files.__len__()
            self.ferm.save()
            util.upload_files(files, str(self.ferm.job_id))
            return self.form_valid()
        else:
            files = request.FILES.getlist('files')
            errs = util.validate_order_form(files, self.ferm.email)
            return self.form_invalid(request, errs)

    def form_valid(self):
        self.request.session['farm'] = model_to_dict(self.ferm)
        self.request.session['newCode'] = self.newCode
        return super(TSViewFormOrder, self).form_valid(self.form)

    def form_invalid(self, request, errs):
        return render(request, 'order/form.html', {'form': self.form, 'errs': errs})


class TSViewOrder(FormView):
    form_class = OrderForm
    template_name = 'order/order-form.html'
    success_url = reverse_lazy('order:success')

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        ferm = form.save(commit=False)
        ferm.job_id = initializer.generate_job_id()
        ferm.save()
        files = request.FILES.getlist('file')
        if form.is_valid():
            initializer.handle(files, str(ferm.job_id))
            return self.form_valid(form)
        else:
            return self.form_invalid(form)

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        ferm = form.save(commit=False)
        ferm.job_id = initializer.generate_job_id()
        ferm.save()
        files = request.FILES.getlist('file')
        if form.is_valid():
            initializer.handle(files, str(ferm.job_id))
            return self.form_valid(form)
        else:
            return self.form_invalid(form)


class TSViewOrder(TemplateView):
    template_name = 'order/order-form.html'

    def get(self, request):
        form = PersonalInfoForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = PersonalInfoForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.id = initializer.generate_job_id()
            post.user = request.user
            post.save()
            etext = form.cleaned_data['email']
            dtext = form.cleaned_data['details']
            form = PersonalInfoForm()
            args = {'form': form, 'etext': etext, 'dtext': dtext}
            return render(request, self.template_name, args)


class TSViewServiceSelection(TemplateView):
    template_name = 'order/service-selection.html'


class TSViewSuccess(TemplateView):
    model = VFModelOrder
    template_name = 'order/success.html'


class TSViewSuccessOrder(TemplateView):
    model = VFModelOrder
    template_name = templates_util.orderSuccessTemplate


def ajax_validate_coupon(request):
    code = request.GET.get('coupon', None)
    data = {'isValid': coupon_util.coupon_exists(VFModelCoupon, code)}
    if data['isValid'] == False:
        data['errorMessage'] = 'The coupon you entered is invalid'
    else:
        koupon = VFModelCoupon.objects.get(code__iexact=code)
        data['discount'] = coupon_util.get_formatted_discount(koupon)
    return JsonResponse(data)


def ajax_validate_coupon(request):
    code = request.GET.get('coupon', None)
    data = {'isValid': util.coupon_exists(code)}
    if data['isValid'] == False:
        data['errorMessage'] = 'The coupon you entered is invalid'
    else:
        koupon = VFModelCoupon.objects.get(code__iexact=code)
        data['discount'] = util.get_formatted_discount(koupon)
    return JsonResponse(data)


@transaction.atomic
def apply_coupon(ferm):
    coupon_util.update_coupons(VFModelCoupon)
    try:
        coupon = coupon_util.get_coupon(VFModelCoupon, ferm.coupon_redeemed)
        price = float(ferm.price[1:])
        discount = float(coupon.discount)
        if coupon.percentage:
            ferm.discount_redeemed = '{}%'.format(discount)
        else:
            if (price > discount):
                ferm.discount_redeemed = '${}'.format(discount)
            else:
                ferm.discount_redeemed = '${}'.format(price)
                newCode, newDiscount = coupon_util.handle_excess_discount(price, discount)
                ferm.coupon_generated = newCode
                ferm.discount_generated = '${}'.format(newDiscount)
                ferm.save()
        coupon.remaining_uses = int(coupon.remaining_uses) - 1
        if not coupon_util.is_valid_coupon(coupon):
            coupon.delete()
        else:
            coupon.save()
        coupon_util.update_coupons(VFModelCoupon)
        return ferm, newCode
    except VFModelCoupon.DoesNotExist:
        ferm.discount_redeemed = '$0.00'
        coupon_util.update_coupons(VFModelCoupon)
        return ferm, ''


@transaction.atomic
def apply_coupon(ferm):
    util.update_coupons()
    try:
        coupon = util.get_coupon(ferm.coupon_redeemed)
        price = float(ferm.price[1:])
        discount = float(coupon.discount)
        if coupon.percentage:
            ferm.discount_redeemed = '{}%'.format(discount)
        else:
            if (price > discount):
                ferm.discount_redeemed = '${}'.format(discount)
            else:
                ferm.discount_redeemed = '${}'.format(price)
                newCode, newDiscount = util.handle_excess_discount(price, discount)
                ferm.coupon_generated = newCode
                ferm.discount_generated = '${}'.format(newDiscount)
                ferm.save()
        coupon.remaining_uses = int(coupon.remaining_uses) - 1
        if not is_valid_coupon(coupon):
            coupon.delete()
        else:
            coupon.save()
        util.update_coupons()
        return ferm, newCode
    except VFModelCoupon.DoesNotExist:
        ferm.discount_redeemed = '$0.00'
        util.update_coupons()
        return ferm, ''


@transaction.atomic
def apply_coupon(ferm):
    util.update_coupons()
    try:
        coupon = util.get_coupon(ferm.coupon_redeemed)
        price = float(ferm.price[1:])
        discount = float(coupon.discount)
        if coupon.percentage:
            ferm.discount_redeemed = '{}%'.format(discount)
        else:
            if (price > discount):
                ferm.discount_redeemed = '${}'.format(discount)
            else:
                ferm.discount_redeemed = '${}'.format(price)
                newCode, newDiscount = util.handle_excess_discount(price, discount)
                ferm.coupon_generated = newCode
                ferm.discount_generated = '${}'.format(newDiscount)
                ferm.save()
        coupon.remaining_uses = int(coupon.remaining_uses) - 1
        if not util.is_valid_coupon(coupon):
            coupon.delete()
        else:
            coupon.save()
        util.update_coupons()
        return ferm, newCode
    except VFModelCoupon.DoesNotExist:
        ferm.discount_redeemed = '$0.00'
        util.update_coupons()
        return ferm, ''


@transaction.atomic
def apply_coupon(orderModel):
    coupon_util.update_coupons(VFModelCoupon)
    try:
        coupon = coupon_util.get_coupon(VFModelCoupon, orderModel.coupon_redeemed)
        price = float(orderModel.price[1:])
        discount = float(coupon.discount)
        if coupon.percentage:
            orderModel.discount_redeemed = '{}%'.format(discount)
        else:
            if (price > discount):
                orderModel.discount_redeemed = '${}'.format(discount)
            else:
                orderModel.discount_redeemed = '${}'.format(price)
                newCode, newDiscount = coupon_util.handle_excess_discount(price, discount)
                orderModel.coupon_generated = newCode
                orderModel.discount_generated = '${}'.format(newDiscount)
                orderModel.save()
        coupon.remaining_uses = int(coupon.remaining_uses) - 1
        if not coupon_util.is_valid_coupon(coupon):
            coupon.delete()
        else:
            coupon.save()
        coupon_util.update_coupons(VFModelCoupon)
        return orderModel, newCode
    except VFModelCoupon.DoesNotExist:
        orderModel.discount_redeemed = '$0.00'
        coupon_util.update_coupons(VFModelCoupon)
        return orderModel, ''


def results(request, *args, **kwargs):
    p = request.path.split('/')
    fileExt = p[-1]
    fileDir = p[-2]
    fileName = '{0}.{1}'.format(VFModelOrder.objects.get(job_id=kwargs['job_id']).project_name, fileExt)
    filePath = '/media/{0}/{1}'.format(fileDir, fileName)
    response = HttpResponse()
    response['Content-Disposition'] = 'attachment; filename={0}'.format(smart_str(fileName))
    response['X-Accel-Redirect'] = smart_str(filePath)
    response['X-Sendfile'] = smart_str(filePath)
    return response


def upload_file(request):
    template_name = 'order/order-form.html'
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            handle(request.FILES.getlist('files'))
            return render(
                request, template_name, {
                    'form':  form, 'etext': form.cleaned_data['title'],
                    'dtext': [str(f) for f in request.FILES.getlist('file')]
                }
            )
    else:
        form = UploadForm()
    return render(request, template_name, {'form': form})
