import datetime
import os
import random
import string
import uuid

from django.utils import timezone

from . import configuration, models

supportedExt = ['wav', 'mp3']


def coupon_exists(code):
    return models.VFModelCoupon.objects.filter(code__iexact=code).exists()


def format_date(date):
    return date.strftime('%m/%d/%Y')


def get_formatted_discount(couponModel):
    if couponModel.percentage:
        return '{}%'.format(couponModel.discount)
    return '${}'.format(couponModel.discount)


def generate_code():
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(10))


def generate_leftover_coupon(couponModel, code, discount):
    now = timezone.now()
    coupon = couponModel(
        code=code, valid_from=now, valid_to=now + timezone.timedelta(days=3650), discount=discount,
        percentage=False, remaining_uses=1
    )
    coupon.save()


def generate_job_id():
    jobID = uuid.uuid1()
    return jobID


def generate_job_id():
    jobID = uuid.uuid1()
    return jobID


def generate_job_id():
    jobID = uuid.uuid1()
    return jobID


def get_coupon(couponModel, code):
    now = timezone.now()
    return couponModel.objects.select_for_update().get(
        code__iexact=code, valid_from__lte=now, valid_to__gte=now,
        remaining_uses__gte=1
    )


def get_next_week():
    return (get_today() + datetime.timedelta(days=7))


def get_today():
    return datetime.datetime.now().date()


def get_unique_code():
    while True:
        code = generate_code()
        try:
            coupon = models.VFModelCoupon.objects.select_for_update().get(code__iexact=code)
            if not is_valid_coupon(coupon):
                coupon.delete()
        except models.VFModelCoupon.DoesNotExist:
            return code


def handle(files, directory):
    projectDir = os.path.join('uploads', directory)
    tracksDir = os.path.join(projectDir, 'Tracks')
    os.mkdir(projectDir)
    os.mkdir(tracksDir)
    for f in files:
        with open(os.path.join(tracksDir, str(f)), 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)


def handle_excess_discount(price, discount):
    newCode = get_unique_code()
    newDiscount = discount - price
    generate_leftover_coupon(newCode, newDiscount)
    return newCode, newDiscount


def is_valid_coupon(coupon):
    now = timezone.now()
    return ((coupon.valid_from < now) and (coupon.valid_to > now) and (coupon.remaining_uses > 1))


def validate_order_form(files, email):
    errs = []
    for f in files:
        name = str(f)
        ext = name.split('.')[-1]
        if name.count('.') > 1:
            errs.append('{0} has multiple extensions. Files must have only one extension.'.format(name))
        if ext not in supportedExt:
            errs.append('{0} files are not currently supported.'.format(ext))
    if str(email).count('@') < 1:
        errs.append('Enter a valid email address.')
    return errs if errs.__len__() > 0 else None


def update_coupons():
    for coupon in models.VFModelCoupon.objects.all():
        if not is_valid_coupon(coupon):
            coupon.delete()


def update_coupons(couponModel):
    couponModel.objects.filter(remaining_uses__lt=1).delete()
    couponModel.objects.filter(valid_to__lt=timezone.now()).delete()


def upload_files(files, directory):
    projectDir = os.path.join(configuration.uploadsPath, directory)
    os.mkdir(projectDir)
    tracksDir = os.path.join(projectDir, configuration.tracksPath)
    os.mkdir(tracksDir)
    _backupDir = os.path.join(configuration.backupPath, directory)
    os.mkdir(_backupDir)
    backupDir = os.path.join(_backupDir, configuration.tracksPath)
    os.mkdir(backupDir)
    resultsDir = os.path.join(configuration.resultsPath, directory)
    os.mkdir(resultsDir)
    for f in files:
        with open(os.path.join(tracksDir, str(f)), 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        with open(os.path.join(backupDir, str(f)), 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)


def upload_files(files, directory):
    projectDir = os.path.join('uploads', directory)
    os.mkdir(projectDir)
    tracksDir = os.path.join(projectDir, 'Tracks')
    os.mkdir(tracksDir)
    backupDir = os.path.join('/mnt/uploads', directory)
    os.mkdir(backupDir)
    backupDir = os.path.join(backupDir, 'Tracks')
    os.mkdir(backupDir)
    resultsDir = os.path.join('results', directory)
    os.mkdir(resultsDir)
    for f in files:
        with open(os.path.join(tracksDir, str(f)), 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        with open(os.path.join(backupDir, str(f)), 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
