import django.contrib.admin
from . import models

# Register your models here.
django.contrib.admin.site.register(models.VFModelCoupon)
django.contrib.admin.site.register(models.VFModelOrder)
