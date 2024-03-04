import datetime
import uuid

import date_util
import django.db.models
from django.utils import timezone

import lozoya.time
from util import date_util as du

today = lozoya.time.format_date(lozoya.time.get_today())
nextWeek = lozoya.time.format_date(lozoya.time.get_next_week())
today = datetime.datetime.now().date()
nextWeek = (today + datetime.timedelta(days=7)).strftime('%m/%d/%Y')
today = today.strftime('%m/%d/%Y')


# Create your models here.
class VFModelCoupon(django.db.models.Model):
    code = django.db.models.CharField(max_length=10, unique=True)
    valid_from = django.db.models.DateTimeField()
    valid_to = django.db.models.DateTimeField()
    discount = django.db.models.FloatField()
    percentage = django.db.models.BooleanField()
    remaining_uses = django.db.models.IntegerField()

    def __str__(self):
        return self.code


class _VFModelOrder(django.db.models.Model):
    job_id = django.db.models.CharField(max_length=36)
    email = django.db.models.EmailField(max_length=64)
    project_name = django.db.models.CharField(max_length=100)
    details = django.db.models.CharField(max_length=500, blank=True)
    reference = django.db.models.CharField(max_length=100, blank=True)
    date_submitted = django.db.models.CharField(max_length=10, default=today)
    date_due = django.db.models.CharField(max_length=10, default=nextWeek)
    price = django.db.models.CharField(max_length=10)
    terms = django.db.models.BooleanField()
    number_of_files = django.db.models.IntegerField(default=0)
    duration = django.db.models.CharField(max_length=7)
    # coupon = django.db.models.ForeignKey(Coupon, related_name='order', null=True, blank=True)
    coupon_redeemed = django.db.models.CharField(max_length=10, blank=True, null=True)
    discount_redeemed = django.db.models.CharField(max_length=4, default=0, blank=True, null=True)
    coupon_generated = django.db.models.CharField(max_length=10, blank=True, null=True)
    discount_generated = django.db.models.CharField(max_length=4, blank=True, null=True)

    def __str__(self):
        return self.job_id

    def clean(self):
        if self.date_due is None:
            self.date_due = nextWeek

    def __getitem__(self, item):
        return getattr(self, item)


class VFModelOrder(_VFModelOrder):
    mix = django.db.models.BooleanField(default=False)
    performance_correction = django.db.models.BooleanField(default=False)
    repair = django.db.models.BooleanField(default=False)
    master = django.db.models.BooleanField(default=False)

    class Meta:
        verbose_name_plural = 'Orders'


class _VFModelOrder(django.db.models.Model):
    job_id = django.db.models.CharField(help_text="", max_length=36)
    email = django.db.models.EmailField(help_text="", max_length=64)
    project_name = django.db.models.CharField(help_text="", max_length=100)
    details = django.db.models.CharField(help_text="", max_length=500, blank=True)
    reference = django.db.models.CharField(help_text="", max_length=100, blank=True)
    date_submitted = django.db.models.CharField(help_text="", max_length=10, default=lozoya.time.ftoday())
    date_due = django.db.models.CharField(help_text="", max_length=10, default=lozoya.time.fnext_week())
    price = django.db.models.CharField(help_text="", max_length=10)
    terms = django.db.models.BooleanField(help_text="", )
    number_of_files = django.db.models.IntegerField(help_text="", default=0)
    duration = django.db.models.CharField(help_text="", max_length=7)
    # coupon = django.db.models.ForeignKey(Coupon, related_name='order', null=True, blank=True)
    coupon_redeemed = django.db.models.CharField(help_text="", max_length=10, blank=True, null=True)
    discount_redeemed = django.db.models.CharField(help_text="", max_length=4, default=0, blank=True, null=True)
    coupon_generated = django.db.models.CharField(help_text="", max_length=10, blank=True, null=True)
    discount_generated = django.db.models.CharField(help_text="", max_length=4, blank=True, null=True)

    def __str__(self):
        return self.job_id

    def clean(self):
        if self.date_due is None:
            self.date_due = lozoya.time.fnext_week()

    def __getitem__(self, item):
        return getattr(self, item)


class VFModelOrder(_VFModelOrder):
    mix = django.db.models.BooleanField(help_text="", default=False)
    performance_correction = django.db.models.BooleanField(help_text="", default=False)
    repair = django.db.models.BooleanField(help_text="", default=False)
    master = django.db.models.BooleanField(help_text="", default=False)

    class Meta:
        verbose_name_plural = 'Orders'


class VFModelOrder(django.db.models.Model):
    job_id = django.db.models.CharField(max_length=36)
    email = django.db.models.EmailField(max_length=64)
    project_name = django.db.models.CharField(max_length=100)
    details = django.db.models.CharField(max_length=500, blank=True)
    reference = django.db.models.CharField(max_length=100, blank=True)
    date_submitted = django.db.models.CharField(max_length=10, default=today)
    date_due = django.db.models.CharField(max_length=10, default=nextWeek)
    price = django.db.models.CharField(max_length=10)
    terms = django.db.models.BooleanField()

    def __str__(self):
        return self.job_id

    def clean(self):
        if self.date_due is None:
            self.date_due = nextWeek

    def get_absolute_url(self):
        return reverse('order:success', kwargs={'job_id': self.job_id})


class VFModelOrder(django.db.models.Model):
    job_id = django.db.models.CharField(max_length=36, default='-1')
    email = django.db.models.EmailField(default='bdp@example.com')
    artist_name = django.db.models.CharField(max_length=100, default='Artist Name')
    song_title = django.db.models.CharField(max_length=100, default='Song Title')
    details = django.db.models.CharField(max_length=500)
    reference_artist = django.db.models.CharField(max_length=100, default='Reference Artist Name')
    reference_song = django.db.models.CharField(max_length=100, default='Reference Song Title')
    date_submitted = django.db.models.DateTimeField(default=timezone.localtime(timezone.now()))
    date_due = django.db.models.DateField()

    def __str__(self):
        return self.job_id


class VFModelOrder(django.db.models.Model):
    mix = django.db.models.BooleanField(help_text="", default=False)
    performance_correction = django.db.models.BooleanField(help_text="", default=False)
    repair = django.db.models.BooleanField(help_text="", default=False)
    master = django.db.models.BooleanField(help_text="", default=False)

    job_id = django.db.models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = django.db.models.EmailField(help_text="", max_length=64)
    project_name = django.db.models.CharField(help_text="", max_length=100)
    details = django.db.models.CharField(help_text="", max_length=500, blank=True)
    reference = django.db.models.CharField(help_text="", max_length=100, blank=True)
    date_submitted = django.db.models.CharField(help_text="", max_length=10, default=lozoya.time.ftoday())
    date_due = django.db.models.CharField(help_text="", max_length=10, default=lozoya.time.fnext_week())
    price = django.db.models.CharField(help_text="", max_length=10)
    terms = django.db.models.BooleanField(help_text="", )
    number_of_files = django.db.models.IntegerField(help_text="", default=0)
    duration = django.db.models.CharField(help_text="", max_length=7)
    # coupon = django.db.models.ForeignKey(Coupon, related_name='order', null=True, blank=True)
    coupon_redeemed = django.db.models.CharField(help_text="", max_length=10, blank=True, null=True)
    discount_redeemed = django.db.models.CharField(help_text="", max_length=4, default=0, blank=True, null=True)
    coupon_generated = django.db.models.CharField(help_text="", max_length=10, blank=True, null=True)
    discount_generated = django.db.models.CharField(help_text="", max_length=4, blank=True, null=True)
    tracks = django.db.models.TextField()

    def __str__(self):
        return self.job_id

    def clean(self):
        if self.date_due is None:
            self.date_due = lozoya.time.fnext_week()

    def __getitem__(self, item):
        return getattr(self, item)

    class Meta:
        verbose_name_plural = 'Orders'


class VFModelCoupon(django.db.models.Model):
    code = django.db.models.CharField(help_text="", max_length=10, unique=True)
    valid_from = django.db.models.DateTimeField(help_text="")
    valid_to = django.db.models.DateTimeField(help_text="")
    discount = django.db.models.FloatField(help_text="")
    percentage = django.db.models.BooleanField(help_text="")
    remaining_uses = django.db.models.IntegerField(help_text="")

    def __str__(self):
        return self.code


class VFModelOrder(django.db.models.Model):
    mix = django.db.models.BooleanField(help_text="", default=False)
    performance_correction = django.db.models.BooleanField(help_text="", default=False)
    repair = django.db.models.BooleanField(help_text="", default=False)
    master = django.db.models.BooleanField(help_text="", default=False)

    job_id = django.db.models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = django.db.models.EmailField(help_text="", max_length=64)
    project_name = django.db.models.CharField(help_text="", max_length=100)
    details = django.db.models.CharField(help_text="", max_length=500, blank=True)
    reference = django.db.models.CharField(help_text="", max_length=100, blank=True)
    date_submitted = django.db.models.CharField(help_text="", max_length=10, default=lozoya.time.ftoday())
    date_due = django.db.models.CharField(help_text="", max_length=10, default=lozoya.time.fnext_week())
    price = django.db.models.CharField(help_text="", max_length=10)
    terms = django.db.models.BooleanField(help_text="", )
    number_of_files = django.db.models.IntegerField(help_text="", default=0)
    duration = django.db.models.CharField(help_text="", max_length=7)
    # coupon = django.db.models.ForeignKey(Coupon, related_name='order', null=True, blank=True)
    coupon_redeemed = django.db.models.CharField(help_text="", max_length=10, blank=True, null=True)
    discount_redeemed = django.db.models.CharField(help_text="", max_length=4, default=0, blank=True, null=True)
    coupon_generated = django.db.models.CharField(help_text="", max_length=10, blank=True, null=True)
    discount_generated = django.db.models.CharField(help_text="", max_length=4, blank=True, null=True)
    tracks = django.db.models.TextField()

    def __str__(self):
        return self.job_id

    def clean(self):
        if self.date_due is None:
            self.date_due = lozoya.time.fnext_week()

    def __getitem__(self, item):
        return getattr(self, item)

    class Meta:
        verbose_name_plural = 'Orders'
