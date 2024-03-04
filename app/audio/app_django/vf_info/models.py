import datetime

import django.db.models
import django.urls

today = datetime.datetime.now().date()
nextWeek = (today + datetime.timedelta(days=7)).strftime('%m/%d/%Y')
today = today.strftime('%m/%d/%Y')


class VFModelMessage(django.db.models.Model):
    message_id = django.db.models.CharField(max_length=36)
    email = django.db.models.EmailField(max_length=64)
    message = django.db.models.CharField(max_length=500, blank=True)
    date_submitted = django.db.models.CharField(max_length=10, default=today)

    def __str__(self):
        return self.message_id

    def get_absolute_url(self):
        return django.urls.reverse('info:success', kwargs={'message_id': self.message_id})
