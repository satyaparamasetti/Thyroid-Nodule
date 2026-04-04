from django.db import models

class UserRegistration(models.Model):
    username = models.CharField(max_length=255)
    userid = models.CharField(max_length=255)
    email = models.CharField(max_length=255)
    password = models.CharField(max_length=255)
    phone_number = models.CharField(max_length=10)
    is_active = models.BooleanField(default=False)


    def __str__(self):
        return self.username