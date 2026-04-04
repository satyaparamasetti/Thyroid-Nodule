from django import forms
from .models import UserRegistration
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

class UserRegistrationForm(forms.ModelForm):
    class Meta:
        model = UserRegistration
        fields = ['username', 'userid', 'email', 'password', 'phone_number']
        widgets = {
            'password': forms.PasswordInput(),
        }

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if UserRegistration.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already registered.")
        return email

    def clean_userid(self):
        userid = self.cleaned_data.get('userid')
        if UserRegistration.objects.filter(userid=userid).exists(): 
            raise forms.ValidationError("This user ID is already taken.")
        return userid

    def clean_phone_number(self):
        phone_number = self.cleaned_data.get('phone_number')
        if not phone_number.isdigit():
            raise forms.ValidationError("Phone number should contain only digits.")
        if len(phone_number) != 10:
            raise forms.ValidationError("Phone number should be exactly 10 digits.")
        return phone_number



class AdminLoginForm(forms.Form):
    userid = forms.CharField(max_length=100)
    password = forms.CharField(widget=forms.PasswordInput)


class UserLoginForm(forms.Form):
    userid = forms.CharField(max_length=100)
    password = forms.CharField(widget=forms.PasswordInput)        