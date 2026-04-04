from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .forms import UserRegistrationForm, AdminLoginForm, UserLoginForm
from .models import UserRegistration


# ── Public landing (index page, not base skeleton) ──────────────────────────
def basepage(request):
    return render(request, 'index.html')          # ← was 'base.html' (skeleton only)


# ── Register ─────────────────────────────────────────────────────────────────
def register_user(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Registration successful! Please wait for admin activation.')
            return redirect('login')
        else:
            messages.error(request, 'Error in form submission. Please correct the errors below.')
    else:
        form = UserRegistrationForm()
    return render(request, 'register.html', {'form': form})


# ── User login ────────────────────────────────────────────────────────────────
def loginpage(request):
    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            user_id  = form.cleaned_data.get('userid')
            password = form.cleaned_data.get('password')
            try:
                user = UserRegistration.objects.get(userid=user_id, password=password)
                if user.is_active:
                    # store user in session so other views know who is logged in
                    request.session['user_id']   = user.id
                    request.session['user_name'] = user.userid
                    return redirect('user_home')      # ← goes to dedicated home view
                else:
                    messages.error(request, 'Your account is inactive. Please contact admin.')
            except UserRegistration.DoesNotExist:
                messages.error(request, 'Invalid user ID or password.')
        else:
            messages.error(request, 'Invalid form submission.')
    else:
        form = UserLoginForm()
    return render(request, 'login.html', {'form': form})


# ── User home (dedicated view, not the login view) ───────────────────────────
def user_home(request):
    # Optional: guard the page so only logged-in users can see it
    if not request.session.get('user_id'):
        return redirect('login')
    return render(request, 'remoteuser/homeuser.html')


# ── Admin login ───────────────────────────────────────────────────────────────
def admin_login(request):
    if request.method == 'POST':
        form = AdminLoginForm(request.POST)
        if form.is_valid():
            user_id  = form.cleaned_data.get('userid')
            password = form.cleaned_data.get('password')
            if user_id == 'admin' and password == 'admin':
                request.session['is_admin'] = True
                return redirect('admin_home')
            else:
                messages.error(request, 'Invalid credentials.')
    else:
        form = AdminLoginForm()
    return render(request, 'admin_login.html', {'form': form})


# ── Admin home ────────────────────────────────────────────────────────────────
def admin_home_page(request):
    data = UserRegistration.objects.all()
    return render(request, 'accounts/list.html', {'data': data})


# ── Activate / Deactivate / Delete ────────────────────────────────────────────
def activate_user(request, user_id):
    user = get_object_or_404(UserRegistration, id=user_id)
    user.is_active = True
    user.save()
    return redirect('admin_home')


def deactivate_user(request, user_id):
    user = get_object_or_404(UserRegistration, id=user_id)
    user.is_active = False
    user.save()
    return redirect('admin_home')


def delete_user(request, user_id):
    user = get_object_or_404(UserRegistration, id=user_id)
    user.delete()
    return redirect('admin_home')


# ── Logout ────────────────────────────────────────────────────────────────────
def logout_view(request):
    request.session.flush()
    return redirect('login')