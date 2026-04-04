from django.urls import path
from . import views
 
urlpatterns = [
    path('register/',           views.register_user,    name='register'),
    path('login/',              views.loginpage,         name='login'),
    path('admin-login/',        views.admin_login,       name='admin_login'),
    path('admin-home/',         views.admin_home_page,   name='admin_home'),
    path('logout/',             views.logout_view,       name='logout'),
    path('user-home/',          views.user_home,         name='user_home'),   # ← new
 
    path('activate/<int:user_id>/',   views.activate_user,   name='activate_user'),
    path('deactivate/<int:user_id>/', views.deactivate_user, name='deactivate_user'),
    path('delete/<int:user_id>/',     views.delete_user,     name='delete_user'),
]
 

