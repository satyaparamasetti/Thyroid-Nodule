from django.urls import path, include
from Accounts import views as av
from remoteuser import views as rv
 
urlpatterns = [
    # Landing page — '' means site root  /
    path('',            av.basepage,       name='home'),        # ← was 'home' path missing root
 
    # Remote-user feature pages
    path('remoteuser/data/',    rv.ImageData,     name='image_data'),
    path('remoteuser/build/',   rv.Training,      name='build'),
    path('remoteuser/scores/',  rv.scores,        name='scores'),
    path('remoteuser/detect/',  rv.classify_image, name='classify_image'),
 
    # All accounts URLs under /accounts/
    path('accounts/', include('Accounts.urls')),
]
