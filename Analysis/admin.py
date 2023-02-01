from django.contrib import admin
from .models import Snippet
from .models import Statistics
from .models import Item
from .models import Localized
from .models import Welcome2


admin.site.register(Snippet)
admin.site.register(Statistics)
admin.site.register(Localized)
admin.site.register(Item)
admin.site.register(Welcome2)