from django.contrib import admin
from detect.djangomodels import SysDict, SysDictItem


# Register your models here.


@admin.register(SysDict)
class SysDictAdmin(admin.ModelAdmin):
    list_display = ('dict_name', 'dict_name_en', 'update_time')


@admin.register(SysDictItem)
class SysDictItemAdmin(admin.ModelAdmin):
    list_display = ('dict_id', 'dict_value', 'dict_label', 'dict_index', 'update_time')
