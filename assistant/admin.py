from django.contrib import admin
from .models import Document, Job, ExperimentRun


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['title', 'state', 'scope', 'effective_from', 'synthetic']
    readonly_fields = [field.name for field in Document._meta.fields]
    def has_add_permission(self, request):
        return False
    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ['id', 'document', 'kind', 'state', 'attempts']
    readonly_fields = [field.name for field in Job._meta.fields]
    def has_add_permission(self, request):
        return False
    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(ExperimentRun)
class ExperimentAdmin(admin.ModelAdmin):
    readonly_fields = [field.name for field in ExperimentRun._meta.fields]
    def has_add_permission(self, request):
        return False
