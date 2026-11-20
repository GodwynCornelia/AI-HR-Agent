from django import forms
from .models import Document


class UploadForm(forms.Form):
    title = forms.CharField(label='Название', max_length=240)
    file = forms.FileField(label='Документ PDF, DOCX или TXT')
    scope = forms.CharField(label='Область применения', initial='Все сотрудники', max_length=240)
    effective_from = forms.DateField(label='Дата вступления в силу', required=False,
                                     widget=forms.DateInput(attrs={'type': 'date'}))
    supersedes = forms.ModelChoiceField(label='Явно заменяет документ', required=False,
                                        queryset=Document.objects.none())
    provenance = forms.CharField(label='Происхождение и условия использования', widget=forms.Textarea)
    synthetic = forms.BooleanField(label='Синтетический учебный документ', initial=True, required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['supersedes'].queryset = Document.objects.exclude(state='deleted')

    def clean_file(self):
        file = self.cleaned_data['file']
        if file.size > 20 * 1024 * 1024:
            raise forms.ValidationError('Максимальный размер — 20 MiB')
        return file
