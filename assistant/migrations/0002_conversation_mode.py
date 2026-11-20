from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies=[('assistant','0001_initial')]
    operations=[migrations.AddField(model_name='conversation',name='mode',field=models.CharField(choices=[('rag','Обычный RAG'),('agent','LangGraph Agent')],default='rag',max_length=16))]
