# Generated by Django 4.2.6 on 2023-10-31 03:36

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ImageContents',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='images/')),
                ('image_uuid', models.UUIDField(default=uuid.uuid4, editable=False, unique=True)),
            ],
        ),
    ]
