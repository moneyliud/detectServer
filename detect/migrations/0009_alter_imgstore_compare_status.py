# Generated by Django 3.2 on 2023-02-23 02:26

import detect.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detect', '0008_imgstore_compare_status'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imgstore',
            name='compare_status',
            field=models.CharField(default=detect.djangomodels.IMG_COMPARE_STATUS['UN_COMPARE'], max_length=20, verbose_name='对比状态'),
        ),
    ]
