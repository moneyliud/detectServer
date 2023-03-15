# Generated by Django 3.2 on 2023-02-24 01:02

import detect.models
from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('detect', '0009_alter_imgstore_compare_status'),
    ]

    operations = [
        migrations.CreateModel(
            name='SysDict',
            fields=[
                ('dict_id', models.AutoField(help_text='字典ID', primary_key=True, serialize=False)),
                ('dict_name', models.CharField(max_length=50, verbose_name='字典名称')),
                ('dict_name_en', models.CharField(max_length=50, verbose_name='字典键英文名')),
                ('dict_value', models.CharField(max_length=50, verbose_name='字典数值')),
                ('dict_label', models.CharField(max_length=50, verbose_name='字典文本')),
                ('dict_index', models.IntegerField(verbose_name='排序位置')),
                ('create_time', models.DateTimeField(default=django.utils.timezone.now, verbose_name='创建时间')),
                ('update_time', models.DateTimeField(auto_now=True, verbose_name='修改时间')),
            ],
            options={
                'db_table': 'sys_dict',
            },
        ),
        migrations.AlterField(
            model_name='imgcompareresult',
            name='result_img',
            field=models.ImageField(blank=True, help_text='图像对比结果', null=True, upload_to=detect.models.result_image_dir_path),
        ),
        migrations.AlterField(
            model_name='imgstore',
            name='compare_status',
            field=models.CharField(default='未对比', max_length=20, verbose_name='对比状态'),
        ),
    ]