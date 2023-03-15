# Generated by Django 3.2 on 2023-02-21 13:14

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ImgStore',
            fields=[
                ('img_id', models.AutoField(help_text='图片ID', primary_key=True, serialize=False)),
                ('product_name', models.CharField(help_text='机型', max_length=20)),
                ('part_no', models.CharField(help_text='组件图号', max_length=100)),
                ('batch_no', models.CharField(help_text='批次', max_length=10)),
                ('plane_no', models.CharField(help_text='架次', max_length=10)),
                ('img_content', models.BinaryField(help_text='图像内容', max_length=52428800)),
                ('img_feature', models.BinaryField(help_text='图像特征', max_length=52428800)),
            ],
            options={
                'db_table': 'img_store',
            },
        ),
    ]