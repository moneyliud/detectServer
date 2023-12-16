import os.path

from django.db import models
import django.utils.timezone as timezone
from enum import Enum


# Create your models here.

def image_dir_path(instance, filename):
    ext = filename.split(".").pop()
    filename = '{0}{1}{2}-{3}.{4}'.format(instance.product_name, instance.batch_no, instance.plane_no,
                                          instance.part_no, ext)
    # print(os.path.join(instance.product_name, instance.batch_no, filename))
    return os.path.join(instance.product_name, instance.batch_no, filename)


def org_image_dir_path(instance, filename):
    ext = filename.split(".").pop()
    filename = 'org_{0}{1}{2}-{3}.{4}'.format(instance.product_name, instance.batch_no, instance.plane_no,
                                              instance.part_no, ext)
    # print(os.path.join(instance.product_name, instance.batch_no, filename))
    return os.path.join(instance.product_name, instance.batch_no, filename)


def result_image_dir_path(instance, filename):
    f_split = filename.split(".")
    ext = f_split.pop()
    head = f_split[0]
    filename = '{0}-{1}-{2}.{3}'.format(head, instance.img_src_id, instance.img_dst_id, ext)
    return os.path.join("result", filename)


class IMG_COMPARE_STATUS(Enum):
    UN_COMPARE = "未对比"
    COMPARING = "对比中"
    NO_DIFFERENCE = "一致"
    HAS_DIFFERENCE = "不一致"
    COMPARE_ERROR = "对比失败"


class ImgStore(models.Model):
    """
    ImgStore 图像数据库
    """
    img_id = models.AutoField(primary_key=True, help_text="图片ID")
    product_name = models.CharField(max_length=20, help_text="机型")
    part_no = models.CharField(max_length=100, help_text="组件图号")
    batch_no = models.CharField(max_length=10, help_text="批次")
    plane_no = models.CharField(max_length=10, help_text="架次")
    img_content = models.ImageField(upload_to=image_dir_path, help_text="图像内容", blank=True, null=True)
    img_content_org = models.ImageField(upload_to=org_image_dir_path, help_text="原始图像内容", blank=True, null=True)
    img_feature = models.BinaryField(max_length=1024 * 1024 * 50, help_text="图像特征")
    is_basic_img = models.BooleanField('是否为基准图像', default=True)
    compare_status = models.CharField('对比状态', max_length=20, default=IMG_COMPARE_STATUS.UN_COMPARE.value)
    create_time = models.DateTimeField('创建时间', default=timezone.now)
    update_time = models.DateTimeField('修改时间', auto_now=True)
    is_detected = models.BooleanField('是否已提取标签', default=False)

    class Meta:
        db_table = "img_store"

    def __str__(self):
        return str(self.img_id) + "," + self.product_name + self.batch_no + self.plane_no + "," + self.part_no


class ImgCompareResult(models.Model):
    img_compare_id = models.AutoField(primary_key=True, help_text="图片ID")
    img_src_id = models.IntegerField('源图片ID')
    img_dst_id = models.IntegerField('对比图片ID')
    result_img = models.ImageField(upload_to=result_image_dir_path, help_text="图像对比结果", blank=True, null=True)
    diff_count = models.IntegerField('比对差异数量')
    compare_result = models.BinaryField(max_length=1024 * 1024 * 50, help_text="对比结果")
    create_time = models.DateTimeField('创建时间', default=timezone.now)
    update_time = models.DateTimeField('修改时间', auto_now=True)

    class Meta:
        db_table = "img_compare_result"


class ImgCompareResultV(models.Model):
    img_compare_id = models.AutoField(primary_key=True, help_text="图片ID")
    img_src_id = models.IntegerField('源图片ID')
    img_dst_id = models.IntegerField('对比图片ID')
    product_name = models.CharField(max_length=20, help_text="机型")
    batch_no_src = models.CharField(max_length=10, help_text="源批次")
    batch_no_dst = models.CharField(max_length=10, help_text="对比批次")
    plane_no_src = models.CharField(max_length=10, help_text="源架次")
    plane_no_dst = models.CharField(max_length=10, help_text="对比架次")
    part_no = models.CharField(max_length=100, help_text="组件图号")
    result_img = models.ImageField(upload_to=image_dir_path, help_text="图像对比结果", blank=True, null=True)
    diff_count = models.IntegerField('比对差异数量')
    compare_result = models.BinaryField(max_length=1024 * 1024 * 50, help_text="对比结果")
    create_time = models.DateTimeField('创建时间', default=timezone.now)
    update_time = models.DateTimeField('修改时间', auto_now=True)

    class Meta:
        managed = False
        db_table = "img_compare_result_v"


class ImgLabel(models.Model):
    label_id = models.AutoField(primary_key=True, help_text="标签ID")
    label_en = models.CharField(max_length=20, help_text="标签英文名")
    label_cn = models.CharField(max_length=20, help_text="标签中文名")
    color = models.CharField(max_length=20, help_text="颜色")
    create_time = models.DateTimeField('创建时间', default=timezone.now)
    update_time = models.DateTimeField('修改时间', auto_now=True)

    class Meta:
        managed = False
        db_table = "img_label"


class ImgLabelMsg(models.Model):
    label_msg_id = models.AutoField(primary_key=True, help_text="标签信息ID")
    img_id = models.IntegerField('图片ID')
    label_id = models.IntegerField('标签ID')
    x = models.FloatField()
    y = models.FloatField()
    w = models.FloatField()
    h = models.FloatField()
    enable = models.IntegerField("是否启用")
    auto_detect = models.IntegerField("是否自动识别的")
    conf = models.FloatField("置信度")
    create_time = models.DateTimeField('创建时间', default=timezone.now)
    update_time = models.DateTimeField('修改时间', auto_now=True)

    def __str__(self):
        return str(self.img_id) + "," + str(self.label_id) + "," + str(self.label_msg_id) + \
               "," + str(self.x) + "," + str(self.y) + "," + str(self.w) + "," + str(self.h) + ","

    class Meta:
        managed = False
        db_table = "img_label_msg"



class SysDict(models.Model):
    dict_id = models.AutoField(primary_key=True, help_text="字典ID")
    dict_name = models.CharField('字典名称', max_length=50)
    dict_name_en = models.CharField('字典英文名', max_length=50)
    create_time = models.DateTimeField('创建时间', default=timezone.now)
    update_time = models.DateTimeField('修改时间', auto_now=True)

    class Meta:
        db_table = "sys_dict"
        verbose_name = "系统字典"
        verbose_name_plural = "系统字典管理"

    def __str__(self):
        return self.dict_name


class SysDictItem(models.Model):
    dict_id = models.ForeignKey(
        SysDict,
        on_delete=models.PROTECT
    )
    dict_item_id = models.AutoField(primary_key=True, help_text="字典条目ID")
    dict_value = models.CharField('字典数值', max_length=50)
    dict_label = models.CharField('字典文本', max_length=50)
    dict_index = models.IntegerField('排序位置')
    create_time = models.DateTimeField('创建时间', default=timezone.now)
    update_time = models.DateTimeField('修改时间', auto_now=True)

    class Meta:
        db_table = "sys_dict_item"
        verbose_name = "字典条目"
        verbose_name_plural = "字典条目管理"
