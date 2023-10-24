from django.urls import path, include

from detect.controller import imageController

urlpatterns = [
    path("test/", imageController.test, name="test1"),
    path("image_upload/", imageController.image_upload, name="image_upload"),
    path("get_image_list/", imageController.get_image_list, name="get_image_list"),
    path("delete_image/", imageController.delete_image, name="delete_image"),
    path("get_compare_result/", imageController.get_compare_result, name="get_compare_result"),
    path("get_dict_item/", imageController.get_dict_item, name="get_dict_item"),
    path("set_basic_img/", imageController.set_basic_img, name="set_basic_img"),
    path("get_image_list_all/", imageController.get_image_list_all, name="get_image_list_all"),
    path("get_image_and_project/", imageController.get_image_and_project, name="get_image_and_project"),
    path("project_empty_image/", imageController.project_empty_image, name="project_empty_image")
    # path("get_camera_stream/", imageController.get_camera_stream, name="get_camera_stream")
]
