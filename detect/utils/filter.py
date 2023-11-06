from django.db.models import Q


def get_filter_by_request(request, model):
    params = {}
    for i in model.__dict__:
        if request.GET.get(i) is not None:
            params[i] = request.GET.get(i)
    query = Q()
    for j in params:
        if "id" in j:
            query.add(Q(**{j + "__exact": params[j]}), Q.AND)
        else:
            query.add(Q(**{j + "__contains": params[j]}), Q.AND)
    return model.objects.filter(query).extra(
        select={"create_time": "date(create_time)",
                "update_time": "date(update_time)"})
    # .extra(
    # select={"create_time": "DATE_FORMAT(create_time,'%%Y-%%m-%%d %%H:%%i:%%s')",
    #         "update_time": "DATE_FORMAT(update_time,'%%Y-%%m-%%d %%H:%%i:%%s')"})
