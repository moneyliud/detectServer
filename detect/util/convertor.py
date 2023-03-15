from copy import deepcopy


def model_obj_to_dict(obj):
    ret = deepcopy(obj.__dict__)
    ret.pop("_state")
    return ret
