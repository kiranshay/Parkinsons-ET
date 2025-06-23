def _flatten_dict(data_dict, tmp_fields, data_dict_flat):
    is_most_inner = True  # Assume we are at most inner element
    for key, item in data_dict.items():
        if isinstance(item, dict):
            is_most_inner = False
            data_dict, tmp_fields, data_dict_flat = _flatten_dict(item, tmp_fields.copy(), data_dict_flat.copy())
        elif isinstance(item, list):
            if isinstance(item[0], dict):
                is_most_inner = False
                for list_item in item:
                    data_dict, tmp_fields, data_dict_flat = _flatten_dict(list_item, tmp_fields.copy(),
                                                                          data_dict_flat.copy())
            else:
                tmp_fields[key] = item
        else:
            tmp_fields[key] = item

    if is_most_inner:
        data_dict_flat.append(tmp_fields)

    return data_dict, tmp_fields, data_dict_flat


def flatten_dict(data_dict):
    tmp_fields = {}
    data_dict_flat = []
    _, _, data_dict_flat = _flatten_dict(data_dict, tmp_fields, data_dict_flat)
    return data_dict_flat
