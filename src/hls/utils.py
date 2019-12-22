import json

def print_tf_global_variables():
    import tensorflow as tf
    print(json.dumps([str(foo) for foo in tf.global_variables()], indent=4))

def print_var_list(var_list):
    print(json.dumps([str(foo) for foo in var_list], indent=4))

def pretty_print(data_structure):
    print(json.dumps(data_structure, indent=4))

def merge_dict_a_into_b(a, b):
    for key in a:
        assert key not in b
        b[key] = a[key]

def get_list_or_None(s, dtype=int):
    if s.strip() == '':
        return None
    else:
        lst = s.strip().split(',')
        return [dtype(x) for x in lst]

def get_list(s):
    lst = get_list_or_None(s)
    if lst is None:
        return []
    else:
        return lst

def None_if_zero(n):
    if n <= 0:
        return None
    else:
        return n

def boolean(s):
    if s == 'True':
        return True
    if s == 'False':
        return False
    raise ValueError('Invalid boolean value: %s' % s)

def set_to_list_of_values_if_None_or_empty(lst, val, num_vals):
    if not lst:
        return [val] * num_vals
    else:
        print(len(lst), num_vals)
        assert len(lst) == num_vals
        return lst
