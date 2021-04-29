
def parse_arg_list(list,out_put_type="str"):
    if out_put_type=='int':
        my_list = [int(item) for item in list.split('-')]
    elif out_put_type=='str':
        my_list = [item for item in list.split('-')]
    elif out_put_type=='float':
        my_list = [float(item) for item in list.split('-')]
    else:
        exit(-998)
    return my_list
