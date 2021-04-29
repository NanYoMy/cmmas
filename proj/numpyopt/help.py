import numpy as np
def zero_padding(in_array, padding_size=0):
    rows, cols = in_array.shape
    padding_array = np.zeros([rows + 2 * padding_size, cols + 2 * padding_size])
    padding_array[padding_size:rows + padding_size, padding_size:cols + padding_size] = in_array
    return padding_array

def zero_padding_3d(in_array, padding_size=0):
    rows, cols ,depth= in_array.shape
    padding_array = np.zeros([rows +  padding_size, cols +  padding_size,depth+padding_size])
    padding_array[0:rows , 0:cols ,0:depth] = in_array
    return padding_array