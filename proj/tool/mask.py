import numpy as np
from keras.utils import  to_categorical

def create_mask(lab,indexs):
    mask = np.zeros(lab.shape, np.uint16)
    for i, idx in enumerate(indexs):
        mask = mask + np.where(lab == idx, 1, 0)
    return mask

