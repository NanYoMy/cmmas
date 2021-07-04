
import numpy as np
smooth = 0.01
def calculate_dice(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_compute(groundtruth,pred ):           #batchsize*channel*W*W
    dice=[]
    for i in [1]:
        dice_i = 2*(np.sum((pred==i)*(groundtruth==i),dtype=np.float32))/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)+0.0001)
        dice=dice+[dice_i]
    if dice[0]>1:
        print("error!!!! dice >1 ")
    return np.array(dice,dtype=np.float32)