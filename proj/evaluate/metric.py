
import numpy as np
smooth = 0.01
from medpy.metric import hd95,assd
'''
0/1的二值化mask可以用这个求解
'''
def calculate_binary_dice(y_true, y_pred,thres=0.5):
    y_true=np.squeeze(y_true)
    y_pred=np.squeeze(y_pred)
    y_true=np.where(y_true>thres,1,0)
    y_pred=np.where(y_pred>thres,1,0)
    return  dc(y_pred,y_true)

def calculate_binary_hd(y_true, y_pred,thres=0.5,spacing=[1,1,1]):
    y_true=np.squeeze(y_true)
    y_pred=np.squeeze(y_pred)
    y_true=np.where(y_true>thres,1,0)
    y_pred=np.where(y_pred>thres,1,0)
    return  assd(y_pred,y_true,spacing)
def dice_compute(groundtruth,pred,labs ):           #batchsize*channel*W*W
    dice=[]
    # for i in [1]:
    for i in labs:
        dice_i = 2*(np.sum((pred==i)*(groundtruth==i),dtype=np.float32))/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)+0.0001)
        dice=dice+[dice_i]
    if dice[0]>1:
        print("error!!!! dice >1 ")
    return np.array(dice,dtype=np.float32)

import SimpleITK as sitk

def neg_jac(flow):
    flow_img = sitk.GetImageFromArray(flow, isVector=True)
    jac_det_filt = sitk.DisplacementFieldJacobianDeterminant(flow_img)
    jac_det = sitk.GetArrayFromImage(jac_det_filt)
    mean_grad_detJ = np.mean(np.abs(np.gradient(jac_det)))
    negative_detJ = np.sum((jac_det < 0))
    return jac_det,mean_grad_detJ,negative_detJ


def computeQualityMeasures(lP, lT):
    quality = dict()
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()

    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["dice"] = dicecomputer.GetDiceCoefficient()

    return quality


from medpy.metric import dc,hd
def dice_and_asd(target_lab, predict_lab, spacing=[1, 1, 1]):
    dice=calculate_binary_dice(target_lab,predict_lab)
    haus=calculate_binary_hd(target_lab,predict_lab,spacing=spacing)
    return dice,haus


from tool.variablename import retrieve_name

def print_mean_and_std(array, info="info",detail=True):
    print("=====%s===="%retrieve_name(array))
    if detail:
        print(array)
    else:
        print(len(array))
    print("mean:%f"%np.mean(array))
    print("std:%f"%np.std(array))


#image metric
def sad(x, y):
    """Sum of Absolute Differences (SAD) between two images."""
    return np.sum(np.abs(x - y))


def ssd(x, y):
    """Sum of Squared Differences (SSD) between two images."""
    return np.sum((x - y) ** 2)


def ncc(x, y):
    """Normalized Cross Correlation (NCC) between two images."""
    return np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std())


def mi(x, y):
    """Mutual Information (MI) between two images."""
    from sklearn.metrics import mutual_info_score
    return mutual_info_score(x.ravel(), y.ravel())


from excelutil.output2excel import read_excel,outpu2excel

def cross_validate(args):
    dict=read_excel(args.res_excel)
    stastic_all(args,dict, '_DS')
    stastic_all(args,dict, '_HD')

def stastic_all(args, dict, type):
    all_fold = []
    for id in range(1, 5):
        list=dict.get((args.MOLD_ID_TEMPLATE + type).replace("#", str(id)))
        if not list is None:
            all_fold = all_fold + list
    outpu2excel(args.res_excel,args.MOLD_ID_TEMPLATE+type,[np.mean(all_fold),np.std(all_fold)])
