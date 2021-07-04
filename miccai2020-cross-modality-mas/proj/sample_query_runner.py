#-*- encoding=utf-8 -*-
#matplotlib inline
from __future__ import print_function

from config import configer
from fusion.patchembbeding import PatchEmbbeding
import os
from config.load_embedding_arg import get_args
import MAS.helpers as helpers

'''
training:使用4个support样本，利用4个query,对模型进行训练
inference:使用4个从train样本中得到的support样本，对剩余的24样本进行评估，
'''


if __name__ == "__main__":
    # path_atlas_img=glob.glob("../../data_vote_man/MR_CT/train/model_24000/*ct_train_1013_image_A_T/atlas_img_mr*")
    # path_atlas_lab=glob.glob("../../data_vote_man/MR_CT/train/model_24000/*ct_train_1013_image_A_T/atlas_lab_mr*")
    #
    # path_target_img=glob.glob("../../data_vote_man/MR_CT/train/*/*ct_train_1013_image_A_T/target_img_ct*")[0]
    # path_target_lab=glob.glob("../../data_vote_man/MR_CT/train/*/*ct_train_1013_image_A_T/target_lab_ct*")[0]
    #
    # ps=PatchSampler(path_target_img,path_target_lab,path_atlas_img,path_atlas_lab)
    # atlas_img_patches, atlas_labs, target_img_patch, target_lab=ps.next_sample()
    # sitk.WriteImage(sitk.GetImageFromArray(target_img_patch),".\\tmp\\"+"-1_label"+str(target_lab)+"target_img_tmp.nii.gz")
    # for i,tmp in enumerate(atlas_img_patches):
    #     sitk.WriteImage(sitk.GetImageFromArray(tmp), ".\\tmp\\"+str(i)+"_label"+str(atlas_labs[i])+"atals_img_tmp.nii.gz")

    #===========================================


    #设置成2,用于训练encoder
    # sq.n_support_sample=2
    # sq=PatchEmbbeding(args=get_args())
    # sq.build_model()
    # sq.train()

    sq2=PatchEmbbeding(args=get_args())
    # sq2.build_model()#测试和训练所用的个数不一样，训练相对可以少，测试比较多
    # sq2.visulize()
    sq2.build_model(12*get_args().n_support_sample[0])#测试和训练所用的个数不一样，训练相对可以少，测试比较多
    sq2.test()
