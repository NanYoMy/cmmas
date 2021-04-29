from sitkImageIO.itkdatareader import  convert_p_img_2_nor_array
import sitkImageIO.itkdatawriter as sitkioReader
from dirutil.helper import mk_or_cleardir,get_name_wo_suffix
from sitkImageIO.itkdatawriter import sitk_write_image,sitk_write_lab
from sitkImageIO.itkdatareader import convert_img_2_nor_array, convert_lab_2_array
from sitkImageIO.itkdatareader import sitk_read_img_lab
import numpy as np
from dirutil.helper import sort_glob
import os
from scipy import  ndimage

from keras.utils import  to_categorical
class FusionNetSampler():
    def __init__(self,args,type):
        self.args = args
        if type=='train':
            self.data_samples = sort_glob(self.args.dataset_dir + "/train/*/")
            self.is_train=True
        elif type=='validate':
            self.data_samples = sort_glob(self.args.dataset_dir + "/validate/*/")
            self.is_train=False
        elif type == 'test':
            self.data_samples = sort_glob(self.args.dataset_dir + "/test/*/")
            self.is_train = False
        else:
            print('unsupport type')
            exit('-999')
        self.index=0
        self.len=len(self.data_samples)

    def next_sample(self):
        target_img_batch=[]
        target_lab_batch=[]
        atlas_img_batch=[]
        atlas_lab_batch=[]
        for i in range(self.args.batch_size):
            fix_imgs, fix_labs, atlases_imgs, atlases_labs=self.get_file()
            # print(fix_imgs)
            if self.is_train==True:
                target_img_data,target_lab_data,atlas_img_data,atlas_lab_data=self.get_dataV2(fix_imgs, fix_labs, atlases_imgs, atlases_labs,True)
            else:
                target_img_data,target_lab_data,atlas_img_data,atlas_lab_data=self.get_dataV2(fix_imgs, fix_labs, atlases_imgs, atlases_labs,False)
            target_img_batch.append(target_img_data)
            target_lab_batch.append(target_lab_data)
            atlas_img_batch.append(atlas_img_data)
            atlas_lab_batch.append(atlas_lab_data)
        target_img_batch=np.stack(target_img_batch,axis=0)
        target_lab_batch=np.stack(target_lab_batch,axis=0)
        atlas_img_batch=np.stack(atlas_img_batch,axis=0)
        atlas_lab_batch=np.stack(atlas_lab_batch,axis=0)
        return target_img_batch,target_lab_batch,atlas_img_batch,atlas_lab_batch

    def get_file(self):
        if self.is_train==False:
            sample_dir=self.data_samples[self.index]
            self.index=(self.index+1)%self.len
        else:
            sample_dir = self.data_samples[np.random.randint(len(self.data_samples))]

        atlases_imgs=(sort_glob(sample_dir + "/*mv_img*.nii.gz"))
        atlases_labs=(sort_glob(sample_dir + "/*mv_lab*.nii.gz"))
        fix_imgs=(sort_glob(sample_dir + "/*fix_img*.nii.gz"))
        fix_labs=(sort_glob(sample_dir + "/*fix_lab*.nii.gz"))
        return fix_imgs,fix_labs,atlases_imgs,atlases_labs

    # def get_data(self,target_img,target_lab,atlas_imgs,atlas_labs):
    #     atlas_img_data=sitk_read_atlas_imgs_array(atlas_imgs)
    #     atlas_lab_data=sitk_read_atlas_labs_array(atlas_labs)
    #     target_img_data=np.expand_dims(sitk_read_image(target_img[0]),axis=-1)
    #     target_lab_data=to_categorical(sitk_read_lab_array(target_lab[0]), num_classes=2)
    #
    #     return target_img_data,target_lab_data,atlas_img_data,atlas_lab_data

    def get_data(self, p_target_img, p_target_lab, p_atlas_imgs, p_atlas_labs,is_aug=True):
        p_imgs= p_target_img + p_atlas_imgs
        p_labs= p_target_lab + p_atlas_labs
        imgs,labs=sitk_read_img_lab(p_imgs, p_labs, is_aug)

        # mk_or_cleardir('./tmp/')
        # for img,lab,p_img,p_lab in zip(imgs,labs,p_imgs,p_labs):
        #     sitk_write_image(img,dir='./tmp/',name=get_name_wo_suffix(p_img))
        #     sitk_write_image(lab,dir='./tmp/',name=get_name_wo_suffix(p_lab))
        imgs = np.stack([convert_img_2_nor_array(img) for img in imgs], axis=-1)
        labs = np.stack([convert_lab_2_array(lab) for lab in labs], axis=-1)

        atlas_imgs= imgs[ ...,1:]
        target_img=imgs[ ...,0]

        atlas_labs= labs[ ...,1:]
        target_lab=labs[ ...,0]

        return np.expand_dims(target_img,axis=-1), to_categorical(target_lab),atlas_imgs,atlas_labs

    def get_dataV2(self, p_target_img, p_target_lab, p_atlas_imgs, p_atlas_labs, is_aug=True):
        p_imgs = p_target_img + p_atlas_imgs
        p_labs = p_target_lab + p_atlas_labs
        imgs, labs = sitk_read_img_lab(p_imgs, p_labs, is_aug)

        # mk_or_cleardir('./tmp/')
        # for img,lab,p_img,p_lab in zip(imgs,labs,p_imgs,p_labs):
        #     sitk_write_image(img,dir='./tmp/',name=get_name_wo_suffix(p_img))
        #     sitk_write_image(lab,dir='./tmp/',name=get_name_wo_suffix(p_lab))
        imgs = np.stack([convert_img_2_nor_array(img) for img in imgs], axis=-1)
        labs = np.stack([convert_lab_2_array(lab) for lab in labs], axis=-1)

        atlas_imgs = imgs[..., 1:]
        target_img = imgs[..., 0]

        atlas_labs = np.mean(labs[..., 1:].astype(np.float32),axis=-1,keepdims=True)
        target_lab = labs[..., 0].astype(np.float32)

        # sitk_write_image(np.squeeze(atlas_labs),dir='./tmp/',name='input')
        # sitk_write_image(np.squeeze(target_lab),dir='./tmp/',name='gt')

        return np.expand_dims(target_img, axis=-1), to_categorical(target_lab), atlas_imgs, atlas_labs

class IntIntSimNetSampler(FusionNetSampler):

    def next_sample(self):
        target_img_batch=[]
        target_lab_batch=[]
        atlas_img_batch=[]
        atlas_lab_batch=[]
        sims_batch=[]
        for i in range(self.args.batch_size):
            fix_imgs, fix_labs, atlases_imgs, atlases_labs=self.get_file()
            # print(fix_imgs)
            if self.is_train==True:
                target_img_data,target_lab_data,atlas_img_data,atlas_lab_data,sim=self.get_dataV2(fix_imgs, fix_labs, atlases_imgs, atlases_labs,True)
            else:
                target_img_data,target_lab_data,atlas_img_data,atlas_lab_data,sim=self.get_dataV2(fix_imgs, fix_labs, atlases_imgs, atlases_labs,False)
            target_img_batch.append(target_img_data)
            target_lab_batch.append(target_lab_data)
            atlas_img_batch.append(atlas_img_data)
            atlas_lab_batch.append(atlas_lab_data)
            sims_batch.append(sim)
        target_img_batch=np.stack(target_img_batch,axis=0)
        target_lab_batch=np.stack(target_lab_batch,axis=0)
        atlas_img_batch=np.stack(atlas_img_batch,axis=0)
        atlas_lab_batch=np.stack(atlas_lab_batch,axis=0)
        sims_batch=np.stack(sims_batch,axis=0)
        return target_img_batch,target_lab_batch,atlas_img_batch,atlas_lab_batch,sims_batch,

    def get_file(self):
        if self.is_train==False:
            sample_dir=self.data_samples[self.index]
            self.index=(self.index+1)%self.len
        else:
            sample_dir = self.data_samples[np.random.randint(len(self.data_samples))]

        atlases_imgs=(sort_glob(sample_dir + "/*mv_img*.nii.gz"))
        atlases_labs=(sort_glob(sample_dir + "/*mv_lab*.nii.gz"))
        fix_imgs=(sort_glob(sample_dir + "/*fix_img*.nii.gz"))
        fix_labs=(sort_glob(sample_dir + "/*fix_lab*.nii.gz"))
        ID=np.random.randint(len(atlases_imgs))
        return fix_imgs,fix_labs,[atlases_imgs[ID]],[atlases_labs[ID]]

    def get_dataV2(self, p_target_img, p_target_lab, p_atlas_imgs, p_atlas_labs, is_aug=True):
        p_imgs = p_target_img + p_atlas_imgs
        p_labs = p_target_lab + p_atlas_labs
        imgs, labs = sitk_read_img_lab(p_imgs, p_labs, is_aug)

        # mk_or_cleardir('./tmp/')
        # for img,lab,p_img,p_lab in zip(imgs,labs,p_imgs,p_labs):
        #     sitk_write_image(img,dir='./tmp/',name=get_name_wo_suffix(p_img))
        #     sitk_write_image(lab,dir='./tmp/',name=get_name_wo_suffix(p_lab))
        imgs = np.stack([convert_img_2_nor_array(img) for img in imgs], axis=-1)
        labs = np.stack([convert_lab_2_array(lab) for lab in labs], axis=-1)

        atlas_img = imgs[..., 1]
        atlas_lab = labs[..., 1]
        target_img = imgs[..., 0]
        target_lab = labs[..., 0]
        conv_target_lab=ndimage.convolve(target_lab,np.ones([self.args.patch_size,self.args.patch_size,self.args.patch_size]))
        conv_atlas_lab=ndimage.convolve(atlas_lab,np.ones([self.args.patch_size,self.args.patch_size,self.args.patch_size]))
        denominator=conv_atlas_lab+conv_target_lab
        numerator=ndimage.convolve(target_lab*atlas_lab,np.ones([self.args.patch_size,self.args.patch_size,self.args.patch_size]))
        sim=2*numerator.astype(np.float32)/(denominator.astype(np.float32)+0.000001)
        # sitk_write_image(np.squeeze(atlas_labs),dir='./tmp/',name='input')
        # sitk_write_image(np.squeeze(target_lab),dir='./tmp/',name='gt')
        return np.expand_dims(target_img, axis=-1), np.expand_dims(target_lab,axis=-1), np.expand_dims(atlas_img,axis=-1), np.expand_dims(atlas_lab,axis=-1),np.expand_dims(sim,axis=-1)

class LabIntSimNetSampler(FusionNetSampler):
    def __init__(self,args,type):
        self.args = args
        if type=='train':
            self.data_samples = sort_glob(self.args.dataset_dir + "/train_fusion_atlas_target/*/")
            self.is_train=True
        elif type=='validate':
            self.data_samples = sort_glob(self.args.dataset_dir + "/atlas_target/*/")
            self.is_train=False
        # elif type == 'test':
        #     self.data_samples = sort_glob(self.args.dataset_dir + "/test/*/")
        #     self.is_train = False
        else:
            print('unsupport type')
            exit('-999')
        self.index=0
        self.len=len(self.data_samples)

    def next_sample(self):
        target_img_batch=[]
        target_lab_batch=[]
        # atlas_img_batch=[]
        atlas_lab_batch=[]
        sims_batch=[]
        for i in range(self.args.batch_size):
            fix_imgs, fix_labs, atlases_imgs, atlases_labs=self.get_file()
            # print(fix_imgs)
            if self.is_train==True:
                target_img_data,target_lab_data,atlas_img_data,atlas_lab_data=self.get_dataV3(fix_imgs, fix_labs, atlases_imgs, atlases_labs,True)
            else:
                target_img_data,target_lab_data,atlas_img_data,atlas_lab_data=self.get_dataV3(fix_imgs, fix_labs, atlases_imgs, atlases_labs,False)
            # 合并多atlas,获取随机选择一个，
            atlas_lab_data=self.merge_atlas_label(atlas_lab_data)
            sim= self.calculate_gt_sim(target_lab_data, atlas_lab_data)

            target_img_batch.append(target_img_data)
            target_lab_batch.append(target_lab_data)
            # atlas_img_batch.append(atlas_img_data)
            atlas_lab_batch.append(atlas_lab_data)
            sims_batch.append(sim)

        target_img_batch=np.stack(target_img_batch,axis=0)
        target_lab_batch=np.stack(target_lab_batch,axis=0)
        # atlas_img_batch=np.stack(atlas_img_batch,axis=0)
        atlas_lab_batch=np.stack(atlas_lab_batch,axis=0)
        sims_batch=np.stack(sims_batch,axis=0)
        # return target_img_batch,target_lab_batch,atlas_img_batch,atlas_lab_batch,sims_batch,
        return target_img_batch,target_lab_batch,None,atlas_lab_batch,sims_batch,

    def next_sample_4_fusion(self):
        target_img_batch = []
        target_lab_batch = []
        atlas_img_batch = []
        atlas_lab_batch = []
        sims_batch = []
        for i in range(self.args.batch_size):
            fix_img, fix_lab, atlases_imgs, atlases_labs = self.get_file()
            # print(fix_imgs)
            # print(fix_labs)
            # print(atlases_imgs)
            # print(atlases_labs)
            if self.is_train == True:
                target_img_data, target_lab_data, atlas_img_data, atlas_lab_data = self.get_data_4_fusion(fix_img, fix_lab,atlases_imgs,atlases_labs, True)
            else:
                target_img_data, target_lab_data, atlas_img_data, atlas_lab_data = self.get_data_4_fusion(fix_img, fix_lab,atlases_imgs,atlases_labs, False)

            sim=[]
            for i in range(atlas_lab_data.shape[-2]):
                sim.append(np.squeeze(self.calculate_gt_sim(target_lab_data, atlas_lab_data[...,i,0])))
            sim=np.stack(sim,axis=-1)

            target_img_batch.append(target_img_data)
            target_lab_batch.append(target_lab_data)
            atlas_img_batch.append(atlas_img_data)
            atlas_lab_batch.append(atlas_lab_data)
            sims_batch.append(sim)

        target_img_batch = np.stack(target_img_batch, axis=0)
        target_lab_batch = np.stack(target_lab_batch, axis=0)
        atlas_img_batch = np.stack(atlas_img_batch, axis=0)
        atlas_lab_batch = np.stack(atlas_lab_batch, axis=0)
        sims_batch = np.stack(sims_batch, axis=0)
        return target_img_batch, target_lab_batch, atlas_img_batch, atlas_lab_batch, np.expand_dims(sims_batch,axis=-1),fix_img,fix_lab

    def get_file(self):
        if self.is_train==False:
            sample_dir=self.data_samples[self.index]
            self.index=(self.index+1)%self.len
        else:
            sample_dir = self.data_samples[np.random.randint(len(self.data_samples))]

        atlases_imgs=(sort_glob(sample_dir + "/*atlas_image*.nii.gz"))
        atlases_labs=(sort_glob(sample_dir + "/*atlas_label*.nii.gz"))
        fix_imgs=(sort_glob(sample_dir + "/*target_image*.nii.gz"))
        fix_labs=(sort_glob(sample_dir + "/*target_label*.nii.gz"))
        return fix_imgs,fix_labs,atlases_imgs,atlases_labs

    def merge_atlas_label(self, atlas_lab):

        atlas_lab=atlas_lab[...,0,:]
        # sitk_write_lab(atlas_lab,dir='../tmp',name='before')
        atlas_lab=self.shift(atlas_lab)
        # sitk_write_lab(atlas_lab,dir='../tmp',name='after')

        # atlas_lab = (np.sum(atlas_lab, axis=-2) > 0).astype(np.int16)

        return atlas_lab

    def calculate_gt_sim(self,target_lab,atlas_lab):
        # 需要生成添加适合的atlas lab
        atlas_lab=np.squeeze(atlas_lab)
        target_lab=np.squeeze(target_lab)
        # atlas_lab = (np.sum(atlas_lab,axis=-1)>0).astype(np.int16)
        diff=atlas_lab-target_lab
        #不为0就是差异
        same=np.where(diff==0,1,0)
        same_sum=ndimage.convolve(same,np.ones([self.args.patch_size,self.args.patch_size,self.args.patch_size]))
        sim=same_sum.astype(np.float32)/(self.args.patch_size*self.args.patch_size*self.args.patch_size)
        return np.expand_dims(sim,-1)

    def get_dataV3(self, p_target_img, p_target_lab, p_atlas_imgs, p_atlas_labs, is_aug=True):

        p_imgs = p_target_img

        p_labs = p_target_lab+[p_atlas_labs[np.random.randint(len(p_atlas_labs))]]


        imgs, labs = sitk_read_img_lab(p_imgs, p_labs, is_aug)

        imgs = np.stack([convert_img_2_nor_array(img) for img in imgs], axis=-1)
        labs = np.stack([convert_lab_2_array(lab) for lab in labs], axis=-1)

        # atlas_img = imgs[..., 1:]
        atlas_lab = labs[..., 1:]
        target_img = imgs[..., 0]
        target_lab = labs[..., 0]
        # return np.expand_dims(target_img, axis=-1), np.expand_dims(target_lab,axis=-1), np.expand_dims(atlas_img,axis=-1), np.expand_dims(atlas_lab,axis=-1)
        return np.expand_dims(target_img, axis=-1), np.expand_dims(target_lab,axis=-1),None, np.expand_dims(atlas_lab,axis=-1)

    def get_data_4_fusion(self, p_target_img, p_target_lab, p_atlas_imgs, p_atlas_labs, is_aug=True):
        p_imgs = p_target_img + p_atlas_imgs
        p_labs = p_target_lab + p_atlas_labs
        imgs, labs = sitk_read_img_lab(p_imgs, p_labs, is_aug)

        imgs = np.stack([convert_img_2_nor_array(img) for img in imgs], axis=-1)
        labs = np.stack([convert_lab_2_array(lab) for lab in labs], axis=-1)

        atlas_img = imgs[..., 1:]
        atlas_lab = labs[..., 1:]
        target_img = imgs[..., 0]
        target_lab = labs[..., 0]
        return np.expand_dims(target_img, axis=-1), np.expand_dims(target_lab,axis=-1), np.expand_dims(atlas_img,axis=-1), np.expand_dims(atlas_lab,axis=-1)


    def shift(self, arr,shift_len=None, fill_value=0):
        if shift_len==None:
            shift_len=(np.random.randint(-4,4),np.random.randint(-4,4),np.random.randint(-4,4))
        arr=np.roll(arr,shift_len,axis=[0,1,3])
        return arr

        # result = np.empty_like(arr)
        # if num > 0:
        #     result[:num] = fill_value
        #     result[num:] = arr[:-num]
        # elif num < 0:
        #     result[num:] = fill_value
        #     result[:num] = arr[-num:]
        # else:
        #     result[:] = arr
        # return result





