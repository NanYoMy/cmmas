import glob

import SimpleITK as sitk
import numpy as np
import scipy
from scipy.stats import zscore
from NIIVisualization.Niiplot import multi_slice_viewer
from file.helper import mkcleardir
from evaluate import  helper
from config.load_embedding_arg import get_args
args=get_args()
class PatchSampler:

    def __init__(self, target_img_dir, target_lab_dir, atlas_imgs_dir, atlas_labs_dir, patch_size=15):
        self.sample_xyz=[]
        self.atlas_img=[]
        self.atlas_lab=[]
        self.p_size=patch_size
        self.sample_id=0
        self.sample_index=-1

        for img,lab in zip(atlas_imgs_dir, atlas_labs_dir):
            self.atlas_img.append(sitk.GetArrayFromImage(sitk.ReadImage(img)))
            self.atlas_lab.append(sitk.GetArrayFromImage(sitk.ReadImage(lab)))

        self.n_atlas=len(self.atlas_lab)

        self.target_img=sitk.GetArrayFromImage(sitk.ReadImage(target_img_dir))
        self.target_lab=sitk.GetArrayFromImage(sitk.ReadImage(target_lab_dir))
        self.find_candidate_patch_center()
        print("the num of patch sample:"+str(len(self.sample_xyz)))

    def find_candidate_patch_center(self):

        all_lab=np.zeros([96,96,96,12],np.uint8)
        for i, lab in enumerate(self.atlas_lab):
            all_lab[:,:,:,i]=lab


        u_lab = np.unique(self.target_lab)
        freq_maps = np.zeros((u_lab.size,) + self.target_lab.shape, dtype=np.float32)
        for i, l in enumerate(u_lab):
            freq_maps[i] += np.sum((all_lab == l).astype(np.float32), axis=3) / float(len(self.atlas_lab))
        mask = np.max(freq_maps, axis=0) < args.consensus_thr[0]
        self.sample_xyz=np.argwhere(mask)

        self.mv_predict_lab=np.zeros(self.target_lab.shape)
        LabelStats = np.zeros((len(u_lab),) +  self.target_lab.shape)
        for i, l in enumerate(u_lab):
            LabelStats[i] = np.sum((all_lab == l).astype(np.int16), axis=3)
        TargetMV = u_lab[np.argmax(LabelStats, axis=0)]
        # 其他的用mv方式来投票
        self.mv_predict_lab[~mask] = TargetMV[~mask]

        # for i in range(96):
        #     for j in range(96):
        #         for k in range(96):
        #
        #             # if np.sum(all_lab[i,j,k,:])<12 and np.sum(all_lab[i,j,k,:])>0:
        #             # if np.sum(all_lab[i,j,k,:])<3 and self.target_lab[i,j,k]==1: 可以验证一些错误
        #             if np.sum(all_lab[i,j,k,:])<7 and np.sum(all_lab[i,j,k,:])>5:
        #             # if len(set(all_lab[i,j,k,:])):
        #                 diff_map[i,j,k]=1
        #                 self.sample_xyz.append([i,j,k])
        #             else:
        #                 diff_map[i, j, k] = 0
        np.random.shuffle(self.sample_xyz)
        sitk.WriteImage(sitk.GetImageFromArray(self.mv_predict_lab.astype(np.uint8)), 'mv_predict.nii.gz')
        sitk.WriteImage(sitk.GetImageFromArray(mask.astype(np.uint8)),"sample_map.nii.gz")
    def mean_atlas(self,data,num_atlas,num_nl_patch):
        for i in range(num_atlas):
            for j in range(num_nl_patch):
                data[i][j]=self.mean_filter(data[i][j])
        return data
        # return data
    def mean_filter(self,patch):
        return scipy.ndimage.filters.convolve(patch, np.full((3, 3, 3), 1.0 / 27))

    def get_random_shift_pos(self,pos,center_shif_rad=7):


        while True:
            out_bound = False
            shift=np.random.randint(-center_shif_rad,center_shif_rad,3)
            new_pos=pos+shift
            for i in range(3):
                if (new_pos[i])+self.p_size//2+1>96 or (new_pos[i])-self.p_size//2 <0 :
                    out_bound=True
            if out_bound==True:
                continue
            else:
                break
        return new_pos


    def next_sample(self, is_test,num_nonlocal_patch=6):
        #获取每张warped image上6个non-local patches
        p_atlas_imgs=np.zeros([self.n_atlas,num_nonlocal_patch,self.p_size,self.p_size,self.p_size])
        p_atlas_labs=np.zeros([self.n_atlas,num_nonlocal_patch,self.p_size,self.p_size,self.p_size],dtype=np.uint16)
        dices=np.zeros([self.n_atlas,num_nonlocal_patch])
        #每次只有前面10个样本，一共40个样本
        self.pos = self.get_sample_pos(is_test)

        if self.pos is None:
            return None,None,None,None,None

        p_target_img,p_target_lab=self.__crop_one_patch(self.pos,self.target_img,self.target_lab)



        for j in range(self.n_atlas):
            new_pos = self.pos  # 第一次用对应的点
            for k in range(num_nonlocal_patch):
                tmp_img,tmp_lab=self.__crop_one_patch(new_pos,self.atlas_img[j],self.atlas_lab[j])
                new_pos = self.get_random_shift_pos(self.pos,10)
                dice=helper.dice_compute(tmp_lab, p_target_lab)
                p_atlas_imgs[j][k]=tmp_img
                p_atlas_labs[j][k]=tmp_lab
                dices[j][k]=dice[0]

        # for i in range(self.n_atlas):
            # flattern=p_atlas_imgs[i].reshape(num_nonlocal_patch*(self.p_size**3))
            # p_atlas_imgs[i]=zscore(flattern).reshape((num_nonlocal_patch,self.p_size,self.p_size,self.p_size))
        #zscore
        for j in range(self.n_atlas):
            p_atlas_imgs[j] = zscore(p_atlas_imgs[j], axis=None)
        p_target_img = zscore(p_target_img, axis=None)

        p_atlas_imgs=self.mean_atlas(p_atlas_imgs,self.n_atlas,num_nonlocal_patch)
        p_target_img=self.mean_filter(p_target_img)


        #dirty code  选择一个作为训练使用

        #以此获取所有的12个图像上，每个图像6个patch
        return p_atlas_imgs,p_atlas_labs,p_target_img,p_target_lab,dices


    def get_sample_pos(self, is_test,dupplicated=False):

        while True:
            out_bound = False

            if is_test:
                self.sample_index=self.sample_index+1
            else:
                self.sample_index = np.random.randint(len(self.sample_xyz))

            #
            if self.sample_index>=len(self.sample_xyz):
                return None


            if dupplicated == True:
                pos = self.sample_xyz[self.sample_index % 20]
            else:
                pos = self.sample_xyz[self.sample_index]

            #check if out of bound
            for i in range(3):
                if (pos[i])+self.p_size//2+1>96 or (pos[i])-self.p_size//2 <0 :
                    out_bound=True
            if out_bound==True:
                continue
            else:
                break

        return pos


    def __crop_one_patch(self,pos,img,lab):

        l_x = pos[0] - self.p_size // 2
        h_x = pos[0] + self.p_size // 2 + 1
        l_y = pos[1] - self.p_size // 2
        h_y = pos[1] + self.p_size // 2 + 1
        l_z = pos[2] - self.p_size // 2
        h_z = pos[2] + self.p_size // 2 + 1
        tmp_img_patch = (img[l_x:h_x, l_y:h_y, l_z:h_z])
        tmp_lab_patch = lab[l_x:h_x, l_y:h_y, l_z:h_z]

        return tmp_img_patch,tmp_lab_patch


    # def normlize_target(self,data):
    #     # data=scipy.ndimage.filters.median_filter(data,(3,3,3))
    #     data=scipy.ndimage.filters.convolve(data, np.full((3, 3, 3), 1.0/27))
    #     # data=data-data[7,7,7]
    #     return (data-data.mean())/data.std()
    #     # return data

    # def next_sample_v2(self, dupplicated=True):
    #
    #
    #     while self.sample_index<len(self.sample_xyz)-1:
    #
    #         # self.sample_index=np.random.randint(len(self.sample_xyz))
    #         # print("random:"+str(rint))
    #         #每次只有前面10个样本，一共40个样本
    #         pos = self.get_random_pos(dupplicated)
    #         out_bound=False
    #
    #         for i in range(3):
    #             if pos[i]+self.p_size//2+1>96 or pos[i]-self.p_size//2 <0 :
    #                 out_bound=True
    #                 break
    #         if out_bound==True:
    #             continue
    #         else:
    #             break
    #     atlas_labs, atlas_patches, target_lab, target_patch,sim = self.__crop_patches(pos)
    #
    #
    #     return atlas_patches, atlas_labs,target_patch,target_lab,sim
    # def __crop_patches(self, pos):
    #     atlas_patches=[]
    #     atlas_center_labs=[]
    #     sim=[]
    #
    #     l_x = pos[0] - self.p_size // 2
    #     h_x = pos[0] + self.p_size // 2 + 1
    #     l_y = pos[1] - self.p_size // 2
    #     h_y = pos[1] + self.p_size // 2 + 1
    #     l_z = pos[2] - self.p_size // 2
    #     h_z = pos[2] + self.p_size // 2 + 1
    #
    #     target_img_patch = self.normlize_target(self.target_img[l_x:h_x, l_y:h_y, l_z:h_z])
    #     target_lab_patch=self.target_lab[l_x:h_x,l_y:h_y,l_z:h_z]
    #     mkcleardir("../tmp/")
    #     sitk.WriteImage(sitk.GetImageFromArray(target_img_patch),"../tmp/target_img.nii.gz")
    #     sitk.WriteImage(sitk.GetImageFromArray(target_lab_patch),"../tmp/target_lab.nii.gz")
    #     target_center_lab = self.target_lab[pos[0], pos[1], pos[2]]
    #
    #
    #     i=0
    #     for img, lab in zip(self.atlas_img, self.atlas_lab):
    #         tmp_img_patch = self.normlize(img[l_x:h_x, l_y:h_y, l_z:h_z])
    #         tmp_lab_patch = lab[l_x:h_x, l_y:h_y, l_z:h_z]
    #         atlas_patches.append(tmp_img_patch)
    #         # multi_slice_viewer(img[l_x:h_x,l_y:h_y,l_z:h_z])
    #         # atlas_lab_patches.append(lab[l_x:h_x,l_y:h_y,l_z:h_z])
    #         dice=helper.calculate_dice(tmp_lab_patch,target_lab_patch)
    #         sim.append(dice)
    #         atlas_center_labs.append(lab[pos[0], pos[1], pos[2]])
    #         sitk.WriteImage(sitk.GetImageFromArray(self.normlize(img[l_x:h_x, l_y:h_y, l_z:h_z])), "../tmp/%d_atlas_img_%f.nii.gz"%(i,dice))
    #         sitk.WriteImage(sitk.GetImageFromArray(lab[l_x:h_x, l_y:h_y, l_z:h_z]), "../tmp/%d_atlas_lab.nii.gz" % (i))
    #         i=i+1
    #         # multi_slice_viewer(lab[l_x:h_x, l_y:h_y, l_z:h_z])
    #     return atlas_center_labs, atlas_patches,  target_center_lab, target_img_patch,sim
    #

from config.load_embedding_arg import get_args

def get_sample_generator(base_dir="",
                           target_id=1013,args=None):


    if args.model_dir[0].find('ct_mr')!=-1:
        print((base_dir + "target_img_mr*") % (str(target_id)))
        path_atlas_img = glob.glob((base_dir + "atlas_img_ct*") % (str(target_id)))
        path_atlas_lab = glob.glob((base_dir + "atlas_lab_ct*") % (str(target_id)))
        path_target_img = glob.glob((base_dir + "target_img_mr*") % (str(target_id)))[0]
        path_target_lab = glob.glob((base_dir + "target_lab_mr*") % (str(target_id)))[0]
    else:
        path_atlas_img = glob.glob((base_dir + "atlas_img_mr*") % (str(target_id)))
        path_atlas_lab = glob.glob((base_dir + "atlas_lab_mr*") % (str(target_id)))
        path_target_img = glob.glob((base_dir + "target_img_ct*") % (str(target_id)))[0]
        path_target_lab = glob.glob((base_dir + "target_lab_ct*") % (str(target_id)))[0]
    ps = PatchSampler(path_target_img, path_target_lab, path_atlas_img, path_atlas_lab,args.patch_size[0])
    return ps