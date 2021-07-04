
import numpy as np
import SimpleITK as sitk
import os
import glob
import configparser
from dataprocessor.tools import resize3DImageV2,resize3DImage,sitkResize3D,sitkResize3DV2
import glob
'''
lazy reader
'''
class LazySitkDataReader:

    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.files = os.listdir(dir_name)
        self.files.sort()
        self.num_data = len(self.files)
        # self.file_objects = [sitk.ReadImage(os.path.join(dir_name, self.files[i])) for i in range(self.num_data)]
        self.data_shape=list(self.get_file_obj(0).GetSize())
        self.num_labels=1
    def get_file_obj(self,case_indices=0):
        # return self.file_objects[case_indices]
        return sitk.ReadImage(os.path.join(self.dir_name, self.files[case_indices]))
    # def get
    def get_file(self,case_indices):
        return self.files[case_indices]

    # def get_data(self, case_indices=0,label_indices=None):
    #     img=sitk.ReadImage(os.path.join(self.dir_name, self.files[case_indices]))
    #     array=sitk.GetArrayFromImage(img)
    #     array=array.swapaxes(0,-1)
    #     return array

    def get_batch_data(self,case_indices=0,label_indices=None):
        pass

'''
greedy reader
'''
class RegistorDataReader:

    def __init__(self, dir_name,is_label=False):
        self.dir_name = dir_name
        self.files = glob.glob(dir_name+"/*.nii.gz")
        self.files.sort()
        self.num_data = len(self.files)

        if self.num_data==0:
            return

        # self.file_objects = [sitk.ReadImage(os.path.join(dir_name, self.files[i])) for i in range(self.num_data)]
        #[print(self.file_objects[itr].shape) for itr in range(self.num_data)]
        #对于普通的MI来说，只有1类，但对于lablel来说确是有多个lable的可能
        # self.num_labels = [sitk.GetArrayViewFromImage(self.file_objects[i]).shape[3] if len(sitk.GetArrayViewFromImage(self.file_objects[i]).shape) == 4
        #                    else 8
        #                    for i in range(self.num_data)]
        self.num_labels=[8 for i in range(self.num_data)]

        self.data_shape = list(self.get_file_object(0).GetSize()[0:3])
        # self.data_shape = [96,96,96]


    def get_num_labels(self, case_indices):
        return [self.num_labels[i] for i in case_indices]

    def get_file_names(self, case_indices):
        return [os.path.basename(self.files[i]).split('.')[0] for i in case_indices]

    def get_file_name(self,case_indices):
        return self.files[case_indices]

    def get_file_objects(self, case_indices):
        return [sitk.ReadImage(self.files[i]) for i in case_indices]

    def get_file_object(self,i):

        img= sitk.ReadImage(self.files[i])
        return img

    # def get_file(self,case_indices):
    #     return sitk.ReadImage(self.files[case_indices])

    # def _register_mv_to_fix(self, mv_ind, fix_img, is_label):
    #     mv_img=self.get_file_object(mv_ind)
    #     initial_transform = sitk.CenteredTransformInitializer(fix_img,
    #                                                           mv_img,
    #                                                           sitk.Euler3DTransform(),
    #                                                           sitk.CenteredTransformInitializerFilter.GEOMETRY)
    #     if is_label:
    #         reg_img = sitk.Resample(mv_img, fix_img, initial_transform, sitk.sitkNearestNeighbor, 0.0,
    #                                            mv_img.GetPixelID())
    #     else:
    #         reg_img = sitk.Resample(mv_img, fix_img, initial_transform, sitk.sitkLinear, 0.0,
    #                                          mv_img.GetPixelID())
    #     # multi_slice_viewer(reg_img)
    #     # multi_slice_viewer(fix_img)
    #     # multi_slice_viewer(mv_img)
    #     return reg_img


    # def get_registed_data(self,case_indices,ref_img,label_indices=None,is_lable=False):
    #     if label_indices is None:  # e.g. images only
    #         data = [sitk.GetArrayFromImage(self._register_mv_to_fix(i, k, is_lable))
    #                 for (i, k) in zip(case_indices, ref_img)]
    #     else:
    #         if len(label_indices) == 1:#get all same label from sample
    #             label_indices *= self.num_data
    #
    #         data = [np.where(sitk.GetArrayFromImage(self._register_mv_to_fix(i, k, is_lable))==j,1,0) if self.num_labels[i] > 1
    #                 else sitk.GetArrayFromImage(self._register_mv_to_fix(i, k, is_lable))
    #                 for (i,j,k) in zip(case_indices, label_indices,ref_img)]
    #         #[print(data[0][i].max()) for i in range(data[0].shape[0])] #查看每组数据的最大值，
    #     return np.expand_dims(np.stack(data, axis=0), axis=4).astype(np.float32)

    def normalize(self, image):
        return sitk.Normalize(image)

    def get_data(self, case_indices=None, label_indices=None, need_img_normalize=True):
        if case_indices is None:
            case_indices = range(self.num_data)
        # todo: check the supplied label_indices smaller than num_labels
        if label_indices is None:  # e.g. images only
            if need_img_normalize==True:
                data = [sitk.GetArrayFromImage(self.normalize(self.get_file_object(i))) for i in case_indices]
            else:
                data = [sitk.GetArrayFromImage(self.get_file_object(i)) for i in case_indices]
        else:
            if len(label_indices) == 1:#get all same label from sample
                label_indices *= self.num_data
            data = [np.where(sitk.GetArrayFromImage(self.get_file_object(i))==j,1,0) if self.num_labels[i] > 1
                    else sitk.GetArrayFromImage(self.get_file_object(i))
                    for (i, j) in zip(case_indices, label_indices)]
            #[print(data[0][i].max()) for i in range(data[0].shape[0])] #查看每组数据的最大值，

        return np.expand_dims(np.stack(data, axis=0), axis=4).astype(np.float32)


class VoteDataReader:
    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.files = self.__get_train_dir(dir_name)
        self.files.sort()
        self.num_data = len(self.files)
        # self.file_objects = [sitk.ReadImage(os.path.join(dir_name, self.files[i])) for i in range(self.num_data)]
        self.data_shape = self.__get_shape()
        self.num_labels = 1
    def __get_train_dir(self,dir_name):
        dirs=glob.glob(dir_name+"/*/*")
        return dirs

    def __get_shape(self):
        atlas_img,_,_,_=self.get_file_obj(0)
        return list(atlas_img.GetSize())

    def get_case_dir(self,case_indices=-1):
        return self.files[case_indices]

    def get_file_by_prefix(self,files,tag):
        for i,f in enumerate(files):
            if f.find(tag)!=-1:
               return f
        return "error_path"

    def get_file_obj(self, case_indices=0):
        # return self.file_objects[case_indices]
        train_dir=self.files[case_indices]
        files=os.listdir(train_dir)
        files.sort()
        target_file=glob.glob(train_dir+"/target*")

        if len(target_file)==2:
            return sitk.ReadImage(os.path.join(train_dir, self.get_file_by_prefix(files,"atlas_img"))), \
                   sitk.ReadImage(os.path.join(train_dir, self.get_file_by_prefix(files,"atlas_lab"))), \
                   sitk.ReadImage(os.path.join(train_dir, self.get_file_by_prefix(files,"target_img"))), \
                   sitk.ReadImage(os.path.join(train_dir, self.get_file_by_prefix(files,"target_lab")))
        else:
            return sitk.ReadImage(os.path.join(train_dir, self.get_file_by_prefix(files,"atlas_img"))), \
                   sitk.ReadImage(os.path.join(train_dir, self.get_file_by_prefix(files,"atlas_lab"))), \
                   sitk.ReadImage(os.path.join(train_dir, self.get_file_by_prefix(files,"target_img"))), \
                   None

    def get_test_file_obj(self, case_indices=0):
        # return self.file_objects[case_indices]
        train_dir=self.files[case_indices]
        files=os.listdir(train_dir)
        files.sort()
        return sitk.ReadImage(os.path.join(train_dir, files[0])), \
               sitk.ReadImage(os.path.join(train_dir, files[1])), \
               sitk.ReadImage(os.path.join(train_dir, files[2]))


    # def get
    def get_file_name(self, case_indices):
        return self.files[case_indices]
    def normalize(self, image,need_img_normalize):
        if need_img_normalize==True:
            return sitk.Normalize(image)
        else:
            return image

    def __convert(self,data):
        return np.expand_dims(np.stack(data, axis=0), axis=4).astype(np.float32)

    def get_4_data(self,case_indices,need_img_normalize=True):
        atlas_imgs=[]
        atlas_labs=[]
        target_imgs=[]
        target_labs=[]
        is_test=False
        for i in case_indices:
            atlas_img,atlas_lab,target_img,target_lab=self.get_file_obj(i)
            if target_lab==None:
                is_test=True
                atlas_imgs.append(sitk.GetArrayFromImage(self.normalize(atlas_img,need_img_normalize)))
                atlas_labs.append(sitk.GetArrayFromImage(atlas_lab))
                target_imgs.append(sitk.GetArrayFromImage(self.normalize(target_img,need_img_normalize)))
            else:
                atlas_imgs.append(sitk.GetArrayFromImage(self.normalize(atlas_img,need_img_normalize)))
                atlas_labs.append(sitk.GetArrayFromImage(atlas_lab))
                target_imgs.append(sitk.GetArrayFromImage(self.normalize(target_img,need_img_normalize)))
                target_labs.append(sitk.GetArrayFromImage(target_lab))
        if is_test==False:
            return self.__convert(atlas_imgs),\
                   self.__convert(atlas_labs),\
                   self.__convert(target_imgs),\
                   self.__convert(target_labs)
        else:
            return self.__convert(atlas_imgs), \
                   self.__convert(atlas_labs), \
                   self.__convert(target_imgs),\
                   None



'''
fusion reader
'''
class FusionSitkDataReader:

    def __init__(self, files):
        self.files =files
        self.files.sort()
        self.num_data = len(self.files)
        # self.file_objects = [sitk.ReadImage(os.path.join(dir_name, self.files[i])) for i in range(self.num_data)]
        self.data_shape=list(self.get_file_obj(0).GetSize())
        self.num_labels=1
    def get_file_obj(self,case_indices=0):
        # return self.file_objects[case_indices]
        print("get file:"+self.files[case_indices])
        return sitk.ReadImage(self.files[case_indices])
    # def get
    def get_file_path(self, case_indices):
        return self.files[case_indices]

    def get_file_name(self,case_indice):
        tmp=self.files[case_indice]
        _,fullname=os.path.split(tmp)
        return fullname
    def get_data(self, case_indices=0,label_indices=None):
        img=sitk.ReadImage(self.files[case_indices])
        array=sitk.GetArrayFromImage(img)
        print("get file:"+self.files[case_indices])
        return array

    def get_batch_data(self,case_indices=0,label_indices=None):
        pass
