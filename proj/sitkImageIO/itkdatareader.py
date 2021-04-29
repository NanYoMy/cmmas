
import numpy as np
import SimpleITK as sitk
import os
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
        print("get dirutil:"+self.files[case_indices])
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
        print("get dirutil:"+self.files[case_indices])
        return array

    def get_batch_data(self,case_indices=0,label_indices=None):
        pass



from scipy.stats import zscore
def convert_p_img_2_nor_array(img_path,z_score=True):
    img = sitk.ReadImage(img_path)
    return convert_img_2_nor_array(img,z_score)
def convert_img_2_nor_array(img,z_score=True):
    img = sitk.RescaleIntensity(img)
    img = sitk.GetArrayFromImage(img)
    if z_score:
        img = zscore(img, axis=None)
    return img
def convert_img_2_scale_array(img):
    img = sitk.RescaleIntensity(img,0,1)
    img = sitk.GetArrayFromImage(img)
    return img
def convert_p_img_2_nor_array_batch(img_paths):
    batches=[]
    for p in img_paths:
        batches.append(np.expand_dims(convert_p_img_2_nor_array(p), axis=-1))
    batches=np.array(batches)
    return batches

def convert_lab_2_array(lab):
    lab=sitk.GetArrayFromImage(lab)
    return lab
def convert_p_lab_2_array(lab_path):
    lab = sitk.ReadImage(lab_path)
    lab=convert_lab_2_array(lab)
    return lab
def convert_p_lab_2_array_batch(img_paths):
    batches=[]
    for p in img_paths:
        batches.append(np.expand_dims(convert_p_lab_2_array(p), axis=-1))
    batches=np.array(batches)
    return batches

import SimpleITK as sitk
from preprocessor.sitkSpatialAugment import augment_imgs_labs
def sitk_read_img_lab(img_paths, lab_paths, is_aug=False):
    imgs=[]
    labs=[]
    for p_img in img_paths:
        imgs.append(sitk.ReadImage(p_img))
    for p_lab in lab_paths:
        labs.append(sitk.ReadImage(p_lab))
    if is_aug==True:
        imgs,labs=augment_imgs_labs(imgs,labs)
    return imgs,labs

def sitk_read_dico_series(file_path = "/data/jianjunming/BEOT/BEOT_1st/B/B13-5219998/"):
    # Dicom序列所在文件夹路径（在我们的实验中，该文件夹下有多个dcm序列，混合在一起）


    # 获取该文件下的所有序列ID，每个序列对应一个ID， 返回的series_IDs为一个列表
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file_path)
    # 查看该文件夹下的序列数量
    nb_series = len(series_IDs)
    print(nb_series)
    # 通过ID获取该ID对应的序列所有切片的完整路径， series_IDs[1]代表的是第二个序列的ID
    # 如果不添加series_IDs[1]这个参数，则默认获取第一个序列的所有切片路径
    # series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_path, series_IDs[1])
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_path )

    # 新建一个ImageSeriesReader对象
    series_reader = sitk.ImageSeriesReader()

    # 通过之前获取到的序列的切片路径来读取该序列
    series_reader.SetFileNames(series_file_names)

    # 获取该序列对应的3D图像
    image3D = series_reader.Execute()

    # 查看该3D图像的尺寸
    print(image3D.GetSize())

    # 将序列保存为单个的DCM或者NRRD文件
    # sitk.WriteImage(image3D, 'img3D.dcm')
    # sitk.WriteImage(image3D, 'img3D.nrrd')
    return image3D

import imageio
from dirutil.helper import sort_glob
def read_png_series(file_path=''):
    files=sort_glob(file_path+"/*.png")
    image3d=[]
    for f in files:
        array=imageio.imread(f)
        # image3d.insert(0,array)
        image3d.append(array)

    image3d=np.stack(image3d,axis=0)
    return image3d

def read_png_seriesV2(file_path=''):
    files=sort_glob(file_path+"/*.png")
    image3d=[]
    for f in files:
        array=imageio.imread(f)
        image3d.insert(0,array)
        # image3d.append(array)

    image3d=np.stack(image3d,axis=0)
    return image3d
