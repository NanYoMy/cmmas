
import SimpleITK  as sitk
import numpy as np

import os

import config.configer
from preprocessor.tools import get_bounding_box
from sitkImageIO.itkdatareader import LazySitkDataReader
from preprocessor.sitkOPtool import paste_roi_image,resample_segmentations
import scipy.ndimage
from preprocessor.sitkOPtool import recast_pixel_val
from preprocessor.tools import sitkResize3DV2

reg_config = config.configer.get_reg_config()





class PreProcessor():

    def __init__(self,img_dir,lable_dir=None):
        self.img=LazySitkDataReader(img_dir)
        if lable_dir:
            self.label=LazySitkDataReader(lable_dir)
        self.NEW_SIZSE=[96]
        # self.LABEL_INDEX=420
        self.structure=-1

    def paste_back_to_ori_img(self, label_index=205):
        # get a lable and img
        self.structure = label_index
        for i in range(self.img.num_data):
            print("process image %d" % (i + 1))
            img = self.img.get_file_obj(i)
            label = self.label.get_file_obj(i)
            label = resample_segmentations(img, label)
            blank_img = sitk.Image(img.GetSize(), label.GetPixelIDValue())
            blank_img.CopyInformation(img)
            label_in_orignial_img = paste_roi_image(blank_img, label)

            convert = sitk.GetArrayFromImage(label_in_orignial_img)
            convert = np.where(convert == 1, label_index, 0)
            convert_img = sitk.GetImageFromArray(convert)
            convert_img.CopyInformation(label_in_orignial_img)

            self.__itksave_after_convert(blank_img, convert_img, i, tag="_paste_roi", need_write_image=False)

    def paste_img_to_img(self, label_index=205):
        # get a lable and img
        self.structure = label_index
        for i in range(self.img.num_data):
            print("process image %d" % (i + 1))
            img = self.img.get_file_obj(i)
            label = self.label.get_file_obj(i)
            label = resample_segmentations(img, label)
            blank_img = sitk.Image(img.GetSize(), label.GetPixelIDValue())
            blank_img.CopyInformation(img)
            label_in_orignial_img = paste_roi_image(blank_img, label)

            convert = sitk.GetArrayFromImage(label_in_orignial_img)
            convert = np.where(convert == 1, label_index, 0)
            convert_img = sitk.GetImageFromArray(convert)
            convert_img.CopyInformation(label_in_orignial_img)

            self.__itksave_after_convert(blank_img, convert_img, i, tag="_paste_roi", need_write_image=False)

    # def resize_label_to_img(self):
    #     pass

    # def preprocess(self):
    #
    #     #get a lable and img
    #     for i in range(self.img.num_data):
    #         print("process image %d"%(i+1))
    #         image=self.img.get_data([i])
    #         label=self.label.get_data([i])
    #         print("original size: ",image.shape)
    #         image, label = self.__crop(image[0, 0:, 0:, 0:, 0], label[0, 0:, 0:, 0:, 0])
    #         #do the process
    #         image,label=self.__resize(image,label)
    #         self.__save(image,label,i)

    def convert(self):
        for i in range(self.label.num_data):
            label = self.label.get_data([i])

    def __reindex_label(self, label):
        Label_Index = [500, 600, 420, 550, 205, 820, 850]
        array = sitk.GetArrayFromImage(label)

        Label_name = ["left ventricle", " right ventricle", "left atrium", "right atrium", "myocardium",
                      "ascending aorta", "pulmonary artery"]
        for i,L in enumerate(Label_Index):
            array = np.where(array ==L , i+1, array)

        new_label=sitk.GetImageFromArray(array)
        new_label.CopyInformation(label)#这个函数关键

        return new_label

    def resize_process(self):
        for i in range(self.img.num_data):
            image = self.img.get_file_obj(i)
            newimage = self.__resample_with_interpolator(image, sitk.sitkLinear, [1,1,1])
            self.__itksave_single_img(newimage, i, tag="_resize")
    # def generate_process(self,label_index=205):
    #     '''
    #       | z    / y
    #       |    /
    #       |  /
    #        --------x
    #     sagittal 在x轴移动
    #     axial 在z轴上移动
    #     coronal在y轴上移动
    #     '''
    #     '''
    #     #MITK  ct_test_2001_image.nii.gz
    #     # axial面变化
    #     z_low=[47]
    #     z_top=[175]
    #     #coronal面的移动变化
    #     y_low=[193]
    #     y_top=[412]
    #     #sagittal面的移动
    #     x_low=[275]
    #     x_top=[442]
    #
    #     '''
    #     ####这个需要写文件顺序，从1开始
    #     # index = [1]
    #     '''
    #     MITK
    #     z    x
    #
    #     y
    #     '''
    #     ####这个安装顺序写相应的z,y,x坐标
    #     index = [21,31] #,,21,31
    #     z_low=[26,53]
    #     z_top=[120,149]
    #     y_low=[184,226]
    #     y_top=[382,364]
    #     x_low=[258,265]
    #     x_top=[423,382]
    #
    #
    #     for i,j in enumerate(index):
    #         image = self.img.get_file_obj(j-1)
    #         blank_img = sitk.Image(image.GetSize(), sitk.sitkUInt16)
    #         blank_img.CopyInformation(image)
    #         array_255 = sitk.GetArrayFromImage(blank_img)
    #         array_255 = np.where(array_255 == 0, label_index, label_index)
    #         img_255 = sitk.GetImageFromArray(array_255)
    #         img_255.CopyInformation(image)
    #
    #         crop_label=img_255[x_low[i]:x_top[i],y_low[i]:y_top[i],z_low[i]:z_top[i]]
    #
    #         label_in_orignial_img = paste_roi_image(blank_img, crop_label)
    #         self.__itksave_generator_label(label_in_orignial_img, j-1, tag="_"+str(label_index)+"//")
    #
    #

    def resize_itk(self):
        for i in range(self.img.num_data):
            image=self.img.get_file_obj(i)
            label=self.label.get_file_obj(i)
            image,label=self.__sitkresample(image,label)
            self.__itksave(image, label, i,tag="_resize")

    def preprocess_itk(self,structure=420):
        self.structure=structure

        for i in range(self.img.num_data):
            print("process image %d"%(i+1))
            if i+1==3:
                print("error")

            image=self.img.get_file_obj(i)
            label=self.label.get_file_obj(i)
            print("original size: ",image.GetSize())

            image,label=self.__sitkresample(image,label,spacing=[1,1,1])
            # self.__itksave(image, label, i)
            print("image size: ", image.GetSize())
            print("label size: ", label.GetSize())

            image, label = self.__sitkcrop(image, label, self.structure)

            label = self.__reindex_label(label)
            print("image size: ", image.GetSize())
            print("label size: ", label.GetSize())
            # image=self.__itknormalize_itensity(image)
            image,label=self.__itk_padding(image,label)
            print("paddign image size: ", image.GetSize())
            print("paddign label size: ", label.GetSize())
            self.__itksave(image,label,i)

    def __sitkresize(self,img,lab,size):
        img=sitkResize3DV2(img,size,sitk.sitkLinear)
        # img=sitkResize3D(img,size)
        lab=sitkResize3DV2(lab,size,sitk.sitkNearestNeighbor)
        # lab=sitkResize3D(lab,size)

        return img,lab

    def __reset_space(self,input,space=[1,1,1]):
        img=sitk.GetImageFromArray(sitk.GetArrayFromImage(input))
        return img

    def preprocess_mannul_itk_img(self,structure=420):
        for i in range(self.img.num_data):
            print("process image %d"%(i+1))
            image=self.img.get_file_obj(i)
            print("original size: ",image.GetSize())
            image=sitkResize3DV2(image,[96,96,96],sitk.sitkLinear)
            print("image size: ", image.GetSize())
            # image=self.__reset_space(image)

            self.__itksave_single_img(image,i,tag='resize')

    def histogram_equalization(self,image,
                               min_target_range=0,
                               max_target_range=1000,
                               use_target_range=True):
        '''
        Histogram equalization of scalar images whose single channel has an integer
        type. The goal is to map the original intensities so that resulting
        histogram is more uniform (increasing the image's entropy).
        Args:
            image (SimpleITK.Image): A SimpleITK scalar image whose pixel type
                                     is an integer (sitkUInt8,sitkInt8...
                                     sitkUInt64, sitkInt64).
            min_target_range (scalar): Minimal value for the target range. If None
                                       then use the minimal value for the scalar pixel
                                       type (e.g. 0 for sitkUInt8).
            max_target_range (scalar): Maximal value for the target range. If None
                                       then use the maximal value for the scalar pixel
                                       type (e.g. 255 for sitkUInt8).
            use_target_range (bool): If true, the resulting image has values in the
                                     target range, otherwise the resulting values
                                     are in [0,1].
        Returns:
            SimpleITK.Image: A scalar image with the same pixel type as the input image
                             or a sitkFloat64 (depending on the use_target_range value).
        '''
        arr = sitk.GetArrayViewFromImage(image)

        i_info = np.iinfo(np.uint8)
        if min_target_range is None:
            min_target_range = i_info.min
        else:
            min_target_range = np.max([i_info.min, min_target_range])
        if max_target_range is None:
            max_target_range = i_info.max
        else:
            max_target_range = np.min([i_info.max, max_target_range])

        min_val = arr.min()
        number_of_bins = arr.max() - min_val + 1
        # using ravel, not flatten, as it does not involve memory copy
        hist = np.bincount((arr - min_val).ravel(), minlength=number_of_bins)
        cdf = np.cumsum(hist)
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
        res = cdf[arr - min_val]
        if use_target_range:
            res = (min_target_range + res * (max_target_range - min_target_range)).astype(arr.dtype)
        return sitk.GetImageFromArray(res)

    def preprocess_mannul_itk_img_label(self, structure):
        self.structure=structure

        for i in range(self.img.num_data):
            print("process image %d"%(i+1))

            image=self.img.get_file_obj(i)
            label=self.label.get_file_obj(i)
            print("original size: ",image.GetSize())

            # image,label=self.__sitkresample(image,label,spacing=[1,1,1])
            # self.__itksave(image, label, i)
            # print("image size: ", image.GetSize())
            # print("label size: ", label.GetSize())

            image,label=self.__sitkresize(image,label,[96,96,96])

            # self.augment_images_intensity(image,i,"aug_img","nii.gz")
            # image=self.histogram_equalization(sitk.Cast(image,sitk.sitkInt16))
            # label

            label = self.__reindex_label(label)
            print("image size: ", image.GetSize())
            print("label size: ", label.GetSize())
            # image=self.__itknormalize_itensity(image)
            # image,label=self.__itk_padding(image,label)
            # print("paddign image size: ", image.GetSize())
            # print("paddign label size: ", label.GetSize())
            # image=self.__reset_space(image)
            # label=self.__reset_space(label)

            self.__itksave(image,label,i,tag='resize')


    def __itknormalize_itensity(self,img):
        nor_img=sitk.Normalize(img)

        return nor_img

    def __itksave_after_convert(self,image,label,i,tag="_crop",need_write_image=True):
        # path1=os.path.dirname(self.img.dir_name)+"//"+str(self.structure)+"//"+os.path.basename(self.img.dir_name)+"_"+tag
        # path2=os.path.dirname(self.label.dir_name)+"//"+str(self.structure)+"//"+os.path.basename(self.label.dir_name)+"_"+tag
        image_path = os.path.join(
            os.path.dirname(self.img.dir_name) + "//" + os.path.basename(self.img.dir_name) + tag + "//",
            os.path.basename(self.img.files[i]))
        label_path = os.path.join(
            os.path.dirname(self.label.dir_name) + "//" + os.path.basename(self.label.dir_name) + tag + "//",
            os.path.basename(self.label.files[i]))
        # image_path = os.path.join(path1, os.path.basename(self.img.files[i]))
        # label_path = os.path.join(path2, os.path.basename(self.label.files[i]))

        if not os.path.exists(os.path.dirname(label_path)):
            os.makedirs(os.path.dirname(label_path))
        if need_write_image==True:
            if not os.path.exists(os.path.dirname(image_path)):
                os.makedirs(os.path.dirname(image_path))
            sitk.WriteImage(image,image_path)

        #扩展维度
        array = sitk.GetArrayFromImage(label)
        tmp=array.astype(np.uint16)
        new_label=sitk.GetImageFromArray(tmp)
        new_label.CopyInformation(label)#这个函数关键
        sitk.WriteImage(new_label,label_path)

    def __itksaveStep(self,image,label,i,tag="crop",need_write_image=True):
        path1=os.path.dirname(self.img.dir_name)+"//"+os.path.basename(self.img.dir_name)+"_"+tag
        path2=os.path.dirname(self.label.dir_name)+"//"+os.path.basename(self.label.dir_name)+"_"+tag
        image_path = os.path.join(path1, os.path.basename(self.img.files[i]))
        label_path = os.path.join(path2, os.path.basename(self.label.files[i]))

        if not os.path.exists(os.path.dirname(label_path)):
            os.makedirs(os.path.dirname(label_path))
        if need_write_image==True:
            if not os.path.exists(os.path.dirname(image_path)):
                os.makedirs(os.path.dirname(image_path))
            array = sitk.GetArrayFromImage(image)
            tmp = array.astype(np.int16)
            new_img = sitk.GetImageFromArray(tmp)
            new_img.CopyInformation(image)  # 这个函数关键
            sitk.WriteImage(new_img,image_path)
        #不是uint16后面出问题
        array = sitk.GetArrayFromImage(label)
        tmp=array.astype(np.uint16)
        new_label=sitk.GetImageFromArray(tmp)
        new_label.CopyInformation(label)#这个函数关键
        sitk.WriteImage(new_label,label_path)

    def __itksave(self,image,label,i,tag="crop",need_write_image=True):
        if need_write_image==True:
            self.__itksave_single_img(image,i,tag)
        self.__itksave_single_label(label,i,tag)

    def __itksave_single_img(self, image, i, tag="_crop"):
        save_dir = os.path.dirname(self.img.dir_name) + "//" + os.path.basename(self.img.dir_name) + "_" + tag

        image_path = os.path.join(save_dir, os.path.basename(self.img.files[i]))
        if not os.path.exists(os.path.dirname(image_path)):
            os.makedirs(os.path.dirname(image_path))
        array = sitk.GetArrayFromImage(image)
        tmp = array.astype(np.int16)
        new_img = sitk.GetImageFromArray(tmp)
        new_img.CopyInformation(image)  # 这个函数关键
        sitk.WriteImage(new_img, image_path)

    def __itksave_single_label(self,label,i,tag="_crop"):
        save_dir = os.path.dirname(self.label.dir_name) + "//" + os.path.basename(self.label.dir_name) + "_" + tag
        label_path = os.path.join(save_dir, os.path.basename(self.label.files[i]))

        if not os.path.exists(os.path.dirname(label_path)):
            os.makedirs(os.path.dirname(label_path))
        # 不是uint16后面出问题
        array = sitk.GetArrayFromImage(label)
        tmp = array.astype(np.uint16)
        new_label = sitk.GetImageFromArray(tmp)
        new_label.CopyInformation(label)  # 这个函数关键
        sitk.WriteImage(new_label, label_path)

    # def __itksave_generator_label(self, image, i, tag="_crop"):
    #     tmp = os.path.dirname(self.img.dir_name)
    #     out = tmp + "//" + os.path.basename(self.img.dir_name).replace('image', 'label') + tag
    #     image_path = os.path.join(out, os.path.basename(self.img.files[i]).replace("image", "label"))
    #     if not os.path.exists(os.path.dirname(image_path)):
    #         os.makedirs(os.path.dirname(image_path))
    #     sitk.WriteImage(image, image_path)

    '''
    对spaceing 重采用，一般大小是1x1x1
    '''
    def __sitkresample(self,image,label,spacing=[1,1,1]):
        newimage = self.__resample_with_interpolator(image,sitk.sitkLinear,spacing)
        newlabel = self.__resample_with_interpolator(label, sitk.sitkNearestNeighbor,spacing)
        return newimage,newlabel

    def __resample_with_interpolator(self, image,interpolator,spacing):
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(interpolator)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        new_spacing = spacing
        resample.SetOutputSpacing(new_spacing)
        orig_size = np.array(image.GetSize(), dtype=np.int)
        orig_spacing = list(image.GetSpacing())
        new_size=[oz*os/nz for oz,os,nz in zip(orig_size,orig_spacing,new_spacing)]
        new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
        new_size = [int(s) for s in new_size]
        resample.SetSize(new_size)
        newimage = resample.Execute(image)
        return newimage

    def __sitkcrop(self, image, label,label_index):

        array=sitk.GetArrayFromImage(label)
        mask=np.where(array==label_index,1,0)
        center=scipy.ndimage.measurements.center_of_mass(mask)
        print(center)
        #z y x
        pos1,pos2=self.__get_crop_bounding_box(center, mask.shape, size=[self.NEW_SIZSE[0] / 2] * 3)
        print("===============")
        print(pos1)
        print(pos2)
        crop_image=image[int(pos1[2]):int(pos2[2]),int(pos1[1]):int(pos2[1]),int(pos1[0]):int(pos2[0])]
        crop_label=label[int(pos1[2]):int(pos2[2]),int(pos1[1]):int(pos2[1]),int(pos1[0]):int(pos2[0])]
        return crop_image,crop_label

    def __sitkcrop_by_label_index(self, image, label,label_index):

        array=sitk.GetArrayFromImage(label)
        mask=np.where(array==label_index,1,0)

        # multi_slice_viewer(mask)
        new_label=sitk.GetImageFromArray(mask)
        new_label.CopyInformation(label)#这个函数关键
        center=scipy.ndimage.measurements.center_of_mass(mask)
        print(center)
        pos1,pos2=self.__get_bounding_box(center,mask.shape,size=[self.NEW_SIZSE[0]/2]*3)
        crop_image=image[pos1[2]:pos2[2],pos1[1]:pos2[1],pos1[0]:pos2[0]]
        crop_label=new_label[pos1[2]:pos2[2],pos1[1]:pos2[1],pos1[0]:pos2[0]]

        # crop_image=sitk.GetArrayViewFromImage(image)[pos1[0]:pos2[0],pos1[1]:pos2[1],pos1[2]:pos2[2]]
        # crop_label = mask[pos1[0]:pos2[0], pos1[1]:pos2[1] ,pos1[2]:pos2[2]]
        # multi_slice_viewer(crop_label)
        # return sitk.GetImageFromArray(crop_image),sitk.GetImageFromArray(crop_label)
        return crop_image,crop_label

    def __itk_padding(self,img,lab):
        padder=sitk.ConstantPadImageFilter()
        padder.SetConstant(-3)
        padder.SetPadLowerBound([0,0,0])
        size=self.NEW_SIZSE[0]
        padder.SetPadUpperBound([size-img.GetSize()[0],size-img.GetSize()[1],size-img.GetSize()[2]])
        pad_img=padder.Execute(img)
        padder.SetConstant(0)
        pad_lab=padder.Execute(lab)
        return pad_img,pad_lab

    def __get_bounding_box(self,center,shape,size=[48,48,48]):

        pos1=np.ceil(center)-np.array(size)
        pos2=np.ceil(center)+np.array(size)
        pos1=np.clip(pos1,np.array([0,0,0]),np.array(shape))
        pos2 = np.clip(pos2, np.array([0, 0, 0]), np.array(shape))

        return pos1.astype(int).tolist(),pos2.astype(int).tolist()

    def __shift_pos(self, pos1, pos2, shape, crop_vol_size=96):
        dist=pos2-pos1
        shift_pos1=[]
        shift_pos2=[]
        print("before fix:")
        print(pos1)
        print(pos2)
        for d,low,high,s in zip(dist.tolist(),pos1.tolist(),pos2.tolist(),shape):
            if d!=crop_vol_size:
                print("error distance"+str(d))
                if low==0:
                    high=crop_vol_size
                elif high==s:
                    low=s-crop_vol_size
            else:

                pass
            shift_pos1.append(low)
            shift_pos2.append(high)
        return shift_pos1,shift_pos2

    def __get_crop_bounding_box(self, center, shape, size=[48, 48, 48]):

        pos1=np.ceil(center)-np.array(size)
        pos2=np.ceil(center)+np.array(size)
        pos1=np.clip(pos1,np.array([0,0,0]),np.array(shape))
        pos2 = np.clip(pos2, np.array([0, 0, 0]), np.array(shape))
        #如果超过了边界就中心移动，满足96个元素
        pos1=pos1.astype(np.int32)
        pos2=pos2.astype(np.int32)

        return self.__shift_pos(pos1,pos2,shape,crop_vol_size=size[0]*2)

    def __crop(self,image,label):

        bbox=get_bounding_box(label)
        crop_img=image[bbox[0].start:bbox[0].stop,bbox[1].start:bbox[1].stop,bbox[2].start:bbox[2].stop]
        crop_label = label[bbox[0].start:bbox[0].stop, bbox[1].start:bbox[1].stop, bbox[2].start:bbox[2].stop]
        # multi_slice_viewer(crop_label)
        # multi_slice_viewer(crop_img)
        print("croped shape: ",crop_label.shape)
        return crop_img,crop_label

    def __resize(self,image,label):

        zoom = [float(nz) / float(oz) for oz, nz in zip(image.shape,self.NEW_SIZSE )]
        new_image = scipy.ndimage.zoom(image, zoom,order=5)
        new_label = scipy.ndimage.zoom(label,zoom,order=0)
        print("scaled size: ",new_label.shape)
        # multi_slice_viewer(new_image)
        # multi_slice_viewer(new_label)
        return  new_image,new_label

    def __convert_label_in_4_dim(self, label):
        intensity = np.sort(np.unique(label))
        print(intensity)
        labels = []
        for i, gray in enumerate(intensity[1:]):
            labels.append(np.copy(label))
            labels[i] = np.where(labels[i] == gray, 1, 0)
        labels_4D = np.stack(labels, -1)
        return  labels_4D

    def augment_images_intensity(self,img,j, output_prefix, output_suffix):
        '''
        Generate intensity modified images from the originals.
        Args:
            image_list (iterable containing SimpleITK images): The images whose intensities we modify.
            output_prefix (string): output dirutil name prefix (dirutil name: output_prefixi_FilterName.output_suffix).
            output_suffix (string): output dirutil name suffix (dirutil name: output_prefixi_FilterName.output_suffix).
        '''

        # Create a list of intensity modifying filters, which we apply to the given images
        filter_list = []

        # Smoothing filters

        filter_list.append(sitk.SmoothingRecursiveGaussianImageFilter())
        filter_list[-1].SetSigma(2.0)

        filter_list.append(sitk.DiscreteGaussianImageFilter())
        filter_list[-1].SetVariance(4.0)

        filter_list.append(sitk.BilateralImageFilter())
        filter_list[-1].SetDomainSigma(4.0)
        filter_list[-1].SetRangeSigma(8.0)

        filter_list.append(sitk.MedianImageFilter())
        filter_list[-1].SetRadius(8)

        # Noise filters using default settings

        # Filter control via SetMean, SetStandardDeviation.
        filter_list.append(sitk.AdditiveGaussianNoiseImageFilter())

        # Filter control via SetProbability
        filter_list.append(sitk.SaltAndPepperNoiseImageFilter())

        # Filter control via SetScale
        filter_list.append(sitk.ShotNoiseImageFilter())

        # Filter control via SetStandardDeviation
        filter_list.append(sitk.SpeckleNoiseImageFilter())

        filter_list.append(sitk.AdaptiveHistogramEqualizationImageFilter())
        filter_list[-1].SetAlpha(1.0)
        filter_list[-1].SetBeta(0.0)

        filter_list.append(sitk.AdaptiveHistogramEqualizationImageFilter())
        filter_list[-1].SetAlpha(0.0)
        filter_list[-1].SetBeta(1.0)

        aug_image_lists = []  # Used only for display purposes in this notebook.

        aug_image_lists.append([f.Execute(img) for f in filter_list])
        for aug_image, f in zip(aug_image_lists[-1], filter_list):
            sitk.WriteImage(aug_image, output_prefix  + str(j)+'_' +
                            f.GetName() + '.' + output_suffix)
        return aug_image_lists


class PostProcessor():
    def __init__(self, target_dir, ref_dir):
        self.target_labels = LazySitkDataReader(target_dir)
        self.references = LazySitkDataReader(ref_dir)

    def de_resize(self, structure):
        self.structure = structure
        save_dir = self.target_labels.dir_name + "_deResize"
        for i in range(self.target_labels.num_data):
            print("process image %d" % (i + 1))

            ref = self.references.get_file_obj(i)
            lab = self.target_labels.get_file_obj(i)
            lab = sitkResize3DV2(lab, ref.GetSize(), sitk.sitkNearestNeighbor)
            self.__itksave_label(lab, i,save_dir)

        return save_dir

    def de_crop(self,structure):
        # get a lable and img
        save_dir = self.target_labels.dir_name + "_deCrop"
        self.structure = structure
        for i in range(self.target_labels.num_data):
            print("process image %d" % (i + 1))
            img = self.references.get_file_obj(i)
            label = self.target_labels.get_file_obj(i)

            label = resample_segmentations(img, label)
            blank_img = sitk.Image(img.GetSize(), label.GetPixelIDValue())
            blank_img.CopyInformation(img)
            label_in_orignial_img = paste_roi_image(blank_img, label)
            # 标签重新转换成205或者其他对应的值
            convert = sitk.GetArrayFromImage(label_in_orignial_img)
            convert = np.where(convert == 1, structure, 0)
            convert_img = sitk.GetImageFromArray(convert)
            convert_img.CopyInformation(label_in_orignial_img)

            self.__itksave_label(convert_img, i, save_dir)
        return save_dir

    def de_rotate(self, structure):
        save_dir = self.target_labels.dir_name + "_deRotate"
        self.structure = structure
        for i in range(self.target_labels.num_data):
            print("processing %d" % (i))
            ref = self.references.get_file_obj(i)
            lab = self.target_labels.get_file_obj(i)
            ref=recast_pixel_val(lab,ref)
            initial_transform = sitk.CenteredTransformInitializer(ref,
                                                                  lab,
                                                                  sitk.Euler3DTransform(),
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)

            lab_resampled = sitk.Resample(lab, ref, initial_transform, sitk.sitkNearestNeighbor, 0,
                                             lab.GetPixelID())

            self.__itksave_label(lab_resampled, i,save_dir)
        return save_dir

    def __itksave_label(self,label,i,save_dir=""):
        label_path = os.path.join(save_dir, os.path.basename(self.target_labels.files[i]))

        if not os.path.exists(os.path.dirname(label_path)):
            os.makedirs(os.path.dirname(label_path))
        # 不是uint16后面出问题
        array = sitk.GetArrayFromImage(label)
        tmp = array.astype(np.uint16)
        new_label = sitk.GetImageFromArray(tmp)
        new_label.CopyInformation(label)  # 这个函数关键
        sitk.WriteImage(new_label, label_path)
