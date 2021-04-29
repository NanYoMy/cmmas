'''
pre-registration
'''
from sitkImageIO.itkdatareader import LazySitkDataReader
import SimpleITK as sitk
import os
from preprocessor.sitkOPtool import recast_pixel_val


class Registrator():
    def __init__(self, mv_image_dir, mv_label_dir=None, ref_img_dir=None):
        self.mv_imgs = LazySitkDataReader(mv_image_dir)
        if mv_label_dir:
            self.mv_labels = LazySitkDataReader(mv_label_dir)
        if ref_img_dir:
            self.ref_img= LazySitkDataReader(ref_img_dir)


    def registration_back_to_individual_space(self):
        for i in range(self.mv_imgs.num_data):
            print("processing %d"%(i))
            
            mv_img=self.mv_imgs.get_file_obj(i)

            mv_lab=self.mv_labels.get_file_obj(i)
            ref_img=self.ref_img.get_file_obj(i)

            ref_img=recast_pixel_val(mv_img,ref_img)

            initial_transform = sitk.CenteredTransformInitializer(ref_img,
                                                                  mv_img,
                                                                  sitk.Euler3DTransform(),
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)
            # mv_label_resampled=mv_lab
            # mv_img_resampled=mv_img
            # uncomment the code below if u wanna preregistration
            mv_label_resampled = sitk.Resample( mv_lab, ref_img,initial_transform, sitk.sitkNearestNeighbor, 0.0,mv_lab.GetPixelID())
            mv_img_resampled = sitk.Resample( mv_img,ref_img, initial_transform, sitk.sitkLinear, 0.0, mv_img.GetPixelID())


            self.__itksave(mv_img,mv_lab,mv_img_resampled,mv_label_resampled,i,tag="_reg_back",need_write_image=False)

    def registraion_to_common_space(self, ref_img, ref_lab,structure=205):
        self.structure=structure
        for i in range(self.mv_imgs.num_data):
            print("processing %d"%(i))
            mv_img=self.mv_imgs.get_file_obj(i)
            mv_lab=self.mv_labels.get_file_obj(i)
            ref_lab=recast_pixel_val(mv_lab,ref_lab)
            ref_img=recast_pixel_val(mv_img,ref_img)

            initial_transform = sitk.CenteredTransformInitializer(ref_lab,
                                                                  mv_lab,
                                                                  sitk.Euler3DTransform(),
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)
            # mv_label_resampled=mv_lab
            # mv_img_resampled=mv_img
            # uncomment the code below if u wanna preregistration
            mv_label_resampled = sitk.Resample( mv_lab, ref_lab,initial_transform, sitk.sitkNearestNeighbor, 0,mv_lab.GetPixelID())

            initial_transform = sitk.CenteredTransformInitializer(ref_img,
                                                                  mv_img,
                                                                  sitk.Euler3DTransform(),
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)

            mv_img_resampled = sitk.Resample( mv_img,ref_img, initial_transform, sitk.sitkLinear, 0, mv_img.GetPixelID())


            self.__itksave(mv_img,mv_lab,mv_img_resampled,mv_label_resampled,i)

    # def pre_registraion(self):
    #     for i in range(self.fix_imgs.num_data):
    #         print("processing %d" % (i))
    #         fix_img = self.fix_imgs.get_file_obj(i)
    #         fix_lab = self.fix_labels.get_file_obj(i)
    #         mv_img = self.mv_imgs.get_file_obj(i)
    #         mv_lab = self.mv_labels.get_file_obj(i)
    #
    #         initial_transform = sitk.CenteredTransformInitializer(fix_img,
    #                                                               mv_img,
    #                                                               sitk.Euler3DTransform(),
    #                                                               sitk.CenteredTransformInitializerFilter.GEOMETRY)
    #         mv_label_resampled=mv_lab
    #         mv_img_resampled=mv_img
    #         # uncomment the code below if u wanna preregistration
    #         # mv_label_resampled = sitk.Resample(mv_lab, fix_img, initial_transform, sitk.sitkNearestNeighbor, 0.0,
    #         #                                    mv_lab.GetPixelID())
    #         # mv_img_resampled = sitk.Resample(mv_img, fix_img, initial_transform, sitk.sitkLinear, 0.0,
    #         #                                  mv_img.GetPixelID())
    #
    #         self.__itksave(fix_img, fix_lab, mv_img_resampled, mv_label_resampled, i)
    # def pre_registraion_each_pair(self):
    #     for i in range(self.fix_imgs.num_data):
    #         for j in range(self.mv_imgs.num_data):
    #             print("processing %d" % (i))
    #             fix_img = self.fix_imgs.get_file_obj(i)
    #             fix_lab = self.fix_labels.get_file_obj(i)
    #             mv_img = self.mv_imgs.get_file_obj(j)
    #             mv_lab = self.mv_labels.get_file_obj(j)
    #
    #             initial_transform = sitk.CenteredTransformInitializer(fix_img,
    #                                                                   mv_img,
    #                                                                   sitk.Euler3DTransform(),
    #                                                                   sitk.CenteredTransformInitializerFilter.GEOMETRY)
    #             # mv_label_resampled=mv_lab
    #             # mv_img_resampled=mv_img
    #             # uncomment the code below if u wanna preregistration
    #             mv_label_resampled = sitk.Resample(mv_lab, fix_img, initial_transform, sitk.sitkNearestNeighbor, 0.0,
    #                                                mv_lab.GetPixelID())
    #             mv_img_resampled = sitk.Resample(mv_img, fix_img, initial_transform, sitk.sitkLinear, 0.0,
    #                                              mv_img.GetPixelID())
    #
    #             self.__itksave_i_j(fix_img, fix_lab, mv_img_resampled, mv_label_resampled, i,j)
    def __itksave(self,fix_img,fix_lab,mv_img,mv_lab,i,tag="_reg",need_write_image=True):
        # fix_img_path = os.path.join(self.mv_imgs.dir_name + "_pre_reg", os.path.basename(self.mv_imgs.files[i]))
        # fix_lab_path = os.path.join(self.mv_labels.dir_name + "_pre_reg", os.path.basename(self.mv_labels.files[i]))
        mv_img_path = os.path.join(os.path.dirname(self.mv_imgs.dir_name) +"//"+os.path.basename(self.mv_imgs.dir_name)+tag+"//", os.path.basename(self.mv_imgs.files[i]))
        mv_lab_path = os.path.join(os.path.dirname(self.mv_labels.dir_name)+"//"+os.path.basename(self.mv_labels.dir_name)+tag+"//", os.path.basename(self.mv_labels.files[i]))
        # if not os.path.exists(os.path.dirname(fix_img_path)):
        #     os.makedirs(os.path.dirname(fix_img_path))
        # if not os.path.exists(os.path.dirname(fix_lab_path)):
        #     os.makedirs(os.path.dirname(fix_lab_path))

        if not os.path.exists(os.path.dirname(mv_lab_path)):
            os.makedirs(os.path.dirname(mv_lab_path))
        # sitk.WriteImage(fix_img,fix_img_path)
        # sitk.WriteImage(fix_lab,fix_lab_path)
        if need_write_image==True:
            if not os.path.exists(os.path.dirname(mv_img_path)):
                os.makedirs(os.path.dirname(mv_img_path))
            sitk.WriteImage(mv_img,mv_img_path)
            # array = sitk.GetArrayFromImage(mv_img)
            # tmp = array.astype(np.int16)
            # new_img = sitk.GetImageFromArray(tmp)
            # new_img.CopyInformation(mv_img)  # 这个函数关键
            # sitk.WriteImage(new_img, mv_img_path)


        sitk.WriteImage(mv_lab,mv_lab_path)
    # def __itksave_i_j(self,fix_img,fix_lab,mv_img,mv_lab,i,j):
    #     fix_img_path = os.path.join(self.fix_imgs.dir_name + "_pre_reg", str(i)+"-"+str(j)+os.path.basename(self.fix_imgs.files[i]))
    #     fix_lab_path = os.path.join(self.fix_labels.dir_name + "_pre_reg", str(i)+"-"+str(j)+os.path.basename(self.fix_labels.files[i]))
    #     mv_img_path = os.path.join(self.mv_imgs.dir_name + "_pre_reg", str(i)+"-"+str(j)+os.path.basename(self.mv_imgs.files[j]))
    #     mv_lab_path = os.path.join(self.mv_labels.dir_name + "_pre_reg", str(i)+"-"+str(j)+os.path.basename(self.mv_labels.files[j]))
    #     if not os.path.exists(os.path.dirname(fix_img_path)):
    #         os.makedirs(os.path.dirname(fix_img_path))
    #     if not os.path.exists(os.path.dirname(fix_lab_path)):
    #         os.makedirs(os.path.dirname(fix_lab_path))
    #     if not os.path.exists(os.path.dirname(mv_img_path)):
    #         os.makedirs(os.path.dirname(mv_img_path))
    #     if not os.path.exists(os.path.dirname(mv_lab_path)):
    #         os.makedirs(os.path.dirname(mv_lab_path))
    #     sitk.WriteImage(fix_img,fix_img_path)
    #     sitk.WriteImage(fix_lab,fix_lab_path)
    #     sitk.WriteImage(mv_img,mv_img_path)
    #     sitk.WriteImage(mv_lab,mv_lab_path)

if __name__=="__main__":
    reg=Registrator("E:\MIA_CODE_DATA\zhuang_data\MMWHS\CT\\train\ct-image_result",
                    "E:\MIA_CODE_DATA\zhuang_data\MMWHS\CT\\train\ct-label_result",
                    "E:\MIA_CODE_DATA\zhuang_data\MMWHS\MRI\mr-image_result",
                    "E:\MIA_CODE_DATA\zhuang_data\MMWHS\MRI\mr-label_result")
    reg.registraion_to_common_space()
