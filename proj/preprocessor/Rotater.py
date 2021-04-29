'''
pre-registration
'''
from sitkImageIO.itkdatareader import LazySitkDataReader
import SimpleITK as sitk
import os
from preprocessor.sitkOPtool import recast_pixel_val
import numpy as np


class Rotater():
    def __init__(self, image_dir, label_dir=None, ref_img_dir=None):
        self.mv_imgs = LazySitkDataReader(image_dir)
        if label_dir:
            self.mv_labels = LazySitkDataReader(label_dir)
        if ref_img_dir:
            self.ref_img = LazySitkDataReader(ref_img_dir)

    def get_rotate_ref_img(self, data):
        dimension = data.GetDimension()


        reference_origin = np.zeros(dimension)
        reference_direction = np.identity(dimension).flatten()

        # Select arbitrary number of pixels per dimension, smallest size that yields desired results
        # or the required size of a pretrained network (e.g. VGG-16 224x224), transfer learning. This will
        # often result in non-isotropic pixel spacing.
        reference_size = [0] * dimension

        reference_spacing = [0] * dimension
        print(data.GetDirection())
        new_size=np.matmul(np.reshape(np.array(data.GetDirection()),[3,3]),np.array(data.GetSize()))
        reference_size[0]=int(abs(new_size[0]))
        reference_size[1]=int(abs(new_size[1]))
        reference_size[2]=int(abs(new_size[2]))

        new_space=np.matmul(np.reshape(np.array(data.GetDirection()),[3,3]),np.array(data.GetSpacing()))
        reference_spacing[0]=float(abs(new_space[0]))
        reference_spacing[1]=float(abs(new_space[1]))
        reference_spacing[2]=float(abs(new_space[2]))

        reference_image = sitk.Image(reference_size, data.GetPixelIDValue())
        reference_image.SetOrigin(reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)

        return reference_image

    # def get_ref_img(self, data):
    #     dimension = data.GetDimension()
    #
    #     # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    #     reference_physical_size = np.zeros(dimension)
    #
    #     reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
    #                                   zip(data.GetSize(), data.GetSpacing(), reference_physical_size)]
    #
    #     # Create the reference image with a zero origin, identity direction cosine matrix and dimension
    #     reference_origin = np.zeros(dimension)
    #     reference_direction = np.identity(dimension).flatten()
    #
    #     # Select arbitrary number of pixels per dimension, smallest size that yields desired results
    #     # or the required size of a pretrained network (e.g. VGG-16 224x224), transfer learning. This will
    #     # often result in non-isotropic pixel spacing.
    #     reference_size = [96] * dimension
    #     reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]
    #
    #     # Another possibility is that you want isotropic pixels, then you can specify the image size for one of
    #     # the axes and the others are determined by this choice. Below we choose to set the x axis to 128 and the
    #     # spacing set accordingly.
    #     # Uncomment the following lines to use this strategy.
    #     # reference_size_x = 128
    #     # reference_spacing = [reference_physical_size[0]/(reference_size_x-1)]*dimension
    #     # reference_size = [int(phys_sz/(spc) + 1) for phys_sz,spc in zip(reference_physical_size, reference_spacing)]
    #
    #     reference_image = sitk.Image(reference_size, data.GetPixelIDValue())
    #     reference_image.SetOrigin(reference_origin)
    #     reference_image.SetSpacing(reference_spacing)
    #     reference_image.SetDirection(reference_direction)
    #
    #     # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as
    #     # this takes into account size, spacing and direction cosines. For the vast majority of images the direction
    #     # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the
    #     # spacing will not yield the correct coordinates resulting in a long debugging session.
    #     reference_center = np.array(
    #         reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))
    #     return reference_image

    def registration_back_to_individual_space(self):
        for i in range(self.mv_imgs.num_data):
            print("processing %d" % (i))

            mv_img = self.mv_imgs.get_file_obj(i)

            mv_lab = self.mv_labels.get_file_obj(i)
            ref_img = self.ref_img.get_file_obj(i)

            ref_img = recast_pixel_val(mv_img, ref_img)

            initial_transform = sitk.CenteredTransformInitializer(ref_img,
                                                                  mv_img,
                                                                  sitk.Euler3DTransform(),
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)
            # mv_label_resampled=mv_lab
            # mv_img_resampled=mv_img
            # uncomment the code below if u wanna preregistration
            mv_label_resampled = sitk.Resample(mv_lab, ref_img, initial_transform, sitk.sitkNearestNeighbor, 0.0,
                                               mv_lab.GetPixelID())
            mv_img_resampled = sitk.Resample(mv_img, ref_img, initial_transform, sitk.sitkLinear, 0.0,
                                             mv_img.GetPixelID())

            self.__itksave(mv_img, mv_lab, mv_img_resampled, mv_label_resampled, i, tag="_reg_back",
                           need_write_image=False)

    def rotate_to_same_direction_image(self, ref_img, ref_lab, structure=205):
        # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/
        # https://simpleitk.readthedocs.io/en/master/registrationOverview.html
        self.structure = structure
        for i in range(self.mv_imgs.num_data):
            print("processing %d" % (i))
            mv_img = self.mv_imgs.get_file_obj(i)

            ref_img = self.get_rotate_ref_img(mv_img)

            initial_transform = sitk.CenteredTransformInitializer(ref_img,
                                                                  mv_img,
                                                                  sitk.Euler3DTransform(),
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)

            mv_img_resampled = sitk.Resample(mv_img, ref_img, initial_transform, sitk.sitkLinear, 0,
                                             mv_img.GetPixelID())

            self.__itksave_image(mv_img_resampled,  i)

    def rotate_to_same_direction_image_label(self, ref_img, ref_lab, structure=205):
        # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/
        # https://simpleitk.readthedocs.io/en/master/registrationOverview.html
        self.structure = structure
        for i in range(self.mv_imgs.num_data):
            print("processing %d" % (i))
            mv_img = self.mv_imgs.get_file_obj(i)
            mv_lab = self.mv_labels.get_file_obj(i)
            # ref_lab=recast_pixel_val(mv_lab,ref_lab)
            ref_lab = self.get_rotate_ref_img(mv_lab)
            ref_img = self.get_rotate_ref_img(mv_img)

            # multi_slice_viewer(mv_img)

            initial_transform = sitk.CenteredTransformInitializer(ref_lab,
                                                                  mv_lab,
                                                                  sitk.Euler3DTransform(),
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)
            # mv_label_resampled=mv_lab
            # mv_img_resampled=mv_img
            # uncomment the code below if u wanna preregistration
            mv_label_resampled = sitk.Resample(mv_lab, ref_lab, initial_transform, sitk.sitkNearestNeighbor, 0,
                                               mv_lab.GetPixelID())

            initial_transform = sitk.CenteredTransformInitializer(ref_img,
                                                                  mv_img,
                                                                  sitk.Euler3DTransform(),
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)

            mv_img_resampled = sitk.Resample(mv_img, ref_img, initial_transform, sitk.sitkLinear, 0,
                                             mv_img.GetPixelID())

            self.__itksave(mv_img, mv_lab, mv_img_resampled, mv_label_resampled, i)

    def __itksave(self, fix_img, fix_lab, mv_img, mv_lab, i, tag="_reg", need_write_image=True):
        # fix_img_path = os.path.join(self.mv_imgs.dir_name + "_pre_reg", os.path.basename(self.mv_imgs.files[i]))
        # fix_lab_path = os.path.join(self.mv_labels.dir_name + "_pre_reg", os.path.basename(self.mv_labels.files[i]))
        mv_img_path = os.path.join(
            os.path.dirname(self.mv_imgs.dir_name) + "//" + os.path.basename(self.mv_imgs.dir_name) + tag + "//",
            os.path.basename(self.mv_imgs.files[i]))
        mv_lab_path = os.path.join(
            os.path.dirname(self.mv_labels.dir_name) + "//" + os.path.basename(self.mv_labels.dir_name) + tag + "//",
            os.path.basename(self.mv_labels.files[i]))
        # if not os.path.exists(os.path.dirname(fix_img_path)):
        #     os.makedirs(os.path.dirname(fix_img_path))
        # if not os.path.exists(os.path.dirname(fix_lab_path)):
        #     os.makedirs(os.path.dirname(fix_lab_path))

        if not os.path.exists(os.path.dirname(mv_lab_path)):
            os.makedirs(os.path.dirname(mv_lab_path))
        # sitk.WriteImage(fix_img,fix_img_path)
        # sitk.WriteImage(fix_lab,fix_lab_path)
        if need_write_image == True:
            if not os.path.exists(os.path.dirname(mv_img_path)):
                os.makedirs(os.path.dirname(mv_img_path))
            sitk.WriteImage(mv_img, mv_img_path)
            # array = sitk.GetArrayFromImage(mv_img)
            # tmp = array.astype(np.int16)
            # new_img = sitk.GetImageFromArray(tmp)
            # new_img.CopyInformation(mv_img)  # 这个函数关键
            # sitk.WriteImage(new_img, mv_img_path)

        sitk.WriteImage(mv_lab, mv_lab_path)
    def __itksave_image(self,mv_img,i,tag="_reg"):
        # fix_img_path = os.path.join(self.mv_imgs.dir_name + "_pre_reg", os.path.basename(self.mv_imgs.files[i]))
        # fix_lab_path = os.path.join(self.mv_labels.dir_name + "_pre_reg", os.path.basename(self.mv_labels.files[i]))
        mv_img_path = os.path.join(
            os.path.dirname(self.mv_imgs.dir_name) + "//" + os.path.basename(self.mv_imgs.dir_name) + tag + "//",
            os.path.basename(self.mv_imgs.files[i]))
        if not os.path.exists(os.path.dirname(mv_img_path)):
            os.makedirs(os.path.dirname(mv_img_path))
        sitk.WriteImage(mv_img, mv_img_path)




if __name__ == "__main__":
    reg = Rotater("E:\MIA_CODE_DATA\zhuang_data\MMWHS\CT\\train\ct-image_result",
                      "E:\MIA_CODE_DATA\zhuang_data\MMWHS\CT\\train\ct-label_result",
                      "E:\MIA_CODE_DATA\zhuang_data\MMWHS\MRI\mr-image_result",
                      "E:\MIA_CODE_DATA\zhuang_data\MMWHS\MRI\mr-label_result")
    reg.rotate_to_same_direction_image_label()
