# import glob
import SimpleITK as sitk
# from preprocessor.tools import get_bounding_box,crop_by_bbox,get_bounding_boxV2,sitkResize3DV2,sitkResample3DV2,get_bounding_box_by_id
import os
import numpy as np
from sitkImageIO.itkdatawriter import sitk_write_image
# from dirutil.helper import mkdir_if_not_exist,mk_or_cleardir,sort_glob
# import shutil
# import random
# from preprocessor.tools import reindex_label,get_rotate_ref_img
# from preprocessor.Registrator import Registrator
# from preprocessor.Rotater import Rotater
# from sitkImageIO.itkdatareader import LazySitkDataReader
from preprocessor.sitkOPtool import paste_roi_image,resample_segmentations
import scipy.ndimage
from preprocessor.sitkOPtool import recast_pixel_val
from preprocessor.tools import sitkResize3DV2
from dirutil.helper import sort_glob
def de_resize(refs,tgts,output_dir):

    for r,t in zip(refs,tgts):
        ref = sitk.ReadImage(r)
        lab = sitk.ReadImage(t)
        lab = sitkResize3DV2(lab, ref.GetSize(), sitk.sitkNearestNeighbor)
        sitk_write_image(lab,dir=output_dir,name=os.path.basename(t))

def de_crop(refs,tgts,output_dir,structure):

    for r,t in zip(refs,tgts):
        img = sitk.ReadImage(r)
        label = sitk.ReadImage(t)
        label = resample_segmentations(img, label)
        blank_img = sitk.Image(img.GetSize(), label.GetPixelIDValue())
        blank_img.CopyInformation(img)
        label_in_orignial_img = paste_roi_image(blank_img, label)
        # 标签重新转换成205或者其他对应的值
        convert = sitk.GetArrayFromImage(label_in_orignial_img).astype(np.uint16)
        convert = np.where(convert == 1, structure, convert)
        convert_img = sitk.GetImageFromArray(convert)
        convert_img.CopyInformation(label_in_orignial_img)
        sitk_write_image(convert_img, dir=output_dir, name=os.path.basename(t))

def de_rotate(refs,tgts,output_dir):
    for r,t in zip(refs,tgts):
        ref = sitk.ReadImage(r)
        lab = sitk.ReadImage(t)
        ref =recast_pixel_val(lab ,ref)
        initial_transform = sitk.CenteredTransformInitializer(ref,
                                                              lab,
                                                              sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

        lab_resampled = sitk.Resample(lab, ref, initial_transform, sitk.sitkNearestNeighbor, 0,
                                      lab.GetPixelID())

        sitk_write_image(lab_resampled, dir=output_dir, name=os.path.basename(t))

def post_process(args):
    base_ref_dir=args.dataset_dir+"/test_target/"
    # target_dir=args.test_dir
    #测试
    # target_dir=args.dataset_dir+"/test_target/rez/lab/"
    target_dirs=sort_glob(args.test_dir+"/atlas_wise/*")
    for target_dir in target_dirs:
        paste_lab_to_ori_space(args, base_ref_dir, target_dir)


from dirutil.helper import get_name_wo_suffix
def paste_lab_to_ori_space(args, base_ref_dir, target_dir):
    print("processing:" + os.path.dirname(target_dir))
    base_dir=(target_dir)
    refs = sort_glob(base_ref_dir + "/rot/img/*.nii.gz")
    tgts = sort_glob(target_dir + "/*label.nii.gz")
    output_dir_de_rez = base_dir + "_deRez"
    de_resize(refs, tgts, output_dir_de_rez)
    refs = sort_glob(base_ref_dir + "/crop/img/*.nii.gz")
    tgts = sort_glob(output_dir_de_rez + "/*.nii.gz")
    output_dir_de_rot = base_dir + "_deRot"
    de_rotate(refs, tgts, output_dir_de_rot)
    refs = sort_glob("../../dataset/MMWHS/%s-test-image" % (args.Ttarget) + "/*.nii.gz")
    tgts = sort_glob(output_dir_de_rot + "/*.nii.gz")
    output_dir_de_crop = base_dir + "_deCrop"
    de_crop(refs, tgts, output_dir_de_crop, args.component)