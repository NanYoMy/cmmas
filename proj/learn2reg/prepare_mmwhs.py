import glob
import SimpleITK as sitk
from preprocessor.tools import get_bounding_box,crop_by_bbox,get_bounding_boxV2,sitkResize3DV2,sitkResample3DV2,get_bounding_box_by_id
import os
import numpy as np
from sitkImageIO.itkdatawriter import sitk_write_image,sitk_write_lab
from dirutil.helper import mkdir_if_not_exist,mk_or_cleardir,sort_glob
import shutil
import random
from dirutil.helper import get_name_wo_suffix
from preprocessor.tools import reindex_label,get_rotate_ref_img
from preprocessor.Registrator import Registrator
from preprocessor.Rotater import Rotater
from sitkImageIO.itkdatareader import LazySitkDataReader
def crop_ROI_data_by_label(imgs, labs, outputdir, lab_id,zscore=False):
    # 裁剪数据
    crop_img_dir=outputdir+"/crop/img"
    mkdir_if_not_exist(crop_img_dir)
    crop_lab_dir=outputdir+"/crop/lab"
    mkdir_if_not_exist(crop_lab_dir)
    crop_img_lab(imgs, labs, crop_img_dir, crop_lab_dir, lab_id)

    #rot数据
    imgs=sort_glob(crop_img_dir+"/*.nii.gz")
    labs=sort_glob(crop_lab_dir+"/*.nii.gz")
    rot_img_dir=outputdir+"/rot/img/"
    mkdir_if_not_exist(rot_img_dir)
    rot_lab_dir=outputdir+"/rot/lab/"
    mkdir_if_not_exist(rot_lab_dir)
    rotate_img_to_same_direction(imgs,labs,rot_img_dir,rot_lab_dir)

    #resize数据
    imgs=sort_glob(rot_img_dir+"/*.nii.gz")
    labs=sort_glob(rot_lab_dir+"/*.nii.gz")
    rez_img_dir=outputdir+"/rez/img/"
    mkdir_if_not_exist(rez_img_dir)
    rez_lab_dir=outputdir+"/rez/lab/"
    mkdir_if_not_exist(rez_lab_dir)
    resize_img_lab(imgs,labs,rez_img_dir,rez_lab_dir,zscore)


    # split(test_imgs_fix_dir+"_crop_reg_rez", test_lab_fix_dir+"_crop_reg_rez", crop_train_imgs_fix_dir+"_crop_reg_rez", crop_train_lab_fix_dir+"_crop_reg_rez")

def split(test_imgs_mv, test_lab_mv, train_imgs_mv, train_lab_mv):
    mk_or_cleardir(test_imgs_mv)
    mk_or_cleardir(test_lab_mv)
    file_img_mv = glob.glob(train_imgs_mv+"/*.nii.gz")
    file_img_mv.sort()
    file_lab_mv = glob.glob(train_lab_mv+"/*.nii.gz")
    file_lab_mv.sort()
    # L = random.sample(range(0, len(file_img_mv)), 8)
    L=range(0,len(file_img_mv))
    for i in L[:8]:
        shutil.move(file_img_mv[i], test_imgs_mv)
        shutil.move(file_lab_mv[i], test_lab_mv)

def resize_img_lab(imgs,labs,output_img,output_lab,zscore):

    for p_img,p_lab in zip(imgs,labs):
        img_obj=sitk.ReadImage(p_img)
        lab_obj=sitk.ReadImage(p_lab)
        resize_lab = sitkResize3DV2(lab_obj, [96, 96, 96], sitk.sitkNearestNeighbor)
        sitk_write_image(resize_lab, dir=output_lab, name=os.path.basename(p_lab))
        resize_img = sitkResize3DV2(img_obj, [96, 96, 96], sitk.sitkLinear)
        if zscore:
            resize_img=sitk.RescaleIntensity(resize_img)
            resize_img=sitk.Normalize(resize_img)
        sitk_write_image(resize_img, dir=output_img, name=os.path.basename(p_img))


def rotate_img_to_same_direction(imgs, labs, output_imgs_dir, output_labs_dir):
    for p_img,p_lab in zip(imgs,labs):
        mv_img = sitk.ReadImage(p_img)
        mv_lab = sitk.ReadImage(p_lab)
        # ref_lab=recast_pixel_val(mv_lab,ref_lab)
        ref_lab = get_rotate_ref_img(mv_lab)
        ref_img = get_rotate_ref_img(mv_img)

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
        sitk_write_image(mv_img_resampled,dir=output_imgs_dir,name=os.path.basename(p_img))
        sitk_write_image(mv_label_resampled, dir=output_labs_dir, name=os.path.basename(p_lab))

def crop_img_lab(input_imgs, input_labs, crop_imgs_dir, crop_lab_fix, id):
    for p_img, p_lab in zip(input_imgs, input_labs):
        lab = sitk.ReadImage(p_lab)

        # 先转化成统一space,保证crop的大小原始物理尺寸一致
        lab = sitkResample3DV2(lab, sitk.sitkNearestNeighbor, [1, 1, 1])
        bbox = get_bounding_box_by_id(sitk.GetArrayFromImage(lab), padding=10,id=id)
        ##extend bbox
        crop_lab = crop_by_bbox(lab, bbox)
        # crop_lab = sitkResize3DV2(crop_lab, [96, 96, 96], sitk.sitkNearestNeighbor)
        sitk_write_image(crop_lab, dir=crop_lab_fix, name=os.path.basename(p_lab))
        #
        img = sitk.ReadImage(p_img)
        img = sitkResample3DV2(img, sitk.sitkLinear, [1, 1, 1])
        crop_img = crop_by_bbox(img, bbox)
        # crop_img = sitkResize3DV2(crop_img, [96, 96, 96], sitk.sitkLinear)
        sitk_write_image(crop_img, dir=crop_imgs_dir, name=os.path.basename(p_img))

def generator_ROI_data_for_3DUnet(args):
    imgs=sort_glob("../../dataset/MMWHS/" + "/%s-image/*.nii.gz" % (args.Ttarget))
    labs=sort_glob("../../dataset/MMWHS/" + "/%s-label/*.nii.gz" % (args.Ttarget))
    crop_ROI_data_by_label(imgs[:12], labs[:12], args.dataset_dir + "/train_target/", args.component)

    imgs=sort_glob("../../dataset/MMWHS/" + "/%s-image/*.nii.gz" % (args.Ttarget))
    labs=sort_glob("../../dataset/MMWHS/" + "/%s-label/*.nii.gz" % (args.Ttarget))
    crop_ROI_data_by_label(imgs[16:], labs[16:], args.dataset_dir + "/validate_target/", args.component)

    imgs=sort_glob("../../dataset/MMWHS/" + "/%s-test-image/*.nii.gz" % (args.Ttarget))
    labs=sort_glob("../../dataset/MMWHS/" + "/%s-test-label/*.nii.gz" % (args.Ttarget))
    crop_ROI_data_by_label(imgs, labs, args.dataset_dir + "/test_target/", args.component)

def generator_ROI_data(args,type='MMWHS'):
    ####
    imgs=sort_glob("../../dataset/%s/%s-image/*.nii.gz" % (type,args.Tatlas))
    labs=sort_glob("../../dataset/%s/%s-label/*.nii.gz" % (type,args.Tatlas))
    crop_ROI_data_by_label(imgs[:20], labs[:20], args.dataset_dir + "/train_atlas/", args.component)

    # imgs=sort_glob("../../dataset/MMWHS/" + "/%s-image/*.nii.gz" % (args.Tatlas))
    # labs=sort_glob("../../dataset/MMWHS/" + "/%s-label/*.nii.gz" % (args.Tatlas))
    # prepare_data(imgs[12:16],labs[12:16],args.dataset_dir+"/train_fuse_atlas/",args.component)

    # imgs=sort_glob("../../dataset/MMWHS/" + "/%s-image/*.nii.gz" % (args.Tatlas))
    # labs=sort_glob("../../dataset/MMWHS/" + "/%s-label/*.nii.gz" % (args.Tatlas))
    # prepare_data(imgs[16:],labs[16:],args.dataset_dir+"/test_fuse_atlas/",args.component)

    #####
    imgs=sort_glob("../../dataset/%s/%s-image/*.nii.gz" % (type,args.Ttarget))
    labs=sort_glob("../../dataset/%s/%s-label/*.nii.gz" % (type,args.Ttarget))
    crop_ROI_data_by_label(imgs[:20], labs[:20], args.dataset_dir + "/train_target/", args.component)

    # imgs=sort_glob("../../dataset/%s/%s-image/*.nii.gz" % (type,args.Ttarget))
    # labs=sort_glob("../../dataset/%s/%s-label/*.nii.gz" % (type,args.Ttarget))
    # crop_ROI_data_by_label(imgs[12:16], labs[12:16], args.dataset_dir + "/train_fuse_target/", args.component)
    #
    # imgs=sort_glob("../../dataset/%s/%s-image/*.nii.gz" % (type,args.Ttarget))
    # labs=sort_glob("../../dataset/%s/%s-label/*.nii.gz" % (type,args.Ttarget))
    # crop_ROI_data_by_label(imgs[16:], labs[16:], args.dataset_dir + "/validate_target/", args.component)

    ####challenge data


    # imgs=sort_glob("../../dataset/%s/%s-test-image/*.nii.gz" % (type,args.Ttarget))
    # labs=sort_glob("../../dataset/%s/%s-test-label/*.nii.gz" % (type,args.Ttarget))
    # crop_ROI_data_by_label(imgs, labs, args.dataset_dir + "/test_target/", args.component)


def prepare_chaos_reg_working_data(args):
    if not os.path.exists(args.dataset_dir):
        mk_or_cleardir(args.dataset_dir)
        generator_ROI_data(args,'chaos')

def prepare_mmwhs_reg_working_data(args):
    if not os.path.exists(args.dataset_dir):
        mk_or_cleardir(args.dataset_dir)
        generator_ROI_data(args)

def prepare_crossvalidation_reg_data(args):
    if not os.path.exists(args.dataset_dir):
        mk_or_cleardir(args.dataset_dir)
        imgs=sort_glob("../../dataset/%s/%s-image/*.nii.gz" % (args.task,args.Tatlas))
        labs=sort_glob("../../dataset/%s/%s-label/*.nii.gz" % (args.task,args.Tatlas))
        crop_ROI_data_by_label(imgs, labs, args.dataset_dir + "/atlas/", args.component)
        #####
        imgs=sort_glob("../../dataset/%s/%s-image/*.nii.gz" % (args.task,args.Ttarget))
        labs=sort_glob("../../dataset/%s/%s-label/*.nii.gz" % (args.task,args.Ttarget))
        crop_ROI_data_by_label(imgs, labs, args.dataset_dir + "/target/", args.component)
from tool.parse import parse_arg_list
def prepare_unsupervised_reg_data(args):
    if not os.path.exists(args.dataset_dir):
        mk_or_cleardir(args.dataset_dir)
        types=parse_arg_list(args.mode)
        for t in types:
            imgs=sort_glob("../../dataset/%s/%s-image/*.nii.gz" % (args.task,t))
            labs=sort_glob("../../dataset/%s/%s-label/*.nii.gz" % (args.task,t))
            crop_ROI_data_by_label(imgs, labs, args.dataset_dir + "/%s/"%(t), args.component)
        #####
# def prepare_mm_whs_data(args):
#     if not os.path.exists(args.dataset_dir):
#         mk_or_cleardir(args.dataset_dir)
#         types=['ct','t1_in_DUAL_mr','t1_out_DUAL_mr','t2SPIR_mr']
#         for t in types:
#             imgs=sort_glob("../../dataset/%s/%s-image/*.nii.gz" % (args.task,t))
#             labs=sort_glob("../../dataset/%s/%s-label/*.nii.gz" % (args.task,t))
#             crop_ROI_data_by_label(imgs, labs, args.dataset_dir + "/%s/"%(t), args.component)

from preprocessor.tools import rescale_one_dir
def prepare_unsupervised_reg_data_for_ant(args):
    if not os.path.exists(args.dataset_dir):
        mk_or_cleardir(args.dataset_dir)
        types=parse_arg_list(args.mode)
        for t in types:
            imgs=sort_glob("../../dataset/%s/%s-image/*.nii.gz" % (args.task,t))
            labs=sort_glob("../../dataset/%s/%s-label/*.nii.gz" % (args.task,t))
            crop_ROI_data_by_label(imgs, labs, args.dataset_dir + "/%s/"%(t), args.component,True)
            images=sort_glob(args.dataset_dir+"/%s/rez/img/*.nii.gz"%(t))
            rescale_one_dir(images)
def generator_ROI_mask(args):

    labs=sort_glob("../../dataset/MMWHS/" + "/%s-test-label/*.nii.gz" % (args.Ttarget))
    for p_lab in labs:
        lab = sitk.ReadImage(p_lab)
        bbox = get_bounding_box_by_id(sitk.GetArrayFromImage(lab), padding=10,id=None)
        # sitk.RegionOfInterest(lab,)
        # crop_lab = crop_by_bbox(lab, bbox)
        array_lab=sitk.GetArrayFromImage(lab)
        array_lab[0:,0:,0:]=0
        array_lab[bbox[0].start:bbox[0].stop + 1, bbox[1].start:bbox[1].stop + 1,bbox[2].start:bbox[2].stop + 1 ]=1
        sitk_write_lab(array_lab,parameter_img=lab,dir='../../dataset/MMWHS'+"/%s-label-test_ROI/"%(args.Ttarget),name=get_name_wo_suffix(p_lab))

def prepare_mmwhs_wholeheart_ROI_mask(args):
    # if not os.path.exists(args.dataset_dir):
    #     mk_or_cleardir(args.dataset_dir)
    generator_ROI_mask(args)

def prepare_3dUnet_ROI_data(args):
    if not os.path.exists(args.dataset_dir):
        mk_or_cleardir(args.dataset_dir)
        generator_ROI_data_for_3DUnet(args)