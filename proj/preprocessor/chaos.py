from dirutil.helper import sort_glob
from sitkImageIO.itkdatawriter import sitk_write_labs,sitk_write_image,sitk_write_lab
from sitkImageIO.itkdatareader import sitk_read_dico_series,read_png_series,read_png_seriesV2
import numpy as np
'''
用于转化2d到3D上
'''
output = "../../../dataset/multi_chaos/"

def generate_3dMR(image_dirs, lab_dirs, type='mr'):
    for img_dir,lab_dir in zip(image_dirs,lab_dirs):
        # files=sort_glob(img_dir+"\\*")
        img=sitk_read_dico_series(img_dir)
        lab=read_png_series(lab_dir)
        sitk_write_image(img,dir=output+"//%s-image//"%(type),name=img_dir.split('\\')[-4]+"_%s_image"%(type))

        lab_low=np.where(lab>=55,1,0)
        lab_up=np.where(lab<=70,1,0)
        lab=lab_low*lab_up
        sitk_write_lab(lab,parameter_img=img,dir=output+"//%s-label//"%(type),name=lab_dir.split('\\')[-3]+"_%s_label"%(type))

def generate_3dCT(image_dirs, lab_dirs, type='ct'):
    for img_dir,lab_dir in zip(image_dirs,lab_dirs):
        # files=sort_glob(img_dir+"\\*")
        img=sitk_read_dico_series(img_dir)
        lab=read_png_seriesV2(lab_dir)
        sitk_write_image(img,dir=output+"//%s-image//"%(type),name=img_dir.split('\\')[-2]+"_%s_image"%(type))
        sitk_write_lab(lab,parameter_img=img,dir=output+"//%s-label//"%(type),name=lab_dir.split('\\')[-2]+"_%s_label"%(type))

if __name__=="__main__":
    image_dirs = sort_glob('E:\\consistent_workspace\\dataset\\Ori_CHOAS\\CHAOS_Train_Sets\\Train_Sets\\CT\\*\\DICOM_anon')
    lab_dirs = sort_glob('E:\\consistent_workspace\\dataset\\Ori_CHOAS\\CHAOS_Train_Sets\\Train_Sets\\CT\\*\\Ground')
    generate_3dCT(image_dirs, lab_dirs, 'ct')

    image_dirs = sort_glob('E:\\consistent_workspace\\dataset\\Ori_CHOAS\\CHAOS_Train_Sets\\Train_Sets\\MR\\*\\T1DUAL\\DICOM_anon\\InPhase')
    lab_dirs= sort_glob('E:\\consistent_workspace\\dataset\\Ori_CHOAS\\CHAOS_Train_Sets\\Train_Sets\\MR\\*\\T1DUAL\\Ground')
    generate_3dMR(image_dirs, lab_dirs, 't1_in_DUAL_mr')

    image_dirs = sort_glob('E:\\consistent_workspace\\dataset\\Ori_CHOAS\\CHAOS_Train_Sets\\Train_Sets\\MR\\*\\T1DUAL\\DICOM_anon\\OutPhase')
    lab_dirs= sort_glob('E:\\consistent_workspace\\dataset\\Ori_CHOAS\\CHAOS_Train_Sets\\Train_Sets\\MR\\*\\T1DUAL\\Ground')
    generate_3dMR(image_dirs, lab_dirs, 't1_out_DUAL_mr')

    image_dirs = sort_glob('E:\\consistent_workspace\\dataset\\Ori_CHOAS\\CHAOS_Train_Sets\\Train_Sets\\MR\\*\\T2SPIR\\DICOM_anon\\')
    lab_dirs= sort_glob('E:\\consistent_workspace\\dataset\\Ori_CHOAS\\CHAOS_Train_Sets\\Train_Sets\\MR\\*\\T2SPIR\\Ground')
    generate_3dMR(image_dirs, lab_dirs, 't2SPIR_mr')
