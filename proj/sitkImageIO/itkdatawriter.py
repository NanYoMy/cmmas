import os

import SimpleITK as sitk
import nibabel as nib
import numpy as np
from dirutil.helper import mk_or_cleardir
import imageio
def sitk_write_images(input_, parameter_img=None, dir=None, name=''):

    if dir is not None:
        if not os.path.exists(dir):
            mk_or_cleardir(dir)
        batch_size = input_.shape[0]
        for idx in range(batch_size):
            if not isinstance(input_, sitk.Image):
                img = sitk.GetImageFromArray(input_[idx, ...])
            else:
                img = input_[idx, ...]
            if parameter_img is not None:
                img.CopyInformation(parameter_img)

            sitk.WriteImage(img, os.path.join(dir, name + '_%s.nii.gz' % idx))

def sitk_write_labs(input_, parameter_img=None, dir=None, name=''):

    input_=np.where(input_ > 0.5, 1, 0)
    input_=input_.astype(np.uint16)
    if dir is not None:
        if not os.path.exists(dir):
            mk_or_cleardir(dir)
        batch_size = input_.shape[0]
        for idx in range(batch_size):
            if not isinstance(input_, sitk.Image):
                img = sitk.GetImageFromArray(input_[idx, ...])
            else:
                img = input_[idx, ...]
            if parameter_img is not None:
                img.CopyInformation(parameter_img)

            sitk.WriteImage(img, os.path.join(dir, name + '_%s.nii.gz' % idx))


def sitk_wirte_ori_image(img,  dir=None, name=''):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if dir is not None:
        sitk.WriteImage(img, os.path.join(dir, name+'.nii.gz'))


def sitk_write_lab(input_, parameter_img=None, dir=None, name=''):
    if dir is not None:
        if not os.path.exists(dir):
            mk_or_cleardir(dir)
        if not isinstance(input_, sitk.Image):
            input_ = np.where(input_ > 0.5, 1, 0)
            input_ = input_.astype(np.uint16)
            img = sitk.GetImageFromArray(input_)
        else:
            img = input_
        if parameter_img is not None:
            img.CopyInformation(parameter_img)
        sitk.WriteImage(img, os.path.join(dir, name + '.nii.gz'))


def sitk_write_multi_lab(input_, parameter_img=None, dir=None, name=''):


    if dir is not None:
        if not os.path.exists(dir):
            mk_or_cleardir(dir)
        if not isinstance(input_, sitk.Image):
            input_ = input_.astype(np.uint16)
            img = sitk.GetImageFromArray(input_)
        else:
            img = input_
        if parameter_img is not None:
            img.CopyInformation(parameter_img)
        sitk.WriteImage(img, os.path.join(dir, name + '.nii.gz'))




def sitk_write_image(input_, parameter_img=None, dir=None, name=''):

    if not os.path.exists(dir):
        os.makedirs(dir)
    if dir is not None:

        if not isinstance(input_,sitk.Image):
            img = sitk.GetImageFromArray(input_)
        else:
            img=input_
        if parameter_img is not None:
            img.CopyInformation(parameter_img)
        sitk.WriteImage(img, os.path.join(dir, name+'.nii.gz'))
    return os.path.join(dir, name+'.nii.gz')

def write_images(input_, file_path=None, file_prefix=''):
    if file_path is not None:
        batch_size = input_.shape[0]
        affine = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
        [nib.save(nib.Nifti1Image(input_[idx, ...], affine),
                  os.path.join(file_path,
                               file_prefix + '%s.nii.gz' % idx))
         for idx in range(batch_size)]

def write_png_image(img,dir,name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if isinstance(img, sitk.Image):
        img=sitk.GetArrayFromImage(img)

    imageio.imwrite(os.path.join(dir,'{}.png'.format(name)), img.astype(np.uint8))

def write_png_lab(img,dir,name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if  isinstance(img, sitk.Image):
        img=sitk.GetArrayFromImage(img)
    img=np.where(img>0,255,0)
    imageio.imwrite(os.path.join(dir,'{}.png'.format(name)), img.astype(np.uint8))
