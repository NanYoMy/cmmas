import random
import numpy as np
import tensorflow as tf
from skimage.transform import rescale,resize
from scipy.ndimage.interpolation import zoom
import SimpleITK as sitk
def get_bounding_box(x):
    """ Calculates the bounding box of a ndarray"""
    mask = x == 0
    bbox = []
    all_axis = np.arange(x.ndim)
    for kdim in all_axis:
        nk_dim = np.delete(all_axis, kdim)
        mask_i = mask.all(axis=tuple(nk_dim))

        dmask_i = np.diff(mask_i)#边缘滤波

        idx_i = np.nonzero(dmask_i)[0]#idx_i[0]不为0的x坐标
        if len(idx_i) > 2:
            # raise ValueError('Algorithm failed, {} does not have more than 2 elements!'.format(idx_i))
            print("more than 2 elements")
            if(len(idx_i)%2==0):
                bbox.append(slice(idx_i[0]+1, idx_i[-1]+1))
            else:
                bbox.append(slice(idx_i[0]+1, len(mask_i)-1))

        elif len(idx_i)==2:
            bbox.append(slice(idx_i[0]+1, idx_i[-1]+1))
        elif len(idx_i)==1:
            print("the bound of label is 1")
            if dmask_i[-1]==False:
                bbox.append(slice(idx_i[0]+1, len(mask_i)-1))
            else:
                bbox.append(slice(0,idx_i[0]-1))
        elif len(idx_i)==0:
            raise RuntimeError("the bound of label is not exist!!!!!!!")
    return bbox
def random_crop_image_and_labels(warp_atlas_image, target_image ,warp_atlas_label,target_label, size=[40.40,40]):
    """Randomly crops `image` together with `labels`.
    Args:
      image: A Tensor with shape [D_1, ..., D_K, N]
      labels: A Tensor with shape [D_1, ..., D_K, M]
      size: A Tensor with shape [K] indicating the crop size.
    Returns:
      A tuple of (cropped_image, cropped_label).
    """
    combined = tf.concat([warp_atlas_image, target_image,warp_atlas_label,target_label], axis=-1)
    # image_shape = tf.shape(image)
    # combined_pad = tf.image.pad_to_bounding_box(
    #     combined, 0, 0,
    #     tf.maximum(size[0], image_shape[0]),
    #     tf.maximum(size[1], image_shape[1]))
    last_atlas_img_dim = tf.shape(warp_atlas_image)[-1]
    last_target_img_dim = tf.shape(target_image)[-1]
    last_atlas_label_dim = tf.shape(warp_atlas_label)[-1]
    last_target_label_dim = tf.shape(target_label)[-1]
    seed=np.random.seed(10000)
    combined_crop = tf.random_crop(
        combined,
        size=tf.concat([size, [last_atlas_img_dim + last_target_img_dim+last_atlas_label_dim+last_target_label_dim]],
                       axis=0),seed=seed)
    return combined_crop[:,:,:, 0:last_atlas_img_dim],\
           combined_crop[:,:, :, last_atlas_img_dim:last_target_img_dim+last_atlas_img_dim],\
           combined_crop[:,:, :, last_target_img_dim+last_atlas_img_dim:last_target_img_dim+last_atlas_img_dim+last_atlas_label_dim],\
           combined_crop[:,:, :, last_target_img_dim+last_atlas_img_dim+last_atlas_label_dim:last_target_img_dim+last_atlas_img_dim+last_atlas_label_dim+last_target_label_dim]
def simple_random_crop_image(warp_atlas_image, target_image, warp_atlas_label, target_label, cut_size):
    """Crop training data."""

    data = tf.stack([warp_atlas_image, target_image, warp_atlas_label,target_label], axis=-1)
    seed=random.seed(10000)
    # Randomly crop a [patch_size, patch_size, patch_size] section of the image.
    image = tf.random_crop( data, [40,40,40, 4],seed=seed)

    [crop_atlas_image, crop_target_image, crop_atlas_label,crop_target_label] = tf.unstack(image, 4, axis=-1)

    return crop_atlas_image, crop_target_image, crop_atlas_label,crop_target_label
def random_crop_for_trainning(atlas_img,atlas_lab,target_img,target_lab):
    while True:
        i,j,k=random.choice([0,1]),random.choice([0,1]),random.choice([0,1])

        target_lab_crop=target_lab[i*48:i*48+48,j*48:j*48+48,k*48:k*48+48]
        if np.sum(target_lab_crop)==0:
            continue
        atlas_lab_crop=atlas_lab[i*48:i*48+48,j*48:j*48+48,k*48:k*48+48]
        atlas_img_crop=atlas_img[i*48:i*48+48,j*48:j*48+48,k*48:k*48+48]
        target_img_crop=target_img[i*48:i*48+48,j*48:j*48+48,k*48:k*48+48]
        return atlas_img_crop,target_img_crop,atlas_lab_crop,target_lab_crop


def resize3DImage(img,new_size):
    scale=img.shape/new_size
    return zoom(img, scale)

def resize3DImageV2(img, new_size):
    return resize(img,new_size)

def sitkResize3D(img,new_size):

    reference_image = sitk.Image(new_size, img.GetPixelIDValue())
    reference_image.SetOrigin(img.GetOrigin())
    reference_image.SetDirection(img.GetDirection())
    reference_image.SetSpacing([sz * spc / nsz for nsz, sz, spc in zip(new_size, img.GetSize(), img.GetSpacing())])
    return sitk.Resample(img, reference_image)
    #sitk.Resample(sitk.SmoothingRecursiveGaussian(grid_image, 2.0)

def sitkResize3DV2(image,new_size,interpolator):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    new_spacing = [sz * spc / nsz for nsz, sz, spc in zip(new_size, image.GetSize(), image.GetSpacing())]
    resample.SetOutputSpacing(new_spacing)
    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = list(image.GetSpacing())
    new_size=[oz*os/nz for oz,os,nz in zip(orig_size,orig_spacing,new_spacing)]
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    newimage = resample.Execute(image)
    return newimage