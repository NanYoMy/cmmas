import random

import numpy as np
# import tensorflow as tf
from skimage.transform import resize
from scipy.ndimage.interpolation import zoom
import SimpleITK as sitk
from keras.utils import to_categorical

def padd(min,max,padding,size):
    start = 0 if min - padding < 0 else min - padding
    stop = (size - 1) if max + padding > (size - 1) else max + padding
    return slice(start, stop)

def get_bounding_boxV2(x,padding=0):
    res=[]
    coor = np.nonzero(x)
    size=np.shape(x)
    for i in range(3):
        xmin = np.min(coor[i])
        xmax = np.max(coor[i])
        res.append(padd(xmin,xmax,padding,size[i]))
    return res

def get_bounding_box_by_id(x,padding=0,id=5):
    res=[]
    # coor = np.nonzero(x)
    if id is not None:
        x=np.where(x==id,1,0)
    coor = np.nonzero(x)
    size=np.shape(x)
    for i in range(3):
        xmin = np.min(coor[i])
        xmax = np.max(coor[i])
        res.append(padd(xmin,xmax,padding,size[i]))
    return res

def get_bounding_box_by_ids(x,padding=0,ids=[200,1220,2112]):
    res=[]
    # coor = np.nonzero(x)
    if isinstance(x,sitk.Image):
        x=sitk.GetArrayFromImage(x)
    out = binarize_numpy_array( x,ids)
    coor = np.nonzero(out)
    size=np.shape(out)
    for i in range(3):
        xmin = np.min(coor[i])
        xmax = np.max(coor[i])
        res.append(padd(xmin,xmax,padding,size[i]))
    return res


def convertArrayToImg(array,para=None):
    img=sitk.GetImageFromArray(array)
    if para is not None:
        img.CopyInformation(para)
    return img
def binarize_img(x,ids ):
    array=sitk.GetArrayFromImage(x)
    out = np.zeros(array.shape, dtype=np.uint16)
    for L in ids:
        out = out + np.where(array == L, 1, 0)

    out_img = convertArrayToImg(out, x)

    return out_img

def binarize_numpy_array(array,ids ):

    out = np.zeros(array.shape, dtype=np.uint16)
    for L in ids:
        out = out + np.where(array == L, 1, 0)


    return out


def get_bounding_box(x,padding=0):
    """ Calculates the bounding box of a ndarray"""
    mask = x == 0
    bbox = []
    res=[]
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

            if dmask_i[-1]==False and dmask_i[0]==False:
                bbox.append(slice(0, len(mask_i) - 1))
            else:
                raise RuntimeError("the bound of label is not exist!!!!!!!")

        #####padding#####
        start=0 if bbox[kdim].start-padding < 0 else bbox[kdim].start-padding
        stop= (len(mask_i) - 1) if bbox[kdim].stop+padding > (len(mask_i) - 1) else bbox[kdim].stop+padding
        res.append(slice(start,stop))

    return res

# def random_crop_image_and_labels(warp_atlas_image, target_image ,warp_atlas_label,target_label, size=[40.40,40]):
#     """Randomly crops `image` together with `labels`.
#     Args:
#       image: A Tensor with shape [D_1, ..., D_K, N]
#       labels: A Tensor with shape [D_1, ..., D_K, M]
#       size: A Tensor with shape [K] indicating the crop size.
#     Returns:
#       A tuple of (cropped_image, cropped_label).
#     """
#     combined = tf.concat([warp_atlas_image, target_image,warp_atlas_label,target_label], axis=-1)
#     # image_shape = tf.shape(image)
#     # combined_pad = tf.image.pad_to_bounding_box(
#     #     combined, 0, 0,
#     #     tf.maximum(size[0], image_shape[0]),
#     #     tf.maximum(size[1], image_shape[1]))
#     last_atlas_img_dim = tf.shape(warp_atlas_image)[-1]
#     last_target_img_dim = tf.shape(target_image)[-1]
#     last_atlas_label_dim = tf.shape(warp_atlas_label)[-1]
#     last_target_label_dim = tf.shape(target_label)[-1]
#     seed=np.random.seed(10000)
#     combined_crop = tf.random_crop(
#         combined,
#         size=tf.concat([size, [last_atlas_img_dim + last_target_img_dim+last_atlas_label_dim+last_target_label_dim]],
#                        axis=0),seed=seed)
#     return combined_crop[:,:,:, 0:last_atlas_img_dim],\
#            combined_crop[:,:, :, last_atlas_img_dim:last_target_img_dim+last_atlas_img_dim],\
#            combined_crop[:,:, :, last_target_img_dim+last_atlas_img_dim:last_target_img_dim+last_atlas_img_dim+last_atlas_label_dim],\
#            combined_crop[:,:, :, last_target_img_dim+last_atlas_img_dim+last_atlas_label_dim:last_target_img_dim+last_atlas_img_dim+last_atlas_label_dim+last_target_label_dim]
# def simple_random_crop_image(warp_atlas_image, target_image, warp_atlas_label, target_label, cut_size):
#     """Crop training data."""
#
#     data = tf.stack([warp_atlas_image, target_image, warp_atlas_label,target_label], axis=-1)
#     seed=random.seed(10000)
#     # Randomly crop a [patch_size, patch_size, patch_size] section of the image.
#     image = tf.random_crop( data, [40,40,40, 4],seed=seed)
#
#     [crop_atlas_image, crop_target_image, crop_atlas_label,crop_target_label] = tf.unstack(image, 4, axis=-1)
#
#     return crop_atlas_image, crop_target_image, crop_atlas_label,crop_target_label
# def random_crop_for_trainning(atlas_img,atlas_lab,target_img,target_lab):
#     while True:
#         i,j,k=random.choice([0,1]),random.choice([0,1]),random.choice([0,1])
#
#         target_lab_crop=target_lab[i*48:i*48+48,j*48:j*48+48,k*48:k*48+48]
#         if np.sum(target_lab_crop)==0:
#             continue
#         atlas_lab_crop=atlas_lab[i*48:i*48+48,j*48:j*48+48,k*48:k*48+48]
#         atlas_img_crop=atlas_img[i*48:i*48+48,j*48:j*48+48,k*48:k*48+48]
#         target_img_crop=target_img[i*48:i*48+48,j*48:j*48+48,k*48:k*48+48]
#         return atlas_img_crop,target_img_crop,atlas_lab_crop,target_lab_crop


def zoom3Darray(img, new_size):
    scale=img.shape/new_size
    return zoom(img, scale)

def resize3DArray(img, new_size):
    return resize(img,new_size)

def sitkResize3D(img,new_size):

    reference_image = sitk.Image(new_size, img.GetPixelIDValue())
    reference_image.SetOrigin(img.GetOrigin())
    reference_image.SetDirection(img.GetDirection())
    reference_image.SetSpacing([sz * spc / nsz for nsz, sz, spc in zip(new_size, img.GetSize(), img.GetSpacing())])
    return sitk.Resample(img, reference_image)
    #sitk.Resample(sitk.SmoothingRecursiveGaussian(grid_image, 2.0)

def get_rotate_ref_img( data):
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

def sitkResample3DV2(image,interpolator,spacing):
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

# def ndarrayResize3D(array,new_size,interpolator):
#     img=sitk.GetImageFromArray(array)
#     img=sitkResize3DV2(img,new_size,interpolator)
#     return sitk.GetArrayFromImage(img)

def crop_by_bbox(img,bbox):
    crop_img = img[bbox[2].start:bbox[2].stop+1,bbox[1].start:bbox[1].stop+1,bbox[0].start:bbox[0].stop+1]
    return crop_img

def reindex_label(label,Label_Index=[200,1220,2221,500, 600]):
    array = sitk.GetArrayFromImage(label)
    for i,L in enumerate(Label_Index):
        array = np.where(array ==L , i+1, array)
    array=to_categorical(array)
    new_label=sitk.GetImageFromArray(array)
    new_label.CopyInformation(label)#这个函数关键

    return new_label

def reindex_label_array(array,Label_Index=[200,1220,2221,500, 600]):

    for i,L in enumerate(Label_Index):
        array = np.where(array ==L , i+1, array)
    array=to_categorical(array)
    return array

def reverse_one_hot(array):
    last_dim=array.shape[-1]
    out_shape=array.shape[:-1]
    y = array.ravel()
    array=array.reshape(y.shape[0]//last_dim,last_dim)
    array=np.argmax(array,axis=1)
    array=array.reshape(out_shape)
    return array


def normalize_mask(img,mask):
    img_array=sitk.GetArrayFromImage(img).astype(np.float32)
    mask_array=sitk.GetArrayFromImage(mask)
    indices=np.where(mask_array>0)
    mean=img_array[indices].mean()
    std=img_array[indices].std()
    img_array[indices]=(img_array[indices]-mean)/std

    #其他的值保持为0
    indices = np.where(mask_array <=0)
    img_array[indices]=0

    return convertArrayToImg(img_array,img)


def rescale_one_dir(pathes, is_image=True):
    for path in pathes:
        img=sitk.ReadImage(path)
        # img=sitk.RescaleIntensity(img)
        img=clipseScaleSitkImage(img)
        sitk.WriteImage(img,path)


def clipseScaleSitkImage(sitk_image,low=5, up=95):
    np_image = sitk.GetArrayFromImage(sitk_image)
    # threshold image between p10 and p98 then re-scale [0-255]
    p0 = np_image.min().astype('float')
    p10 = np.percentile(np_image, low)
    p99 = np.percentile(np_image, up)
    p100 = np_image.max().astype('float')
    # logger.info('p0 {} , p5 {} , p10 {} , p90 {} , p98 {} , p100 {}'.format(p0,p5,p10,p90,p98,p100))
    sitk_image = sitk.Threshold(sitk_image,
                                lower=p10,
                                upper=p100,
                                outsideValue=p10)
    sitk_image = sitk.Threshold(sitk_image,
                                lower=p0,
                                upper=p99,
                                outsideValue=p99)
    sitk_image = sitk.RescaleIntensity(sitk_image,
                                       outputMinimum=0,
                                       outputMaximum=255)
    return sitk_image


def clipScaleImage(name, low=5, up=95):
    sitk_image = sitk.ReadImage(name, sitk.sitkFloat32)
    return clipseScaleSitkImage(sitk_image,low,up)