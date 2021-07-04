
import numpy as np
import nibabel as nib
import os
import logger.Logger as tflog
import SimpleITK as sitk

# 0 - get configs
from config.configer import reg_config, vote_config
from dataprocessor.itkdatareader import RegistorDataReader




def get_data_readers(dir_image0, dir_image1, dir_label0=None, dir_label1=None):

    reader_image0 = RegistorDataReader(dir_image0)
    reader_image1 = RegistorDataReader(dir_image1)

    reader_label0 = RegistorDataReader(dir_label0) if dir_label0 is not None else None
    reader_label1 = RegistorDataReader(dir_label1) if dir_label1 is not None else None

    # some checks
    # if not (reader_image0.num_data == reader_image1.num_data):
    #     raise Exception('Unequal num_data between images0 and images1!')
    # if dir_label0 is not None:
    #     if not (reader_image0.num_data == reader_label0.num_data):
    #         raise Exception('Unequal num_data between images0 and labels0!')
    #     if not (reader_image0.data_shape == reader_label0.data_shape):
    #         raise Exception('Unequal data_shape between images0 and labels0!')
    # if dir_label1 is not None:
    #     if not (reader_image1.num_data == reader_label1.num_data):
    #         raise Exception('Unequal num_data between images1 and labels1!')
    #     if not (reader_image1.data_shape == reader_label1.data_shape):
    #         raise Exception('Unequal data_shape between images1 and labels1!')
    #     if dir_label0 is not None:
    #         if not (reader_label0.num_labels == reader_label1.num_labels):
    #             raise Exception('Unequal num_labels between labels0 and labels1!')

    return reader_image0, reader_image1, reader_label0, reader_label1

def get_vote_reader(base_dir):
    pass

def random_transform_generator(batch_size, corner_scale=.1):
    offsets = np.tile([[[1., 1., 1.],
                        [1., 1., -1.],
                        [1., -1., 1.],
                        [-1., 1., 1.]]],
                      [batch_size, 1, 1]) * np.random.uniform(0, corner_scale, [batch_size, 4, 3])
    new_corners = np.transpose(np.concatenate((np.tile([[[-1., -1., -1.],
                                                         [-1., -1., 1.],
                                                         [-1., 1., -1.],
                                                         [1., -1., -1.]]],
                                                       [batch_size, 1, 1]) + offsets,
                                               np.ones([batch_size, 4, 1])), 2), [0, 1, 2])  # O = T I
    src_corners = np.tile(np.transpose([[[-1., -1., -1., 1.],
                                         [-1., -1., 1., 1.],
                                         [-1., 1., -1., 1.],
                                         [1., -1., -1., 1.]]], [0, 1, 2]), [batch_size, 1, 1])
    transforms = np.array([np.linalg.lstsq(src_corners[k], new_corners[k], rcond=-1)[0]
                           for k in range(src_corners.shape[0])])
    transforms = np.reshape(np.transpose(transforms[:][:, :][:, :, :3], [0, 2, 1]), [-1, 1, 12])
    return transforms


def initial_transform_generator(batch_size):
    identity = identity_transform_vector()
    transforms = np.reshape(np.tile(identity, batch_size), [batch_size, 1, 12])
    return transforms


def identity_transform_vector():
    identity = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
    return identity.flatten()


def get_padded_shape(size, stride):
    return [int(np.ceil(size[i] / stride)) for i in range(len(size))]



def print_result(arr):
    log = tflog.getLogger("inference", reg_config)
    nparr=np.array(arr)
    log.info("mean:"+str(np.mean(nparr)))
    log.info("std:"+str(np.std(nparr)))



def sitk_write_images(input_, parameter_img, file_path=None, file_postfix=''):
    if file_path is not None:
        batch_size = input_.shape[0]
        for idx in range(batch_size):
            img=sitk.GetImageFromArray(input_[idx, ...])
            img.CopyInformation(parameter_img)
            sitk.WriteImage(img, os.path.join(file_path, file_postfix+ '%s.nii.gz' % idx))

def sitk_write_image(input_, parameter_img, file_path=None, file_postfix=''):

    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if file_path is not None:
        img = sitk.GetImageFromArray(input_)
        img.CopyInformation(parameter_img)
        sitk.WriteImage(img, os.path.join(file_path, file_postfix))

def write_images(input_, file_path=None, file_prefix=''):
    if file_path is not None:
        batch_size = input_.shape[0]
        affine = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
        [nib.save(nib.Nifti1Image(input_[idx, ...], affine),
                  os.path.join(file_path,
                               file_prefix + '%s.nii.gz' % idx))
         for idx in range(batch_size)]


