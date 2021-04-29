import tensorflow as tf
from .grid import batch_mgrid
from .warp import batch_warp2d, batch_warp3d


def batch_displacement_warp2d(imgs, vector_fields,vector_fileds_in_pixel_space=True):
    """
    warp images by free form transformation

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, n_channel]
    vector_fields : tf.Tensor
        [n_batch, 2, xlen, ylen]

    Returns
    -------
    output : tf.Tensor
    warped imagees
        [n_batch, xlen, ylen, n_channel]
    """
    vector_fields_transposed = tf.transpose(vector_fields, [0, 3, 1, 2])
    n_batch = imgs.get_shape()[0]
    xlen = imgs.get_shape()[1]
    ylen = imgs.get_shape()[2]

    grids = batch_mgrid(n_batch, xlen, ylen)

    if vector_fileds_in_pixel_space:
        # Scale the vector field from [0, xlen][0, ylen] to [-1.,1.][-1.,1.]
        vector_fields_transposed_rescaled = tf.stack([
            (vector_fields_transposed[:, 0, :, :] /
            (tf.to_float(xlen)-1.) ),
            (vector_fields_transposed[:, 1, :, :] /
            (tf.to_float(ylen)-1.) )], 1)
        T_g = grids + vector_fields_transposed_rescaled
    else:
        T_g = grids + vector_fields_transposed

    output = batch_warp2d(imgs, T_g)
    return output


# def batch_displacement_warp3d_fix_size(imgs, vector_fields):
#     """
#     warp images by displacement vector fields
#
#     Parameters
#     ----------
#     imgs : tf.Tensor
#         images to be warped
#         [n_batch, xlen, ylen, zlen, n_channel]
#     vector_fields : tf.Tensor
#         [n_batch, 3, xlen, ylen, zlen]
#
#     Returns
#     -------
#     output : tf.Tensor
#     warped imagees
#         [n_batch, xlen, ylen, zlen, n_channel]
#     """
#     vector_fields=tf.transpose(vector_fields,[0,4,1,2,3])
#     n_batch = 1
#     xlen = 96
#     ylen = 96
#     zlen = 96
#
#     grids = batch_mgrid(n_batch, xlen, ylen, zlen)
#
#     T_g = grids + vector_fields
#     output = batch_warp3d_3c(imgs, T_g)
#     return output

def batch_displacement_warp3dV2(imgs, vector_fields,vector_fileds_in_pixel_space=True):
    """
    warp images by displacement vector fields

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, zlen, n_channel]
    vector_fields : tf.Tensor
        [n_batch, 3, xlen, ylen, zlen]

    Returns
    -------
    output : tf.Tensor
    warped imagees
        [n_batch, xlen, ylen, zlen, n_channel]
    """

    vector_fields_transposed = tf.transpose(vector_fields,[0,4,1,2,3])
    n_batch = imgs.get_shape().as_list()[0]
    xlen = imgs.get_shape().as_list()[1]
    ylen = imgs.get_shape().as_list()[2]
    zlen = imgs.get_shape().as_list()[3]

    grids = batch_mgrid(n_batch, xlen, ylen, zlen)

    if vector_fileds_in_pixel_space:
        # Scale the vector field from to [-1.,1.][-1.,1.]
        vector_fields_transposed_rescaled = tf.stack([
            (2.*vector_fields_transposed[:, 0, :, :]) /
            (tf.to_float(xlen)-1.) ,
            (2. *vector_fields_transposed[:, 1, :, :]) /
            (tf.to_float(ylen)-1.) ,
            (2. *vector_fields_transposed[:, 2, :, :]) /
             (tf.to_float(zlen) - 1.)], 1)
        T_g = grids + vector_fields_transposed_rescaled
    else:
        T_g = grids + vector_fields_transposed

    output = batch_warp3d(imgs, T_g)
    return output



def batch_displacement_warp3d(imgs, vector_fields,num_batch=1,x=96,y=96,z=96):
    """
    warp images by displacement vector fields

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, zlen, n_channel]
    vector_fields : tf.Tensor
        [n_batch, 3, xlen, ylen, zlen]

    Returns
    -------
    output : tf.Tensor
    warped imagees
        [n_batch, xlen, ylen, zlen, n_channel]
    """
    vector_fields=tf.transpose(vector_fields,[0,4,1,2,3])
    n_batch = imgs.get_shape()[0]
    xlen = imgs.get_shape()[1]
    ylen = imgs.get_shape()[2]
    zlen = imgs.get_shape()[3]

    grids = batch_mgrid(n_batch, xlen, ylen, zlen)

    T_g = grids + vector_fields
    output = batch_warp3d(imgs, T_g)
    return output
