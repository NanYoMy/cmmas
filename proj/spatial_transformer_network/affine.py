import tensorflow as tf
from spatial_transformer_network.grid import batch_mgrid
from spatial_transformer_network.warp import batch_warp2d, batch_warp3d


def batch_affine_warp2d(imgs, theta):
    """
    affine transforms 2d images

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, n_channel]
    theta : tf.Tensor
        parameters of affine transformation
        [n_batch, 6]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, n_channel]
    """
    n_batch = imgs.get_shape()[0]
    xlen = imgs.get_shape()[1]
    ylen = imgs.get_shape()[2]
    theta = tf.reshape(theta, [-1, 2, 3])
    matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 2])# batch,2,2
    t = tf.slice(theta, [0, 0, 2], [-1, -1, -1])#batch,2,1

    grids = batch_mgrid(n_batch, xlen, ylen)#1,2,5,5
    coords = tf.reshape(grids, [n_batch, 2, -1])#1,2,25

    T_g = tf.matmul(matrix, coords) + t#旋转平移
    # T_g = tf.batch_matmul(matrix, coords) + t#旋转平移
    T_g = tf.reshape(T_g, [n_batch, 2, xlen, ylen])
    output = batch_warp2d(imgs, T_g)
    return output


def batch_affine_warp3d(imgs, theta):
    """
    affine transforms 3d images

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, zlen, n_channel]
    theta : tf.Tensor
        parameters of affine transformation
        [n_batch, 12]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, zlen, n_channel]
    """
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    zlen = tf.shape(imgs)[3]
    theta = tf.reshape(theta, [-1, 3, 4])
    matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 3])
    t = tf.slice(theta, [0, 0, 3], [-1, -1, -1])

    grids = batch_mgrid(n_batch, xlen, ylen, zlen)
    grids = tf.reshape(grids, [n_batch, 3, -1])

    T_g = tf.matmul(matrix, grids) + t
    T_g = tf.reshape(T_g, [n_batch, 3, xlen, ylen, zlen])
    output = batch_warp3d(imgs, T_g)
    return output


ROTATION_STD = 0.1
TRANSLATION_MAX = 0.1
def rotation_translation_3D(gt):
    batch_size = gt.get_shape().as_list()[0]
    basis_exp = tf.random_normal(shape=[batch_size, 3, 3], stddev=ROTATION_STD)
    skew_exp = basis_exp - tf.transpose(basis_exp, perm=[0, 2, 1])
    rotation = tf.linalg.expm(skew_exp)

    translation = tf.random_uniform(shape=[batch_size, 3, 1], minval=-TRANSLATION_MAX, maxval=TRANSLATION_MAX)
    theta = tf.concat([rotation, translation], axis=-1)

    rot_gt = batch_affine_warp3d(gt, theta)

    return rot_gt

def rotation_translation_2D(gt):
    batch_size = gt.get_shape().as_list()[0]
    basis_exp = tf.random_normal(shape=[batch_size, 2, 2], stddev=ROTATION_STD)
    skew_exp = basis_exp - tf.transpose(basis_exp, perm=[0, 2, 1])
    rotation = tf.linalg.expm(skew_exp)

    translation = tf.random_uniform(shape=[batch_size, 2, 1], minval=-TRANSLATION_MAX, maxval=TRANSLATION_MAX)
    theta = tf.concat([rotation, translation], axis=-1)# 1，2，3

    rot_gt = batch_affine_warp2d(gt, theta)


    return rot_gt

if __name__ == '__main__':
    """
    for test

    the result will be

    the original image
    [[  0.   1.   2.   3.   4.]
     [  5.   6.   7.   8.   9.]
     [ 10.  11.  12.  13.  14.]
     [ 15.  16.  17.  18.  19.]
     [ 20.  21.  22.  23.  24.]]

    identity warped
    [[  0.   1.   2.   3.   4.]
     [  5.   6.   7.   8.   9.]
     [ 10.  11.  12.  13.  14.]
     [ 15.  16.  17.  18.  19.]
     [ 20.  21.  22.  23.  24.]]

    zoom in warped
    [[  6.    6.5   7.    7.5   8. ]
     [  8.5   9.    9.5  10.   10.5]
     [ 11.   11.5  12.   12.5  13. ]
     [ 13.5  14.   14.5  15.   15.5]
     [ 16.   16.5  17.   17.5  18. ]]
    """
    import numpy as np
    img = tf.to_float(np.arange(25).reshape(1, 5, 5, 1))
    identity_matrix = tf.to_float([1, 0, 0, 0, 1, 0])
    zoom_in_matrix = identity_matrix * 0.5
    identity_warped = batch_affine_warp2d(img, identity_matrix)
    zoom_in_warped = batch_affine_warp2d(img, zoom_in_matrix)
    random_warp=rotation_translation_2D(img)
    with tf.Session() as sess:
        print(sess.run(img[0, :, :, 0]))
        print(sess.run(identity_warped[0, :, :, 0]))
        print(sess.run(zoom_in_warped[0, :, :, 0]))
        print(sess.run(random_warp[0, :, :, 0]))


