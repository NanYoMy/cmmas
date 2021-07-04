import numpy as np
import tensorflow as tf
from .losses import gauss_kernel1d,separable_filter3d
from spatial_transformer_network.displacement import batch_displacement_warp3d



def warp_image_ddf(vol,ph_random_ddf):
    ddf_list=tf.unstack(ph_random_ddf,axis=-1)
    ddf_list=[separable_filter3d(tf.expand_dims(ddf,axis=-1),gauss_kernel1d(10)) for ddf in ddf_list]
    warp_ddf=tf.concat(ddf_list,-1)
    my_warp_mv = batch_displacement_warp3d(vol, warp_ddf)
    return my_warp_mv

def warp_image_affine(vol, theta):
    return resample_linear(vol, warp_grid(get_reference_grid(vol.get_shape()[1:4]), theta))

#theta是仿射变化矩阵,这个函数把grid进行了一个theta的变化，相当于把grid变成了一个warpped的grid
def warp_grid(grid, theta):
    # grid=grid_reference
    num_batch = int(theta.get_shape()[0])
    theta = tf.cast(tf.reshape(theta, (-1, 3, 4)), 'float32')
    size = grid.get_shape().as_list()
    grid = tf.concat([tf.transpose(tf.reshape(grid, [-1, 3])), tf.ones([1, size[0]*size[1]*size[2]])], axis=0)
    grid = tf.reshape(tf.tile(tf.reshape(grid, [-1]), [num_batch]), [num_batch, 4, -1])
    grid_warped = tf.matmul(theta, grid)
    return tf.reshape(tf.transpose(grid_warped, [0, 2, 1]), [num_batch, size[0], size[1], size[2], 3])

#sample_coords:=batchsize*64*64*60*3
def resample_linear(inputs, sample_coords):

    input_size = inputs.get_shape().as_list()[1:-1]#input图像的大小
    spatial_rank = inputs.get_shape().ndims - 2
    xy = tf.unstack(sample_coords, axis=len(sample_coords.get_shape())-1)#grid mesh进行unstack
    index_voxel_coords = [tf.floor(x) for x in xy]#index_voxel_coodrs[0] grid mesh的所有x坐标值，index_voxel_coodrs[1]grid mesh的所有y坐标值

    def boundary_replicate(sample_coords0, input_size0):
        return tf.maximum(tf.minimum(sample_coords0, input_size0 - 1), 0)
    #裁剪无效的空间，对于z方向上，最大值是60，但对于warping过的数据，有可能超过范围
    spatial_coords = [boundary_replicate(tf.cast(x, tf.int32), input_size[idx])
                      for idx, x in enumerate(index_voxel_coords)]
    spatial_coords_plus1 = [boundary_replicate(tf.cast(x+1., tf.int32), input_size[idx])
                            for idx, x in enumerate(index_voxel_coords)]

    weight = [tf.expand_dims(x - tf.cast(i, tf.float32), -1) for x, i in zip(xy, spatial_coords)]
    weight_c = [tf.expand_dims(tf.cast(i, tf.float32) - x, -1) for x, i in zip(xy, spatial_coords_plus1)]

    sz = spatial_coords[0].get_shape().as_list()
    batch_coords = tf.tile(tf.reshape(tf.range(sz[0]), [sz[0]] + [1] * (len(sz) - 1)), [1] + sz[1:])
    sc = (spatial_coords, spatial_coords_plus1)
    binary_codes = [[int(c) for c in format(i, '0%ib' % spatial_rank)] for i in range(2**spatial_rank)]

    make_sample = lambda bc: tf.gather_nd(inputs, tf.stack([batch_coords] + [sc[c][i] for i, c in enumerate(bc)], -1))
    samples = [make_sample(bc) for bc in binary_codes]

    def pyramid_combination(samples0, weight0, weight_c0):
        if len(weight0) == 1:
            return samples0[0]*weight_c0[0]+samples0[1]*weight0[0]
        else:
            return pyramid_combination(samples0[::2], weight0[:-1], weight_c0[:-1]) * weight_c0[-1] + \
                   pyramid_combination(samples0[1::2], weight0[:-1], weight_c0[:-1]) * weight0[-1]

    return pyramid_combination(samples, weight, weight_c)


def get_reference_grid(grid_size):
    return tf.to_float(tf.stack(tf.meshgrid(
        [i for i in range(grid_size[0])],
        [j for j in range(grid_size[1])],
        [k for k in range(grid_size[2])],
        indexing='ij'), axis=3))

def augment_data_by_DDF(ph_MV_image, ph_MV_label, ph_random_ddf):
    ddf_list = tf.unstack(ph_random_ddf, axis=-1)
    ddf_list = [separable_filter3d(tf.expand_dims(ddf, axis=-1), gauss_kernel1d(20)) for ddf in ddf_list]
    mv_ddf = tf.concat(ddf_list, -1)
    warp_mv = batch_displacement_warp3d(ph_MV_image, mv_ddf)
    warp_real_mv_label = batch_displacement_warp3d(ph_MV_label, mv_ddf)
    warp_mv_label = warp_real_mv_label >= 0.5
    warp_mv_label=tf.cast(warp_mv_label,tf.float32)
    return warp_mv, warp_mv_label

def compute_binary_diff(input1,input2):
    mask1 = input1 >= 0.5
    mask2 = input2 >= 0.5
    return tf.bitwise.bitwise_xor(tf.to_int32(mask2),tf.to_int32(mask1))

def compute_binary_dice(input1, input2):
    mask1 = input1 >= 0.5
    mask2 = input2 >= 0.5
    vol1 = tf.reduce_sum(tf.to_float(mask1), axis=[1, 2, 3, 4])
    vol2 = tf.reduce_sum(tf.to_float(mask2), axis=[1, 2, 3, 4])
    dice = tf.reduce_sum(tf.to_float(mask1 & mask2), axis=[1, 2, 3, 4])*2 / (vol1+vol2)
    return dice




def compute_centroid_distance(input1, input2, grid=None):
    if grid is None:
        grid = get_reference_grid(input1.get_shape()[1:4])

    def compute_centroid(mask, grid0):
        return tf.stack([tf.reduce_mean(tf.boolean_mask(grid0, mask[i, ..., 0] >= 0.5), axis=0)
                         for i in range(mask.shape[0].value)], axis=0)
    c1 = compute_centroid(input1, grid)
    c2 = compute_centroid(input2, grid)
    return tf.sqrt(tf.reduce_sum(tf.square(c1-c2), axis=1))


def normalize_mask_image(atlas_image,atlas_label):
    mask_atlas_image = atlas_image * atlas_label
    mean = np.sum(mask_atlas_image) / np.sum(atlas_label)
    var = np.sum(((atlas_image - mean) * atlas_label) ** 2)/np.sum(atlas_label)
    normal = ((atlas_image - mean) / np.sqrt(var)) * atlas_label
    return normal