"""
This is a tutorial example.
"""
import tensorflow as tf
import MAS.utils as util


def tensor_of_warp_volumes_by_ddf(input_, ddf):
    grid_warped = util.get_reference_grid(ddf.shape[1:4]) + ddf#这种情况ddf会被转化为tf.constant https://stackoverflow.com/questions/49239136/what-happens-when-you-add-a-tensor-with-a-numpy-array
    warped = util.resample_linear(tf.cast(input_,dtype=tf.float32),grid_warped)
    return warped

def tensor_of_warp_volumes_by_fix_and_inverse_fix_ddf(input_, ph_FIX_ddf, ph_Inverse_FIX_ddf):
    warped=tensor_of_warp_volumes_by_ddf(input_,ph_FIX_ddf)
    return tensor_of_warp_volumes_by_ddf(warped,ph_Inverse_FIX_ddf)
    # ddf=tf.add(invers_fix_ddf,mv_ddf)
    # return tensor_of_warp_volumes_by_ddf(input_,ddf)
def tensor_of_compute_binary_dice(warped_labels, data_fix_label):
    dice = util.compute_binary_dice(tf.cast(warped_labels,dtype=tf.float32),tf.cast(data_fix_label, tf.float32))
    return  dice

def tensor_of_compute_centroid_distance(warped_labels, data_fix_label):
    dist=util.compute_centroid_distance(tf.cast(warped_labels,dtype=tf.float32),tf.cast(data_fix_label, tf.float32))
    return dist