import numpy as np
import tensorflow as tf

from tf_help.utils import separable_filter3d, gauss_kernel1d


def build_loss(similarity_type, similarity_scales, regulariser_type, regulariser_weight,
               label_moving, label_fixed, network_type, ddf):
    label_similarity = multi_scale_loss(label_fixed, label_moving, similarity_type.lower(), similarity_scales)
    if network_type.lower() == 'global':
        ddf_regularisation = tf.constant(0.0)
    else:
        ddf_regularisation = tf.reduce_mean(local_displacement_energy(ddf, regulariser_type, regulariser_weight))
    return tf.reduce_mean(label_similarity), ddf_regularisation

def build_consistent_loss_zero(similarity_type, similarity_scales, regulariser_type, regulariser_weight,consistent_weight,
                          warp_MV_label, warp_FIX_label, network_type, ddf_MV_ref, ddf_FIX_ref):
    label_similarity = multi_scale_loss(warp_MV_label, warp_FIX_label, similarity_type.lower(), similarity_scales)
    ddf_regularisation_FIX = tf.reduce_mean(local_displacement_energy(ddf_FIX_ref, regulariser_type, regulariser_weight))
    ddf_regularisation_MV = tf.reduce_mean(local_displacement_energy(ddf_MV_ref, regulariser_type, regulariser_weight))
    consistent = tf.reduce_sum(tf.abs(tf.add(ddf_MV_ref, ddf_FIX_ref)), axis=4)
    consistent = tf.reduce_mean(consistent, [1, 2, 3])
    return tf.reduce_mean(label_similarity), ddf_regularisation_MV,ddf_regularisation_FIX,consistent

def build_consistent_loss_minimize(similarity_type, similarity_scales, regulariser_type, regulariser_weight,consistent_weight,
                          warp_MV_label, warp_FIX_label, network_type, ddf_MV_ref, ddf_FIX_ref):
    label_similarity = multi_scale_loss(warp_MV_label, warp_FIX_label, similarity_type.lower(), similarity_scales)
    ddf_regularisation_FIX = tf.reduce_mean(local_displacement_energy(ddf_FIX_ref, regulariser_type, regulariser_weight))
    ddf_regularisation_MV = tf.reduce_mean(local_displacement_energy(ddf_MV_ref, regulariser_type, regulariser_weight))
    consistent=tf.reduce_sum(tf.square(tf.add(ddf_MV_ref, ddf_FIX_ref)), axis=4)
    consistent=tf.reduce_mean(consistent,[1,2,3])
    return tf.reduce_mean(label_similarity), ddf_regularisation_MV,ddf_regularisation_FIX,consistent*consistent_weight

def build_consistent_loss_inverse(similarity_type, similarity_scales, regulariser_type, regulariser_weight, consistent_weight,
                                  warp_MV_label, warp_FIX_label, network_type, ddf_MV_FIX, ddf_FIX_MV):
    label_similarity = multi_scale_loss(warp_MV_label, warp_FIX_label, similarity_type.lower(), similarity_scales)
    ddf_regularisation_FIX = tf.reduce_mean(local_displacement_energy(ddf_FIX_MV, regulariser_type, regulariser_weight))
    ddf_regularisation_MV = tf.reduce_mean(local_displacement_energy(ddf_MV_FIX, regulariser_type, regulariser_weight))
    consistent=tf.reduce_sum(tf.square(tf.add(ddf_MV_FIX, ddf_FIX_MV)), axis=4)
    consistent=tf.reduce_mean(consistent,[1,2,3])
    return tf.reduce_mean(label_similarity), ddf_regularisation_MV,ddf_regularisation_FIX,consistent*consistent_weight

def weighted_binary_cross_entropy(ts, ps, pw=1, eps=1e-6):
    ps = tf.clip_by_value(ps, eps, 1-eps)
    return -tf.reduce_sum(
        tf.concat([ts*pw, 1-ts], axis=4)*tf.log(tf.concat([ps, 1-ps], axis=4)),
        axis=4, keep_dims=True)
def weighted_2Dbinary_cross_entropy(ts, ps, pw=1, eps=1e-6):
    ps = tf.clip_by_value(ps, eps, 1-eps)
    return -tf.reduce_sum(
        tf.concat([ts*pw, 1-ts], axis=3)*tf.log(tf.concat([ps, 1-ps], axis=3)),
        axis=3, keep_dims=True)

def dice_simple(ts, ps, eps_vol=1e-6):
    numerator = tf.reduce_sum(ts*ps, axis=[1, 2, 3, 4]) * 2
    denominator = tf.reduce_sum(ts, axis=[1, 2, 3, 4]) + tf.reduce_sum(ps, axis=[1, 2, 3, 4])+eps_vol
    return numerator/denominator


def dice_generalised(ts, ps, weights):
    ts2 = tf.concat([ts, 1-ts], axis=4)
    ps2 = tf.concat([ps, 1-ps], axis=4)
    numerator = 2 * tf.reduce_sum(tf.reduce_sum(ts2*ps2, axis=[1, 2, 3]) * weights, axis=1)
    denominator = tf.reduce_sum((tf.reduce_sum(ts2, axis=[1, 2, 3]) +
                                 tf.reduce_sum(ps2, axis=[1, 2, 3])) * weights, axis=1)
    return numerator/denominator


def jaccard_simple(ts, ps, eps_vol=1e-6):
    numerator = tf.reduce_sum(ts*ps, axis=[1, 2, 3, 4])
    denominator = tf.reduce_sum(tf.square(ts), axis=[1, 2, 3, 4]) + \
                  tf.reduce_sum(tf.square(ps), axis=[1, 2, 3, 4]) - numerator + eps_vol
    return numerator/denominator


def cauchy_kernel1d(sigma):  # this is an approximation
    if sigma == 0:
        return 0
    else:
        tail = int(sigma*5)
        # k = tf.reciprocal(([((x/sigma)**2+1)*sigma*3.141592653589793 for x in range(-tail, tail+1)]))
        k = tf.reciprocal([((x/sigma)**2+1) for x in range(-tail, tail + 1)])
        return k / tf.reduce_sum(k)

def single_scale_loss(label_fixed, label_moving, loss_type):
    if loss_type == 'cross-entropy':
        label_loss_batch = tf.reduce_mean(weighted_binary_cross_entropy(label_fixed, label_moving), axis=[1, 2, 3, 4])
    elif loss_type == 'mean-squared':
        label_loss_batch = tf.reduce_mean(tf.squared_difference(label_fixed, label_moving), axis=[1, 2, 3, 4])
    elif loss_type == 'dice':
        label_loss_batch = - dice_simple(label_fixed, label_moving)
    elif loss_type == 'jaccard':
        label_loss_batch = 1 - jaccard_simple(label_fixed, label_moving)
    else:
        raise Exception('Not recognised label correspondence loss!')
    return label_loss_batch


def multi_scale_loss(label_fixed, label_moving, loss_type, loss_scales):
    label_loss_all = tf.stack(
        [single_scale_loss(
            separable_filter3d(label_fixed, gauss_kernel1d(s)),
            separable_filter3d(label_moving, gauss_kernel1d(s)), loss_type)
            for s in loss_scales],
        axis=1)
    return tf.reduce_mean(label_loss_all, axis=1)

def restore_loss(fix,warp):
    return tf.reduce_mean(tf.abs(fix-warp))

def restore_loss2(fix,warp):
    return tf.reduce_mean((tf.square(fix-warp)))


EPS = 1.0e-6
def NCC(t, p, kernel_size=7):
    kernel_vol = kernel_size ** 3
    # [dim1, dim2, dim3, d_in, d_out]
    # ch must be evenly divisible by d_in
    filters = tf.ones(shape=[kernel_size, kernel_size, kernel_size, 1, 1])
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    # t = y_true, p = y_pred
    # (batch, dim1, dim2, dim3, ch)
    t2 = t * t
    p2 = p * p
    tp = t * p
    # sum over kernel
    # (batch, dim1, dim2, dim3, 1)
    t_sum = tf.nn.conv3d(t, filter=filters, strides=strides, padding=padding)
    p_sum = tf.nn.conv3d(p, filter=filters, strides=strides, padding=padding)
    t2_sum = tf.nn.conv3d(t2, filter=filters, strides=strides, padding=padding)
    p2_sum = tf.nn.conv3d(p2, filter=filters, strides=strides, padding=padding)
    tp_sum = tf.nn.conv3d(tp, filter=filters, strides=strides, padding=padding)
    # average over kernel
    # (batch, dim1, dim2, dim3, 1)
    t_avg = t_sum / kernel_vol
    p_avg = p_sum / kernel_vol

    # normalized cross correlation between t and p
    # sum[(t - mean[t]) * (p - mean[p])] / std[t] / std[p]
    # denoted by num / denom
    # assume we sum over N values
    # num = sum[t * p - mean[t] * p - t * mean[p] + mean[t] * mean[p]]
    #     = sum[t*p] - sum[t] * sum[p] / N * 2 + sum[t] * sum[p] / N
    #     = sum[t*p] - sum[t] * sum[p] / N
    #     = sum[t*p] - sum[t] * mean[p] = cross
    # the following is actually squared ncc
    # shape = (batch, dim1, dim2, dim3, 1)
    cross = tp_sum - p_avg * t_sum-t_avg*p_sum+p_avg*t_avg
    cross=tf.nn.conv3d(cross,filter=filters,strides=strides,padding=padding)
    t_var = t2_sum - 2*t_avg * t_sum+t_avg*t_avg  # std[t] ** 2
    t_var=tf.sqrt(tf.nn.conv3d(t_var,filter=filters,strides=strides,padding=padding))
    p_var = p2_sum - 2*p_avg * p_sum+p_avg*p_avg  # std[p] ** 2
    p_var=tf.sqrt(tf.nn.conv3d(p_var,filter=filters,strides=strides,padding=padding))
    ncc = (cross ) / (t_var * p_var + EPS)
    return tf.reduce_mean(ncc, axis=[1, 2, 3, 4])

def LNCC2(I, J, win=None):
    # get dimension of volume
    # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    ndims = len(I.get_shape().as_list()) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    if win is None:
        win = [5] * ndims

    # get convolution function
    conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

    # compute CC squares
    I2 = I * I
    J2 = J * J
    IJ = I * J

    # compute filters
    in_ch = J.get_shape().as_list()[-1]
    sum_filt = tf.ones(win+[ in_ch, 1])
    strides = 1
    if ndims > 1:
        strides = [1] * (ndims + 2)

    # compute local sums via convolution
    padding = 'SAME'
    I_sum = conv_fn(I, sum_filt, strides, padding)
    J_sum = conv_fn(J, sum_filt, strides, padding)
    I2_sum = conv_fn(I2, sum_filt, strides, padding)
    J2_sum = conv_fn(J2, sum_filt, strides, padding)
    IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

    # compute cross correlation
    win_size = np.prod(win) * in_ch
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size  # TODO: simplify this
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + EPS)
    # cc=tf.sqrt(cc)
    # cc= conv_fn(cc, sum_filt, strides, padding)
    return tf.reduce_mean(cc, axis=[1, 2, 3, 4])
    # return mean cc for each entry in batch
    # return tf.reduce_mean(.batch_flatten(cc), axis=-1)
def anti_folding(ddf):


    dIdx = ddf[:, 1:, :, :,0] - ddf[:, :-1, :, :,0]+1
    dIdy = ddf[:, :, 1:, :,1] - ddf[:, :, :-1, :,1]+1
    dIdz = ddf[:, :, :, 1:,2] - ddf[:, :, :, :-1,2]+1
    fold_loss_x=tf.nn.relu(-dIdx)*(dIdx*dIdx)
    fold_loss_y=tf.nn.relu(-dIdy)*(dIdy*dIdy)
    fold_loss_z=tf.nn.relu(-dIdz)*(dIdz*dIdz)
    anti_fold_loss=tf.reduce_mean(fold_loss_x,[0,1,2,3])+tf.reduce_mean(fold_loss_y,[0,1,2,3])+tf.reduce_mean(fold_loss_z,[0,1,2,3])

    return anti_fold_loss


def anti_folding2D(ddf):

    dIdx = ddf[:, 1:, :, 0] - ddf[:, :-1, :, 0]+1
    dIdy = ddf[:, :, 1:, 1] - ddf[:, :, :-1, 1]+1

    fold_loss_x=tf.nn.relu(-dIdx)*(dIdx*dIdx)
    fold_loss_y=tf.nn.relu(-dIdy)*(dIdy*dIdy)
    anti_fold_loss=tf.reduce_mean(fold_loss_x,[0,1,2])+tf.reduce_mean(fold_loss_y,[0,1,2])
    return anti_fold_loss

def local_displacement_energy(ddf, energy_type, energy_weight):

    def gradient_dx(fv): return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(fv): return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(fv): return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(Txyz, fn):
        channal=Txyz.shape[-1]
        return tf.stack([fn(Txyz[..., i]) for i in range(channal)], axis=4)

    def compute_gradient_norm(displacement, flag_l1=False):
        dTdx = gradient_txyz(displacement, gradient_dx)
        dTdy = gradient_txyz(displacement, gradient_dy)
        dTdz = gradient_txyz(displacement, gradient_dz)
        if flag_l1:
            norms = tf.abs(dTdx) + tf.abs(dTdy) + tf.abs(dTdz)
        else:
            norms = dTdx**2 + dTdy**2 + dTdz**2
        return tf.reduce_mean(norms, [1, 2, 3, 4])

    def compute_bending_energy(displacement):
        dTdx = gradient_txyz(displacement, gradient_dx)
        dTdy = gradient_txyz(displacement, gradient_dy)
        dTdz = gradient_txyz(displacement, gradient_dz)
        dTdxx = gradient_txyz(dTdx, gradient_dx)
        dTdyy = gradient_txyz(dTdy, gradient_dy)
        dTdzz = gradient_txyz(dTdz, gradient_dz)
        dTdxy = gradient_txyz(dTdx, gradient_dy)
        dTdyz = gradient_txyz(dTdy, gradient_dz)
        dTdxz = gradient_txyz(dTdx, gradient_dz)
        return tf.reduce_mean(dTdxx**2 + dTdyy**2 + dTdzz**2 + 2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2, [1, 2, 3, 4])

    if energy_weight:
        if energy_type == 'bending':
            energy = compute_bending_energy(ddf)
        elif energy_type == 'gradient-l2':
            energy = compute_gradient_norm(ddf)
        elif energy_type == 'gradient-l1':
            energy = compute_gradient_norm(ddf, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')
    else:
        energy = tf.constant(0.0)

    return energy*energy_weight

def local_displacement_energy2DV2(ddf,energy_type):
    dxdx=ddf[:,2:,:]+ddf[:,:-2,:]-2*ddf[:,1:-1,:]
    dydy=ddf[:,:,2:]+ddf[:,:,:-2]-2*ddf[:,:,1:-1]
    return tf.reduce_mean(tf.reduce_mean(dxdx ** 2, [1, 2, 3]) + tf.reduce_mean(dydy ** 2, [1, 2, 3]))

def local_displacement_energy2D(ddf, energy_type):



    def gradient_dx(fv): return (fv[:, 1:, :] - fv[:, :-1, :])

    def gradient_dy(fv): return (fv[:, :, 1:] - fv[:, :, :-1])

    def gradient_txy(Txy, fn):
        channal=Txy.shape[-1]
        return tf.stack([fn(Txy[..., i]) for i in range(channal)], axis=3)

    def compute_gradient_norm(displacement, flag_l1=False):
        dTdx = gradient_txy(displacement, gradient_dx)
        dTdy = gradient_txy(displacement, gradient_dy)

        if flag_l1:
            norms = tf.abs(dTdx) + tf.abs(dTdy)
        else:
            norms = dTdx**2 + dTdy**2
        return tf.reduce_mean(norms, [1, 2, 3])

    def compute_bending_energy(displacement):
        dTdx = gradient_txy(displacement, gradient_dx)
        dTdy = gradient_txy(displacement, gradient_dy)
        dTdxx = gradient_txy(dTdx, gradient_dx)
        dTdyy = gradient_txy(dTdy, gradient_dy)
        dTdxy = gradient_txy(dTdx, gradient_dy)
        return tf.reduce_mean(dTdxx**2, [1, 2, 3]) + tf.reduce_mean(dTdyy**2  , [1, 2, 3])


    if energy_type == 'bending':
        energy = compute_bending_energy(ddf)
    elif energy_type == 'gradient-l2':
        energy = compute_gradient_norm(ddf)
    elif energy_type == 'gradient-l1':
        energy = compute_gradient_norm(ddf, flag_l1=True)
    else:
        raise Exception('Not recognised local regulariser!')


    return tf.reduce_mean(energy)

# def ncc(x, y):
#     """Normalized Cross Correlation (NCC) between two images."""
#     return np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std())
def global_ncc(x,y):
    axis=[1,2,3,4]
    mean_x=tf.reduce_mean(x,axis=axis)
    mean_y=tf.reduce_mean(y,axis=axis)

    a=(x-mean_x)*(y-mean_y)
    b=tf.sqrt(tf.reduce_mean(tf.square(x-mean_x),axis=axis)*tf.reduce_mean(tf.square(y-mean_y),axis=axis))
    return tf.reduce_mean((a+EPS)/(b+EPS),axis=axis)

def ncc_3d(x, y):
    mean_x = tf.reduce_mean(x, [1, 2, 3, 4], keepdims=True)
    mean_y = tf.reduce_mean(y, [1, 2, 3, 4], keepdims=True)
    mean_x2 = tf.reduce_mean(tf.square(x), [1, 2, 3, 4], keepdims=True)
    mean_y2 = tf.reduce_mean(tf.square(y), [1, 2, 3, 4], keepdims=True)
    stddev_x = tf.reduce_sum(tf.sqrt(mean_x2 - tf.square(mean_x)), [1, 2, 3, 4], keepdims=True)
    stddev_y = tf.reduce_sum(tf.sqrt(mean_y2 - tf.square(mean_y)), [1, 2, 3, 4], keepdims=True)
    return tf.reduce_mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))


if __name__=='__main__':
    import numpy as np
    import tensorflow as tf
    ddf=np.zeros((9,9),dtype=np.float32)
    for i in range(9):
        for j in range(9):
            if i %2==0:
                ddf[i,j]= np.random.random() * 8
            else:
                ddf[i, j] = np.random.random()
    print(ddf)
    ddf=np.expand_dims(np.expand_dims(ddf,axis=-1),axis=0)
    a=tf.convert_to_tensor(ddf)
    gradient=local_displacement_energy2D(a, 'bending')
    with tf.Session() as sess:
        print(sess.run(gradient))