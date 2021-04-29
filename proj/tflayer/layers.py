

import tensorflow as tf


def batch_norm(x, momentum=0.9, epsilon=1e-5, train=True, name="batch_norm"):
    # return batch_instance_norm(x, name)
    return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon, scale=True,
                                        is_training=train, scope=name)



def conv_bn_relu_3D(x, out_channel, ks=3 ,train=True, name='conv'):
    y=tf.layers.conv3d(x,out_channel,ks,padding='same',name=name+'_c')
    y=batch_norm(y,train=train,name=name+"_bn")
    y=tf.nn.relu(y)
    return y

def conv_bn_relue_pool_3D(x, out_channel, ks=3, s=2 ,train=True, name='conv'):
    y=conv_bn_relu_3D(x, out_channel, ks, train, name)
    y=tf.layers.max_pooling3d(y,s,s)
    return y

def deconv_bn_relu_3D(x, dim, ks=3, s=1, is_train=True, name='de'):
    x=tf.layers.conv3d_transpose(x, dim,ks,s ,padding='same', use_bias=False,name=name + '_c')
    x = batch_norm(x, train=is_train, name=name + '_bn')
    x = tf.nn.relu(x)
    return x