import tensorflow as tf
import MAS.layers as layer
import MAS.utils as util
import MAS.losses as loss
from spatial_transformer_network.displacement import  batch_displacement_warp3d_fix_size,batch_displacement_warp3d
def build_network(network_type, **kwargs):
    type_lower = network_type.lower()

    if type_lower=='bend':
        return ConsistentLocalNet(**kwargs,lambda_consis=0.0,lambda_ben=0.3)
    elif type_lower=='consistent':
        return ConsistentLocalNet(**kwargs, lambda_consis=0.3, lambda_ben=0.0)
    elif type_lower=='one_ddf':
        return OneDDF(**kwargs,  lambda_ben=0.2)
    elif type_lower=='mix':
        return ConsistentLocalNet(**kwargs, lambda_consis=0.3, lambda_ben=0.2)
    else:
        print("error no network tyep exist")
        exit(997)
    # return GlobalNet(**kwargs)
    # if type_lower=="inverse":
    #     return ConsistentLocalNet(**kwargs)
    # elif type_lower=="global":
    #     return GlobalNet(**kwargs)
    # else:
    #     print("the network type is invalid")
def build_global_network(network_type, **kwargs):
    return GlobalNet(**kwargs)

class BaseNet:

    def __init__(self, minibatch_size, MV_image, FIX_image,ph_is_training):
        self.minibatch_size = minibatch_size
        self.image_size = MV_image.shape.as_list()[1:4]
        self.grid_ref = util.get_reference_grid(self.image_size)
        self.grid_warped = tf.zeros_like(self.grid_ref)
        self.image_FIX = FIX_image
        self.image_MV = MV_image
        self.input_layer = tf.concat([layer.resize_volume(FIX_image, self.image_size), MV_image], axis=4)
        self.is_train=ph_is_training


    def build_loss(self, config, warp_MV_label, warp_FIX_label, ddf_MV_ref, ddf_FIX_ref):
        pass

class ConsistentLocalNet(BaseNet):
    def warp_MV_image(self, input_):
        return util.resample_linear(input_, self.grid_warped_MV_FIX)
    def warp_FIX_image(self, input_):
        return util.resample_linear(input_, self.grid_warped_FIX_MV)

    def __init__(self, lambda_ben=0.0,lambda_consis=0.3,ddf_levels=None, **kwargs):
        BaseNet.__init__(self, **kwargs,)
        # defaults
        self.lambda_bend=lambda_ben
        self.lambda_consis=lambda_consis
        self.grid_warped_MV_FIX = tf.zeros_like(self.grid_ref)  # initial zeros are safer for debug
        self.grid_warped_FIX_MV = tf.zeros_like(self.grid_ref)  # initial zeros are safer for debug

        self.ddf_levels = [0, 1, 2, 3, 4] if ddf_levels is None else ddf_levels
        self.num_channel_initial = 32
        #32,64,128,256,512
        nc = [int(self.num_channel_initial*(2**i)) for i in range(5)]
        h0, hc0 = layer.downsample_resnet_block(self.is_train,self.input_layer, 2, nc[0], k_conv0=[7, 7, 7], name='local_down_0')
        h1, hc1 = layer.downsample_resnet_block(self.is_train,h0, nc[0], nc[1], name='local_down_1')
        h2, hc2 = layer.downsample_resnet_block(self.is_train,h1, nc[1], nc[2], name='local_down_2')
        h3, hc3 = layer.downsample_resnet_block(self.is_train,h2, nc[2], nc[3], name='local_down_3')
        #这个代码是对应文章中 fig.4 中的哪个卷积块？
        hm = [layer.conv3_block(self.is_train,h3, nc[3], nc[4], name='local_deep_4')]
        min_level = min(self.ddf_levels)
        hm += [layer.upsample_resnet_block(self.is_train,hm[0], hc3, nc[4], nc[3], name='local_up_3')] if min_level < 4 else []
        hm += [layer.upsample_resnet_block(self.is_train,hm[1], hc2, nc[3], nc[2], name='local_up_2')] if min_level < 3 else []
        hm += [layer.upsample_resnet_block(self.is_train,hm[2], hc1, nc[2], nc[1], name='local_up_1')] if min_level < 2 else []
        hm += [layer.upsample_resnet_block(self.is_train,hm[3], hc0, nc[1], nc[0], name='local_up_0')] if min_level < 1 else []
        ddf_list=[layer.ddf_summand(hm[4-idx], nc[idx], self.image_size, name='ddf1_sum_%d' % idx)for idx in self.ddf_levels]
        ddf_list=tf.stack(ddf_list,axis=5)
        self.ddf_MV_FIX = tf.reduce_sum(ddf_list, axis=5)

        ddf_list2=[layer.ddf_summand(hm[4-idx], nc[idx], self.image_size, name='ddf2_sum_%d' % idx)for idx in self.ddf_levels]
        ddf_list2=tf.stack(ddf_list2,axis=5)
        self.ddf_FIX_MV = tf.reduce_sum(ddf_list2, axis=5)

        #好像bn层没有效果
        # self.ddf_MV_FIX=tf.layers.conv3d(self.ddf, 3,3, padding="same", name="ddf_W_MV_FIX")
        # self.ddf_FIX_MV = tf.layers.conv3d(self.ddf, 3, 3, padding="same", name="ddf_W_FIX_MV")
        #self.ddf = tf.reduce_sum(ddf_feature, axis=5)
        self.grid_warped_MV_FIX= self.grid_ref + self.ddf_MV_FIX
        self.grid_warped_FIX_MV= self.grid_ref + self.ddf_FIX_MV
    def build_loss(self, config, MV_label, FIX_label, ):
        self.warped_MV_label = self.warp_MV_image(MV_label)  # warp the moving label with the predicted ddf
        self.warped_warped_MV_label = self.warp_FIX_image(self.warped_MV_label)
        self.warped_FIX_label = self.warp_FIX_image(FIX_label)
        self.warped_warped_FIX_label = self.warp_MV_image(self.warped_FIX_label)

        self.label_similarity_warp_mv_fix = tf.reduce_mean(loss.multi_scale_loss(FIX_label, self.warped_MV_label, config['Loss']['similarity_type'].lower(),
                                                 config['Loss']['similarity_scales']))

        self.label_similarity_warp_fix_mv = tf.reduce_mean(loss.multi_scale_loss(MV_label, self.warped_FIX_label, config['Loss']['similarity_type'].lower(),
                                                 config['Loss']['similarity_scales']))

        self.label_similarity_warp_warp_fix_fix = tf.reduce_mean(loss.multi_scale_loss(FIX_label, self.warped_warped_FIX_label,
                                                   config['Loss']['similarity_type'].lower(),
                                                   config['Loss']['similarity_scales']))

        self.label_similarity_warp_warp_mv_mv = tf.reduce_mean(loss.multi_scale_loss(MV_label, self.warped_warped_MV_label,
                                                   config['Loss']['similarity_type'].lower(),
                                                   config['Loss']['similarity_scales']))
        # self.label_similarity_warp_warp_mv_mv=tf.constant(0.0)
        # self.label_similarity_warp_warp_fix_fix=tf.constant(0.0)
        # self.ICE1 = batch_displacement_warp3d_fix_size(self.ddf_MV_FIX,self.ddf_FIX_MV)+self.ddf_FIX_MV
        # self.ICE2 = batch_displacement_warp3d_fix_size(self.ddf_FIX_MV, self.ddf_MV_FIX)+self.ddf_MV_FIX

        # self.consistent_loss=tf.reduce_mean(tf.reduce_mean(tf.abs(self.ICE1),axis=4),axis=[0,1,2,3])+tf.reduce_mean(tf.reduce_mean(tf.abs(self.ICE2),axis=4),axis=[0,1,2,3])
        # self.consistent_loss=tf.reduce_mean((tf.reduce_mean(tf.abs(self.ICE1)+tf.abs(self.ICE2),axis=4),[1,2,3]))/2
        # self.consistent_loss=tf.square(tf.add(self.label_similarity_warp_mv_fix,-self.label_similarity_warp_fix_mv))
        self.consistent_loss=tf.constant(0.0)

        self.anti_folding_loss= loss.anti_folding(self.ddf_FIX_MV) + loss.anti_folding(self.ddf_MV_FIX)



        self.ddf_regularisation_FIX = tf.reduce_mean(
            loss.local_displacement_energy(self.ddf_FIX_MV, config['Loss']['regulariser_type'], 1.0))
        self.ddf_regularisation_MV = tf.reduce_mean(
            loss.local_displacement_energy(self.ddf_MV_FIX, config['Loss']['regulariser_type'], 1.0))

        self.train_op = tf.train.AdamOptimizer(config['Train']['learning_rate']).minimize(
            1*self.label_similarity_warp_mv_fix+
            1*self.label_similarity_warp_fix_mv+
            self.lambda_consis*self.label_similarity_warp_warp_fix_fix+
            self.lambda_consis*self.label_similarity_warp_warp_mv_mv+
            self.lambda_bend*self.ddf_regularisation_FIX +
            self.lambda_bend*self.ddf_regularisation_MV)
            # 0*self.consistent_loss+
            # 0*self.anti_folding_loss



class OneDDF(BaseNet):
    def warp_MV_image(self, input_):
        return util.resample_linear(input_, self.grid_warped_MV_FIX)
    def warp_FIX_image(self, input_):
        return util.resample_linear(input_, self.grid_warped_FIX_MV)

    def __init__(self, lambda_ben=0.0,lambda_consis=0.3,ddf_levels=None, **kwargs):
        BaseNet.__init__(self, **kwargs,)
        # defaults
        self.lambda_bend=lambda_ben
        self.lambda_consis=lambda_consis
        self.grid_warped_MV_FIX = tf.zeros_like(self.grid_ref)  # initial zeros are safer for debug
        self.grid_warped_FIX_MV = tf.zeros_like(self.grid_ref)  # initial zeros are safer for debug

        self.ddf_levels = [0, 1, 2, 3, 4] if ddf_levels is None else ddf_levels
        self.num_channel_initial = 32
        #32,64,128,256,512
        nc = [int(self.num_channel_initial*(2**i)) for i in range(5)]
        h0, hc0 = layer.downsample_resnet_block(self.is_train,self.input_layer, 2, nc[0], k_conv0=[7, 7, 7], name='local_down_0')
        h1, hc1 = layer.downsample_resnet_block(self.is_train,h0, nc[0], nc[1], name='local_down_1')
        h2, hc2 = layer.downsample_resnet_block(self.is_train,h1, nc[1], nc[2], name='local_down_2')
        h3, hc3 = layer.downsample_resnet_block(self.is_train,h2, nc[2], nc[3], name='local_down_3')
        #这个代码是对应文章中 fig.4 中的哪个卷积块？
        hm = [layer.conv3_block(self.is_train,h3, nc[3], nc[4], name='local_deep_4')]
        min_level = min(self.ddf_levels)
        hm += [layer.upsample_resnet_block(self.is_train,hm[0], hc3, nc[4], nc[3], name='local_up_3')] if min_level < 4 else []
        hm += [layer.upsample_resnet_block(self.is_train,hm[1], hc2, nc[3], nc[2], name='local_up_2')] if min_level < 3 else []
        hm += [layer.upsample_resnet_block(self.is_train,hm[2], hc1, nc[2], nc[1], name='local_up_1')] if min_level < 2 else []
        hm += [layer.upsample_resnet_block(self.is_train,hm[3], hc0, nc[1], nc[0], name='local_up_0')] if min_level < 1 else []
        ddf_list=[layer.ddf_summand(hm[4-idx], nc[idx], self.image_size, name='ddf1_sum_%d' % idx)for idx in self.ddf_levels]
        ddf_list=tf.stack(ddf_list,axis=5)
        self.ddf_MV_FIX = tf.reduce_sum(ddf_list, axis=5)

        ddf_list2=[layer.ddf_summand(hm[4-idx], nc[idx], self.image_size, name='ddf2_sum_%d' % idx)for idx in self.ddf_levels]
        ddf_list2=tf.stack(ddf_list2,axis=5)
        self.ddf_FIX_MV = tf.reduce_sum(ddf_list2, axis=5)

        #好像bn层没有效果
        # self.ddf_MV_FIX=tf.layers.conv3d(self.ddf, 3,3, padding="same", name="ddf_W_MV_FIX")
        # self.ddf_FIX_MV = tf.layers.conv3d(self.ddf, 3, 3, padding="same", name="ddf_W_FIX_MV")
        #self.ddf = tf.reduce_sum(ddf_feature, axis=5)
        self.grid_warped_MV_FIX= self.grid_ref + self.ddf_MV_FIX
        self.grid_warped_FIX_MV= self.grid_ref + self.ddf_FIX_MV
    def build_loss(self, config, MV_label, FIX_label, ):
        self.warped_MV_label = self.warp_MV_image(MV_label)  # warp the moving label with the predicted ddf
        self.warped_warped_MV_label = self.warp_FIX_image(self.warped_MV_label)
        self.warped_FIX_label = self.warp_FIX_image(FIX_label)
        self.warped_warped_FIX_label = self.warp_MV_image(self.warped_FIX_label)

        self.label_similarity_warp_mv_fix = tf.reduce_mean(loss.multi_scale_loss(FIX_label, self.warped_MV_label, config['Loss']['similarity_type'].lower(),
                                                 config['Loss']['similarity_scales']))

        self.label_similarity_warp_fix_mv = tf.reduce_mean(loss.multi_scale_loss(MV_label, self.warped_FIX_label, config['Loss']['similarity_type'].lower(),
                                                 config['Loss']['similarity_scales']))

        self.label_similarity_warp_warp_fix_fix = tf.reduce_mean(loss.multi_scale_loss(FIX_label, self.warped_warped_FIX_label,
                                                   config['Loss']['similarity_type'].lower(),
                                                   config['Loss']['similarity_scales']))

        self.label_similarity_warp_warp_mv_mv = tf.reduce_mean(loss.multi_scale_loss(MV_label, self.warped_warped_MV_label,
                                                   config['Loss']['similarity_type'].lower(),
                                                   config['Loss']['similarity_scales']))
        # self.label_similarity_warp_warp_mv_mv=tf.constant(0.0)
        # self.label_similarity_warp_warp_fix_fix=tf.constant(0.0)
        # self.ICE1 = batch_displacement_warp3d_fix_size(self.ddf_MV_FIX,self.ddf_FIX_MV)+self.ddf_FIX_MV
        # self.ICE2 = batch_displacement_warp3d_fix_size(self.ddf_FIX_MV, self.ddf_MV_FIX)+self.ddf_MV_FIX

        # self.consistent_loss=tf.reduce_mean(tf.reduce_mean(tf.abs(self.ICE1),axis=4),axis=[0,1,2,3])+tf.reduce_mean(tf.reduce_mean(tf.abs(self.ICE2),axis=4),axis=[0,1,2,3])
        # self.consistent_loss=tf.reduce_mean((tf.reduce_mean(tf.abs(self.ICE1)+tf.abs(self.ICE2),axis=4),[1,2,3]))/2
        # self.consistent_loss=tf.square(tf.add(self.label_similarity_warp_mv_fix,-self.label_similarity_warp_fix_mv))
        self.consistent_loss=tf.constant(0.0)

        self.anti_folding_loss= loss.anti_folding(self.ddf_FIX_MV) + loss.anti_folding(self.ddf_MV_FIX)



        self.ddf_regularisation_FIX = tf.reduce_mean(
            loss.local_displacement_energy(self.ddf_FIX_MV, config['Loss']['regulariser_type'], 1.0))
        self.ddf_regularisation_MV = tf.reduce_mean(
            loss.local_displacement_energy(self.ddf_MV_FIX, config['Loss']['regulariser_type'], 1.0))

        self.train_op = tf.train.AdamOptimizer(config['Train']['learning_rate']).minimize(
            1*self.label_similarity_warp_mv_fix+
            # 1*self.label_similarity_warp_fix_mv+
            # self.lambda_consis*self.label_similarity_warp_warp_fix_fix+
            # self.lambda_consis*self.label_similarity_warp_warp_mv_mv+
            # self.lambda_bend*self.ddf_regularisation_FIX +
            self.lambda_bend*self.ddf_regularisation_MV)
            # 0*self.consistent_loss+
            # 0*self.anti_folding_loss


class GlobalNet(BaseNet):

    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        # defaults
        self.num_channel_initial_global = 8
        self.transform_initial = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]

        nc = [int(self.num_channel_initial_global * (2 ** i)) for i in range(5)]
        h0, hc0 = layer.downsample_resnet_block(self.is_train,self.input_layer, 2, nc[0], k_conv0=[7, 7, 7], name='global_down_0')
        h1, hc1 = layer.downsample_resnet_block(self.is_train,h0, nc[0], nc[1], name='global_down_1')
        h2, hc2 = layer.downsample_resnet_block(self.is_train,h1, nc[1], nc[2], name='global_down_2')
        h3, hc3 = layer.downsample_resnet_block(self.is_train,h2, nc[2], nc[3], name='global_down_3')
        h4 = layer.conv3_block(self.is_train,h3, nc[3], nc[4], name='global_deep_4')
        self.theta = layer.fully_connected(h4, 12, self.transform_initial, name='global_project_0')

        self.grid_warped = util.warp_grid(self.grid_ref, self.theta)
        #这个地方为啥又减去
        self.ddf = self.grid_warped - self.grid_ref

    def warp_image(self,image,ddf):
        return util.resample_linear(image, ddf)

    def build_loss(self, config, MV_label, FIX_label,):
        self.warped_MV_label = self.warp_image(MV_label,self.grid_warped)  # warp the moving label with the predicted ddf

        self.loss_similarity_warp_mv_fix = tf.reduce_mean(
            loss.multi_scale_loss(FIX_label, self.warped_MV_label, config['Loss']['similarity_type'].lower(),
                                  config['Loss']['similarity_scales']))
        # self.ddf_regularisation =tf.reduce_mean(
        #     loss.local_displacement_energy(self.grid_warped, config['Loss']['regulariser_type'], 1.0))

        self.train_op = tf.train.AdamOptimizer(config['Train']['learning_rate']).minimize(
            1 * self.loss_similarity_warp_mv_fix)

# class CompositeNet(BaseNet):
#
#     def __init__(self, **kwargs):
#         BaseNet.__init__(self, **kwargs)
#         # defaults
#         self.ddf_levels = [0]
#
#         global_net = GlobalNet(**kwargs)
#         local_net = Consistent_Subject_To_Zero(minibatch_size=self.minibatch_size,
#                                                image_moving=global_net.warp_image(None),
#                                                image_fixed=self.image_MV,
#                                                ddf_levels=self.ddf_levels)
#
#         self.grid_warped = global_net.grid_warped + local_net.ddf
#         self.ddf = self.grid_warped - self.grid_ref

