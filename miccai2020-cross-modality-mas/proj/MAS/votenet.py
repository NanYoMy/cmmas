
import tensorflow as tf
import MAS.layers as layer
import MAS.utils as util
import MAS.losses as loss


class VoteNet():
    def __init__(self, config=None):

        self.config = config

    def build_network(self,warp_atlas_image,target_image,is_training):

        self.image_size = warp_atlas_image.shape.as_list()[1:4]
        self.image_target = target_image
        self.image_warp_atlas = warp_atlas_image
        self.input_layer = tf.concat([layer.resize_volume(target_image, self.image_size), warp_atlas_image], axis=4)
        #
        # self.input_layer=self.image_target-self.image_warp_atlas
        self.is_train=is_training
        # defaults
        self.num_channel_initial = 32
        #32,64,128,256,512
        nc = [int(self.num_channel_initial*(2**i)) for i in range(5)]
        h0, hc0 = layer.downsample_resnet_block(self.is_train,self.input_layer, 2, nc[0], k_conv0=[7, 7, 7], name='local_down_0')
        h1, hc1 = layer.downsample_resnet_block(self.is_train,h0, nc[0], nc[1], name='local_down_1')
        h2, hc2 = layer.downsample_resnet_block(self.is_train,h1, nc[1], nc[2], name='local_down_2')
        h3, hc3 = layer.downsample_resnet_block(self.is_train,h2, nc[2], nc[3], name='local_down_3')
        #这个代码是对应文章中 fig.4 中的哪个卷积块？
        hm = layer.conv3_block(self.is_train,h3, nc[3], nc[4], name='local_deep_4')
        hv1 = layer.upsample_resnet_block(self.is_train,hm, hc3, nc[4], nc[3], name='local_up_3')
        hv2= layer.upsample_resnet_block(self.is_train,hv1, hc2, nc[3], nc[2], name='local_up_2')
        hv3= layer.upsample_resnet_block(self.is_train,hv2, hc1, nc[2], nc[1], name='local_up_1')
        hv4= layer.upsample_resnet_block(self.is_train,hv3, hc0, nc[1], nc[0], name='local_up_0')
        ho1=layer.conv3_block(self.is_train, hv4, nc[0], nc[0] / 2, name='out_conv_1')
        self.out=layer.conv3_output_layer(self.is_train, ho1, nc[0] / 2, 1, name='out_conv_2', need_BN=False)
        # 输出0/1


    def __get_cost(self,Y_gt,Y_pred ):
        smooth = 1e-7
        Z, H, W, C = Y_gt.get_shape().as_list()[1:]
        pred_flat = tf.reshape(Y_pred, [-1, H * W * Z])
        gt_flat = tf.reshape(Y_gt, [-1, H * W * Z])
        intersection = tf.reduce_sum(pred_flat * gt_flat, axis=1)
        denominator = tf.reduce_sum(gt_flat, axis=1) + tf.reduce_sum(pred_flat, axis=1)
        loss = (2.0 * intersection + smooth) / (denominator + smooth)
        return 1-tf.reduce_mean(loss)

    def build_loss(self,ph_atlas_label,ph_target_label ,ph_gt_mask):
        # self.loss=self.__get_cost(gt_mask,self.out)
        # self.gt_mask = tf.multiply(ph_atlas_label, ph_target_label)
        # self.out=tf.multiply(ph_atlas_label,self.out)
        # self.loss = tf.reduce_mean(loss.multi_scale_loss(self.gt_mask, self.out,'cross-entropy',[0]))

        # self.out=tf.multiply(ph_atlas_label,self.out)
        # self.out=tf.multiply(ph_atlas_label,self.out)
        # self.gt_intersect=tf.multiply(ph_atlas_label,self.gt_intersect)

        self.gt_intersect = tf.multiply(ph_atlas_label, ph_target_label)
        # self.gt_intersect=ph_gt_mask
        # self.out=tf.multiply(ph_atlas_label,self.out)
        #处理权重
        self.loss = tf.reduce_mean(loss.multi_scale_loss(self.gt_intersect, self.out, "dice", [0,1]))
        self.train_op = tf.train.AdamOptimizer(self.config['Train']['learning_rate']).minimize(self.loss)

    def train(self,target_image,warped_atlas_image,gt_mask):
        pass

    def predict(self,target_image,warped_atlas_image):
        pass
