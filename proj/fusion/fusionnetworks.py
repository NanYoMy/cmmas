import time
import os
from excelutil.output2excel import outpu2excel
import tensorflow as tf
from multiseqseg.dice_loss import soft_dice_loss
from evaluate.metric import print_mean_and_std
from medpy.metric import dc
import tf_help.layers as layer

from learn2reg.sampler import Sampler
from model.base_model import BaseModel
from sitkImageIO.itkdatawriter import sitk_write_images,sitk_write_labs,sitk_write_lab,sitk_write_image
import numpy as np
from logger.Logger import getLoggerV3
from evaluate.metric import calculate_binary_dice,neg_jac,calculate_binary_hd
from dirutil.helper import mk_or_cleardir,get_name_wo_suffix
from config.Defines import Get_Name_By_Index
from learn2reg.challenge_sampler import CHallengeSampler
from dirutil.helper import get_name_wo_suffix
import SimpleITK as sitk

from fusion.sampler_network import FusionNetSampler
from tflayer.layers import conv_bn_relue_pool_3D, deconv_bn_relu_3D
'''
单纯用标签的方法
'''
class LabelFusionNet(BaseModel):
    def __init__(self,sess,args):
        self.sess=sess
        self.args=args
        self.train_sampler=FusionNetSampler(args, True)
        self.valid_sampler=FusionNetSampler(args, False)

        if args.phase.find('train')>=0:
            self.is_train=True
        else:
            self.is_train=False
        self.build_network()


    def build_network(self):
        with tf.variable_scope(self.__class__.__name__):
            self.ph_atlas=tf.placeholder(dtype=tf.float32, shape=[None, 96, 96, 96, 1], name='atlas')
            self.ph_gt=tf.placeholder(dtype=tf.float32,shape=[None,96,96,96,2],name='weighted_atlas')
            dim = [8, 16, 32,64]
            x=self.ph_atlas
            for d in dim:
                x=conv_bn_relue_pool_3D(x,d,3,2,self.is_train,'conv_'+str(d))
            x = tf.layers.flatten(x)
            code = tf.layers.dense(x, self.args.latent)
            x = tf.layers.dense(code, (96//2**len(dim)) * (96//2**len(dim)) *(96//2**len(dim))*dim[-1] )
            x = tf.reshape(x, [-1, (96//2**len(dim)), (96//2**len(dim)),(96//2**len(dim)), dim[-1]])
            dim.reverse()
            for d in dim[1:]:
                x=deconv_bn_relu_3D(x,d,3,2,self.is_train,name='deconv'+str(d))
            # x = deconv_bn_relu_3D(x, self.args.input_channel,3,2,self.is_train, name='conv_out')
            x = deconv_bn_relu_3D(x, 4,3,2,self.is_train, name='conv_out')
            self.output = tf.layers.conv3d(x, 2, 3, padding='SAME', name='conv_out')
            self.saver=tf.train.Saver(max_to_keep=10)

    def build_cost(self):
        # self.g_loss = tf.reduce_mean(soft_dice_loss(self.binary_shape, self.output, axis=[1, 2,3]))
        ####logits是什么？ https://stackoverflow.com/questions/40871797/tensorflow-softmax-cross-entropy-with-logits-asks-for-unscaled-log-probabilities
        ###https://stats.stackexchange.com/questions/52825/what-does-the-logit-value-actually-mean
        self.g_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ph_gt, logits=self.output ))

    def meger_all(self):
        tf.summary.scalar('loss',self.g_loss)
        tf.summary.image('gt',tf.cast(tf.expand_dims(self.ph_gt[:,:,:,48,1],axis=-1)*255,tf.uint8))
        tf.summary.image('output_0',tf.cast(tf.expand_dims(tf.argmax(self.output,axis=-1)[:,:,:,48],axis=-1)*255,tf.uint8))
        self.summary=tf.summary.merge_all()

    def train(self):
        self.build_cost()
        self.meger_all()
        g_optim = tf.train.AdamOptimizer(self.args.lr, beta1=self.args.beta1).minimize(self.g_loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)
        start_time = time.time()
        print(tf.trainable_variables())
        if self.args.continue_train and self.load(self.args.checkpoint_dir):
            print("An existing model was found in the checkpoint directory.")
        else:
            print("An existing model was NOT found in the checkpoint directory. Initializing a new one.")
        for itr in range(self.args.iteration):
            target_img,target_lab,atlas_imgs,atlas_labs=self.train_sampler.next_sample()
            feed_dict={self.ph_atlas:atlas_labs,self.ph_gt:target_lab}
            summary,_=self.sess.run([self.summary,g_optim],feed_dict=feed_dict)
            self.writer.add_summary(summary,itr)
            if np.mod(itr, self.args.print_freq) == 1:
                self.sample_network(itr)
            if np.mod(itr, self.args.save_freq) == 1:
                self.save(self.args.checkpoint_dir, itr)

    def validate(self):
        self.build_cost()
        self.meger_all()
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        dices= []
        for i in range(self.valid_sampler.len):
            dices.append(self.sample_network(i))
        print_mean_and_std(dices)

    def sample_network(self,itr):
        # p_target_img, p_target_lab, p_atlas_imgses, p_atlas_labses=self.valid_sampler.get_file()
        # target_img, target_lab, atlas_imgses, atlas_labses=self.valid_sampler.get_data(p_target_img, p_target_lab, p_atlas_imgses, p_atlas_labses)
        target_img, target_lab, atlas_imgses, atlas_labses=self.valid_sampler.next_sample()
        feed_dict = {self.ph_atlas: atlas_labses, self.ph_gt: target_lab}
        summary,out,gt = self.sess.run([self.summary,self.output,self.ph_gt], feed_dict=feed_dict)
        out=np.argmax(out,axis=-1)
        sitk_write_lab(out[0,...],dir=self.args.sample_dir,name=str(itr)+"pred")
        gt=np.argmax(gt, axis=-1)
        sitk_write_lab(gt[0,...],dir=self.args.sample_dir,name=str(itr)+"gt")
        sitk_write_image(np.squeeze(target_img[0,...]),dir=self.args.sample_dir,name=str(itr)+"img")
        dice=dc(out[0,...],gt[0,...])
        print("dc:%f"%(dice))
        return dice
'''
用概率标签和一张target图像的方法
'''
class FusionNet(BaseModel):

    def __init__(self,sess,args):
        BaseModel.__init__(self, sess,args)
        # self.sess=sess
        self.args=args
        if self.args.phase.find('train')>=0:
            self.is_train=True
        else:
            self.is_train=False
        self.image_size = [self.args.image_size,self.args.image_size,self.args.image_size]
        self.train_sampler = FusionNetSampler(self.args, True)
        self.validate_sampler = FusionNetSampler(self.args, False)
        self.build_network()
        self.summary()

    def build_network(self):

        self.ph_target_image = tf.placeholder(tf.float32, [self.args.batch_size] + self.image_size + [1])
        self.ph_warp_label = tf.placeholder(tf.float32,[self.args.batch_size] + self.image_size+ [1])
        self.ph_gt_label= tf.placeholder(tf.float32,[self.args.batch_size] + self.image_size+ [2])

        # self.ph_moving_affine = tf.placeholder(tf.float32, [self.args.batch_size] + [1,12])  # 数据进行augment,4x4矩阵，但是最后四个参数为0001，所以一共12个参数
        # self.ph_fixed_affine = tf.placeholder(tf.float32, [self.args.batch_size] + [1,12])
        # self.ph_random_ddf=tf.placeholder(tf.float32,[self.args.batch_size] + self.image_size+ [3])
        #
        # self.ph_MV_label = tf.placeholder(tf.float32,[self.args.batch_size] + self.image_size + [1])
        # self.ph_FIX_label = tf.placeholder(tf.float32,[self.args.batch_size ]+ self.image_size + [1])

        #data augmentation
        # self.input_MV_label = util.warp_image_affine(self.ph_MV_label, self.ph_moving_affine)  # data augmentation
        # self.input_MV_image = util.warp_image_affine(self.ph_MV_image, self.ph_moving_affine)  # data augmentation
        # self.input_FIX_label = util.warp_image_affine(self.ph_FIX_label, self.ph_fixed_affine)  # data augmentation
        # self.input_FIX_image = util.warp_image_affine(self.ph_FIX_image, self.ph_fixed_affine)  # data augmentation
        # self.input_MV_image,self.input_MV_label=util.augment_3Ddata_by_affine(self.ph_MV_image,self.ph_MV_label,self.ph_moving_affine)
        # self.input_FIX_image,self.input_FIX_label=util.augment_3Ddata_by_affine(self.ph_FIX_image,self.ph_FIX_label,self.ph_fixed_affine)
        # self.input_FIX_image,self.input_FIX_label=util.augment_3Ddata_by_DDF(self.ph_FIX_image,self.ph_FIX_label,self.ph_random_ddf)

        self.input_layer = tf.concat([self.ph_target_image, self.ph_warp_label], axis=-1)

        self.ddf_levels = [0, 1, 2, 3, 4]
        self.num_channel_initial = self.args.num_channel_initial
        # 32,64,128,256,512
        nc = [int(self.num_channel_initial * (2 ** i)) for i in range(5)]
        h0, hc0 = layer.downsample_resnet_block(self.is_train, self.input_layer, 2, nc[0], k_conv0=[7, 7, 7],name='local_down_0')
        h1, hc1 = layer.downsample_resnet_block(self.is_train, h0, nc[0], nc[1], name='local_down_1')
        h2, hc2 = layer.downsample_resnet_block(self.is_train, h1, nc[1], nc[2], name='local_down_2')
        h3, hc3 = layer.downsample_resnet_block(self.is_train, h2, nc[2], nc[3], name='local_down_3')

        hm=layer.conv3_block(self.is_train, h3, nc[3], nc[4], name='local_deep_4')
        hm = layer.upsample_resnet_block(self.is_train, hm, hc3, nc[4], nc[3], name='local_up_3')
        hm = layer.upsample_resnet_block(self.is_train, hm, hc2, nc[3], nc[2], name='local_up_2')
        hm = layer.upsample_resnet_block(self.is_train, hm, hc1, nc[2], nc[1], name='local_up_1')
        hm = layer.upsample_resnet_block(self.is_train, hm, hc0, nc[1], nc[0], name='local_up_0')
        self.out=tf.layers.conv3d(hm,2,3,padding='SAME',name='out_conv')

        self.prob=tf.nn.softmax(self.out)
        self.predit=tf.argmax(self.out,axis=-1)
        self.ce_loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.out, labels=self.ph_gt_label)
        # self.dice_loss= tf.reduce_mean(loss.multi_scale_loss(self.ph_gt_label, self., 'dice',[0, 1, 2, 4]))
        self.dice_loss=tf.reduce_mean(soft_dice_loss(self.ph_gt_label, self.prob, axis=[1, 2,3]))
        self.ce_loss=tf.reduce_mean(self.ce_loss)#奇怪为啥不用这个也能工作
        self.train_op = tf.train.AdamOptimizer(self.args.lr).minimize(10*self.ce_loss+self.dice_loss)


        # hm = [layer.conv3_block(self.is_train, h3, nc[3], nc[4], name='local_deep_4')]
        # min_level = min(self.ddf_levels)
        # hm += [layer.upsample_resnet_block(self.is_train, hm[0], hc3, nc[4], nc[3], name='local_up_3')] if min_level < 4 else []
        # hm += [layer.upsample_resnet_block(self.is_train, hm[1], hc2, nc[3], nc[2], name='local_up_2')] if min_level < 3 else []
        # hm += [layer.upsample_resnet_block(self.is_train, hm[2], hc1, nc[2], nc[1], name='local_up_1')] if min_level < 2 else []
        # hm += [layer.upsample_resnet_block(self.is_train, hm[3], hc0, nc[1], nc[0], name='local_up_0')] if min_level < 1 else []
        #ddf_list = [layer.ddf_summand(hm[4 - idx], nc[idx], self.image_size, name='ddf1_sum_%d' % idx) for idx in self.ddf_levels]
        # ddf_list = tf.stack(ddf_list, axis=5)
        # self.ddf_MV_FIX = tf.reduce_sum(ddf_list, axis=5)
        # ddf_list2 = [layer.ddf_summand(hm[4 - idx], nc[idx], self.image_size, name='ddf2_sum_%d' % idx) for idx in self.ddf_levels]
        # ddf_list2 = tf.stack(ddf_list2, axis=5)
        # self.ddf_FIX_MV = tf.reduce_sum(ddf_list2, axis=5)

        # self.ddf_FIX_MV=tf.nn.tanh(self.ddf_FIX_MV)
        # self.ddf_MV_FIX=tf.nn.tanh(self.ddf_MV_FIX)

        # self.grid_warped_MV_FIX = self.grid_ref + self.ddf_MV_FIX
        # self.grid_warped_FIX_MV = self.grid_ref + self.ddf_FIX_MV

        #
        #create loss
        # self.warped_mv_img=self.__warp_MV_image(self.input_MV_image)
        # self.warped_warped_MV_img= self.__warp_FIX_image(self.warped_mv_img)
        # self.warped_MV_label = self.__warp_MV_image(self.input_MV_label)  # warp the moving label with the predicted ddf
        # self.warped_warped_MV_label = self.__warp_FIX_image(self.warped_MV_label)
        #
        #
        # self.warped_fix_img=self.__warp_FIX_image(self.input_FIX_image)
        # self.warped_warped_Fix_image=self.__warp_MV_image(self.warped_fix_img)
        # self.warped_FIX_label = self.__warp_FIX_image(self.input_FIX_label)
        # self.warped_warped_FIX_label = self.__warp_MV_image(self.warped_FIX_label)
        #
        # self.loss_warp_mv_fix = tf.reduce_mean(
        #     loss.multi_scale_loss(self.input_FIX_label, self.warped_MV_label, 'dice',[0, 1, 2, 4]))
        #
        # self.loss_warp_fix_mv = tf.reduce_mean(
        #     loss.multi_scale_loss(self.input_MV_label, self.warped_FIX_label, 'dice',[0, 1, 2, 4]))

        #label restore
        # self.loss_restore_fix = 0.1 * tf.reduce_mean(
        #     loss.multi_scale_loss(self.input_FIX_label, self.warped_warped_FIX_label,'dice',[0, 1, 2, 4]))
        #
        # self.loss_restore_mv = 0.1 * tf.reduce_mean(
        #     loss.multi_scale_loss(self.input_MV_label, self.warped_warped_MV_label,'dice',[0, 1, 2, 4]))

        #image restore
        # self.loss_restore_fix=  0.1* restore_loss(self.input_FIX_image, self.warped_warped_Fix_image)
        # self.loss_restore_mv= 0.1 * restore_loss(self.input_MV_image, self.warped_warped_MV_img)
        #
        #
        # self.anti_folding_loss = loss.anti_folding(self.ddf_FIX_MV) + loss.anti_folding(self.ddf_MV_FIX)
        #
        # self.ddf_regu_FIX = tf.reduce_mean(
        #     loss.local_displacement_energy(self.ddf_FIX_MV, 'bending', 100))
        # self.ddf_regu_MV = self.lambda_bend * tf.reduce_mean(
        #     loss.local_displacement_energy(self.ddf_MV_FIX, 'bending', 100))

        # self.train_op = tf.train.AdamOptimizer(self.args.lr).minimize(
        #     self.loss_warp_mv_fix +
        #     self.loss_warp_fix_mv +
        #     self.loss_restore_fix +
        #     self.loss_restore_mv +
        #     self.ddf_regu_FIX +
        #     self.ddf_regu_MV)
        # self.logger.debug("build network finish")
    def summary(self):
        #统计各个loss
        # tf.summary.scalar('warp_mv-fix', self.loss_warp_mv_fix)
        # tf.summary.scalar('warp_fix-mv', self.loss_warp_fix_mv)
        # tf.summary.scalar('restore-mv', self.loss_restore_mv)
        # tf.summary.scalar('restore-fix', self.loss_restore_fix)
        # tf.summary.scalar('bending_mv', self.ddf_regu_MV)
        # tf.summary.scalar('bending_fix', self.ddf_regu_FIX)
        # tf.summary.scalar('anti-fold',self.anti_folding_loss)

        tf.summary.image("gt",tf.expand_dims(self.ph_gt_label[:,:,:,48,1],-1))
        tf.summary.image("warp_label",tf.expand_dims(self.ph_warp_label[:,:,:,48,0],-1))
        tf.summary.image("target_img",tf.expand_dims(self.ph_target_image[:,:,:,48,0],-1))
        tf.summary.image("predict",tf.cast(tf.expand_dims(self.predit[:,:,:,48],-1),dtype=tf.float32))
        tf.summary.scalar('loss', self.ce_loss)
        tf.summary.scalar('loss', self.dice_loss)
        # all
        #不支持3d数据
        # tf.summary.image('atlas',self.ph_MV_label*255,max_outputs=4)
        # tf.summary.image('target',self.ph_FIX_label*255,max_outputs=4)
        # tf.summary.image('warp_atlas',self.warped_MV_label*255,max_outputs=4)

        self.summary=tf.summary.merge_all()

    def train(self):
        self.is_train=True

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)
        self.saver = tf.train.Saver()
        for itr in range(self.args.iteration):
            target_img_batch,target_lab_batch,atlas_img_batch,atlas_lab_batch=self.train_sampler.next_sample()

            trainFeed = {self.ph_target_image: target_img_batch,
                         self.ph_warp_label: atlas_lab_batch,
                         self.ph_gt_label:  target_lab_batch
                         }

            _,summary,gt,predict=self.sess.run([self.train_op, self.summary,self.ph_gt_label,self.predit], feed_dict=trainFeed)
            self.writer.add_summary(summary,global_step=itr)
            print("working  itr: %d dice = %f"%(itr,calculate_binary_dice(np.argmax(gt,axis=-1),predict)))
            self.writer.add_summary(summary, itr)
            if np.mod(itr, self.args.print_freq) == 1:
                self.__sample( itr)
            if np.mod(itr, self.args.save_freq) == 1:
                self.save(self.args.checkpoint_dir, itr)
    def __sample(self,iter):

        target_img_batch, target_lab_batch, atlas_img_batch, atlas_lab_batch = self.validate_sampler.next_sample()

        trainFeed = {self.ph_target_image: target_img_batch,
                     self.ph_warp_label: atlas_lab_batch,
                     self.ph_gt_label: target_lab_batch
                     }

        target_img,warp_label,gt,pred = self.sess.run([self.ph_target_image,self.ph_warp_label,self.ph_gt_label, self.predit], feed_dict=trainFeed)

        sitk_write_images(target_img,dir=self.args.sample_dir,name=str(iter)+"target_img")
        sitk_write_images(warp_label,dir=self.args.sample_dir,name=str(iter)+"warp_label")
        sitk_write_labs(np.argmax(gt,axis=-1),dir=self.args.sample_dir,name=str(iter)+"gt")
        sitk_write_labs(pred,dir=self.args.sample_dir,name=str(iter)+"pred")
        acc=calculate_binary_dice(np.argmax(gt,axis=-1),pred)
        self.logger.debug("acc:"+str(acc))
        return acc

    def validate(self):
        self.is_train=False
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        sims=[]
        sim_warp_mv_fix_all=[]
        for i in range(self.validate_sampler.len):
            sim=self.__sample(i)
            sims.append(sim)
        print("%s total_sim %s->%s " % (Get_Name_By_Index(self.args.component),self.args.Tatlas,self.args.Ttarget))
        print_mean_and_std(sims)

'''
求解两两图像之间的相似度
'''
from fusion.sampler_network import IntIntSimNetSampler
'''
计算一张target图片和一张atlas图片的相似度
'''
class IntIntSimNet(BaseModel):

    def __init__(self,sess,args):
        BaseModel.__init__(self, sess,args)
        self.args=args
        if self.args.phase.find('train')>=0:
            self.is_train=True
        else:
            self.is_train=False
        self.image_size = [self.args.image_size,self.args.image_size,self.args.image_size]
        self.train_sampler = IntIntSimNetSampler(self.args, True)
        self.validate_sampler = IntIntSimNetSampler(self.args, False)
        self.build_network()
        self.summary()

    def build_network(self):

        self.ph_target_image = tf.placeholder(tf.float32, [self.args.batch_size] + self.image_size + [1])
        self.ph_target_label= tf.placeholder(tf.float32, [self.args.batch_size] + self.image_size + [1])
        self.ph_atlas_image= tf.placeholder(tf.float32, [self.args.batch_size] + self.image_size + [1])
        self.ph_atlas_label = tf.placeholder(tf.float32, [self.args.batch_size] + self.image_size + [1])
        self.ph_sim= tf.placeholder(tf.float32, [self.args.batch_size] + self.image_size + [1])

        self.input_layer = tf.concat([self.ph_target_image, self.ph_atlas_image], axis=-1)

        self.ddf_levels = [0, 1, 2, 3, 4]
        self.num_channel_initial = self.args.num_channel_initial
        # 32,64,128,256,512
        nc = [int(self.num_channel_initial * (2 ** i)) for i in range(5)]
        h0, hc0 = layer.downsample_resnet_block(self.is_train, self.input_layer, 2, nc[0], k_conv0=[7, 7, 7],name='local_down_0')
        h1, hc1 = layer.downsample_resnet_block(self.is_train, h0, nc[0], nc[1], name='local_down_1')
        h2, hc2 = layer.downsample_resnet_block(self.is_train, h1, nc[1], nc[2], name='local_down_2')
        h3, hc3 = layer.downsample_resnet_block(self.is_train, h2, nc[2], nc[3], name='local_down_3')
        #multi scale similarity
        hm = [layer.conv3_block(self.is_train, h3, nc[3], nc[4], name='local_deep_4')]
        min_level = min(self.ddf_levels)
        hm += [layer.upsample_resnet_block(self.is_train, hm[0], hc3, nc[4], nc[3], name='local_up_3')] if min_level < 4 else []
        hm += [layer.upsample_resnet_block(self.is_train, hm[1], hc2, nc[3], nc[2], name='local_up_2')] if min_level < 3 else []
        hm += [layer.upsample_resnet_block(self.is_train, hm[2], hc1, nc[2], nc[1], name='local_up_1')] if min_level < 2 else []
        hm += [layer.upsample_resnet_block(self.is_train, hm[3], hc0, nc[1], nc[0], name='local_up_0')] if min_level < 1 else []
        sim_list = [layer.sim_summand(hm[4 - idx], nc[idx], self.image_size, name='ddf1_sum_%d' % idx) for idx in self.ddf_levels]
        sim_list = tf.stack(sim_list, axis=5)
        self.out=tf.reduce_sum(sim_list,axis=5)
        # self.out=tf.layers.conv3d(sim_list,1,3,padding='SAME',name='out_conv')
        self.prob=tf.nn.sigmoid(self.out)
        self.ce_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.ph_sim)
        self.ce_loss=tf.reduce_mean(self.ce_loss)#奇怪为啥不用这个也能工作
        self.train_op = tf.train.AdamOptimizer(self.args.lr).minimize(self.ce_loss)


    def summary(self):

        tf.summary.image("sim", tf.expand_dims(self.ph_sim[:, :, :, 48, 0], -1))
        tf.summary.image("out", tf.expand_dims(self.prob[:, :, :, 48, 0], -1))
        tf.summary.scalar('loss', self.ce_loss)
        self.summary=tf.summary.merge_all()

    def train(self):
        self.is_train=True

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)
        self.saver = tf.train.Saver()
        for itr in range(self.args.iteration):
            target_img_batch,target_lab_batch,atlas_img_batch,atlas_lab_batch,sim_batch=self.train_sampler.next_sample()

            trainFeed = {self.ph_target_image: target_img_batch,
                         self.ph_target_label:target_lab_batch,
                         self.ph_atlas_image: atlas_img_batch,
                         self.ph_atlas_label:atlas_lab_batch,
                         self.ph_sim:  sim_batch
                         }

            _,summary=self.sess.run([self.train_op, self.summary], feed_dict=trainFeed)
            self.writer.add_summary(summary,global_step=itr)
            # self.logger.debug("step %d :loss fix->mv=%f, mv->fix=%f; consis fix=%f,consist_mv=%f; anti=%f; fix_bending=%f; mv_bending=%f"%(glob_step,loss_fix_mv,loss_mv_fix,consis_fix,consis_mv,anti,reg_fix,reg_mv))
            if np.mod(itr, self.args.print_freq) == 1:
                self.__sample( itr)
            if np.mod(itr, self.args.save_freq) == 1:
                self.save(self.args.checkpoint_dir, itr)
    def __sample(self, itr):

        target_img_batch, target_lab_batch, atlas_img_batch, atlas_lab_batch,sim_batch = self.validate_sampler.next_sample()

        trainFeed = {self.ph_target_image: target_img_batch,
                     self.ph_target_label: target_lab_batch,
                     self.ph_atlas_image: atlas_img_batch,
                     self.ph_atlas_label: atlas_lab_batch,
                     self.ph_sim: sim_batch
                     }

        target_img,atlas_img,gt_sim,pred= self.sess.run([self.ph_target_image, self.ph_atlas_image, self.ph_sim, self.prob], feed_dict=trainFeed)

        sitk_write_labs(target_lab_batch, dir=self.args.sample_dir, name=str(itr) + "target_lab")
        sitk_write_labs(atlas_lab_batch, dir=self.args.sample_dir, name=str(itr) + "atlas_lab")
        sitk_write_images(target_img, dir=self.args.sample_dir, name=str(itr) + "target_img")
        sitk_write_images(atlas_img, dir=self.args.sample_dir, name=str(itr) + "atlas_img")
        sitk_write_images(gt_sim, dir=self.args.sample_dir, name=str(itr) + "gt_sim")
        sitk_write_images(pred, dir=self.args.sample_dir, name=str(itr) + "pred_sim")


    def validate(self):
        self.is_train=False
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        sims=[]
        for i in range(self.validate_sampler.len):
            sim=self.__sample(i)
            sims.append(sim)
        print("%s total_sim %s->%s " % (Get_Name_By_Index(self.args.component),self.args.Tatlas,self.args.Ttarget))
        print_mean_and_std(sims)


from fusion.sampler_network import LabIntSimNetSampler
class LabIntSimNet(BaseModel):

    def __init__(self,sess,args):
        BaseModel.__init__(self, sess,args)
        # self.sess=sess
        self.args=args
        if args.phase == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.image_size = [self.args.image_size,self.args.image_size,self.args.image_size]
        self.train_sampler = LabIntSimNetSampler(self.args, 'train')
        self.validate_sampler = LabIntSimNetSampler(self.args, 'validate')
        self.build_network()
        self.summary()

    def build_network(self):

        self.global_step=tf.Variable(0,trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.args.lr, self.global_step, self.args.decay_freq, 0.96, staircase=True)

        self.ph_target_image = tf.placeholder(tf.float32, [self.args.batch_size] + self.image_size + [1])
        self.ph_target_label= tf.placeholder(tf.float32, [self.args.batch_size] + self.image_size + [1])
        self.ph_atlas_label = tf.placeholder(tf.float32, [self.args.batch_size] + self.image_size + [1])
        #for supervised training
        self.ph_gt_dicesim= tf.placeholder(tf.float32, [self.args.batch_size] + self.image_size + [1])

        self.input_layer = tf.concat([self.ph_target_image, self.ph_atlas_label], axis=-1)

        self.ddf_levels = [0, 1, 2, 3, 4]
        self.num_channel_initial = self.args.num_channel_initial
        # 32,64,128,256,512
        nc = [int(self.num_channel_initial * (2 ** i)) for i in range(5)]
        h0, hc0 = layer.downsample_resnet_block(self.is_train, self.input_layer, 2, nc[0], k_conv0=[7, 7, 7],name='local_down_0')
        h1, hc1 = layer.downsample_resnet_block(self.is_train, h0, nc[0], nc[1], name='local_down_1')
        h2, hc2 = layer.downsample_resnet_block(self.is_train, h1, nc[1], nc[2], name='local_down_2')
        h3, hc3 = layer.downsample_resnet_block(self.is_train, h2, nc[2], nc[3], name='local_down_3')

        hm = [layer.conv3_block(self.is_train, h3, nc[3], nc[4], name='local_deep_4')]
        min_level = min(self.ddf_levels)
        hm += [layer.upsample_resnet_block(self.is_train, hm[0], hc3, nc[4], nc[3], name='local_up_3')] if min_level < 4 else []
        hm += [layer.upsample_resnet_block(self.is_train, hm[1], hc2, nc[3], nc[2], name='local_up_2')] if min_level < 3 else []
        hm += [layer.upsample_resnet_block(self.is_train, hm[2], hc1, nc[2], nc[1], name='local_up_1')] if min_level < 2 else []
        hm += [layer.upsample_resnet_block(self.is_train, hm[3], hc0, nc[1], nc[0], name='local_up_0')] if min_level < 1 else []
        sim_list = [layer.sim_summand(hm[4 - idx], nc[idx], self.image_size, name='ddf1_sum_%d' % idx) for idx in self.ddf_levels]
        sim_list = tf.stack(sim_list, axis=5)
        self.out = tf.reduce_sum(sim_list, axis=5)
        # self.out=tf.layers.conv3d(sim_list,1,3,padding='SAME',name='out_conv')
        self.predict_sim = tf.nn.sigmoid(self.out)

        predict_mask=self.predict_sim>0.2
        predict_mask=tf.to_float(predict_mask)
        self.predict_label=predict_mask*self.ph_atlas_label


        # self.ce_loss=tf.reduce_mean(tf.square(self.predict_sim-self.ph_gt_dicesim))
        self.ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.ph_gt_dicesim)
        self.ce_loss = tf.reduce_mean(self.ce_loss) #为啥这个地方不用reduce_mean也能工作？？？？？
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.ce_loss,global_step=self.global_step)

    def summary(self):
        tf.summary.image("gt_sim", tf.expand_dims(self.ph_gt_dicesim[:, :, :, 48, 0], -1))
        tf.summary.image("gt_label", tf.expand_dims(self.ph_target_label[:, :, :, 48, 0], -1))
        tf.summary.image("atlas_label", tf.expand_dims(self.ph_atlas_label[:, :, :, 48, 0], -1))
        tf.summary.image("predict_label", tf.expand_dims(self.predict_label[:, :, :, 48, 0], -1))
        tf.summary.image("predict_sim", tf.expand_dims(self.predict_sim[:, :, :, 48, 0], -1))
        tf.summary.scalar('loss', self.ce_loss)
        self.summary=tf.summary.merge_all()

    def train(self):
        self.is_train=True
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)
        self.saver = tf.train.Saver()
        for itr in range(self.args.iteration):
            target_img_batch,target_lab_batch,_,atlas_lab_batch,sim_batch=self.train_sampler.next_sample()

            trainFeed = {self.ph_target_image: target_img_batch,
                         self.ph_target_label:target_lab_batch,
                         self.ph_atlas_label: atlas_lab_batch,
                         self.ph_gt_dicesim: sim_batch
                         }

            _,summary,pred_label,gt_label=self.sess.run([self.train_op,self.summary,self.predict_label,self.ph_target_label], feed_dict=trainFeed)
            self.writer.add_summary(summary,global_step=itr)
            self.writer.add_summary(summary, itr)
            self.logger.debug("step %d : dice=%f"%(itr,calculate_binary_dice(pred_label,gt_label)))
            if np.mod(itr, self.args.print_freq) == 1:
                self.logger.debug(self.sess.run(self.learning_rate))
                print(self.sess.run(self.global_step))
                self.__sample(itr)
            if np.mod(itr, self.args.save_freq) == 1:
                self.save(self.args.checkpoint_dir, itr)
    def __sample(self, itr):

        target_img_batch, target_lab_batch, atlas_img_batch, atlas_lab_batch,sim_batch = self.validate_sampler.next_sample()

        trainFeed = {self.ph_target_image: target_img_batch,
                     self.ph_target_label: target_lab_batch,
                     self.ph_atlas_label: atlas_lab_batch,
                     self.ph_gt_dicesim: sim_batch
                     }
        target_img,atlas_label,gt,target_lab,pred_lab,pred_sim = self.sess.run([self.ph_target_image, self.ph_atlas_label, self.ph_gt_dicesim,self.ph_target_label,self.predict_label,self.predict_sim], feed_dict=trainFeed)

        sitk_write_images(target_img, dir=self.args.sample_dir, name=str(itr) + "target_img")
        sitk_write_labs(atlas_label, dir=self.args.sample_dir, name=str(itr) + "atlas_label")
        sitk_write_images(gt, dir=self.args.sample_dir, name=str(itr) + "gt_sim")
        sitk_write_images(pred_sim, dir=self.args.sample_dir, name=str(itr) + "pred_sim")
        sitk_write_labs(target_lab, dir=self.args.sample_dir, name=str(itr) + "target_label")
        sitk_write_labs(pred_lab, dir=self.args.sample_dir, name=str(itr) + "pred_label")
        pre_acc=calculate_binary_dice(target_lab,atlas_label)
        acc=calculate_binary_dice(target_lab,pred_lab,0.2)
        self.logger.debug("pre_acc %f -> acc %f"%(pre_acc,acc))
        return acc

    def validate(self):
        self.is_train=False
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        sims=[]
        for i in range(self.validate_sampler.len):
            sim=self.__sample(i)
            sims.append(sim)
        print("%s total_sim %s->%s " % (Get_Name_By_Index(self.args.component),self.args.Tatlas,self.args.Ttarget))
        print_mean_and_std(sims)

    def fusion(self):
        self.is_train = False
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        DSs= []
        HDs= []
        start=time.time()
        for i in range(self.validate_sampler.len):
            ds,hd=self.fusion_one_target(i)
            DSs.append(ds)
            HDs.append(hd)
        end=time.time()
        print("runtime: %.2f"%((end-start)/self.validate_sampler.len))
        print("%s total_sim %s->%s " % (Get_Name_By_Index(self.args.component),self.args.Tatlas,self.args.Ttarget))
        outpu2excel(self.args.res_excel,self.args.MOLD_ID+"_DS",DSs)
        outpu2excel(self.args.res_excel,self.args.MOLD_ID+"_HD",HDs)
        print_mean_and_std(DSs)
        print_mean_and_std(HDs)


    def fusion_one_target(self,itr):
        target_img_batch, target_lab_batch, atlas_img_batch, atlas_lab_batch,sim_batch,p_fix_img,p_fix_lab = self.validate_sampler.next_sample_4_fusion()
        sims=[]

        param=sitk.ReadImage(p_fix_img[0])
        sitk_write_lab(np.squeeze(target_lab_batch.astype(np.uint8)),param,dir=self.args.validate_dir,name=get_name_wo_suffix(p_fix_lab[0]))
        sitk_write_image(np.squeeze(target_img_batch),param,dir=self.args.validate_dir,name=get_name_wo_suffix(p_fix_img[0]))

        for i in range(atlas_lab_batch.shape[-2]):
            trainFeed = {self.ph_target_image: target_img_batch,
                         self.ph_target_label: target_lab_batch,
                         self.ph_atlas_label: atlas_lab_batch[...,i,:],
                         self.ph_gt_dicesim: sim_batch[...,i,:]
                         }
            target_img, atlas_label, gt, target_lab, pred_lab, pred_sim = self.sess.run(
                [self.ph_target_image, self.ph_atlas_label, self.ph_gt_dicesim, self.ph_target_label,
                 self.predict_label, self.predict_sim], feed_dict=trainFeed)
            sims.append(pred_sim)
            sitk_write_lab(np.squeeze(atlas_lab_batch[...,i,:]),param,dir=self.args.validate_dir,name=get_name_wo_suffix(p_fix_lab[0].replace('label','label_'+str(i))))
            sitk_write_image(np.squeeze(pred_sim),param,dir=self.args.validate_dir,name=get_name_wo_suffix(p_fix_lab[0]).replace('label','sim_'+str(i)))


        sims=np.stack(sims,-1)
        u_lab = np.unique(target_lab.astype(np.uint8))
        LabelStats = np.zeros((len(u_lab),) + np.squeeze(target_lab).shape)
        for i, lab in enumerate(u_lab):
            LabelStats[i] = np.sum((np.squeeze(atlas_lab_batch) == lab).astype(np.int16) * np.squeeze(sims), axis=-1)
        fusion_label = u_lab[np.argmax(LabelStats, axis=0)]
        ds=calculate_binary_dice(fusion_label,target_lab_batch)
        hd=calculate_binary_hd(fusion_label,target_lab_batch,spacing=param.GetSpacing())


        # sitk_write_image(np.squeeze(target_img_batch),param,dir=os.path.dirname(p_fix_lab[0]),name=get_name_wo_suffix(p_fix_img[0]))
        sitk_write_lab(np.squeeze(fusion_label).astype(np.uint8),param,dir=os.path.dirname(p_fix_lab[0]),name=get_name_wo_suffix(p_fix_lab[0]).replace('label','net_fusion_label'))
        return ds,hd




