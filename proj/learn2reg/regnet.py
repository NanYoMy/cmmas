from visualization.heatmap import plot_heat,save_heat
import time
from evaluate.metric import print_mean_and_std
from excelutil.output2excel import outpu2excel
import tensorflow as tf
import tf_help.layers as layer
import tf_help.utils as util
import tf_help.losses as loss
from learn2reg.sampler import Sampler
from model.base_model import BaseModel
import numpy as np
from sitkImageIO.itkdatawriter import sitk_write_images,sitk_write_lab,sitk_write_image,sitk_write_labs
import numpy as np
from logger.Logger import getLoggerV3
from evaluate.metric import calculate_binary_dice,neg_jac,calculate_binary_hd
from dirutil.helper import mk_or_cleardir,get_name_wo_suffix,mkdir_if_not_exist
from config.Defines import Get_Name_By_Index
from dirutil.helper import get_name_wo_suffix
import SimpleITK as sitk
from tf_help.losses import restore_loss,restore_loss2
from fusion.entropyhelper import conditional_entropy_label_over_image
import os
class LabReg(BaseModel):

    def __init__(self,sess,args,fold=1):
        BaseModel.__init__(self, sess,args)
        # self.sess=sess
        # self.args=args
        self.minibatch_size = self.args.batch_size
        self.image_size = [self.args.image_size,self.args.image_size,self.args.image_size]
        if args.phase == 'train':
            self.is_train = True
        else:
            self.is_train=False
        self.train_sampler = Sampler(self.args, 'train')
        self.validate_sampler = Sampler(self.args, 'validate')
        self.logger=getLoggerV3('learn2reg',self.args.log_dir)
        self.build_network()
        self.summary()

    def warp_MV_image(self, input_):
        return util.resample_linear(input_, self.grid_warped_MV_FIX)
        # return batch_displacement_warp3dV2(input_,self.ddf_MV_FIX,True)
        # return batch_displacement_warp3dV2(input_,self.ddf_MV_FIX,False)

    def warp_FIX_image(self, input_):
        return util.resample_linear(input_, self.grid_warped_FIX_MV)
        # return batch_displacement_warp3dV2(input_, self.ddf_FIX_MV,True)
        # return batch_displacement_warp3dV2(input_, self.ddf_FIX_MV,False)

    def build_network(self):

        self.global_step=tf.Variable(0,trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.args.lr, self.global_step, self.args.decay_freq, 0.96, staircase=True)

        self.grid_ref = util.get_reference_grid(self.image_size)
        self.grid_warped_MV_FIX = tf.zeros_like(self.grid_ref)  # initial zeros are safer for debug
        self.grid_warped_FIX_MV = tf.zeros_like(self.grid_ref)  # initial zeros are safer for debug

        self.ph_MV_image = tf.placeholder(tf.float32,[self.args.batch_size] + self.image_size+ [1])
        self.ph_FIX_image = tf.placeholder(tf.float32,[self.args.batch_size] + self.image_size+ [1])
        self.ph_moving_affine = tf.placeholder(tf.float32, [self.args.batch_size] + [1,12])  # 数据进行augment,4x4矩阵，但是最后四个参数为0001，所以一共12个参数
        self.ph_fixed_affine = tf.placeholder(tf.float32, [self.args.batch_size] + [1,12])
        self.ph_random_ddf=tf.placeholder(tf.float32,[self.args.batch_size] + self.image_size+ [3])

        self.ph_MV_label = tf.placeholder(tf.float32,[self.args.batch_size] + self.image_size + [1])
        self.ph_FIX_label = tf.placeholder(tf.float32,[self.args.batch_size ]+ self.image_size + [1])

        #data augmentation
        # self.input_MV_label = util.warp_image_affine(self.ph_MV_label, self.ph_moving_affine)  # data augmentation
        # self.input_MV_image = util.warp_image_affine(self.ph_MV_image, self.ph_moving_affine)  # data augmentation
        # self.input_FIX_label = util.warp_image_affine(self.ph_FIX_label, self.ph_fixed_affine)  # data augmentation
        # self.input_FIX_image = util.warp_image_affine(self.ph_FIX_image, self.ph_fixed_affine)  # data augmentation

        self.input_MV_image,self.input_MV_label=util.augment_3Ddata_by_affine(self.ph_MV_image,self.ph_MV_label,self.ph_moving_affine)
        self.input_FIX_image,self.input_FIX_label=util.augment_3Ddata_by_affine(self.ph_FIX_image,self.ph_FIX_label,self.ph_fixed_affine)
        # self.input_FIX_image,self.input_FIX_label=util.augment_3Ddata_by_DDF(self.ph_FIX_image,self.ph_FIX_label,self.ph_random_ddf)
        self.input_layer = tf.concat([layer.resize_volume(self.input_MV_image, self.image_size), self.input_FIX_image], axis=4)
        self.lambda_bend = self.args.lambda_ben
        self.lambda_consis = self.args.lambda_consis
        self.ddf_levels = [0, 1, 2, 3, 4]
        self.num_channel_initial = self.args.num_channel_initial
        # 32,64,128,256,512
        nc = [int(self.num_channel_initial * (2 ** i)) for i in range(5)]
        h0, hc0 = layer.downsample_resnet_block(self.is_train, self.input_layer, 2, nc[0], k_conv0=[7, 7, 7],
                                                name='local_down_0')
        h1, hc1 = layer.downsample_resnet_block(self.is_train, h0, nc[0], nc[1], name='local_down_1')
        h2, hc2 = layer.downsample_resnet_block(self.is_train, h1, nc[1], nc[2], name='local_down_2')
        h3, hc3 = layer.downsample_resnet_block(self.is_train, h2, nc[2], nc[3], name='local_down_3')
        # 这个代码是对应文章中 fig.4 中的哪个卷积块？
        hm = [layer.conv3_block(self.is_train, h3, nc[3], nc[4], name='local_deep_4')]
        min_level = min(self.ddf_levels)
        hm += [layer.upsample_resnet_block(self.is_train, hm[0], hc3, nc[4], nc[3],
                                           name='local_up_3')] if min_level < 4 else []
        hm += [layer.upsample_resnet_block(self.is_train, hm[1], hc2, nc[3], nc[2],
                                           name='local_up_2')] if min_level < 3 else []
        hm += [layer.upsample_resnet_block(self.is_train, hm[2], hc1, nc[2], nc[1],
                                           name='local_up_1')] if min_level < 2 else []
        hm += [layer.upsample_resnet_block(self.is_train, hm[3], hc0, nc[1], nc[0],
                                           name='local_up_0')] if min_level < 1 else []
        ddf_list = [layer.ddf_summand(hm[4 - idx], nc[idx], self.image_size, name='ddf1_sum_%d' % idx) for idx in
                    self.ddf_levels]
        ddf_list = tf.stack(ddf_list, axis=5)

        self.ddf_MV_FIX = tf.reduce_sum(ddf_list, axis=5)
        # outchannal=ddf_list.get_shape().as_list()[-1]
        # ddf_list=layer.SElayer(ddf_list,outchannal,1,'ddf1')
        # self.ddf_MV_FIX=tf.reduce_mean(ddf_list,axis=5)

        ddf_list2 = [layer.ddf_summand(hm[4 - idx], nc[idx], self.image_size, name='ddf2_sum_%d' % idx) for idx in
                     self.ddf_levels]
        ddf_list2 = tf.stack(ddf_list2, axis=5)

        self.ddf_FIX_MV = tf.reduce_sum(ddf_list2, axis=5)
        # outchannal=ddf_list2.get_shape().as_list()[-1]
        # ddf_list2=layer.SElayer(ddf_list2,outchannal,1,'ddf2')
        # self.ddf_FIX_MV=tf.reduce_mean(ddf_list2,axis=5)

        # self.ddf_FIX_MV=tf.nn.tanh(self.ddf_FIX_MV)
        # self.ddf_MV_FIX=tf.nn.tanh(self.ddf_MV_FIX)

        self.grid_warped_MV_FIX = self.grid_ref + self.ddf_MV_FIX
        self.grid_warped_FIX_MV = self.grid_ref + self.ddf_FIX_MV

        #
        #create loss
        self.warped_MV_image=self.warp_MV_image(self.input_MV_image)
        self.restore_MV_img= self.warp_FIX_image(self.warped_MV_image)
        self.warped_MV_label = self.warp_MV_image(self.input_MV_label)  # warp the moving label with the predicted ddf
        self.warped_warped_MV_label = self.warp_FIX_image(self.warped_MV_label)


        self.warped_fix_img=self.warp_FIX_image(self.input_FIX_image)
        self.restore_FIX_image=self.warp_MV_image(self.warped_fix_img)
        self.warped_FIX_label = self.warp_FIX_image(self.input_FIX_label)
        self.warped_warped_FIX_label = self.warp_MV_image(self.warped_FIX_label)

        self.loss_warp_mv_fix = tf.reduce_mean(
            loss.multi_scale_loss(self.input_FIX_label, self.warped_MV_label, 'dice',[0, 1, 2, 4]))

        self.loss_warp_fix_mv = tf.reduce_mean(
            loss.multi_scale_loss(self.input_MV_label, self.warped_FIX_label, 'dice',[0, 1, 2, 4]))

        #label restore
        # self.loss_restore_fix = 0.1 * tf.reduce_mean(
        #     loss.multi_scale_loss(self.input_FIX_label, self.warped_warped_FIX_label,'dice',[0, 1, 2, 4]))
        #
        # self.loss_restore_mv = 0.1 * tf.reduce_mean(
        #     loss.multi_scale_loss(self.input_MV_label, self.warped_warped_MV_label,'dice',[0, 1, 2, 4]))

        #image restore
        self.loss_restore_fix= self.args.lambda_consis * restore_loss(self.input_FIX_image, self.restore_FIX_image)
        self.loss_restore_mv= self.args.lambda_consis* restore_loss(self.input_MV_image, self.restore_MV_img)


        self.anti_folding_loss = loss.anti_folding(self.ddf_FIX_MV) + loss.anti_folding(self.ddf_MV_FIX)

        self.ddf_regu_FIX = tf.reduce_mean(
            loss.local_displacement_energy(self.ddf_FIX_MV, 'bending', 0.0))
        self.ddf_regu_MV = self.lambda_bend * tf.reduce_mean(
            loss.local_displacement_energy(self.ddf_MV_FIX, 'bending', 0.0))

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss_warp_mv_fix +
            self.loss_warp_fix_mv +
            self.loss_restore_fix +
            self.loss_restore_mv +
            self.ddf_regu_FIX +
            self.ddf_regu_MV,global_step=self.global_step)
        self.logger.debug("build network finish")
    def summary(self):
        #统计各个loss
        tf.summary.scalar('warp_mv-fix', self.loss_warp_mv_fix)
        tf.summary.scalar('warp_fix-mv', self.loss_warp_fix_mv)
        tf.summary.scalar('restore-mv', self.loss_restore_mv)
        tf.summary.scalar('restore-fix', self.loss_restore_fix)
        tf.summary.scalar('bending_mv', self.ddf_regu_MV)
        tf.summary.scalar('bending_fix', self.ddf_regu_FIX)
        tf.summary.scalar('anti-fold',self.anti_folding_loss)

        tf.summary.image("fix_img",tf.expand_dims(self.input_FIX_image[:,:,:,48,0],-1))
        tf.summary.image("warp_mv_img", tf.expand_dims(self.warped_MV_image[:, :, :, 48, 0], -1))
        tf.summary.image("mv_image",tf.expand_dims(self.input_MV_image[:,:,:,48,0],-1))
        tf.summary.image("warp_fix_image",tf.expand_dims(self.warped_fix_img[:,:,:,48,0],-1))
        tf.summary.image("restore_fix", tf.expand_dims(self.restore_FIX_image[:, :, :, 48, 0], -1))
        tf.summary.image("restore_mv", tf.expand_dims(self.restore_MV_img[:, :, :, 48, 0], -1))
        # all
        #不支持3d数据
        # tf.summary.image('atlas',self.ph_MV_label*255,max_outputs=4)
        # tf.summary.image('target',self.ph_FIX_label*255,max_outputs=4)
        # tf.summary.image('warp_atlas',self.warped_MV_label*255,max_outputs=4)

        self.summary=tf.summary.merge_all()

    def __swap(self, fix, mv):
        return mv, fix
    def train(self):
        self.is_train=True

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)
        self.saver = tf.train.Saver()
        for glob_step in range(self.args.iteration):
            fix_imgs,fix_labs,mv_imgs,mv_labs=self.train_sampler.next_sample()
            # 随机交换
            # if np.random.randint(2) == 1:
            #     fix_imgs, mv_imgs =self.__swap(fix_imgs,mv_imgs)
            #     fix_labs, mv_labs =self.__swap(fix_labs,mv_labs)

            trainFeed = {self.ph_MV_image: mv_imgs,
                         self.ph_FIX_image: fix_imgs,
                         self.ph_MV_label:  mv_labs,
                         self.ph_FIX_label: fix_labs,
                         self.ph_moving_affine: util.random_transform_generator(self.args.batch_size),
                         self.ph_fixed_affine: util.random_transform_generator(self.args.batch_size),
                         self.ph_random_ddf:util.random_ddf_generator(self.args.batch_size,self.image_size)}

            _,loss_fix_mv,loss_mv_fix,consis_fix,consis_mv,anti,reg_fix,reg_mv,summary=self.sess.run([self.train_op,
                                                                                                      self.loss_warp_fix_mv,
                                                                                                      self.loss_warp_mv_fix,
                                                                                                      self.loss_restore_fix,
                                                                                                      self.loss_restore_mv,
                                                                                                      self.anti_folding_loss,
                                                                                                      self.ddf_regu_FIX,
                                                                                                      self.ddf_regu_MV, self.summary], feed_dict=trainFeed)
            self.logger.debug("step %d :loss fix->mv=%f, mv->fix=%f; consis fix=%f,consist_mv=%f; anti=%f; fix_bending=%f; mv_bending=%f"%(glob_step,loss_fix_mv,loss_mv_fix,consis_fix,consis_mv,anti,reg_fix,reg_mv))
            self.writer.add_summary(summary, glob_step)
            if np.mod(glob_step, self.args.print_freq) == 1:
                self.sample(glob_step)
            if np.mod(glob_step, self.args.save_freq) == 1:
                self.save(self.args.checkpoint_dir, glob_step)
    def sample(self, iter):
        fix_imgs, fix_labs, mv_imgs, mv_labs=self.validate_sampler.next_sample()
        trainFeed = {self.ph_MV_image: mv_imgs,
                     self.ph_FIX_image: fix_imgs,
                     self.ph_MV_label: mv_labs,
                     self.ph_FIX_label: fix_labs,
                     self.ph_moving_affine: util.initial_transform_generator(self.args.batch_size),
                     self.ph_fixed_affine: util.initial_transform_generator(self.args.batch_size),
                     self.ph_random_ddf: util.init_ddf_generator(self.args.batch_size, self.image_size)
                     }

        # sitk_write_images(fix_labs.astype(np.uint16), None, self.args.sample_dir, '%d_fix_lab' % (iter))
        # sitk_write_images(mv_labs.astype(np.uint16),None,self.args.sample_dir,'%d_mv_lab'%(iter))
        input_fix_imgs,input_mv_imgs=self.sess.run([self.input_FIX_image,self.input_MV_image],feed_dict=trainFeed)
        sitk_write_images(input_fix_imgs,None,self.args.sample_dir,'%d_input_fix_img'%(iter))
        sitk_write_images(input_mv_imgs,None,self.args.sample_dir,'%d_input_mv_img'%(iter))

        warped_fix_imgs,warped_mv_imgs=self.sess.run([self.warped_fix_img, self.warped_MV_image], feed_dict=trainFeed)
        sitk_write_images(warped_mv_imgs,None,self.args.sample_dir,'%d_warp_mv_img'%(iter))
        sitk_write_images(warped_fix_imgs,None,self.args.sample_dir,'%d_warp_fix_img'%(iter))

        fix_imgs,mv_imgs=self.sess.run([self.ph_FIX_image,self.ph_MV_image],feed_dict=trainFeed)
        sitk_write_images(mv_imgs,None,self.args.sample_dir,'%d_mv_img'%(iter))
        sitk_write_images(fix_imgs,None,self.args.sample_dir,'%d_fix_img'%(iter))

        restore_mv_img,restore_fix_image=self.sess.run([self.restore_MV_img, self.restore_FIX_image], feed_dict=trainFeed)
        sitk_write_images(restore_fix_image,None,self.args.sample_dir,'%d_restore_fix_img'%(iter))
        sitk_write_images(restore_mv_img,None,self.args.sample_dir,'%d_restore_mv_img'%(iter))


        input_mv_label, input_fix_label = self.sess.run([self.input_MV_label, self.input_FIX_label], feed_dict=trainFeed)
        sitk_write_labs(input_fix_label, None, self.args.sample_dir, '%d_input_fix_lab' % (iter))
        sitk_write_labs(input_mv_label,None,self.args.sample_dir,'%d_input_mv_lab'%(iter))

        warp_fix_labs,warp_mv_labs=self.sess.run([self.warped_FIX_label,self.warped_MV_label], feed_dict=trainFeed)
        sitk_write_labs(warp_fix_labs, None, self.args.sample_dir, '%d_warp_fix_lab' % (iter))
        sitk_write_labs(warp_mv_labs,None,self.args.sample_dir,'%d_warp_mv_lab'%(iter))

        # fix_labs,mv_labs=self.sess.run([self.ph_FIX_label,self.ph_MV_label], feed_dict=trainFeed)
        # fix_labs=np.where(fix_labs>0.5,1,0)
        # mv_labs=np.where(mv_labs>0.5,1,0)
        sim_warp_fix_mv=calculate_binary_dice(warp_fix_labs,input_mv_label)
        sim_warp_mv_fix=calculate_binary_dice(warp_mv_labs,input_fix_label)

        anti_folder,loss_similarity1,loss_similarity2=self.sess.run([self.anti_folding_loss, self.loss_warp_fix_mv, self.loss_warp_mv_fix], feed_dict=trainFeed)
        ddf_fix_mv,ddf_mv_fix=self.sess.run([self.ddf_FIX_MV,self.ddf_MV_FIX],feed_dict=trainFeed)



        _,_,neg1=neg_jac(ddf_fix_mv[0,...])
        _,_,neg2=neg_jac(ddf_mv_fix[0,...])
        self.logger.debug("global_step %d: antifold_loss=%f dice : warp_fix_mv=%f warp_mv_fix=%f"%(iter,anti_folder,sim_warp_fix_mv,sim_warp_mv_fix))
        self.logger.debug("neg_jac %d %d"%(neg1,neg2))


        return sim_warp_mv_fix,sim_warp_fix_mv

    def validate(self):
        self.is_train=False
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # self.gen_sampler=Sampler(self.args,'validate')
        self.generator_sample_targetwise(self.validate_sampler, self.args.sample_dir + "/atlas_target/")

        # sim_warp_fix_mv_all=[]
        # sim_warp_mv_fix_all=[]
        # for i in range(self.validate_sampler.nb_pairs):
        #     sim_warp_mv_fix,sim_warp_fix_mv=self.sample(i)
        #     sim_warp_mv_fix_all.append(sim_warp_mv_fix)
        #     sim_warp_fix_mv_all.append(sim_warp_fix_mv)
        # self.logger.debug("%s total_sim %s->%s " % (Get_Name_By_Index(self.args.component),self.args.Tatlas,self.args.Ttarget))
        # self.logger.debug("mean:%f"%np.mean(sim_warp_mv_fix_all))
        # self.logger.debug("std:%f"%np.std(sim_warp_mv_fix_all))
        # self.logger.debug("%s total_sim %s->%s " % (Get_Name_By_Index(self.args.component),self.args.Ttarget,self.args.Tatlas))
        # self.logger.debug("mean:%f"%np.mean(sim_warp_fix_mv_all))
        # self.logger.debug("std:%f"%np.std(sim_warp_fix_mv_all))

    # def test(self):
    #     self.is_train = False
    #     init_op = tf.global_variables_initializer()
    #     self.saver = tf.train.Saver()
    #     self.sess.run(init_op)
    #     if self.load(self.args.checkpoint_dir):
    #         print(" [*] Load SUCCESS")
    #     else:
    #         print(" [!] Load failed...")
    #
    #     genSample = Sampler(self.args, 'test')
    #     # self.generator_sample_targetwise(genSample, self.args.test_dir + "/target_wise/")
    #     self.generator_sample_atlaswise(genSample, self.args.test_dir + "/atlas_wise/")
    # 生成样本
    # def generate(self):
    #     self.is_train = False
    #     init_op = tf.global_variables_initializer()
    #     self.saver = tf.train.Saver()
    #     self.sess.run(init_op)
    #     if self.load(self.args.checkpoint_dir):
    #         print(" [*] Load SUCCESS")
    #     else:
    #         print(" [!] Load failed...")
    #
    #     genSample = Sampler(self.args, 'train_sim')
    #     self.generator_sample_targetwise(genSample, self.args.fusion_dataset_dir + "/train/target_")
    #     # self.generator_sample(genSample,fix_range[4:],mv_range,self.args.sim_dataset_dir + "/valid/target_")
    #
    #     genSample = Sampler(self.args, 'test')
    #     self.generator_sample_targetwise(genSample, self.args.fusion_dataset_dir + "/test/target_")

    def generate_4_fusion(self):
        self.is_train = False
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        # genSample = Sampler(self.args, 'validate')
        # for index in range(genSample.len_fix):
        #     img_fix, lab_fix = genSample.img_fix[index], genSample.lab_fix[index]
        #     self.gen_warp_atlas('validate', genSample, index, img_fix, lab_fix, False)
        #这里生成数据用于训练模型
        genSample = Sampler(self.args, 'gen_fusion_train')
        self.generator_sample_targetwise(genSample, self.args.sample_dir + "/train_fusion_atlas_target/",output_stat=False)
        #生成验证的数据
        self.generator_sample_targetwise(self.validate_sampler, self.args.sample_dir + "/atlas_target/",output_stat=False)
        # for img_fix, lab_fix in zip(genSample.img_fix, genSample.lab_fix):
        #     output_dir = "/train_fusion_atlas_target/"+ get_name_wo_suffix(img_fix)
        #     mk_or_cleardir(output_dir)
        #     print(output_dir)
        #     for img_mv, lab_mv in zip(genSample.img_mv, genSample.lab_mv):
        #         _, _, _= self.gen_one_batch(genSample, img_fix, lab_fix, img_mv, lab_mv, output_dir)

    # def gen_warp_atlas(self, dir, genSample, i, img_fix, lab_fix, is_aug=True):
    #     output_dir = self.args.gen_dir + "/" + dir + "/target_" + str(i) + "_" + get_name_wo_suffix(img_fix)
    #     mk_or_cleardir(output_dir)
    #     params = []
    #     input_fix_imgs = []
    #     input_fix_labels = []
    #     warp_mv_imgs = []
    #     warp_mv_labels = []
    #     losses = []
    #     sims = []
    #     for img_mv, lab_mv in zip(genSample.img_mv, genSample.lab_mv):
    #         fix_imgs, fix_labs, mv_imgs, mv_labs = genSample.get_batch_data_V2([img_mv], [img_fix], [lab_mv], [lab_fix])
    #         trainFeed = self.create_feed_dict(fix_imgs, fix_labs, mv_imgs, mv_labs, is_aug)
    #         input_mv_label, \
    #         input_fix_label, \
    #         warp_mv_label, \
    #         input_mv_img, \
    #         input_fix_img, \
    #         warp_mv_img = self.sess.run([self.input_MV_label,
    #                                      self.input_FIX_label,
    #                                      self.warped_MV_label,
    #                                      self.input_MV_image,
    #                                      self.input_FIX_image,
    #                                      self.warped_MV_image], feed_dict=trainFeed)
    #         # input_fix_label=np.where(input_fix_label>0.5,1,0)
    #         # warp_mv_label=np.where(warp_mv_label>0.5,1,0)
    #         sims.append(calculate_binary_dice(input_fix_label[0, ...], warp_mv_label[0, ...]))
    #         param = sitk.ReadImage(img_fix)
    #         params.append(param)
    #         input_fix_imgs.append(input_fix_img[0, ...])
    #         input_fix_labels.append(input_fix_label[0, ...])
    #         warp_mv_imgs.append(warp_mv_img[0, ...])
    #         warp_mv_labels.append(warp_mv_label[0, ...])
    #         losses.append(conditional_entropy_label_over_image(np.squeeze(input_fix_img[0,...]),np.squeeze(warp_mv_label[0,...])))
    #         #生成每个warp的图片
    #         sitk_write_image(warp_mv_img[0,...], param, dir=output_dir, name=get_name_wo_suffix(img_mv))
    #         sitk_write_lab(warp_mv_label[0,...], param, dir=output_dir, name=get_name_wo_suffix(lab_mv))
    #
    #     sitk_write_image(input_fix_imgs[0], params[0], dir=output_dir, name=get_name_wo_suffix(img_fix))
    #     sitk_write_lab(input_fix_labels[0], params[0], dir=output_dir, name=get_name_wo_suffix(lab_fix))
    #     self.logger.debug(sims)
    #     indexs = np.argsort(losses)
    #     self.logger.debug("%s %f -> %f" % (output_dir, np.mean(sims), np.mean([sims[ind] for ind in indexs[:5]])))

    '''
    同一个target,有多个atlas进行分分割
    '''
    def generator_sample_targetwise(self, genSample, outputdir,output_stat=True):
        ds_all=[]
        hd_all=[]
        jt_all=[]

        start=time.time()
        for img_fix,lab_fix in zip(genSample.img_fix,genSample.lab_fix):
            output_dir = outputdir + get_name_wo_suffix(img_fix)
            mkdir_if_not_exist(output_dir)
            print(output_dir)
            for img_mv,lab_mv in zip(genSample.img_mv,genSample.lab_mv):
                ds,hd,jt=self.gen_one_batch(genSample, img_fix,lab_fix, img_mv,lab_mv, output_dir)
                # print("sim= %f"%ds)
                ds_all.append(ds)
                hd_all.append(hd)
                jt_all.append(jt)
            # print(ds_all)
        end=time.time()
        print("runtime %.2f "%((end-start)/(len(genSample.img_fix)*len(genSample.img_mv))))
        self.logger.debug("%s total_sim %s->%s " % (Get_Name_By_Index(self.args.component),self.args.Tatlas,self.args.Ttarget))


        if output_stat==True:
            outpu2excel(self.args.res_excel,self.args.MOLD_ID+"_DS",ds_all)
            outpu2excel(self.args.res_excel,self.args.MOLD_ID+"_HD",hd_all)
            print_mean_and_std(ds_all)
            print_mean_and_std(hd_all)
        # self.logger.debug("jt mean:%f (%f)"%(np.mean(jt_all),np.std(jt_all)))

    '''
    同一个atlas, 对多个atlas进行分割
    '''
    # def generator_sample_atlaswise(self, genSample, outputdir):
    #     for img_mv,lab_mv in zip(genSample.img_mv,genSample.lab_mv):
    #         output_dir = outputdir + get_name_wo_suffix(img_mv)
    #         mk_or_cleardir(output_dir)
    #         print(output_dir)
    #         for img_fix,lab_fix in zip(genSample.img_fix,genSample.lab_fix):
    #             self.gen_test(genSample, img_fix,lab_fix, img_mv,lab_mv, output_dir)

    # def gen_test(self, genSample,img_fix,lab_fix, img_mv,lab_mv, output_dir):
    #     fix_imgs, fix_labs, mv_imgs, mv_labs = genSample.get_batch_data_V2([img_mv],[img_fix],[lab_mv],[lab_fix])
    #     trainFeed = self.create_feed_dict(fix_imgs, fix_labs, mv_imgs, mv_labs)
    #     input_mv_label, \
    #     input_fix_label, \
    #     warp_mv_label, \
    #     input_mv_img, \
    #     input_fix_img, \
    #     warp_mv_img = self.sess.run([self.input_MV_label,
    #                                  self.input_FIX_label,
    #                                  self.warped_MV_label,
    #                                  self.input_MV_image,
    #                                  self.input_FIX_image,
    #                                  self.warped_MV_image], feed_dict=trainFeed)
    #     param=sitk.ReadImage(img_fix)
    #     sitk_write_image(input_fix_img[0,...], param, output_dir, get_name_wo_suffix(img_fix))
    #     # sitk_write_lab(input_fix_label[0,...], param, output_dir,get_name_wo_suffix(img_fix).replace('image','label'))
    #
    #     # sitk_write_image(warp_mv_img[0,...], param, output_dir,get_name_wo_suffix(img_mv))
    #     sitk_write_lab(warp_mv_label[0,...],param , output_dir,get_name_wo_suffix(img_fix).replace('image','label'))
    #
    #     ddf_fix_mv, ddf_mv_fix = self.sess.run([self.ddf_FIX_MV, self.ddf_MV_FIX], feed_dict=trainFeed)
    #     _, _, neg1 = neg_jac(ddf_fix_mv[0, ...])
    #     _, _, neg2 = neg_jac(ddf_mv_fix[0, ...])
    #     self.logger.debug("neg_jac %d %d" % (neg1, neg2))

    def gen_one_batch(self, genSample,img_fix,lab_fix, img_mv,lab_mv, output_dir):
        fix_imgs, fix_labs, mv_imgs, mv_labs = genSample.get_batch_data_V2([img_mv],[img_fix],[lab_mv],[lab_fix])
        feed = self.create_feed_dict(fix_imgs, fix_labs, mv_imgs, mv_labs)
        input_mv_label, \
        input_fix_label, \
        warp_mv_label, \
        input_mv_img, \
        input_fix_img, \
        warp_mv_img = self.sess.run([self.input_MV_label,
                                     self.input_FIX_label,
                                     self.warped_MV_label,
                                     self.input_MV_image,
                                     self.input_FIX_image,
                                     self.warped_MV_image], feed_dict=feed)


        param=sitk.ReadImage(img_fix)

        # U,V=self.sess.run([self.ddf_MV_FIX,self.ddf_FIX_MV],feed_dict=feed)
        # sitk_write_image(U[0,...], param, output_dir, get_name_wo_suffix(img_fix).replace('image','U_DDF'))
        # sitk_write_image(V[0,...], param, output_dir, get_name_wo_suffix(img_fix).replace('image','V_DDF'))

        sitk_write_image(input_fix_img[0,...], param, output_dir, get_name_wo_suffix(img_fix).replace('image','target_image'))
        sitk_write_lab(input_fix_label[0,...], param, output_dir,get_name_wo_suffix(lab_fix).replace('label','target_label'))

        sitk_write_image(warp_mv_img[0,...], param, output_dir,get_name_wo_suffix(img_mv).replace('image','atlas_image'))
        sitk_write_lab(warp_mv_label[0,...],param , output_dir,get_name_wo_suffix(lab_mv).replace('label','atlas_label'))
        sitk_write_lab(input_mv_label[0,...],param , output_dir,get_name_wo_suffix(lab_mv).replace('label','ori_atlas_label'))

        resotre_mv,restore_fix=self.sess.run([self.restore_MV_img,self.restore_FIX_image],feed_dict=feed)
        sitk_write_image(input_mv_img[0,...], param, output_dir,get_name_wo_suffix(img_mv))
        sitk_write_image(resotre_mv[0,...], param, output_dir,get_name_wo_suffix(img_mv).replace('image','restore'))
        sitk_write_image(restore_fix[0,...], param, output_dir,get_name_wo_suffix(img_fix).replace('image','restore'))

        # contour= np.where(warp_mv_label> 0.5, 1, 0)
        # contour= contour.astype(np.uint16)
        # contour=sitk.GetImageFromArray(np.squeeze(contour))

        # contour=sitk.LabelContour(contour,True)
        # sitk_write_lab(sitk.GetArrayFromImage(contour),param , output_dir,get_name_wo_suffix(img_mv).replace('image','contour'))

        ddf_fix_mv, ddf_mv_fix = self.sess.run([self.ddf_FIX_MV, self.ddf_MV_FIX], feed_dict=feed)
        _, _, neg1 = neg_jac(ddf_fix_mv[0, ...])
        _, _, neg2 = neg_jac(ddf_mv_fix[0, ...])
        # self.logger.debug("neg_jac %d %d" % (neg1, neg2))
        ds=calculate_binary_dice(warp_mv_label,input_fix_label)
        hd=calculate_binary_hd(warp_mv_label,input_fix_label,spacing=param.GetSpacing())
        return ds,hd,neg2

    def create_feed_dict(self, fix_imgs, fix_labs, mv_imgs, mv_labs, is_aug=False):

        if is_aug==False:
            trainFeed = {self.ph_MV_image: mv_imgs,
                         self.ph_FIX_image: fix_imgs,
                         self.ph_MV_label: mv_labs,
                         self.ph_FIX_label: fix_labs,
                         self.ph_moving_affine: util.initial_transform_generator(self.args.batch_size),
                         self.ph_fixed_affine: util.initial_transform_generator(self.args.batch_size),
                         self.ph_random_ddf: util.init_ddf_generator(self.args.batch_size, self.image_size)
                         }
        else:
            trainFeed = {self.ph_MV_image: mv_imgs,
                         self.ph_FIX_image: fix_imgs,
                         self.ph_MV_label: mv_labs,
                         self.ph_FIX_label: fix_labs,
                         self.ph_moving_affine: util.random_transform_generator(self.args.batch_size),
                         self.ph_fixed_affine: util.random_transform_generator(self.args.batch_size),
                         self.ph_random_ddf: util.random_ddf_generator(self.args.batch_size, self.image_size)
                         }
        return trainFeed

class LabAttentionReg(LabReg):
    def build_network(self):

        self.global_step=tf.Variable(0,trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.args.lr, self.global_step, self.args.decay_freq, 0.96, staircase=True)

        self.grid_ref = util.get_reference_grid(self.image_size)
        self.grid_warped_MV_FIX = tf.zeros_like(self.grid_ref)  # initial zeros are safer for debug
        self.grid_warped_FIX_MV = tf.zeros_like(self.grid_ref)  # initial zeros are safer for debug

        self.ph_MV_image = tf.placeholder(tf.float32,[self.args.batch_size] + self.image_size+ [1])
        self.ph_FIX_image = tf.placeholder(tf.float32,[self.args.batch_size] + self.image_size+ [1])
        self.ph_moving_affine = tf.placeholder(tf.float32, [self.args.batch_size] + [1,12])  # 数据进行augment,4x4矩阵，但是最后四个参数为0001，所以一共12个参数
        self.ph_fixed_affine = tf.placeholder(tf.float32, [self.args.batch_size] + [1,12])
        self.ph_random_ddf=tf.placeholder(tf.float32,[self.args.batch_size] + self.image_size+ [3])

        self.ph_MV_label = tf.placeholder(tf.float32,[self.args.batch_size] + self.image_size + [1])
        self.ph_FIX_label = tf.placeholder(tf.float32,[self.args.batch_size ]+ self.image_size + [1])

        self.input_MV_image,self.input_MV_label=util.augment_3Ddata_by_affine(self.ph_MV_image,self.ph_MV_label,self.ph_moving_affine)
        self.input_FIX_image,self.input_FIX_label=util.augment_3Ddata_by_affine(self.ph_FIX_image,self.ph_FIX_label,self.ph_fixed_affine)
        # self.input_FIX_image,self.input_FIX_label=util.augment_3Ddata_by_DDF(self.ph_FIX_image,self.ph_FIX_label,self.ph_random_ddf)
        self.input_layer = tf.concat([layer.resize_volume(self.input_MV_image, self.image_size), self.input_FIX_image], axis=4)
        self.lambda_bend = self.args.lambda_ben
        self.lambda_consis = self.args.lambda_consis
        self.ddf_levels = [0, 1, 2, 3, 4]
        self.num_channel_initial = self.args.num_channel_initial
        # 32,64,128,256,512
        nc = [int(self.num_channel_initial * (2 ** i)) for i in range(5)]
        h0, hc0 = layer.downsample_resnet_block(self.is_train, self.input_layer, 2, nc[0], k_conv0=[7, 7, 7],
                                                name='local_down_0')
        h1, hc1 = layer.downsample_resnet_block(self.is_train, h0, nc[0], nc[1], name='local_down_1')
        h2, hc2 = layer.downsample_resnet_block(self.is_train, h1, nc[1], nc[2], name='local_down_2')
        h3, hc3 = layer.downsample_resnet_block(self.is_train, h2, nc[2], nc[3], name='local_down_3')
        # 这个代码是对应文章中 fig.4 中的哪个卷积块？
        hm = [layer.conv3_block(self.is_train, h3, nc[3], nc[4], name='local_deep_4')]
        min_level = min(self.ddf_levels)
        gated_h1, self.gated1= layer.att_upsample_resnet_block(self.is_train, hm[0], hc3, nc[4], nc[3], name='local_up_3') #if min_level < 4 else None,None
        hm += [gated_h1]

        gated_h2, self.gated2= layer.att_upsample_resnet_block(self.is_train, hm[1], hc2, nc[3], nc[2], name='local_up_2') #if min_level < 3 else None,None
        hm += [gated_h2]

        gated_h3,self.gated3= layer.att_upsample_resnet_block(self.is_train, hm[2], hc1, nc[2], nc[1], name='local_up_1') #if min_level < 2 else None,None
        hm += [gated_h3]

        gated_h4,self.gated4= layer.att_upsample_resnet_block(self.is_train, hm[3], hc0, nc[1], nc[0], name='local_up_0') #if min_level < 1 else None,None
        hm += [gated_h4]

        ddf_list = [layer.ddf_summand(hm[4 - idx], nc[idx], self.image_size, name='ddf1_sum_%d' % idx) for idx in self.ddf_levels]
        ddf_list = tf.stack(ddf_list, axis=5)

        self.ddf_MV_FIX = tf.reduce_sum(ddf_list, axis=5)
        # outchannal=ddf_list.get_shape().as_list()[-1]
        # ddf_list=layer.SElayer(ddf_list,outchannal,1,'ddf1')
        # self.ddf_MV_FIX=tf.reduce_mean(ddf_list,axis=5)

        ddf_list2 = [layer.ddf_summand(hm[4 - idx], nc[idx], self.image_size, name='ddf2_sum_%d' % idx) for idx in
                     self.ddf_levels]
        ddf_list2 = tf.stack(ddf_list2, axis=5)

        self.ddf_FIX_MV = tf.reduce_sum(ddf_list2, axis=5)
        # outchannal=ddf_list2.get_shape().as_list()[-1]
        # ddf_list2=layer.SElayer(ddf_list2,outchannal,1,'ddf2')
        # self.ddf_FIX_MV=tf.reduce_mean(ddf_list2,axis=5)

        # self.ddf_FIX_MV=tf.nn.tanh(self.ddf_FIX_MV)
        # self.ddf_MV_FIX=tf.nn.tanh(self.ddf_MV_FIX)

        self.grid_warped_MV_FIX = self.grid_ref + self.ddf_MV_FIX
        self.grid_warped_FIX_MV = self.grid_ref + self.ddf_FIX_MV

        #
        #create loss
        self.warped_MV_image=self.warp_MV_image(self.input_MV_image)
        self.restore_MV_img= self.warp_FIX_image(self.warped_MV_image)
        self.warped_MV_label = self.warp_MV_image(self.input_MV_label)  # warp the moving label with the predicted ddf
        self.warped_warped_MV_label = self.warp_FIX_image(self.warped_MV_label)


        self.warped_fix_img=self.warp_FIX_image(self.input_FIX_image)
        self.restore_FIX_image=self.warp_MV_image(self.warped_fix_img)
        self.warped_FIX_label = self.warp_FIX_image(self.input_FIX_label)
        self.warped_warped_FIX_label = self.warp_MV_image(self.warped_FIX_label)

        self.loss_warp_mv_fix = tf.reduce_mean(
            loss.multi_scale_loss(self.input_FIX_label, self.warped_MV_label, 'dice',[0, 1, 2, 4]))

        self.loss_warp_fix_mv = tf.reduce_mean(
            loss.multi_scale_loss(self.input_MV_label, self.warped_FIX_label, 'dice',[0, 1, 2, 4]))

        #label restore
        # self.loss_restore_fix = 0.1 * tf.reduce_mean(
        #     loss.multi_scale_loss(self.input_FIX_label, self.warped_warped_FIX_label,'dice',[0, 1, 2, 4]))
        # #
        # self.loss_restore_mv = 0.1 * tf.reduce_mean(
        #     loss.multi_scale_loss(self.input_MV_label, self.warped_warped_MV_label,'dice',[0, 1, 2, 4]))

        #image restore
        self.loss_restore_fix= self.args.lambda_consis  * restore_loss(self.input_FIX_image, self.restore_FIX_image)
        self.loss_restore_mv=  self.args.lambda_consis * restore_loss(self.input_MV_image, self.restore_MV_img)


        self.anti_folding_loss = loss.anti_folding(self.ddf_FIX_MV) + loss.anti_folding(self.ddf_MV_FIX)

        self.ddf_regu_FIX = tf.reduce_mean(
            loss.local_displacement_energy(self.ddf_FIX_MV, 'bending', 0.0))
        self.ddf_regu_MV = self.lambda_bend * tf.reduce_mean(
            loss.local_displacement_energy(self.ddf_MV_FIX, 'bending', 0.0))

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss_warp_mv_fix +
            self.loss_warp_fix_mv +
            self.loss_restore_fix +
            self.loss_restore_mv,
            # self.ddf_regu_FIX +
            # self.ddf_regu_MV,
            global_step=self.global_step)
        self.logger.debug("build network finish")

    def show_gated_info(self):
        self.is_train=False
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        genSample=Sampler(self.args,'validate')
        for p_img_fix,p_lab_fix in zip(genSample.img_fix,genSample.lab_fix):
            output_dir = self.args.gated_att_dir+ get_name_wo_suffix(p_img_fix)
            mk_or_cleardir(output_dir)
            print(output_dir)
            for p_img_mv,p_lab_mv in zip(genSample.img_mv,genSample.lab_mv):
                fix_imgs, fix_labs, mv_imgs, mv_labs = genSample.get_batch_data_V2([p_img_mv], [p_img_fix], [p_lab_mv],[p_lab_fix])
                trainFeed = self.create_feed_dict(fix_imgs, fix_labs, mv_imgs, mv_labs)
                gated1,gated2, gated3, gated4 = self.sess.run([self.gated1, self.gated2, self.gated3, self.gated4], feed_dict=trainFeed)
                input_mv,input_fix= self.sess.run([self.input_MV_image,self.input_FIX_image], feed_dict=trainFeed)
                sitk_write_image(np.squeeze(input_mv),dir=output_dir,name=get_name_wo_suffix(p_img_mv))
                sitk_write_image(np.squeeze(input_fix),dir=output_dir,name=get_name_wo_suffix(p_img_fix))
                # sitk_write_image(np.squeeze(np.mean(gated1,axis=-1)),dir=output_dir,name="gate1"+get_name_wo_suffix(p_img_mv)+"_"+get_name_wo_suffix(p_img_fix))
                # sitk_write_image(np.squeeze(np.mean(gated2,axis=-1)),dir=output_dir,name="gate2"+get_name_wo_suffix(p_img_mv)+"_"+get_name_wo_suffix(p_img_fix))
                # sitk_write_image(np.squeeze(np.mean(gated3,axis=-1)),dir=output_dir,name="gate3"+get_name_wo_suffix(p_img_mv)+"_"+get_name_wo_suffix(p_img_fix))
                sitk_write_image(np.squeeze(np.mean(gated4,axis=-1)),dir=output_dir,name="gate4_"+get_name_wo_suffix(p_img_mv)+"_"+get_name_wo_suffix(p_img_fix))

                '''
                输出热力图
                '''
                # plot_heat(np.squeeze(gated4[...,0])[48,:,:])
                gated4=np.squeeze(np.mean(gated4,axis=-1))
                # gated4=np.swapaxes(gated4,2,0)
                save_heat((gated4[48,:,:]),dir=output_dir,name=get_name_wo_suffix(p_img_mv)+'_axial')
                save_heat(np.flip(gated4[:,48,:],0),dir=output_dir,name=get_name_wo_suffix(p_img_mv)+'_sagittal')
                save_heat(np.flip(gated4[:,:,48],0),dir=output_dir,name=get_name_wo_suffix(p_img_mv)+'_coronal')





