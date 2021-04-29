from dirutil.helper import get_name_wo_suffix,mk_or_cleardir
from evaluate.metric import print_mean_and_std
from sitkImageIO.itkdatawriter import sitk_write_image,sitk_write_lab
from labelreg.losses import restore_loss2

import labelreg.losses as loss
import tensorflow as tf
import labelreg.layers as layer
import labelreg.utils as util
from learn2reg.loss import gauss_gradientSimilarity
from model.base_model import BaseModel
from learn2reg.regnet import LabReg
import numpy as np
from sitkImageIO.itkdatawriter import sitk_write_labs,sitk_write_images
import SimpleITK as sitk
from evaluate.metric import calculate_binary_dice,neg_jac
from medpy.metric import  dc,hd
from learn2reg.gen_sampler import GenSampler
from config.Defines import Get_Name_By_Index
from abdomen.abdomen_sampler import AbdomenSampler
from logger.Logger import getLoggerV3
class AffineReg(LabReg):

    def create_feed_dict(self, fix_imgs, fix_labs, mv_imgs, mv_labs, is_aug=False):

        if is_aug == False:
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
                         self.ph_moving_affine: util.random_transform_generator(self.args.batch_size,0.15),
                         self.ph_fixed_affine: util.random_transform_generator(self.args.batch_size,0.15),
                         self.ph_random_ddf: util.random_ddf_generator(self.args.batch_size, self.image_size)
                         }
        return trainFeed

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

        self.input_layer = tf.concat([layer.resize_volume(self.input_MV_image, self.image_size), self.input_FIX_image], axis=4)
        self.transform_initial = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]

        nc = [int(self.args.num_channel_initial* (2 ** i)) for i in range(5)]
        h0, hc0 = layer.downsample_resnet_block(self.is_train, self.input_layer, 2, nc[0], k_conv0=[7, 7, 7],
                                                name='global_down_0')
        h1, hc1 = layer.downsample_resnet_block(self.is_train, h0, nc[0], nc[1], name='global_down_1')
        h2, hc2 = layer.downsample_resnet_block(self.is_train, h1, nc[1], nc[2], name='global_down_2')
        h3, hc3 = layer.downsample_resnet_block(self.is_train, h2, nc[2], nc[3], name='global_down_3')
        h4 = layer.conv3_block(self.is_train, h3, nc[3], nc[4], name='global_deep_4')
        self.theta_fw = layer.fully_connected(h4, 12, self.transform_initial, name='global_project_0')
        self.theta_bw = layer.fully_connected(h4, 12, self.transform_initial, name='global_project_1')
        # out=tf.layers.flatten(h4)
        # out=tf.layers.dense(out,1024)
        # out=tf.layers.dense(out,256)
        # xyz=tf.layers.dense(out,3)
        # angles=tf.nn.tanh(tf.layers.dense(out,3))

        self.grid_warped_fw = util.warp_grid(self.grid_ref, self.theta_fw)
        # 这个地方为啥又减去,
        self.ddf_fw = self.grid_warped_fw - self.grid_ref

        self.grid_warped_bw= util.warp_grid(self.grid_ref, self.theta_bw)
        # 这个地方为啥又减去,
        self.ddf_bw = self.grid_warped_bw - self.grid_ref

        self.warped_MV_image= self.warp_image(self.input_MV_image,self.grid_warped_fw)  # warp the moving label with the predicted ddf
        self.warped_FIX_image= self.warp_image(self.input_FIX_image,self.grid_warped_bw)  # warp the moving label with the predicted ddf

        # self.resotre_MV_label=self.warp_image(self.warped_MV_label,self.grid_warped_bw)
        self.restore_MV_image= self.warp_image(self.warped_MV_image,self.grid_warped_bw)

        # self.resotre_FIX_label=self.warp_image(self.warped_FIX_label,self.grid_warped_fw)
        self.restore_FIX_image= self.warp_image(self.warped_FIX_image,self.grid_warped_fw)

        #这里可以让restore_fix_image* restore_fix_label，因为在形变的时候，图像四周容易生成空白，
        self.ddf_regularisation1= self.args.lambda_consis * restore_loss2(self.input_FIX_image, self.restore_FIX_image)
        self.ddf_regularisation2=  self.args.lambda_consis * restore_loss2(self.input_MV_image, self.restore_MV_image)
        self.ddf_regularisation=self.ddf_regularisation1+self.ddf_regularisation2

        # self.restore_MV_label= self.warp_image(self.warped_MV_label,self.grid_warped_bw)  # warp the moving label with the predicted ddf


        # self.warped_MV_image = self.warp_MV_image(self.input_MV_image)
        # self.warped_MV_label = self.warp_MV_image(self.input_MV_label)  # warp the moving label with the predicted ddf

        self.warped_MV_label= self.warp_image(self.input_MV_label,self.grid_warped_fw)  # warp the moving label with the predicted ddf
        self.warped_FIX_label= self.warp_image(self.input_FIX_label,self.grid_warped_bw)  # warp the moving label with the predicted ddf
        self.grad_loss_fw = tf.reduce_mean(loss.multi_scale_loss(self.input_FIX_label, self.warped_MV_label, 'dice', [0, 1, 2, 4,8]))
        self.grad_loss_bw = tf.reduce_mean(loss.multi_scale_loss(self.input_MV_label, self.warped_FIX_label, 'dice', [0, 1, 2, 4,8]))
        self.grad_loss=self.grad_loss_fw+self.grad_loss_bw

        self.train_op = tf.train.AdamOptimizer(self.args.lr).minimize(self.grad_loss+self.ddf_regularisation)


        # self.ddf_regularisation = tf.reduce_mean(
        #     loss.local_displacement_energy(self.ddf, 'bending', 0))

        # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
        #     self.grad_loss+
        #     self.ddf_regularisation,
        #     global_step=self.global_step)

        # gau_loss_0=gauss_gradientSimilarity(0)
        # gau_loss_1=gauss_gradientSimilarity(1)
        # gau_loss_1_5=gauss_gradientSimilarity(2)
        # gau_loss_2=gauss_gradientSimilarity(3)
        # self.grad_loss_0=gau_loss_0(self.warped_MV_image, self.input_FIX_image)
        # self.grad_loss_1=gau_loss_1(self.warped_MV_image, self.input_FIX_image)
        # self.grad_loss_1_5=gau_loss_1_5(self.warped_MV_image, self.input_FIX_image)
        # self.grad_loss_2=gau_loss_2(self.warped_MV_image, self.input_FIX_image)
        # self.grad_loss=(self.grad_loss_1+self.grad_loss_1_5+self.grad_loss_2)/3
        # self.ddf_regularisation =tf.reduce_mean(loss.local_displacement_energy(self.grid_warped, "bending", 0.0))

    def warp_image(self, image, ddf):
        return util.resample_linear(image, ddf)

    def train(self):
        self.is_train = True
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)
        self.saver = tf.train.Saver()
        for glob_step in range(self.args.iteration):
            fix_imgs, fix_labs, mv_imgs, mv_labs = self.train_sampler.next_sample()
            trainFeed = self.create_feed_dict(fix_imgs, fix_labs, mv_imgs, mv_labs, is_aug=True)
            _, theta,grad, ddf_reg, summary = self.sess.run([self.train_op, self.theta_bw,self.grad_loss, self.ddf_regularisation,self.summary_all], feed_dict=trainFeed)
            self.writer.add_summary(summary, glob_step)
            self.logger.debug("step %d: grad=%f,ddf_reg=%f" % (glob_step, grad, ddf_reg))
            print(theta)

            if np.mod(glob_step, self.args.print_freq) == 1:
                self.sample(glob_step)
            if np.mod(glob_step, self.args.save_freq) == 1:
                self.save(self.args.checkpoint_dir, glob_step)

    def sample(self, iter):
        # fix_imgs, fix_labs, mv_imgs, mv_labs=self.validate_sampler.next_sample()
        p_img_mvs, p_img_fixs, p_lab_mvs, p_lab_fixs = self.validate_sampler.get_batch_file()
        fix_imgs, fix_labs, mv_imgs, mv_labs = self.validate_sampler.get_batch_data_V2(p_img_mvs, p_img_fixs, p_lab_mvs,
                                                                                       p_lab_fixs)
        trainFeed = self.create_feed_dict(fix_imgs, fix_labs, mv_imgs, mv_labs, is_aug=False)

        input_mv_label, input_fix_label = self.sess.run([self.input_MV_label, self.input_FIX_label],
                                                        feed_dict=trainFeed)
        np.where(input_mv_label > 0.5, 1, 0)
        np.where(input_fix_label > 0.5, 1, 0)
        sitk_write_images(input_fix_label.astype(np.uint16), None, self.args.sample_dir, '%d_input_fix_lab' % (iter))
        sitk_write_images(input_mv_label.astype(np.uint16), None, self.args.sample_dir, '%d_input_mv_lab' % (iter))

        warp_mv_labs = self.sess.run( self.warped_MV_label, feed_dict=trainFeed)
        warp_mv_labs=np.where(warp_mv_labs > 0.5, 1, 0)
        sitk_write_images(warp_mv_labs.astype(np.uint16), None, self.args.sample_dir, '%d_warp_mv_lab' % (iter))

        fix_labs, mv_labs = self.sess.run([self.ph_FIX_label, self.ph_MV_label], feed_dict=trainFeed)
        fix_labs=np.where(fix_labs > 0.5, 1, 0)
        mv_labs=np.where(mv_labs > 0.5, 1, 0)
        # sitk_write_images(fix_labs.astype(np.uint16), None, self.args.sample_dir, '%d_fix_lab' % (iter))
        # sitk_write_images(mv_labs.astype(np.uint16),None,self.args.sample_dir,'%d_mv_lab'%(iter))
        input_fix_imgs, input_mv_imgs = self.sess.run([self.input_FIX_image, self.input_MV_image], feed_dict=trainFeed)
        sitk_write_images(input_fix_imgs, None, self.args.sample_dir, '%d_input_fix_img' % (iter))
        sitk_write_images(input_mv_imgs, None, self.args.sample_dir, '%d_input_mv_img' % (iter))

        warped_mv_imgs = self.sess.run(self.warped_MV_image, feed_dict=trainFeed)
        sitk_write_images(warped_mv_imgs, None, self.args.sample_dir, '%d_warp_mv_img' % (iter))

        fix_imgs, mv_imgs = self.sess.run([self.ph_FIX_image, self.ph_MV_image], feed_dict=trainFeed)
        sitk_write_images(mv_imgs, None, self.args.sample_dir, '%d_mv_img' % (iter))
        sitk_write_images(fix_imgs, None, self.args.sample_dir, '%d_fix_img' % (iter))

        sim_warp_mv_fix = calculate_binary_dice(warp_mv_labs, fix_labs)
        para = sitk.ReadImage(p_lab_fixs[0])
        # hd_warp_mv_fix = hd(np.squeeze(warp_mv_labs[0, ...]), np.squeeze(fix_labs[0, ...]),voxelspacing=para.GetSpacing())
        hd_warp_mv_fix=999
        return sim_warp_mv_fix,hd_warp_mv_fix

    def validate(self):
        self.is_train = False
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        genSample = GenSampler(self.args, 'validate')
        # self.generator_sample_targetwise(genSample, )
        outputdir=self.args.sample_dir + "/atlas_target/"
        ds_all = []
        bf_ds_all = []
        jt_all = []
        for img_fix, lab_fix in zip(genSample.img_fix, genSample.lab_fix):
            output_dir = outputdir + get_name_wo_suffix(img_fix)
            mk_or_cleardir(output_dir)
            print(output_dir)
            for img_mv, lab_mv in zip(genSample.img_mv, genSample.lab_mv):
                bfds,ds= self.gen_one_batch(genSample, img_fix, lab_fix, img_mv, lab_mv, output_dir)
                # print("sim= %f"%ds)
                ds_all.append(ds)
                bf_ds_all.append(bfds)
            # print(ds_all)
        self.logger.debug("%s total_sim %s->%s " % (Get_Name_By_Index(self.args.component), self.args.Tatlas, self.args.Ttarget))
        print_mean_and_std(bf_ds_all)
        print_mean_and_std(ds_all)
        # warp_mv_2_fixs = []
        # hd_warp_mv_2_fixs = []
        # warp_fix_2_mvs = []
        # hd_warp_fix_2_mvs = []
        # neg_mv_2_fixs = []
        # neg_fix_2_mvs = []
        # for i in range(self.validate_sampler.nb_pairs):
        #     sim_warp_mv_2_fix, sim_warp_fix_2_mv, neg_mv_2_fix_field, neg_fix_2_mv_field, hd_mv_2_fix, hd_fix_2_mv = self.sample(
        #         i)
        #     warp_mv_2_fixs.append(sim_warp_mv_2_fix)
        #     warp_fix_2_mvs.append(sim_warp_fix_2_mv)
        #     neg_mv_2_fixs.append(neg_mv_2_fix_field)
        #     neg_fix_2_mvs.append(neg_fix_2_mv_field)
        #     hd_warp_fix_2_mvs.append(hd_fix_2_mv)
        #     hd_warp_mv_2_fixs.append(hd_mv_2_fix)
        #
        # print(Get_Name_By_Index(self.args.component))
        # print("=============%s->%s================" % (self.args.Tatlas, self.args.Ttarget))
        # print("dice: %f" % (np.mean(warp_mv_2_fixs)))
        # print("std: %f" % (np.std(warp_mv_2_fixs)))
        #
        # print("hd: %f" % (np.mean(hd_warp_mv_2_fixs)))
        # print("std: %f" % (np.std(hd_warp_mv_2_fixs)))
        #
        # print("neg:%f" % (np.mean(neg_mv_2_fixs)))
        #
        # print("=============%s->%s================" % (self.args.Ttarget, self.args.Tatlas))
        # print("dice: %f" % (np.mean(warp_fix_2_mvs)))
        # print("std: %f" % (np.std(warp_fix_2_mvs)))
        # print("hd: %f" % (np.mean(hd_warp_fix_2_mvs)))
        # print("std: %f" % (np.std(hd_warp_fix_2_mvs)))
        # print("neg:%f" % (np.mean(neg_fix_2_mvs)))

    def gen_one_batch(self, genSample,img_fix,lab_fix, img_mv,lab_mv, output_dir):
        fix_imgs, fix_labs, mv_imgs, mv_labs = genSample.get_batch_data_V2([img_mv],[img_fix],[lab_mv],[lab_fix])
        feed = self.create_feed_dict(fix_imgs, fix_labs, mv_imgs, mv_labs)
        input_mv_label, \
        input_fix_label, \
        warp_mv_label, \
        input_mv_img, \
        input_fix_img, \
        warp_mv_img,\
        fw,\
        bw= self.sess.run([self.input_MV_label,
                                     self.input_FIX_label,
                                     self.warped_MV_label,
                                     self.input_MV_image,
                                     self.input_FIX_image,
                                     self.warped_MV_image,
                                     self.theta_bw,
                                     self.theta_fw], feed_dict=feed)
        # param=sitk.ReadImage(img_fix)
        param=None
        sitk_write_image(input_fix_img[0,...], param, output_dir, get_name_wo_suffix(img_fix).replace('image','target_image'))
        sitk_write_lab(input_fix_label[0,...], param, output_dir,get_name_wo_suffix(lab_fix).replace('label','target_label'))

        sitk_write_image(warp_mv_img[0,...], param, output_dir,get_name_wo_suffix(img_mv).replace('image','atlas_image'))
        sitk_write_lab(warp_mv_label[0,...],param , output_dir,get_name_wo_suffix(lab_mv).replace('label','atlas_label'))

        resotre_mv,restore_fix=self.sess.run([self.restore_MV_image,self.restore_FIX_image],feed_dict=feed)
        sitk_write_image(input_mv_img[0,...], param, output_dir,get_name_wo_suffix(img_mv))
        sitk_write_image(resotre_mv[0,...], param, output_dir,get_name_wo_suffix(img_mv).replace('image','restore'))
        sitk_write_image(restore_fix[0,...], param, output_dir,get_name_wo_suffix(img_fix).replace('image','restore'))

        print(fw)
        # contour= np.where(warp_mv_label> 0.5, 1, 0)
        # contour= contour.astype(np.uint16)
        # contour=sitk.GetImageFromArray(np.squeeze(contour))

        # contour=sitk.LabelContour(contour,True)
        # sitk_write_lab(sitk.GetArrayFromImage(contour),param , output_dir,get_name_wo_suffix(img_mv).replace('image','contour'))

        # ddf_fix_mv, ddf_mv_fix = self.sess.run([self.ddf_FIX_MV, self.ddf_MV_FIX], feed_dict=feed)
        # _, _, neg1 = neg_jac(ddf_fix_mv[0, ...])
        # _, _, neg2 = neg_jac(ddf_mv_fix[0, ...])
        ds=calculate_binary_dice(warp_mv_label,input_fix_label)
        bf_ds=calculate_binary_dice(input_mv_label,input_fix_label)
        return bf_ds,ds

    def summary(self):
        tf.summary.scalar("grad_loss",self.grad_loss)
        tf.summary.scalar("bend_loss_MV",self.ddf_regularisation)
        tf.summary.image("fix_img",tf.expand_dims(self.input_FIX_image[:,:,:,48,0],-1))
        tf.summary.image("warp_mv_img", tf.expand_dims(self.warped_MV_image[:, :, :, 48, 0], -1))
        tf.summary.image("mv_image",tf.expand_dims(self.input_MV_image[:,:,:,48,0],-1))
        self.summary_all=tf.summary.merge_all()

class AbdomenAffineReg(AffineReg):
    def __init__(self,sess,args):
        AffineReg.__init__(self, sess, args)
        # self.sess=sess
        # self.args=args
        # self.minibatch_size = self.args.batch_size
        # self.image_size = [self.args.image_size, self.args.image_size, self.args.image_size]
        # if args.phase == 'train':
        #     self.is_train = True
        # else:
        #     self.is_train = False
        self.train_sampler = AbdomenSampler(self.args, 'train')
        self.validate_sampler = AbdomenSampler(self.args, 'validate')
        # self.logger = getLoggerV3('learn2reg', self.args.log_dir)
        # self.build_network()
        # self.summary()
    def sample(self, iter):
        fix_imgs, fix_labs, mv_imgs, mv_labs=self.validate_sampler.next_sample()
        # p_img_mvs, p_img_fixs, p_lab_mvs, p_lab_fixs = self.validate_sampler.get_batch_file()
        # fix_imgs, fix_labs, mv_imgs, mv_labs = self.validate_sampler.get_batch_data_V2(p_img_mvs, p_img_fixs, p_lab_mvs,
        #                                                                                p_lab_fixs)
        trainFeed = self.create_feed_dict(fix_imgs, fix_labs, mv_imgs, mv_labs, is_aug=False)

        input_mv_label, input_fix_label = self.sess.run([self.input_MV_label, self.input_FIX_label],
                                                        feed_dict=trainFeed)
        np.where(input_mv_label > 0.5, 1, 0)
        np.where(input_fix_label > 0.5, 1, 0)
        sitk_write_images(input_fix_label.astype(np.uint16), None, self.args.sample_dir, '%d_input_fix_lab' % (iter))
        sitk_write_images(input_mv_label.astype(np.uint16), None, self.args.sample_dir, '%d_input_mv_lab' % (iter))

        warp_mv_labs = self.sess.run(self.warped_MV_label, feed_dict=trainFeed)
        warp_mv_labs = np.where(warp_mv_labs > 0.5, 1, 0)
        sitk_write_images(warp_mv_labs.astype(np.uint16), None, self.args.sample_dir, '%d_warp_mv_lab' % (iter))

        fix_labs, mv_labs = self.sess.run([self.ph_FIX_label, self.ph_MV_label], feed_dict=trainFeed)
        fix_labs = np.where(fix_labs > 0.5, 1, 0)
        mv_labs = np.where(mv_labs > 0.5, 1, 0)
        # sitk_write_images(fix_labs.astype(np.uint16), None, self.args.sample_dir, '%d_fix_lab' % (iter))
        # sitk_write_images(mv_labs.astype(np.uint16),None,self.args.sample_dir,'%d_mv_lab'%(iter))
        input_fix_imgs, input_mv_imgs = self.sess.run([self.input_FIX_image, self.input_MV_image], feed_dict=trainFeed)
        sitk_write_images(input_fix_imgs, None, self.args.sample_dir, '%d_input_fix_img' % (iter))
        sitk_write_images(input_mv_imgs, None, self.args.sample_dir, '%d_input_mv_img' % (iter))

        warped_mv_imgs = self.sess.run(self.warped_MV_image, feed_dict=trainFeed)
        sitk_write_images(warped_mv_imgs, None, self.args.sample_dir, '%d_warp_mv_img' % (iter))

        fix_imgs, mv_imgs = self.sess.run([self.ph_FIX_image, self.ph_MV_image], feed_dict=trainFeed)
        sitk_write_images(mv_imgs, None, self.args.sample_dir, '%d_mv_img' % (iter))
        sitk_write_images(fix_imgs, None, self.args.sample_dir, '%d_fix_img' % (iter))

        sim_warp_mv_fix = calculate_binary_dice(warp_mv_labs, fix_labs)
        # para = sitk.ReadImage(p_lab_fixs[0])
        # hd_warp_mv_fix = hd(np.squeeze(warp_mv_labs[0, ...]), np.squeeze(fix_labs[0, ...]),voxelspacing=para.GetSpacing())
        hd_warp_mv_fix = 999
        return sim_warp_mv_fix, hd_warp_mv_fix

    def validate(self):

        self.is_train = False
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        genSample = AbdomenSampler(self.args, 'validate')
        # self.generator_sample_targetwise(genSample, )
        outputdir = self.args.sample_dir + "/atlas_target/"
        ds_all = []
        bf_ds_all = []
        jt_all = []
        for img_fix, lab_fix in zip(genSample.img_fix, genSample.lab_fix):
            output_dir = outputdir + get_name_wo_suffix(img_fix)
            mk_or_cleardir(output_dir)
            print(output_dir)
            for img_mv, lab_mv in zip(genSample.img_mv, genSample.lab_mv):
                bfds, ds = self.gen_one_batch(genSample, img_fix, lab_fix, img_mv, lab_mv, output_dir)
                # print("sim= %f"%ds)
                ds_all.append(ds)
                bf_ds_all.append(bfds)
            # print(ds_all)
        self.logger.debug(
            "%s total_sim %s->%s " % (Get_Name_By_Index(self.args.component), self.args.Tatlas, self.args.Ttarget))
        print_mean_and_std(bf_ds_all)
        print_mean_and_std(ds_all)