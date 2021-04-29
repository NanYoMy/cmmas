from medpy.metric import dc
from excelutil.output2excel import outpu2excel
import tensorflow as tf
import labelreg.layers as layer
import labelreg.utils as util
import labelreg.losses as loss
from learn2reg.sampler import Sampler
from model.base_model import BaseModel
import numpy as np
from sitkImageIO.itkdatawriter import sitk_write_images,sitk_write_lab,sitk_write_image,sitk_write_labs
from spatial_transformer_network.displacement import batch_displacement_warp3dV2
import preprocessor.tools as tool
import numpy as np
from logger.Logger import getLoggerV3
from evaluate.metric import calculate_binary_dice,neg_jac,calculate_binary_hd
from dirutil.helper import mk_or_cleardir,get_name_wo_suffix
from config.Defines import Get_Name_By_Index
from learn2reg.challenge_sampler import CHallengeSampler
from dirutil.helper import get_name_wo_suffix
import SimpleITK as sitk
from learn2reg.gen_sampler import GenSampler
from labelreg.losses import restore_loss
from fusion.entropyhelper import conditional_entropy_label_over_image
import os

class OneDDFLabReg(BaseModel):

    def __init__(self,sess,args):
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


        self.grid_warped_MV_FIX = self.grid_ref + self.ddf_MV_FIX
        # self.grid_warped_FIX_MV = self.grid_ref + self.ddf_FIX_MV

        #
        #create loss
        self.warped_MV_image=self.warp_MV_image(self.input_MV_image)
        self.warped_MV_label = self.warp_MV_image(self.input_MV_label)  # warp the moving label with the predicted ddf

        self.loss_warp_mv_fix = tf.reduce_mean(
            loss.multi_scale_loss(self.input_FIX_label, self.warped_MV_label, 'dice',[0, 1, 2, 4]))

        self.ddf_regu_MV =  tf.reduce_mean(
            loss.local_displacement_energy(self.ddf_MV_FIX, 'bending', self.args.lambda_ben))

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss_warp_mv_fix +
            self.ddf_regu_MV,
            global_step=self.global_step)
        self.logger.debug("build network finish")
    def summary(self):
        #统计各个loss
        tf.summary.scalar('warp_mv-fix', self.loss_warp_mv_fix)
        tf.summary.scalar('bending_mv', self.ddf_regu_MV)
        tf.summary.image("fix_img",tf.expand_dims(self.input_FIX_image[:,:,:,48,0],-1))
        tf.summary.image("warp_mv_img", tf.expand_dims(self.warped_MV_image[:, :, :, 48, 0], -1))
        tf.summary.image("mv_image",tf.expand_dims(self.input_MV_image[:,:,:,48,0],-1))
        self.summary=tf.summary.merge_all()

    def train(self):
        self.is_train=True

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)
        self.saver = tf.train.Saver()
        for glob_step in range(self.args.iteration):
            fix_imgs,fix_labs,mv_imgs,mv_labs=self.train_sampler.next_sample()
            trainFeed = {self.ph_MV_image: mv_imgs,
                         self.ph_FIX_image: fix_imgs,
                         self.ph_MV_label:  mv_labs,
                         self.ph_FIX_label: fix_labs,
                         self.ph_moving_affine: util.random_transform_generator(self.args.batch_size),
                         self.ph_fixed_affine: util.random_transform_generator(self.args.batch_size),
                         self.ph_random_ddf:util.random_ddf_generator(self.args.batch_size,self.image_size)}

            _,loss_fix_mv,reg_mv,summary=self.sess.run([self.train_op,self.loss_warp_mv_fix,self.ddf_regu_MV, self.summary], feed_dict=trainFeed)
            self.logger.debug("step %d :loss fix->mv=%f,  mv_bending=%f"%(glob_step,loss_fix_mv,reg_mv))
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

        input_mv_label, input_fix_label = self.sess.run([self.input_MV_label, self.input_FIX_label], feed_dict=trainFeed)
        sitk_write_labs(input_fix_label, None, self.args.sample_dir, '%d_input_fix_lab' % (iter))
        sitk_write_labs(input_mv_label,None,self.args.sample_dir,'%d_input_mv_lab'%(iter))

        warp_mv_labs=self.sess.run(self.warped_MV_label, feed_dict=trainFeed)
        # warp_mv_labs=np.where(warp_mv_labs>0.5,1,0)
        sitk_write_labs(warp_mv_labs,None,self.args.sample_dir,'%d_warp_mv_lab'%(iter))

        warped_mv_imgs=self.sess.run( self.warped_MV_image, feed_dict=trainFeed)
        sitk_write_images(warped_mv_imgs,None,self.args.sample_dir,'%d_warp_mv_img'%(iter))

        sim_warp_mv_fix=calculate_binary_dice(warp_mv_labs,fix_labs)
        ddf_mv_fix=self.sess.run(self.ddf_MV_FIX,feed_dict=trainFeed)
        _,_,neg2=neg_jac(ddf_mv_fix[0,...])
        self.logger.debug("global_step %d: dice :  warp_mv_fix=%f"%(iter,sim_warp_mv_fix))
        self.logger.debug("neg_jac %d"%(neg2))
        return sim_warp_mv_fix
    def validate(self):
        self.is_train=False
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        genSample=Sampler(self.args,'validate')
        self.generator_sample_targetwise(genSample, self.args.sample_dir + "/atlas_target/")

        # sim_warp_mv_fix_all=[]
        # for i in range(self.validate_sampler.nb_pairs):
        #     sim_warp_mv_fix=self.sample(i)
        #     sim_warp_mv_fix_all.append(sim_warp_mv_fix)
        # self.logger.debug("%s total_sim %s->%s " % (Get_Name_By_Index(self.args.component),self.args.Tatlas,self.args.Ttarget))
        # self.logger.debug("mean:%f"%np.mean(sim_warp_mv_fix_all))
        # self.logger.debug("std:%f"%np.std(sim_warp_mv_fix_all))

    def test(self):
        self.is_train = False
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        genSample = Sampler(self.args, 'test')
        # self.generator_sample_targetwise(genSample, self.args.test_dir + "/target_wise/")
        self.generator_sample_atlaswise(genSample, self.args.test_dir + "/atlas_wise/")

    # 生成样本
    def generate(self):
        self.is_train = False
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        genSample = Sampler(self.args, 'train_sim')
        self.generator_sample_targetwise(genSample, self.args.fusion_dataset_dir + "/train/target_")
        # self.generator_sample(genSample,fix_range[4:],mv_range,self.args.sim_dataset_dir + "/valid/target_")

        genSample = Sampler(self.args, 'test')
        self.generator_sample_targetwise(genSample, self.args.fusion_dataset_dir + "/test/target_")

    def generate_4_fusion(self):
        self.is_train = False
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        genSample = Sampler(self.args, 'validate')
        for index in range(genSample.len_fix):
            img_fix, lab_fix = genSample.img_fix[index], genSample.lab_fix[index]
            self.gen_warp_atlas('validate', genSample, index, img_fix, lab_fix, False)

        #这里只生
        genSample = Sampler(self.args, 'fusion')
        for i in range(genSample.len_fix):
            img_fix, lab_fix = genSample.img_fix[i], genSample.lab_fix[i]
            self.gen_warp_atlas('train', genSample, i, img_fix, lab_fix)

    def gen_warp_atlas(self, dir, genSample, i, img_fix, lab_fix, is_aug=True):
        output_dir = self.args.gen_dir + "/" + dir + "/target_" + str(i) + "_" + get_name_wo_suffix(img_fix)
        mk_or_cleardir(output_dir)
        params = []
        input_fix_imgs = []
        input_fix_labels = []
        warp_mv_imgs = []
        warp_mv_labels = []
        losses = []
        sims = []
        for img_mv, lab_mv in zip(genSample.img_mv, genSample.lab_mv):
            fix_imgs, fix_labs, mv_imgs, mv_labs = genSample.get_batch_data_V2([img_mv], [img_fix], [lab_mv], [lab_fix])
            trainFeed = self.create_feed_dict(fix_imgs, fix_labs, mv_imgs, mv_labs, is_aug)
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
                                         self.warped_MV_image], feed_dict=trainFeed)
            input_fix_label=np.where(input_fix_label>0.5,1,0)
            warp_mv_label=np.where(warp_mv_label>0.5,1,0)
            sims.append(calculate_binary_dice(input_fix_label[0, ...], warp_mv_label[0, ...]))
            param = sitk.ReadImage(img_fix)
            params.append(param)
            input_fix_imgs.append(input_fix_img[0, ...])
            input_fix_labels.append(input_fix_label[0, ...])
            warp_mv_imgs.append(warp_mv_img[0, ...])
            warp_mv_labels.append(warp_mv_label[0, ...])
            losses.append(conditional_entropy_label_over_image(np.squeeze(input_fix_img[0,...]),np.squeeze(warp_mv_label[0,...])))
        indexs = np.argsort(losses)
        for ind in indexs:
            sitk_write_image(warp_mv_imgs[ind], params[ind], dir=output_dir, name=str(ind) + "_mv_img")
            sitk_write_lab(warp_mv_labels[ind], params[ind], dir=output_dir, name=str(ind) + "_mv_lab")
        sitk_write_image(input_fix_imgs[0], params[0], dir=output_dir, name=str(0) + "_fix_img")
        sitk_write_lab(input_fix_labels[0], params[0], dir=output_dir, name=str(0) + "_fix_lab")
        self.logger.debug(sims)
        self.logger.debug("%s %f -> %f" % (output_dir, np.mean(sims), np.mean([sims[ind] for ind in indexs[:5]])))

    '''
    同一个target,有多个atlas进行分分割
    '''
    def generator_sample_targetwise(self, genSample, outputdir):
        all_ds=[]
        all_hd=[]
        for img_fix,lab_fix in zip(genSample.img_fix,genSample.lab_fix):
            output_dir = outputdir + get_name_wo_suffix(img_fix)
            mk_or_cleardir(output_dir)
            print(output_dir)
            for img_mv,lab_mv in zip(genSample.img_mv,genSample.lab_mv):
                ds,hd=self.gen_one_batch(genSample, img_fix,lab_fix, img_mv,lab_mv, output_dir)
                all_ds.append(ds)
                all_hd.append(hd)
        outpu2excel(self.args.res_excel,self.args.MOLD_ID+"_DS",all_ds)
        outpu2excel(self.args.res_excel,self.args.MOLD_ID+"_HD",all_hd)

    '''
    同一个atlas, 对多个atlas进行分割
    '''
    def generator_sample_atlaswise(self, genSample, outputdir):
        for img_mv,lab_mv in zip(genSample.img_mv,genSample.lab_mv):
            output_dir = outputdir + get_name_wo_suffix(img_mv)
            mk_or_cleardir(output_dir)
            print(output_dir)
            for img_fix,lab_fix in zip(genSample.img_fix,genSample.lab_fix):
                self.gen_test(genSample, img_fix,lab_fix, img_mv,lab_mv, output_dir)

    def gen_test(self, genSample,img_fix,lab_fix, img_mv,lab_mv, output_dir):
        fix_imgs, fix_labs, mv_imgs, mv_labs = genSample.get_batch_data_V2([img_mv],[img_fix],[lab_mv],[lab_fix])
        trainFeed = self.create_feed_dict(fix_imgs, fix_labs, mv_imgs, mv_labs)
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
                                     self.warped_MV_image], feed_dict=trainFeed)
        param=sitk.ReadImage(img_fix)
        sitk_write_image(input_fix_img[0,...], param, output_dir, get_name_wo_suffix(img_fix))
        sitk_write_lab(warp_mv_label[0,...],param , output_dir,get_name_wo_suffix(img_fix).replace('image','label'))
        ddf_mv_fix = self.sess.run( self.ddf_MV_FIX, feed_dict=trainFeed)
        _, _, neg = neg_jac(ddf_mv_fix[0, ...])
        self.logger.debug("neg_jac %d" % (neg))

    def gen_one_batch(self, genSample,img_fix,lab_fix, img_mv,lab_mv, output_dir):
        fix_imgs, fix_labs, mv_imgs, mv_labs = genSample.get_batch_data_V2([img_mv],[img_fix],[lab_mv],[lab_fix])
        trainFeed = self.create_feed_dict(fix_imgs, fix_labs, mv_imgs, mv_labs)
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
                                     self.warped_MV_image], feed_dict=trainFeed)
        param=sitk.ReadImage(img_fix)
        sitk_write_image(input_fix_img[0,...], param, output_dir, get_name_wo_suffix(img_fix))
        sitk_write_lab(input_fix_label[0,...], param, output_dir,get_name_wo_suffix(img_fix).replace('image','label'))

        sitk_write_image(warp_mv_img[0,...], param, output_dir,get_name_wo_suffix(img_mv))
        sitk_write_lab(warp_mv_label[0,...],param , output_dir,get_name_wo_suffix(img_mv).replace('image','label'))

        ddf_mv_fix = self.sess.run( self.ddf_MV_FIX, feed_dict=trainFeed)
        _, _, neg2 = neg_jac(ddf_mv_fix[0, ...])
        self.logger.debug("neg_jac  %d" % ( neg2))
        ds=calculate_binary_dice(warp_mv_label,input_fix_label)
        hd=calculate_binary_hd(warp_mv_label,input_fix_label,spacing=param.GetSpacing())
        return ds,hd

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


class OneDDFAttLabReg(OneDDFLabReg):


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
        gated_h1, self.gated1 = layer.att_upsample_resnet_block(self.is_train, hm[0], hc3, nc[4], nc[3], name='local_up_3')  # if min_level < 4 else None,None
        hm += [gated_h1]

        gated_h2, self.gated2 = layer.att_upsample_resnet_block(self.is_train, hm[1], hc2, nc[3], nc[2],name='local_up_2')  # if min_level < 3 else None,None
        hm += [gated_h2]

        gated_h3, self.gated3 = layer.att_upsample_resnet_block(self.is_train, hm[2], hc1, nc[2], nc[1],name='local_up_1')  # if min_level < 2 else None,None
        hm += [gated_h3]

        gated_h4, self.gated4 = layer.att_upsample_resnet_block(self.is_train, hm[3], hc0, nc[1], nc[0],name='local_up_0')  # if min_level < 1 else None,None
        hm += [gated_h4]
        ddf_list = [layer.ddf_summand(hm[4 - idx], nc[idx], self.image_size, name='ddf1_sum_%d' % idx) for idx in
                    self.ddf_levels]
        ddf_list = tf.stack(ddf_list, axis=5)
        self.ddf_MV_FIX = tf.reduce_sum(ddf_list, axis=5)

        self.grid_warped_MV_FIX = self.grid_ref + self.ddf_MV_FIX
        # self.grid_warped_FIX_MV = self.grid_ref + self.ddf_FIX_MV

        #create loss
        self.warped_MV_image=self.warp_MV_image(self.input_MV_image)
        self.warped_MV_label = self.warp_MV_image(self.input_MV_label)  # warp the moving label with the predicted ddf

        self.loss_warp_mv_fix = tf.reduce_mean(
            loss.multi_scale_loss(self.input_FIX_label, self.warped_MV_label, 'dice',[0, 1, 2, 4]))

        self.ddf_regu_MV = self.args.lambda_ben*tf.reduce_mean(loss.local_displacement_energy(self.ddf_MV_FIX, 'bending',1))

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss_warp_mv_fix +
            self.ddf_regu_MV,
            global_step=self.global_step)
        self.logger.debug("build network finish")
