import tensorflow as tf
import platform
import random
# from tensorflow.python import  debug as tf_debug
import MAS.helpers as helper
import MAS.regnet as network
import MAS.utils as util
import os
import logger.Logger as tflogger
import numpy as np

from config import configer
from config.Defines import LABEL_INDEX

#-1- set logger
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if platform.system()=="Linux":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# 0 - get configs
reg_config = configer.get_reg_config()

tflog=tflogger.getLogger("train", reg_config)

# 1 - data
reader_MV_image, reader_FIX_image, reader_MV_label, reader_FIX_label = helper.get_data_readers(
    reg_config['Data']['dir_moving_image'],
    reg_config['Data']['dir_fixed_image'],
    reg_config['Data']['dir_moving_label'],
    reg_config['Data']['dir_fixed_label'])


# 2 - graph
ph_MV_image = tf.placeholder(tf.float32, [reg_config['Train']['minibatch_size']] + reader_MV_image.data_shape + [1])
ph_FIX_image = tf.placeholder(tf.float32, [reg_config['Train']['minibatch_size']] + reader_FIX_image.data_shape + [1])
ph_moving_affine = tf.placeholder(tf.float32, [reg_config['Train']['minibatch_size']] + [1, 12])#数据进行augment,4x4矩阵，但是最后四个参数为0001，所以一共12个参数
ph_fixed_affine = tf.placeholder(tf.float32, [reg_config['Train']['minibatch_size']] + [1, 12])
ph_is_train=tf.placeholder(tf.bool)
# 通过设置affine参数进行数据的augmentation，moving和fixed的图片都进行了affine变化
input_MV_image = util.warp_image_affine(ph_MV_image, ph_moving_affine)  # data augmentation
input_FIX_image = util.warp_image_affine(ph_FIX_image, ph_fixed_affine)  # data augmentation

# predicting ddf,利用augmented之后的数据进行训练
reg_net = network.build_network(network_type=reg_config['Network']['network_type'],
                                minibatch_size=reg_config['Train']['minibatch_size'],
                                MV_image=input_MV_image,
                                FIX_image=input_FIX_image,
                                ph_is_training=ph_is_train)

# loss
ph_MV_label = tf.placeholder(tf.float32, [reg_config['Train']['minibatch_size']] + reader_MV_image.data_shape + [1])
ph_FIX_label = tf.placeholder(tf.float32, [reg_config['Train']['minibatch_size']] + reader_FIX_image.data_shape + [1])
input_MV_label = util.warp_image_affine(ph_MV_label, ph_moving_affine)  # data augmentation
input_FIX_label = util.warp_image_affine(ph_FIX_label, ph_fixed_affine)  # data augmentation

#这里reg_net中的DDF已经对label进行warping


reg_net.build_loss(reg_config, MV_label=input_MV_label, FIX_label=input_FIX_label)
# utility nodes - for information only
dice_before_warp=util.compute_binary_dice(input_MV_label, input_FIX_label)

dice_warp_mv_fix = util.compute_binary_dice(input_FIX_label,reg_net.warped_MV_label)
dice_warp_fix_mv = util.compute_binary_dice(input_MV_label,reg_net.warped_FIX_label)

dist_warp_fix_mv = util.compute_centroid_distance(reg_net.warped_FIX_label, input_MV_label)
dist_warp_mv_fix = util.compute_centroid_distance(reg_net.warped_MV_label, input_FIX_label)



#label_diff=util.compute_binary_diff(warped_moving_label,input_fixed_label)
# 3 - training
# 训练完一次所有的数据需要的多少个batch
num_minibatch = int(reader_MV_label.num_data / reg_config['Train']['minibatch_size'])
mv_train_indices = [i for i in range(reader_MV_label.num_data)]
fx_train_indices = [i for i in range(reader_MV_label.num_data)]
saver = tf.train.Saver(max_to_keep=40)

# 这么说单独用myo，会比用多个效果好
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tflog.info("train start!")
    # ckpt = tf.train.get_checkpoint_state(os.path.dirname(reg_config['Train']['file_model_save']))
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess,ckpt.model_checkpoint_path)
    #     print("Model restored...")
    # else:
    #     print('No Model')
    #tensorflow 的调试
    #sess=tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.graph.finalize()#让graph保持不变，防止内存溢出
    for step in range(reg_config['Train']['total_iterations']):
        if step in range(0, reg_config['Train']['total_iterations'], num_minibatch):
            # 从mv和fx中随机采样
            random.shuffle(mv_train_indices)
            random.shuffle(fx_train_indices)

        minibatch_idx = step % num_minibatch
        mv_case_indices = mv_train_indices[minibatch_idx * reg_config['Train']['minibatch_size']:(minibatch_idx + 1) * reg_config['Train']['minibatch_size']]
        fx_case_indices = fx_train_indices[minibatch_idx * reg_config['Train']['minibatch_size']:(minibatch_idx + 1) * reg_config['Train']['minibatch_size']]

        label_indices = [random.randrange(8)+1 for i in mv_case_indices]

        # print(fx_case_indices)
        # print(mv_case_indices)
        # print(label_indices)
        # fx_case_indices=[0]
        # mv_case_indices=[9]
        # label_indices=[1]

        # fx_case_indices=[1]
        # mv_case_indices=[10]
        #只使用myo
        label_indices=[LABEL_INDEX]
        #LV
        # label_indices=[1]
        # values=[1,5]
        # label_indices=[random.choice(values)]


        feed_fix=reader_FIX_image.get_data(fx_case_indices)
        feed_fix_label=reader_FIX_label.get_data(fx_case_indices,label_indices)
        feed_mv=reader_MV_image.get_data(mv_case_indices,)
        feed_mv_label=reader_MV_label.get_data(mv_case_indices,label_indices)

        ####需要判断feed_mv_label 和 feed_fix_label的比例
        # if np.sum(feed_fix_label)/(np.sum(feed_mv_label)+0.00001)>2 or np.sum(feed_fix_label)/(np.sum(feed_mv_label)+0.00001)<0.5:
        #     tflog.info("area_ratio:"+str(np.sum(feed_fix_label)/(np.sum(feed_mv_label)+0.00001)))
        #     tflog.info("skip because the registration area is too small  "+str(np.sum(feed_fix_label))+":"+str(np.sum(feed_mv_label)))
        #     tflog.info(mv_case_indices[0])
        #     tflog.info(fx_case_indices[0])
        #     tflog.info(label_indices[0])
        #     continue


        #数据进行随机的交换warp
        if np.random.randint(0,2)==1:
            tmp=feed_fix
            feed_fix=feed_mv
            feed_mv=tmp
            tmp=feed_fix_label
            feed_fix_label=feed_mv_label
            feed_mv_label=tmp

        trainFeed = {ph_MV_image: feed_mv,
                     ph_FIX_image:feed_fix,
                     ph_MV_label: feed_mv_label,
                     ph_FIX_label: feed_fix_label,
                     ph_moving_affine: helper.random_transform_generator(reg_config['Train']['minibatch_size']),
                     ph_fixed_affine: helper.random_transform_generator(reg_config['Train']['minibatch_size']),
                     ph_is_train:True}


        sess.run(reg_net.train_op, feed_dict=trainFeed)


        if step in range(0, reg_config['Train']['total_iterations'], reg_config['Train']['freq_info_print']):
            loss_consistent_test,sim_warp_mv_fix_test, sim_warp_fix_mv_test, sim_warp_warp_mv_mv_test, \
            sim_warp_warp_fix_fix_test, reg_fix_test, reg_mv_test, dice_bf_warp_test, \
            dice_warp_fix_mv_test, dice_warp_mv_fix_test,anti_folding_loss_test = sess.run( \
                [reg_net.consistent_loss,\
                 reg_net.label_similarity_warp_mv_fix, \
                 reg_net.label_similarity_warp_fix_mv, \
                 reg_net.label_similarity_warp_warp_mv_mv, \
                 reg_net.label_similarity_warp_warp_fix_fix, \
                 reg_net.ddf_regularisation_FIX, \
                 reg_net.ddf_regularisation_MV, \
                 dice_before_warp, \
                 dice_warp_fix_mv, \
                 dice_warp_mv_fix,
                 reg_net.anti_folding_loss], feed_dict=trainFeed)

            tflog.info('Step %d : Loss=%f (consistent_loss=%f ,dice_bf_warp=%f, sim=%f , spatial_loss_mv_fix=%f , spatial_loss_fix_mv=%f,dice_warp_fix_mv=%f,dice_warp_mv_fix=%f regulariser_fix=%f, regulariser_mv=%f, sim_warp_warp_mv_mv=%f,sim_warp_warp_fix_fix=%f,anti_folding=%f)' %\
                  (step,\
                   loss_consistent_test+1-sim_warp_mv_fix_test + 1-sim_warp_fix_mv_test + 1-sim_warp_warp_mv_mv_test + 1-sim_warp_warp_fix_fix_test+reg_fix_test+reg_mv_test, \
                   loss_consistent_test, \
                   dice_bf_warp_test,
                   (dice_warp_fix_mv_test + dice_warp_mv_fix_test) / 2, \
                   sim_warp_mv_fix_test,\
                   sim_warp_fix_mv_test,\
                   dice_warp_fix_mv_test,\
                   dice_warp_mv_fix_test,\
                   reg_fix_test,\
                   reg_mv_test,sim_warp_warp_mv_mv_test,sim_warp_warp_fix_fix_test,
                   anti_folding_loss_test))
            # tflog.info("setp=%d loss_consistent=%f sim_warp_mv_fix=%f sim_warp_fix_mv=%f"%(step,loss_consistent_test,sim_warp_mv_fix_test,sim_warp_fix_mv_test))

            tflog.info('Image-label indices: %s - %s - %s' % (mv_case_indices,fx_case_indices, label_indices))

        if step in range(0, reg_config['Train']['total_iterations'], reg_config['Train']['freq_model_save']):
            save_path = saver.save(sess, reg_config['Train']['file_model_save'] + str(step), write_meta_graph=False)
            tflog.info("Model saved in: %s" % save_path)
            ###############################################################################################
            warped_MV_label_test, warped_FIX_label_test, warp_warp_MV_label_test, warp_warp_FIX_label_test=sess.run([reg_net.warped_MV_label, reg_net.warped_FIX_label, reg_net.warped_warped_MV_label, reg_net.warped_warped_FIX_label], feed_dict=trainFeed)
            input_FIX_label_test,input_MV_label_test=sess.run([input_FIX_label,input_MV_label],feed_dict=trainFeed)
            affin_mv_img_test,affin_fix_img_test=sess.run([input_MV_image,input_FIX_image],feed_dict=trainFeed)

            np.where(warped_MV_label_test < 0.5, 0, 1)
            np.where(warped_FIX_label_test < 0.5, 0, 1)
            np.where(warp_warp_MV_label_test < 0.5, 0, 1)
            np.where(warp_warp_FIX_label_test < 0.5, 0, 1)

            warped_MV_label_test = warped_MV_label_test.astype(np.uint16)
            warped_FIX_label_test = warped_FIX_label_test.astype(np.uint16)
            warp_warp_MV_label_test = warp_warp_MV_label_test.astype(np.uint16)
            warp_warp_FIX_label_test = warp_warp_FIX_label_test.astype(np.uint16)
            input_FIX_label_test = input_FIX_label_test.astype(np.uint16)
            input_MV_label_test = input_MV_label_test.astype(np.uint16)



            helper.write_images(warped_MV_label_test, reg_config['Inference']['dir_save'], "train" + str(step) + 'warped_MV_label')
            # helper.write_images(warped_FIX_label_test, config['Inference']['dir_save'],  "train"+str(step) + 'warped_FIX_label')
            # helper.write_images(warp_warp_MV_label_test, config['Inference']['dir_save'],  "train"+str(step) + 'warp_warped_MV_label')
            # helper.write_images(warp_warp_FIX_label_test, config['Inference']['dir_save'],  "train"+str(step) + 'warp_warped_FIX_label')

            #feed fix img
            # helper.write_images(feed_fix, config['Inference']['dir_save'], "train" + str(step) + 'feed_fix_img')
            # helper.write_images(feed_fix_label.astype(np.int32), config['Inference']['dir_save'],  "train"+str(step) + 'feed_FIX_label')
            #feed mv
            # helper.write_images(feed_mv, config['Inference']['dir_save'],  "train"+str(step) + 'feed_mv_img')
            # helper.write_images(feed_mv_label.astype(np.int32), config['Inference']['dir_save'], "train" + str(step) + 'feed_MV_label')

            #orignial mv
            # helper.write_images(reader_MV_image.get_data(mv_case_indices), config['Inference']['dir_save'], "train" + str(step) + 'mv_orignal_mv_img')
            # helper.write_images(reader_MV_label.get_data(mv_case_indices,label_indices).astype(np.int32), config['Inference']['dir_save'],"train" + str(step) + 'mv_orignal_mv_label')
            # orignial fx
            # helper.write_images(reader_FIX_image.get_data(fx_case_indices), config['Inference']['dir_save'],
            #                     "train" + str(step) + 'fx_orignal_fx_img')
            # helper.write_images(reader_FIX_label.get_data(fx_case_indices, label_indices).astype(np.int32),
            #                     config['Inference']['dir_save'], "train" + str(step) + 'fx_orignal_fx_label')

            #affine
            # helper.write_images(affin_mv_img_test, config['Inference']['dir_save'],"train" + str(step) + 'mv_affine_')
            helper.write_images(affin_fix_img_test, reg_config['Inference']['dir_save'], "train" + str(step) + 'fix_affine_')

            # helper.write_images(input_FIX_label_test, config['Inference']['dir_save'],  "train"+str(step) + 'affine_FIX_label')
            # helper.write_images(input_MV_label_test, config['Inference']['dir_save'],  "train"+str(step) + 'affine_MV_label')


    tflog.info("finish train")
###################################################################################################################
'''
run testing code
'''