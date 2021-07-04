import tensorflow as tf
import sys
import MAS.helpers as helper
import MAS.regnet as network
import MAS.apps as app
import MAS.utils as util
import logger.Logger as  tflog
import  numpy as np
import os
from config import configer
import platform
from config.Defines import LABEL_INDEX
from file.helper import mkdir
from file.SampleSplitter import split_vote,isVoteTestDir
Debug=False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if platform.system()=="Linux":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 0 - get configs


def do_sample_gen_inference(model_id, vote_config, ATLAS = [1000 + i + 1 for i in range(12)], TARGET=[1017, 1018, 1019, 1020], istest=False):
    istrain=not istest
    log = tflog.getLogger("inference", vote_config)
    # 1 - images to register
    reader_Atlas_images, reader_Target_images, reader_Atlas_labels, reader_Target_labels = helper.get_data_readers(
        vote_config['Inference']['dir_atlas_image'],
        vote_config['Inference']['dir_target_image'],
        vote_config['Inference']['dir_atlas_label'],
        vote_config['Inference']['dir_target_label'])
    total_bf_dice = []
    total_af_dice_mv_fix = []
    total_distant_mv_fix = []
    INFER_BATCH_SIZE = 1
    # 2 - graph
    # network for predicting ddf only
    ph_atlas_image = tf.placeholder(tf.float32, [INFER_BATCH_SIZE] + reader_Atlas_images.data_shape + [1])
    ph_target_image = tf.placeholder(tf.float32, [INFER_BATCH_SIZE] + reader_Target_images.data_shape + [1])
    ph_atlas_label = tf.placeholder(tf.float32, [INFER_BATCH_SIZE] + reader_Atlas_labels.data_shape + [1])
    ph_target_label = tf.placeholder(tf.float32, [INFER_BATCH_SIZE] + reader_Atlas_labels.data_shape + [1])
    ph_unnormlized_atlas_image=tf.placeholder(tf.float32, [INFER_BATCH_SIZE] + reader_Atlas_images.data_shape + [1])
    ph_unnormlized_target_image=tf.placeholder(tf.float32, [INFER_BATCH_SIZE] + reader_Atlas_images.data_shape + [1])

    ph_atlas_affine = tf.placeholder(tf.float32, [vote_config['Train']['minibatch_size']] + [1,12])  # 数据进行augment,4x4矩阵，但是最后四个参数为0001，所以一共12个参数
    ph_target_affine = tf.placeholder(tf.float32, [vote_config['Train']['minibatch_size']] + [1,12])

    input_atlas_image = util.warp_image_affine(ph_atlas_image, ph_atlas_affine)  # data augmentation
    input_atlas_label = util.warp_image_affine(ph_atlas_label, ph_atlas_affine)  # data augmentation

    input_target_image = util.warp_image_affine(ph_target_image, ph_target_affine)  # data augmentation
    input_target_label = util.warp_image_affine(ph_target_label, ph_target_affine)  # data augmentation

    input_unnormalized_atlas_image=util.warp_image_affine(ph_unnormlized_atlas_image, ph_atlas_affine)
    input_unnormalized_target_image=util.warp_image_affine(ph_unnormlized_target_image, ph_target_affine)

    ph_is_train = tf.placeholder(tf.bool)
    Batch_NormaLize_tag=True
    # minibatch_size 是2张图，ph表示place holder image
    reg_net = network.build_network(network_type=vote_config['Network']['network_type'],
                                    minibatch_size=reader_Atlas_images.num_data,
                                    MV_image=input_atlas_image,
                                    FIX_image=input_target_image,
                                    ph_is_training=ph_is_train)
    # warped ct mr image
    warp_atlas_image = app.tensor_of_warp_volumes_by_ddf(input_unnormalized_atlas_image, reg_net.ddf_MV_FIX)
    # warped ct mr label
    warp_atlas_label = app.tensor_of_warp_volumes_by_ddf(input_atlas_label, reg_net.ddf_MV_FIX)

    # warped ct mr image
    warp_target_image = app.tensor_of_warp_volumes_by_ddf(input_unnormalized_target_image, reg_net.ddf_FIX_MV)
    # warped ct mr label
    warp_target_label = app.tensor_of_warp_volumes_by_ddf(input_target_label, reg_net.ddf_FIX_MV)


    # dice before warp
    tensor_bf_dice = app.tensor_of_compute_binary_dice(input_atlas_label, input_target_label)
    # dice after warp
    # tensor_af_dice_fix_mv = app.tensor_of_compute_binary_dice(warp_FIX_label, ph_MV_label)
    # tensor_dist_fix_mv = app.tensor_of_compute_centroid_distance(warp_FIX_label,ph_MV_label)
    tensor_af_dice_mv_fix = app.tensor_of_compute_binary_dice(warp_atlas_label, input_target_label)
    tensor_dist_mv_fix = app.tensor_of_compute_centroid_distance(warp_atlas_label,input_target_label)
    # restore the trained weights

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, vote_config['Generator']['reg_model']+str(model_id))
        # fn = lambda x, code=',': reduce(lambda x, y: [str(i)+code+str(j) for i in x for j in y], x)
        # sess.graph.finalize()
        for i in range(0,reader_Atlas_images.num_data):

            for j in range(0,reader_Target_images.num_data):
                # 3 - compute ddf
                               # mkdir(out_put_dir)

                out_put_dir = vote_config['Generator']['output_train_dir'] + "model_" + str(model_id) + "//" + \
                              reader_Atlas_images.get_file_names([i])[0] + "_" + \
                              reader_Target_images.get_file_names([j])[0]

                case_moving = i
                case_fixed = j
                feed_target=reader_Target_images.get_data(case_indices=[case_fixed])
                feed_atlas=reader_Atlas_images.get_data(case_indices=[case_moving])

                feed_unnormalized_target=reader_Target_images.get_data(case_indices=[case_fixed],need_img_normalize=False)
                feed_unnormlized_atlas=reader_Atlas_images.get_data(case_indices=[case_moving],need_img_normalize=False)

                feed_atlas_label=reader_Atlas_labels.get_data(case_indices=[case_moving], label_indices=[LABEL_INDEX])

                feed_atlas_affine = helper.initial_transform_generator(vote_config['Train']['minibatch_size'])
                feed_target_affine = helper.initial_transform_generator(vote_config['Train']['minibatch_size'])


#####训练数据生成
                if reader_Target_labels.num_data>0:
                    #5表示myo
                    feed_target_label = reader_Target_labels.get_data(case_indices=[case_fixed], label_indices=[LABEL_INDEX])
                    #添加了ph_fix_label
                    test_feed={
                        ph_atlas_image: feed_atlas,
                        ph_atlas_label: feed_atlas_label,
                        ph_target_image: feed_target,
                        ph_target_label: feed_target_label,
                        ph_unnormlized_atlas_image: feed_unnormlized_atlas,
                        ph_unnormlized_target_image: feed_unnormalized_target,
                        ph_atlas_affine: feed_atlas_affine,
                        ph_target_affine: feed_target_affine,
                        ph_is_train: Batch_NormaLize_tag
                    }
##################################

                    mkdir(out_put_dir + "_A_T")

                    warped_atlas_images_test, warp_atlas_label_test = sess.run([warp_atlas_image, warp_atlas_label],feed_dict=test_feed)
                    input_target_images_test, input_target_label_test = sess.run([input_unnormalized_target_image, input_target_label],feed_dict=test_feed)
                    # 输出 affine->warp的image,label

                    # 按照target进行保存
                    parameter = reader_Target_images.get_file_objects([j])[0]
                    filename_label = reader_Atlas_labels.get_file_names([i])[0]
                    filename_img = reader_Atlas_images.get_file_names([i])[0]

                    # warp_atlas_label_test=np.where(warp_atlas_label_test>0,1,0)
                    # input_target_label_test=np.where(input_target_label_test>0,1,0)

                    helper.sitk_write_images(warped_atlas_images_test.astype(np.int16),parameter,out_put_dir+"_A_T//", "atlas_img_"+filename_img)
                    helper.sitk_write_images(warp_atlas_label_test.astype(np.uint16),parameter,out_put_dir+"_A_T//", "atlas_lab_"+filename_label)
                    # 输出 affine-fix : image,label

                    # parameter=reader_Target_images.get_file_objects([j])[0]
                    filename_label = reader_Target_labels.get_file_names([j])[0]
                    filename_img = reader_Target_images.get_file_names([j])[0]


                    helper.sitk_write_images(input_target_images_test.astype(np.int16),parameter,out_put_dir+"_A_T//", "target_img_"+filename_img)
                    helper.sitk_write_images(input_target_label_test.astype(np.uint16),parameter,out_put_dir+"_A_T//", "target_lab_"+filename_label)

                    # 输出 affine-fix : image,label
                    bf_dice_test, af_dice_test_mv_fix,dis_test_mv_fix = sess.run([tensor_bf_dice,tensor_af_dice_mv_fix,tensor_dist_mv_fix], feed_dict=test_feed)
                    total_bf_dice.append(bf_dice_test)
                    # total_af_dice_fix_mv.append(af_dice_test_fix_mv)
                    total_af_dice_mv_fix.append(af_dice_test_mv_fix)
                    # total_distant_fix_mv.append(dis_test_fix_mv)
                    total_distant_mv_fix.append(dis_test_mv_fix)
                    print("bf_warp" + str(bf_dice_test))
                    # print("af_warp_fix_mv: " + str(af_dice_test_fix_mv))
                    print("af_warp_mv_fix: " + str(af_dice_test_mv_fix))
                    # print("dist_fix_mv" + str(dis_test_fix_mv))
                    print("dist_mv_fix" + str(dis_test_mv_fix))

        if reader_Target_labels.num_data > 0:
            # 写出所有的数据
            log.info(",".join(map(str, total_bf_dice)))
            helper.print_result(total_bf_dice)
            # log.info(",".join(map(str, total_af_dice_fix_mv)))
            # helper.print_result(total_af_dice_fix_mv)
            log.info(",".join(map(str, total_af_dice_mv_fix)))
            helper.print_result(total_af_dice_mv_fix)

            # log.info(",".join(map(str, total_distant_fix_mv)))
            # helper.print_result(total_distant_fix_mv)
            log.info(",".join(map(str, total_distant_mv_fix)))
            helper.print_result(total_distant_mv_fix)
        sess.close()



if __name__=="__main__":
    ATLAS = [1000+i + 1 for i in range(12)]
    TARGET = [1017,1018,1019,1020]
    vote_config = configer.get_vote_config()
    if len(sys.argv)==4:
        do_sample_gen_inference(sys.argv[3], vote_config)
    else:
        # tf.reset_default_graph()
        # do_sample_gen_inference(0,vote_config)
        # tf.reset_default_graph()
        # do_sample_gen_inference(500,vote_config)

        # do_sample_gen_inference(24000, vote_config)
        # tf.reset_default_graph()
        # do_sample_gen_inference(23000,vote_config)
        # tf.reset_default_graph()
        # do_sample_gen_inference(22000,vote_config)
        # tf.reset_default_graph()
        # do_sample_gen_inference(21000,vote_config)
        # tf.reset_default_graph()
        # do_sample_gen_inference(20000,vote_config)
        # tf.reset_default_graph()
        # do_sample_gen_inference(19000,vote_config)
        # tf.reset_default_graph()
        # do_sample_gen_inference(18000,vote_config)
        # tf.reset_default_graph()
        # do_sample_gen_inference(17000,vote_config)
        # tf.reset_default_graph()
        # do_sample_gen_inference(16000,vote_config)
        # tf.reset_default_graph()
        # do_sample_gen_inference(11500,vote_config)
        # tf.reset_default_graph()
        # do_sample_gen_inference(14000,vote_config)
        # tf.reset_default_graph()
        # do_sample_gen_inference(13000,vote_config)
        # tf.reset_default_graph()
        # do_sample_gen_inference(12000,vote_config)
        # tf.reset_default_graph()
        # do_sample_gen_inference(11000,vote_config)
        tf.reset_default_graph()
        do_sample_gen_inference(10000,vote_config)

    tf.reset_default_graph()

    split_vote(vote_config['Generator']['output_train_dir'],
               vote_config['Generator']['output_test_dir'],
               ATLAS,
               TARGET)# 随机产生4个