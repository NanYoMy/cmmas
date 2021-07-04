from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
import time
import SimpleITK as sitk
from file.helper import mk_or_cleardir
from logger.Logger import getLoggerV2
import config.configer
from dataprocessor.sampler import get_sample_generator
import MAS.helpers as helper
import logger.Logger as tflogger
from fusion.entropyhelper import mutual_information
from evaluate.helper import dice_compute

ckpt_path = './ckpt/%s' % os.path.basename(__file__)
def print_setting():
    pass


n_train_itr = 3000
#12个atlas


class PatchEmbbeding():
    def __init__(self,args):
        self.args=args
        self.logger = getLoggerV2('embedding',args=args)
        self.patch_size=[args.patch_size[0]]*3
        self.thres=args.thres[0]
        self.n_support_sample = args.n_support_sample[0]
        self.n_query_sample = args.n_query_sample[0]
        self.channal = 1
        # 每个atlas获取一个patch
        self.one_support = 1
        # 一个target一个patch
        self.one_query = 1
        # data_addr = sorted(glob.glob('.\\data\\Skeleton\\data\\*.mat'))  # all data
        # self.test_dataset, self.train_dataset = prepar_data(data_addr, n_classes)
        # print(self.train_dataset.shape)  # (17, 32, 60, 40, 3)
        # print(self.test_dataset.shape)  # (10, 32, 60, 40, 3)

    def __write_out_img(self,img,path):
        sitk.WriteImage(sitk.GetImageFromArray(img),path)
    def __write_out_atlas(self,atlas_img,atlas_lab,dices):
        for i in range(dices.shape[0]):
            self.__write_out_img(atlas_img[i],"../tmp/%d_atlas_img_%f.nii.gz"%(i,dices[i]))
            self.__write_out_img(atlas_lab[i],"../tmp/%d_atlas_lab.nii.gz" % (i))
    def __write_out_target(self,targt_img,targt_lab):
        self.__write_out_img(targt_img,'../tmp/target_img.nii.gz')
        self.__write_out_img(targt_lab,'../tmp/target_lab.nii.gz')

    def acc_cum(self, target_lab, atlas_lab, sim):
        p_c=self.patch_size[0] // 2
        target_center = target_lab[p_c, p_c, p_c]
        cum_py_1=0
        cum_py_0=0
        for i,s in enumerate(sim):
            if atlas_lab[i][p_c,p_c,p_c]==1:
                cum_py_1=cum_py_1+s
            else:
                cum_py_0 = cum_py_0 + s

        pre=1 if cum_py_1>cum_py_0 else 0

        if pre==target_center:
            return 1
        else:
            return 0

    def acc_maxlike(self, target_lab, atlas_lab, sim):
        p_c=self.patch_size[0] // 2
        target_center = target_lab[p_c, p_c, p_c]
        i=np.argmax(sim)

        if atlas_lab[i][p_c,p_c,p_c]==target_center:
            return 1
        else:
            return 0

    def acc_MV(self,target_lab,atalas_lab):

        atalas_lab=np.reshape(atalas_lab,[atalas_lab.shape[0]*atalas_lab.shape[1],atalas_lab.shape[2],atalas_lab.shape[3],atalas_lab.shape[4]])
        p_c=self.patch_size[0] // 2
        target_center = target_lab[p_c, p_c, p_c]
        cum_1 = 0
        cum_0 = 0
        for i in range(atalas_lab.shape[0]):
            if atalas_lab[i,p_c, p_c, p_c] == 1:
                cum_1 = cum_1 + 1
            else:
                cum_0 = cum_0 + 1

        pre = 1 if cum_1 > cum_0 else 0

        return  pre

    def acc_nlvw2(self, atlas_patches, atlas_labs, target_patch, target_lab):

        mi=np.zeros([atlas_patches.shape[0],atlas_patches.shape[1]])
        for i in range(atlas_patches.shape[0]):
            for j in range(atlas_patches.shape[1]):
                mi[i,j]=mutual_information(atlas_patches[i,j],target_patch)
        return self.argmax_contribution(atlas_labs, mi, target_lab)

    def acc_nlvw(self,atlas_patches, atlas_labs, target_patch, target_lab,regurization=0.001):
        '''
        Nonlocal patch-based label fusion for hippocampus segmentation
        '''
        l2_norms=np.zeros([atlas_patches.shape[0],atlas_patches.shape[1]])
        for i in range(atlas_patches.shape[0]):
            for j in range(atlas_patches.shape[1]):
                l2_norms[i,j]=np.sum((atlas_patches[i,j]-target_patch)**2,axis=None)
        weights= np.exp(- l2_norms / (l2_norms.min() + regurization))
        # print(weights)
        return self.argmax_contribution(atlas_labs, weights, target_lab)

    def acc_nlbeta(self,atlas_patches, atlas_labs, target_patch, target_lab,regurization=0.001):

        l2_norms=np.zeros([atlas_patches.shape[0],atlas_patches.shape[1]])
        for i in range(atlas_patches.shape[0]):
            for j in range(atlas_patches.shape[1]):
                l2_norms[i,j]=np.sum((atlas_patches[i,j]-target_patch)**2,axis=None)
        weights=np.exp(- l2_norms * regurization)
        return self.argmax_contribution(atlas_labs, weights, target_lab)

    def largest_indices(self,ary, n):
        """Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)

    def argmax_contribution(self, atlas_labs, weight, target_lab,is_top_k=True):

        unique_lab = np.unique(target_lab)
        p_c = self.patch_size[0] // 2
        target_center_lab = target_lab[p_c, p_c, p_c]
        atlas_center_lab=atlas_labs[:, :, p_c, p_c, p_c]

        #用那种方法来做
        if is_top_k==True:
            lag_ind=self.largest_indices(weight,7)
            top_k_atlas_lab=[atlas_center_lab[i][j] for i, j in zip(lag_ind[0],lag_ind[1])]
            top_k_atlas_wei=[weight[i][j] for i, j in zip(lag_ind[0],lag_ind[1])]
        else:
            top_k_atlas_lab=atlas_center_lab
            top_k_atlas_wei=weight

        weighted_votes = [np.sum(top_k_atlas_wei * ( top_k_atlas_lab== i).astype(np.float)) for i in unique_lab]
        predict = unique_lab[np.argmax(weighted_votes)]
        return  predict

    def euclidean_distance(self,query=None, prototype=None):  # a是query b是protypical
        # a.shape = Class_Number*Query_per_class x D
        # b.shape = Class_Number x D
        N, D = tf.shape(query)[0], tf.shape(query)[1]
        M = tf.shape(prototype)[0]
        query = tf.tile(tf.expand_dims(query, axis=1), (1, M, 1))
        prototype = tf.tile(tf.expand_dims(prototype, axis=0), (N, 1, 1))
        return tf.reduce_mean(tf.square(query - prototype), axis=2)

    def encoder(self,x, h_dim=8, scopename='encoder',reuse=False):
        with tf.variable_scope(scopename, reuse=reuse):  # reuse非常有用，可以避免设置
            net=tf.layers.conv3d(x,8,kernel_size=5,padding='same')
            net=tf.nn.relu(net)
            net = tf.layers.max_pooling3d(net, 2, strides=2)
            net=tf.layers.conv3d(net,16,kernel_size=3,padding='same')
            net=tf.nn.relu(net)
            net = tf.layers.max_pooling3d(net, 2, strides=2)
            net=tf.layers.conv3d(net,32,kernel_size=3,padding='same')
            net=tf.nn.relu(net)
            # net = tf.layers.max_pooling3d(net, 2, strides=2)
            # net=tf.layers.conv3d(net,64,kernel_size=3,padding='valid')
            # net=tf.nn.relu(net)
            # net = tf.layers.max_pooling3d(net, 2, strides=2)
            net = tf.layers.flatten(
                net)  # tf.contrib.layers.flatten(P)这个函数就是把P保留第一个维度，把第一个维度包含的每一子张量展开成一个行向量，返回张量是一个二维的
            return net
            # net=tf.layers.conv3d(x,8,kernel_size=3,padding='same')
            # net=tf.nn.relu(net)
            # net=tf.layers.conv3d(net,16,kernel_size=2,padding='same')
            # net=tf.nn.relu(net)
            # net=tf.layers.conv3d(net,32,kernel_size=2,padding='same')
            # net=tf.nn.relu(net)
            #
            # net=tf.layers.conv3d(net,64,kernel_size=3,padding='same')
            # net=tf.nn.relu(net)
            # net = tf.layers.max_pooling3d(net, 2, strides=2)
            #
            # net=tf.layers.conv3d(net,64,kernel_size=3,padding='same')
            # net=tf.nn.relu(net)
            # net = tf.layers.max_pooling3d(net, 2, strides=2)
            # # net = tf.layers.max_pooling3d(net, 2, strides=2)
            # # net=tf.layers.conv3d(net,64,kernel_size=3,padding='valid')
            # # net=tf.nn.relu(net)
            # # net = tf.layers.max_pooling3d(net, 2, strides=2)
            # net = tf.layers.flatten(
            #     net)  # tf.contrib.layers.flatten(P)这个函数就是把P保留第一个维度，把第一个维度包含的每一子张量展开成一个行向量，返回张量是一个二维的
            # return net
            # net1 = tf.layers.conv3d(x, h_dim, kernel_size=2, padding='same')
            # net1 = tf.nn.relu(net1)
            # net2 = tf.layers.conv3d(net1, h_dim * 2, kernel_size=2, padding='same')
            # net2 = tf.nn.relu(net2)
            #
            # net3 = tf.concat([net1, net2], axis=4)
            # net3 = tf.layers.conv3d(net3, h_dim * 4, kernel_size=2, padding='same')
            # net3 = tf.nn.relu(net3)
            #
            # net4 = tf.concat([net1, net2, net3], axis=4)
            #
            # net4 = tf.layers.conv3d(net4, h_dim * 8, kernel_size=2, padding='same')
            # net4 = tf.nn.relu(net4)
            #
            # net4 = tf.layers.conv3d(net4, h_dim * 6, kernel_size=3,
            #                         padding='SAME')  # 64 filters, each filter will generate a feature map.
            # net4 = tf.contrib.layers.batch_norm(net4, updates_collections=None, decay=0.99, scale=True, center=True)
            # net4 = tf.nn.relu(net4)
            # net4 = tf.layers.max_pooling3d(net4, 2, strides=2)
            #
            # net4 = tf.layers.conv3d(net4, h_dim * 4, kernel_size=3,
            #                         padding='SAME')  # 64 filters, each filter will generate a feature map.
            # net4 = tf.contrib.layers.batch_norm(net4, updates_collections=None, decay=0.99, scale=True, center=True)
            # net4 = tf.nn.relu(net4)
            # net4 = tf.layers.max_pooling3d(net4, 2, strides=2)
            # net = tf.layers.flatten(
            #     net4)  # tf.contrib.layers.flatten(P)这个函数就是把P保留第一个维度，把第一个维度包含的每一子张量展开成一个行向量，返回张量是一个二维的
            # return net
        # if self.patch_size[0]==7:
        #
        # else:
        #     with tf.variable_scope(scopename, reuse=reuse):  # reuse非常有用，可以避免设置
        #         net=tf.layers.conv3d(x,8,kernel_size=5,padding='same')
        #         net=tf.nn.relu(net)
        #         net = tf.layers.max_pooling3d(net, 2, strides=2)
        #         net=tf.layers.conv3d(net,16,kernel_size=3,padding='same')
        #         net=tf.nn.relu(net)
        #         net = tf.layers.max_pooling3d(net, 2, strides=2)
        #         net=tf.layers.conv3d(net,32,kernel_size=3,padding='same')
        #         net=tf.nn.relu(net)
        #         # net = tf.layers.max_pooling3d(net, 2, strides=2)
        #         # net=tf.layers.conv3d(net,64,kernel_size=3,padding='valid')
        #         # net=tf.nn.relu(net)
        #         # net = tf.layers.max_pooling3d(net, 2, strides=2)
        #         net = tf.layers.flatten(
        #             net)  # tf.contrib.layers.flatten(P)这个函数就是把P保留第一个维度，把第一个维度包含的每一子张量展开成一个行向量，返回张量是一个二维的
        #         return net

    def build_model(self, n_support_sample=None):
        print_setting()
        tf.reset_default_graph()
        if n_support_sample is None:
            n_support_sample=self.n_support_sample
        self.x = tf.placeholder(tf.float32, [n_support_sample, self.one_support] + self.patch_size + [self.channal], name='x')
        self.q = tf.placeholder(tf.float32, [self.n_query_sample, self.one_query]+self.patch_size+[self.channal], name='q')
        x_shape = tf.shape(self.x)
        q_shape = tf.shape(self.q)
        # 训练的时候具有support sample的参数
        num_support_sample, num_support_one = x_shape[0], x_shape[1]  # num_class num_support_sample
        num_query_sample,num_query_one = q_shape[0] ,q_shape[1]  # num_query_sample
        # y为label数据由外部导入
        # y = tf.placeholder(tf.int64, [n_way, n_support],name='y')
        self.y_n_hot = tf.placeholder(tf.int64, [self.n_query_sample, self.one_query, n_support_sample], name='y_n_hot')  # dimesion of each one_hot vector

        emb_x = self.encoder(tf.reshape(self.x, [num_support_sample * num_support_one]+self.patch_size+[self.channal]), 8, scopename='encoder',reuse=False)
        emb_dim = tf.shape(emb_x)[-1]  # the last dimesion
        # CLASS_NUM,128
        # 这个地方，不能简单的计算平均，要看y的值
        emb_x = tf.reshape(emb_x, [num_support_sample, num_support_one, emb_dim])
        emb_x = tf.reduce_mean(emb_x, axis=1)  # 计算每一类的均值，每一个类的样本都通过CNN映射到高维度空间



        # CLASS_NUM*QUERY_NUM_PER_CLASS,128
        emb_q = self.encoder(tf.reshape(self.q, [num_query_sample * num_query_one]+self.patch_size+[self.channal]), 8, scopename='encoder',reuse=True)
        # emb_q = self.encoder(tf.reshape(self.q, [num_query_sample * num_query_one, im_x, im_y, im_z,channal]), h_dim, z_dim,reuse=True)

        dists = self.euclidean_distance(emb_q, emb_x)

        # log_pY= 的index=1的元素为 {exp(s_i,c_1),exp(s_i,c_2)....exp(s_i,c_n)}/\Sigma{exp(s_i,c_1),exp(s_i,c_2)....exp(s_i,c_n)}
        # 也就是s_i的对应每个类别的概率，
        self.dist=tf.reshape(dists,[num_query_sample, num_query_one, -1])
        self.prob=tf.reshape(tf.nn.softmax(-dists),
                             [num_query_sample, num_query_one, -1])
        # self.log_p_y = tf.reshape(tf.nn.log_softmax(-dists),
        #                      [num_query_sample, num_query_one, -1])  # -1表示自动计算剩余维度，paper中公式2 log_softmax 默认 axis=-1
        # 其实这里并不是真正意义上的cross_entropy
        #
        self.ce_loss = -tf.reduce_mean(tf.reshape(tf.log(tf.reduce_sum(tf.multiply(tf.to_float(self.y_n_hot), self.prob), axis=-1)), [-1]),
                                  name='loss')  # reshpae(a,[-1])会展开所有维度, ce_loss=cross entropy
        tf.add_to_collection('loss', self.ce_loss)
        self.train_op = tf.train.AdamOptimizer().minimize(self.ce_loss)

    def train(self):

        sess = tf.InteractiveSession()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver = tf.train.Saver()
        base_dir=self.args.train_atlas_target[0]
        sample_generator = [get_sample_generator(base_dir,target_id=i, args=self.args) for i in [1013, 1014, 1015, 1016]]
        # sample_generator = [get_sample_generator(base_dir,target_id=i, args=self.args) for i in [1013]]

        time.clock()
        acc_avg=0.0
        mv_acc_avg=0.0
        for itr in range(n_train_itr):
            '''
            随机产生一个数组，包含0-n_classes,取期中n_way个类
            '''
            # atlas_patches, atlas_labs, target_patch, target_lab,sim = sample_generator[np.random.randint(0,len(sample_generator))].next_sample(
            #     num_nonlocal_patch=6, dupplicated=False)
            g=sample_generator[np.random.randint(0, len(sample_generator))]
            atlas_patches, atlas_labs, target_patch, target_lab,sim = g.next_sample(
                num_nonlocal_patch=self.n_support_sample, is_test=False)
            #check是否有问题
            atlas_patches, atlas_labs, sim = self.__slice_for_network(atlas_patches, atlas_labs, sim)

            binary_sim=[(1 if i > self.thres else 0) for i in sim]
            if np.sum(binary_sim)==0 or np.sum(binary_sim)==len(binary_sim):
                continue
            # max_idx=np.argmax(sim)
            # min_idx=np.argmin(sim)
            support = np.zeros([self.n_support_sample, self.one_support]+self.patch_size+[self.channal], dtype=np.float32)
            query = np.zeros([self.n_query_sample, self.one_query]+self.patch_size+[self.channal], dtype=np.float32)
            y_n_labels = np.zeros([self.n_query_sample, self.one_query, self.n_support_sample], np.uint8)

            for i,p in enumerate(atlas_patches):
                support[i,0]=np.expand_dims(p,axis=4).astype(np.float32)
            # support[0, 0]=np.expand_dims(atlas_patches[max_idx],axis=4).astype(np.float32)
            # support[1, 0]=np.expand_dims(atlas_patches[min_idx],axis=4).astype(np.float32)

            query[0,0]=np.expand_dims(target_patch,axis=4).astype(np.float32)

            y_n_labels[0,0]=binary_sim
            # y_n_labels[0,0]=[1,0]

            _, ls = sess.run([self.train_op, self.ce_loss], feed_dict={self.x: support, self.q: query, self.y_n_hot: y_n_labels})
            y_n_hot, predict= sess.run([self.y_n_hot, self.prob], feed_dict={self.x: support, self.q: query, self.y_n_hot: y_n_labels})
            # self.accV2(target_lab,atlas_labs,predict)
            sim_str = ['{:.2f}'.format(x) for x in sim]
            self.logger.debug(sim_str)
            self.logger.debug(predict)
            # if (epi + 1) %50 == 0:
            self.logger.debug('[ episode {}/{}] => loss: {:.3f}'.format(itr + 1, n_train_itr, ls))
        # print("avg_acc=%f" % (acc_avg / n_train_itr))
        # print("avg_acc=%f" % (mv_acc_avg / n_train_itr))
        self.logger.info("training time %s" % time.clock())
        save_path = saver.save(sess, self.args.model_dir[0], write_meta_graph=False)
        self.logger.info("save model to"+save_path)
        sess.close()

    def __slice_for_network(self, p_atlas_imgs, p_atlas_labs, dices, index=None):
        ret_atlas_img=[]
        ret_atlas_lab=[]
        ret_dices=[]

        if index is None:
            j=np.random.randint(p_atlas_imgs.shape[0])
        else:
            j=index
        for i in range(self.n_support_sample):
            ret_atlas_img.append(p_atlas_imgs[j][i])
            ret_atlas_lab.append(p_atlas_labs[j][i])
            ret_dices.append(dices[j][i])
        return ret_atlas_img,ret_atlas_lab,ret_dices

    def visulize(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        saver.restore(sess,self.args.model_dir[0])
        base_dir = self.args.test_atlas_target[0]
        sample_generator = [get_sample_generator(base_dir, target_id=i, args=self.args) for i in
                            [1017, 1018, 1019, 1020]]
        # sample_generator = [get_sample_generator(base_dir,target_id=i, args=self.args) for i in [1013]]

        time.clock()
        acc_avg = 0.0
        mv_acc_avg = 0.0
        for itr in range(n_train_itr):
            '''
            随机产生一个数组，包含0-n_classes,取期中n_way个类
            '''
            # atlas_patches, atlas_labs, target_patch, target_lab,sim = sample_generator[np.random.randint(0,len(sample_generator))].next_sample(
            #     num_nonlocal_patch=6, dupplicated=False)
            g = sample_generator[np.random.randint(0, len(sample_generator))]
            np_atlas_patches, np_atlas_labs, target_patch, target_lab, np_sim = g.next_sample(
                num_nonlocal_patch=self.n_support_sample, is_test=False)
            # check是否有问题
            atlas_patches, atlas_labs, sim = self.__slice_for_network(np_atlas_patches, np_atlas_labs, np_sim,0)

            binary_sim = [(1 if i > self.thres else 0) for i in sim]
            if np.sum(binary_sim) == 0 or np.sum(binary_sim) == len(binary_sim):
                continue
            # max_idx=np.argmax(sim)
            # min_idx=np.argmin(sim)
            support = np.zeros([self.n_support_sample, self.one_support] + self.patch_size + [self.channal],
                               dtype=np.float32)
            query = np.zeros([self.n_query_sample, self.one_query] + self.patch_size + [self.channal], dtype=np.float32)
            y_n_labels = np.zeros([self.n_query_sample, self.one_query, self.n_support_sample], np.uint8)

            for i, p in enumerate(atlas_patches):
                support[i, 0] = np.expand_dims(p, axis=4).astype(np.float32)
            # support[0, 0]=np.expand_dims(atlas_patches[max_idx],axis=4).astype(np.float32)
            # support[1, 0]=np.expand_dims(atlas_patches[min_idx],axis=4).astype(np.float32)

            query[0, 0] = np.expand_dims(target_patch, axis=4).astype(np.float32)

            y_n_labels[0, 0] = binary_sim
            # y_n_labels[0,0]=[1,0]

            # _, ls = sess.run([self.train_op, self.ce_loss],
            #                  feed_dict={self.x: support, self.q: query, self.y_n_hot: y_n_labels})
            y_n_hot, predict = sess.run([self.y_n_hot, self.prob],
                                        feed_dict={self.x: support, self.q: query, self.y_n_hot: y_n_labels})
            # self.accV2(target_lab,atlas_labs,predict)
            sim_str = ['{:.2f}'.format(x) for x in sim]
            self.logger.debug(sim_str)
            self.logger.debug(predict)
            mk_or_cleardir('../tmp/')
            self.__write_out_target(target_patch, target_lab)
            self.__write_out_atlas(np_atlas_patches[0], np_atlas_labs[0], np_sim[0])
            # if (epi + 1) %50 == 0:
            # self.logger.debug('[ episode {}/{}] => loss: {:.3f}'.format(itr + 1, n_train_itr, ls))
        # print("avg_acc=%f" % (acc_avg / n_train_itr))
        # print("avg_acc=%f" % (mv_acc_avg / n_train_itr))
        self.logger.info("training time %s" % time.clock())
        save_path = saver.save(sess, self.args.model_dir[0], write_meta_graph=False)
        self.logger.info("save model to" + save_path)
        sess.close()
    def test(self):
        print("test")
        sess=tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        saver.restore(sess,self.args.model_dir[0])
        base_dir = self.args.test_atlas_target[0]
        # base_dir = self.args.train_atlas_target[0]
        # sample_generator = [get_sample_generator(base_dir, target_id=i, args=self.args) for i in [1017, 1018, 1019, 1020]]
        # for i in [1013, 1014, 1015, 1016]:
        for i in [1017, 1018, 1019, 1020]:
        # for i in [1017]:
            g = get_sample_generator(base_dir,target_id=i, args=self.args)

            predict_target=np.zeros([3,96,96,96],dtype=np.uint16)
            gold_standard=g.target_lab
            for i in range(3):
                predict_target[i] = g.mv_predict_lab


                # sample_generator = get_sample_generator(target_id=1013)
            time.clock()
            acc=np.zeros([3],dtype=np.float32)
            all=[0.0,0.0,0.0]
            count=0
            while True:
                '''
                随机产生一个数组，包含0-n_classes,取期中n_way个类
                '''

                atlas_patches, atlas_labs, target_patch, target_lab,sim = g.next_sample(
                    num_nonlocal_patch=self.n_support_sample, is_test=True)
                if atlas_patches is None:
                    break
                pos=g.pos

                target_patch_center=target_lab[self.patch_size[0]//2,self.patch_size[1]//2,self.patch_size[2]//2]
                # atlas_patches=np.expand_dims(atlas_patches[0,:,:,:,:],axis=0)
                # atlas_labs=np.expand_dims(atlas_labs[0,:,:,:,:],axis=0)
                # sim=np.expand_dims(sim[0,:],axis=0)

                # MV的方法
                pred=self.acc_MV(target_lab,atlas_labs)

                acc[0] = acc[0] + ( 1 if pred==target_patch_center else 0 )
                predict_target[0][pos[0],pos[1],pos[2]]=pred



                # NLVW的方法
                pred=self.acc_nlvw(atlas_patches, atlas_labs, target_patch, target_lab)
                acc[1] = acc[1] + ( 1 if pred==target_patch_center else 0 )
                predict_target[1][pos[0],pos[1],pos[2]]=pred

                # Network
                pred, loss = self.acc_net(sess, atlas_labs, atlas_patches, sim, target_lab, target_patch)
                acc[2] = acc[2] + ( 1 if pred==target_patch_center else 0 )
                predict_target[2][pos[0],pos[1],pos[2]]=pred


                count=count+1
                # print(acc)
                if count%100==0:
                    self.logger.info(acc/count)
                # self.logger.debug(ac1, ac2, ac3)
                #
                # mk_or_cleardir('../tmp/')
                # self.__write_out_target(target_patch, target_lab)
                # self.__write_out_atlas(atlas_patches[0], atlas_labs[0], sim[0])

            # print("testing time %s" % time.clock())
            self.logger.info("mv_acc_avg =%f" % (acc[0] / count))
            self.logger.info("nlvw_acc_avg =%f" % (acc[1] / count))
            self.logger.info("patch embbeding avg_acc=%f" % (acc[2] / count))
            self.logger.info("final result")
            for i in range(3):
                method_i_acc=dice_compute(gold_standard, predict_target[i])
                self.logger.info(method_i_acc)
                all[i]=all[i]+method_i_acc
        # self.logger.info("all result"+all[i]/4.0)
        self.logger.info( all / 4.0)
        sess.close()



    def acc_net(self,sess, atlas_labs, atlas_patches, sim, target_lab, target_patch):
        # atlas_patches, atlas_labs, sim = self.__slice_for_network(atlas_patches, atlas_labs, sim)
        # support1 = np.zeros([self.n_support_sample, self.one_support] + self.patch_size + [self.channal],dtype=np.float32)
        # query1 = np.zeros([self.n_query_sample, self.one_query] + self.patch_size + [self.channal], dtype=np.float32)
        # y_n_labels1 = np.zeros([self.n_query_sample, self.one_query, self.n_support_sample], np.uint8)
        query = np.expand_dims(np.expand_dims(np.expand_dims(target_patch, axis=3),axis=0),axis=0).astype(np.float32)
        shape=atlas_patches.shape
        support=np.expand_dims(np.expand_dims(np.reshape(atlas_patches,[shape[0]*shape[1],shape[2],shape[3],shape[4]]),axis=4),axis=1).astype(np.float32)

        # for i, p in enumerate(atlas_patches):
        #     support[i, 0] = np.expand_dims(p, axis=4).astype(np.float32)
        shape=sim.shape
        sim=np.expand_dims(np.expand_dims(np.reshape(sim,[shape[0]*shape[1]]),axis=0),axis=0)

        # binary_sim = [(1 if i > self.thres else 0) for i in sim]
        # 全为0或者全为1
        # if np.sum(binary_sim)==0 or np.sum(binary_sim)==len(binary_sim):
        #     continue
        y_n_labels = np.where(sim>self.thres,1,0)
        predict, dist, loss = sess.run([self.prob, self.dist, self.ce_loss],
                                           feed_dict={self.x: support, self.q: query, self.y_n_hot: y_n_labels})
        # ac = self.acc_cum(target_lab, atlas_labs, )
        weight=predict[0][0]
        weight_str = ['{:.2f}'.format(x) for x in weight]
        shape=atlas_labs.shape
        weight=np.reshape(weight,[shape[0],shape[1]])
        dist=np.reshape(dist,[shape[0],shape[1]])
        # predict=self.argmax_contribution(atlas_labs,-dist,target_lab)

        unique_lab = np.unique(target_lab)
        p_c = self.patch_size[0] // 2
        target_center_lab = target_lab[p_c, p_c, p_c]
        atlas_center_lab=atlas_labs[:, :, p_c, p_c, p_c]
        lag_ind = self.largest_indices(-dist, 5)
        top_k_atlas_lab = [atlas_center_lab[i][j] for i, j in zip(lag_ind[0], lag_ind[1])]
        top_k_atlas_wei = [-dist[i][j] for i, j in zip(lag_ind[0], lag_ind[1])]
        weighted_votes = [np.sum(( top_k_atlas_lab== i).astype(np.float)) for i in unique_lab]
        predict = unique_lab[np.argmax(weighted_votes)]


        self.logger.debug(sim)
        self.logger.debug(dist)
        # self.logger.debug(predict)
        return predict, loss
