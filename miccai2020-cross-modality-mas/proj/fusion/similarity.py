import numpy as np
import tensorflow as tf
import MAS.layers as layers
from dataprocessor.sampler import PatchSampler, get_sample_generator
import glob
import os
import time
class SimilarityNet():
    def __init__(self,config):
        self.config = config

        self.input_data_shape=[7]*3
        self.__build_model()

    def __network(self,atlas_p,target_p):
        '''
        定义网络结构
        :param atlas_p:
        :param target_p:
        :return:
        '''
        # self.input_layer = tf.concat([target_p, atlas_p], axis=4)
        # h1=layers.conv3_relu_pool_block(self.input_layer,2,4,name="cv_1")
        # h2=layers.conv3_relu_pool_block(h1,4,8,name="cv_2")
        # h3=layers.fully_connected(h2,16,name='fc_1')
        # h3=tf.nn.relu(h3)
        # h4=layers.fully_connected(h3,2,name='fc_2')
        # h4=tf.nn.softmax(h4)
        self.input_layer = tf.concat([target_p, atlas_p], axis=4)
        net=tf.layers.conv3d(self.input_layer,8,3,name="cv_1",padding="SAME")
        net=tf.nn.relu(net)
        net=tf.layers.max_pooling3d(net,2,2)
        net = tf.layers.conv3d(net, 16, 3, name="cv_2",padding="SAME")
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling3d(net, 2,2)
        net=tf.layers.flatten(net)
        net=tf.layers.dense(net,16,name='fc_1')
        net=tf.nn.relu(net)
        net = tf.layers.dense(net, 2, name='fc_2')

        return net

    def __build_model(self):
        '''
        定义网络的输入，输出，loss
        :return:
        '''
        self.ph_atlas_p = tf.placeholder(tf.float32,
                                        [self.config['Train']['minibatch_size']] + self.input_data_shape + [1])
        self.ph_target_p = tf.placeholder(tf.float32,
                                         [self.config['Train']['minibatch_size']] + self.input_data_shape + [1])
        self.ph_sim_lab = tf.placeholder(tf.float32,
                                         [self.config['Train']['minibatch_size']]+[2])

        self.out =self.__network(self.ph_atlas_p, self.ph_target_p)
        self.p_y= tf.nn.softmax(self.out)

        # self.one_hot_lab=tf.one_hot(self.ph_sim_lab, depth=2)
        self.ce_loss=-tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(tf.to_float(self.ph_sim_lab), tf.log(self.p_y)), -1), [-1]))
        # self.ce_loss=tf.losses.softmax_cross_entropy(self.one_hot_lab,self.out)
        self.train_op = tf.train.AdamOptimizer().minimize(self.ce_loss)
        # self.acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(self.p_y, axis=-1), self.ph_sim_lab)), name='acc')

    def __get_sample_index(self,sim):


        # need=np.random.randint(0,2)
        for i,s in enumerate(sim):
            if s==self.need:
                self.need=not self.need
                return i
        return None





    def train(self):
        # initialize all variables
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # saver to save model
        self.saver = tf.train.Saver()
        # summary writer
        self.writer = tf.summary.FileWriter(os.path.dirname(self.config['Train']['file_model_save']), sess.graph)
        # ckpt = tf.train.get_checkpoint_state(os.path.dirname(vote_config['Train']['file_model_save']))
        # if ckpt and ckpt.model_checkpoint_path:
        #     # saver.restore(sess,ckpt.model_checkpoint_path)
        #     saver.restore(sess, vote_config['Inference']['file_model_saved'] + str(2500))
        #     print("Model restored...")
        # else:
        #     print('No Model')
        #tensorflow 的调试
        #sess=tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.graph.finalize()#让graph保持不变，防止内存溢出
        # ps = [get_sample_generator(target_id=i) for i in [1013, 1014, 1015, 1016]]
        ps = [get_sample_generator(target_id=i) for i in [1015]]
        sample_count=0
        self.need=0
        avg_acc=0.0
        for step in range(self.config['Train']['total_iterations']):
            # pass

            atlas_img_patches, atlas_labs, target_img_patch, target_lab,sim = ps[np.random.randint(0,len(ps))].next_sample(
                is_test=False)
            # np.expand_dims(np.stack(data, axis=0), axis=4).astype(np.float32)
            lab=[(True if i>0.8 else False) for i in sim]
            index=self.__get_sample_index(lab)
            if index==None:
                continue

            sim = [(1 if i > 0.8 else 0) for i in sim]

            feed_img_patch=np.expand_dims(np.stack([target_img_patch],axis=0),axis=4).astype(np.float32)
            feed_atlas_patch=np.expand_dims(np.stack([atlas_img_patches[index]],axis=0),axis=4).astype(np.float32)
            feed_sim_lab=np.stack([[sim[index],1-sim[index]]],axis=0).astype(np.float32)
            trainFeed = {self.ph_target_p: feed_img_patch,
                         self.ph_atlas_p: feed_atlas_patch,
                         self.ph_sim_lab:feed_sim_lab}

            sess.run(self.train_op, feed_dict=trainFeed)
            loss,p_y=sess.run([self.ce_loss,self.p_y],feed_dict=trainFeed)
            acc=0.0
            # avg_acc=avg_acc+acc
            # print("step:"+str(step))

            # one_hot_lab,out,p_y=sess.run([self.one_hot_lab, self.out,self.p_y], feed_dict=trainFeed)
            print([sim[index],1-sim[index]])
            if sim[index]>1-sim[index]:
                sample_count=sample_count+1
            else:
                sample_count = sample_count - 1
            print("loss: %f " % (loss))
        # print(avg_acc/self.config['Train']['total_iterations'])
        save_path = self.saver.save(sess, self.config['Train']['file_model_save'], write_meta_graph=False)
        print("save model to"+save_path)
        print("pos %d",sample_count)
        sess.close()


    def test(self):
        print("test")
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, self.config['Train']['file_model_save'])
        base_dir = "../../data_vote_man/MR_CT/test/model_24000/*ct_train_%s_image_A_T/"
        ps = [get_sample_generator(target_id=i) for i in [1016]]
        # sample_generator = get_sample_generator(target_id=1014)
        time.clock()

        n_test_itr=2500
        for itr in range(n_test_itr):
            '''
            随机产生一个数组，包含0-n_classes,取期中n_way个类
            '''
            atlas_img_patches, atlas_labs, target_img_patch, target_lab, sim = \
                ps[np.random.randint(0, len(ps))].next_sample(is_test=False)
            # np.expand_dims(np.stack(data, axis=0), axis=4).astype(np.float32)

            index = np.random.randint(0, len(atlas_img_patches))
            feed_img_patch = np.expand_dims(np.stack([target_img_patch], axis=0), axis=4).astype(np.float32)
            feed_atlas_patch = np.expand_dims(np.stack([atlas_img_patches[index]], axis=0), axis=4).astype(np.float32)

            binary_sim = [(1 if i > 0.8 else 0) for i in sim]
            feed_sim_lab = np.stack([[binary_sim[index], 1 - binary_sim[index]]], axis=0).astype(np.float32)
            testFeed = {self.ph_target_p: feed_img_patch,
                         self.ph_atlas_p: feed_atlas_patch,
                         self.ph_sim_lab: feed_sim_lab}

            loss,p_y=sess.run([self.ce_loss,self.p_y],feed_dict=testFeed)
            acc = 0.0
            # avg_acc=avg_acc+acc
            # print("step:"+str(step))

            # one_hot_lab,out,p_y=sess.run([self.one_hot_lab, self.out,self.p_y], feed_dict=trainFeed)
            # print(one_hot_lab)
            # print(p_y)
            print([sim[index],1-sim[index]])
            print(p_y)
            print("loss: %f "%(loss))

        # print(avg_acc/self.config['Train']['total_iterations'])
        sess.close()


