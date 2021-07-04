'''

'''
from fusion.layer import conv3d, deconv3d, max_pool3d, crop_and_concat, weight_xavier_init, bias_variable
import tensorflow as tf
import numpy as np
import os


def _create_conv_net(X, image_z, image_width, image_height, image_channel, drop_conv):
    inputX = tf.reshape(X, [-1, image_width, image_height, image_z, image_channel])  # shape=(?, 32, 32, 1)
    # UNet model
    # layer1->convolution
    W1_1 = weight_xavier_init(shape=[3, 3, 3, image_channel, 32], n_inputs=3 * 3 * 3 * image_channel, n_outputs=32)
    B1_1 = bias_variable([32])
    conv1_1 = tf.nn.dropout(tf.nn.relu(conv3d(inputX, W1_1) + B1_1), drop_conv)

    W1_2 = weight_xavier_init(shape=[3, 3, 3, 32, 32], n_inputs=3 * 3 * 3 * 32, n_outputs=32)
    B1_2 = bias_variable([32])
    conv1_2 = tf.nn.dropout(tf.nn.relu(conv3d(conv1_1, W1_2) + B1_2), drop_conv)

    pool1 = max_pool3d(conv1_2)
    # layer2->convolution
    W2_1 = weight_xavier_init(shape=[3, 3, 3, 32, 64], n_inputs=3 * 3 * 3 * 32, n_outputs=64)
    B2_1 = bias_variable([64])
    conv2_1 = tf.nn.dropout(tf.nn.relu(conv3d(pool1, W2_1) + B2_1), drop_conv)

    W2_2 = weight_xavier_init(shape=[3, 3, 3, 64, 64], n_inputs=3 * 3 * 3 * 64, n_outputs=64)
    B2_2 = bias_variable([64])
    conv2_2 = tf.nn.dropout(tf.nn.relu(conv3d(conv2_1, W2_2) + B2_2), drop_conv)

    pool2 = max_pool3d(conv2_2)

    # layer3->convolution
    W3_1 = weight_xavier_init(shape=[3, 3, 3, 64, 128], n_inputs=3 * 3 * 3 * 64, n_outputs=128)
    B3_1 = bias_variable([128])
    conv3_1 = tf.nn.dropout(tf.nn.relu(conv3d(pool2, W3_1) + B3_1), drop_conv)

    W3_2 = weight_xavier_init(shape=[3, 3, 3, 128, 128], n_inputs=3 * 3 * 3 * 128, n_outputs=128)
    B3_2 = bias_variable([128])
    conv3_2 = tf.nn.dropout(tf.nn.relu(conv3d(conv3_1, W3_2) + B3_2), drop_conv)

    pool3 = max_pool3d(conv3_2)

    # layer4->convolution
    W4_1 = weight_xavier_init(shape=[3, 3, 3, 128, 256], n_inputs=3 * 3 * 3 * 128, n_outputs=256)
    B4_1 = bias_variable([256])
    conv4_1 = tf.nn.dropout(tf.nn.relu(conv3d(pool3, W4_1) + B4_1), drop_conv)

    W4_2 = weight_xavier_init(shape=[3, 3, 3, 256, 256], n_inputs=3 * 3 * 3 * 256, n_outputs=256)
    B4_2 = bias_variable([256])
    conv4_2 = tf.nn.dropout(tf.nn.relu(conv3d(conv4_1, W4_2) + B4_2), drop_conv)

    pool4 = max_pool3d(conv4_2)

    # layer5->convolution
    W5_1 = weight_xavier_init(shape=[3, 3, 3, 256, 512], n_inputs=3 * 3 * 3 * 256, n_outputs=512)
    B5_1 = bias_variable([512])
    conv5_1 = tf.nn.dropout(tf.nn.relu(conv3d(pool4, W5_1) + B5_1), drop_conv)

    W5_2 = weight_xavier_init(shape=[3, 3, 3, 512, 512], n_inputs=3 * 3 * 3 * 512, n_outputs=512)
    B5_2 = bias_variable([512])
    conv5_2 = tf.nn.dropout(tf.nn.relu(conv3d(conv5_1, W5_2) + B5_2), drop_conv)

    # layer6->deconvolution
    W6 = weight_xavier_init(shape=[3, 3, 3, 256, 512], n_inputs=3 * 3 * 3 * 512, n_outputs=256)
    B6 = bias_variable([256])
    dconv1 = tf.nn.relu(deconv3d(conv5_2, W6) + B6)
    dconv_concat1 = crop_and_concat(conv4_2, dconv1)

    # layer7->convolution
    W7_1 = weight_xavier_init(shape=[3, 3, 3, 512, 256], n_inputs=3 * 3 * 3 * 512, n_outputs=256)
    B7_1 = bias_variable([256])
    conv7_1 = tf.nn.dropout(tf.nn.relu(conv3d(dconv_concat1, W7_1) + B7_1), drop_conv)

    W7_2 = weight_xavier_init(shape=[3, 3, 3, 256, 256], n_inputs=3 * 3 * 3 * 256, n_outputs=256)
    B7_2 = bias_variable([256])
    conv7_2 = tf.nn.dropout(tf.nn.relu(conv3d(conv7_1, W7_2) + B7_2), drop_conv)

    # layer8->deconvolution
    W8 = weight_xavier_init(shape=[3, 3, 3, 128, 256], n_inputs=3 * 3 * 3 * 256, n_outputs=128)
    B8 = bias_variable([128])
    dconv2 = tf.nn.relu(deconv3d(conv7_2, W8) + B8)
    dconv_concat2 = crop_and_concat(conv3_2, dconv2)

    # layer9->convolution
    W9_1 = weight_xavier_init(shape=[3, 3, 3, 256, 128], n_inputs=3 * 3 * 3 * 256, n_outputs=128)
    B9_1 = bias_variable([128])
    conv9_1 = tf.nn.dropout(tf.nn.relu(conv3d(dconv_concat2, W9_1) + B9_1), drop_conv)

    W9_2 = weight_xavier_init(shape=[3, 3, 3, 128, 128], n_inputs=3 * 3 * 3 * 128, n_outputs=128)
    B9_2 = bias_variable([128])
    conv9_2 = tf.nn.dropout(tf.nn.relu(conv3d(conv9_1, W9_2) + B9_2), drop_conv)

    # layer10->deconvolution
    W10 = weight_xavier_init(shape=[3, 3, 3, 64, 128], n_inputs=3 * 3 * 3 * 128, n_outputs=64)
    B10 = bias_variable([64])
    dconv3 = tf.nn.relu(deconv3d(conv9_2, W10) + B10)
    dconv_concat3 = crop_and_concat(conv2_2, dconv3)

    # layer11->convolution
    W11_1 = weight_xavier_init(shape=[3, 3, 3, 128, 64], n_inputs=3 * 3 * 3 * 128, n_outputs=64)
    B11_1 = bias_variable([64])
    conv11_1 = tf.nn.dropout(tf.nn.relu(conv3d(dconv_concat3, W11_1) + B11_1), drop_conv)

    W11_2 = weight_xavier_init(shape=[3, 3, 3, 64, 64], n_inputs=3 * 3 * 3 * 64, n_outputs=64)
    B11_2 = bias_variable([64])
    conv11_2 = tf.nn.dropout(tf.nn.relu(conv3d(conv11_1, W11_2) + B11_2), drop_conv)

    # layer 12->deconvolution
    W12 = weight_xavier_init(shape=[3, 3, 3, 32, 64], n_inputs=3 * 3 * 3 * 64, n_outputs=32)
    B12 = bias_variable([32])
    dconv4 = tf.nn.relu(deconv3d(conv11_2, W12) + B12)
    dconv_concat4 = crop_and_concat(conv1_2, dconv4)

    # layer 13->convolution
    W13_1 = weight_xavier_init(shape=[3, 3, 3, 64, 32], n_inputs=3 * 3 * 3 * 64, n_outputs=32)
    B13_1 = bias_variable([32])
    conv13_1 = tf.nn.dropout(tf.nn.relu(conv3d(dconv_concat4, W13_1) + B13_1), drop_conv)

    W13_2 = weight_xavier_init(shape=[3, 3, 3, 32, 32], n_inputs=3 * 3 * 3 * 32, n_outputs=32)
    B13_2 = bias_variable([32])
    conv13_2 = tf.nn.dropout(tf.nn.relu(conv3d(conv13_1, W13_2) + B13_2), drop_conv)

    # layer14->output
    W14 = weight_xavier_init(shape=[1, 1, 1, 32, 1], n_inputs=1 * 1 * 1 * 32, n_outputs=1)
    B14 = bias_variable([1])
    output_map = tf.nn.sigmoid(conv3d(conv13_2, W14) + B14)

    return output_map


# Serve data by batches
def _next_batch(train_images, train_labels, batch_size, index_in_epoch):
    start = index_in_epoch
    index_in_epoch += batch_size

    num_examples = train_images.shape[0]
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end], index_in_epoch


class unet3dModule(object):
    """
    A unet3d implementation

    :param image_height: number of height in the input image
    :param image_width: number of width in the input image
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param costname: name of the cost function.Default is "dice coefficient"
    """

    def __init__(self, image_z, image_height, image_width, channels=1, costname="dice coefficient"):
        self.image_z = image_z
        self.image_width = image_width
        self.image_height = image_height
        self.channels = channels
        self.X = tf.placeholder("float", shape=[None, image_z, image_height, image_width, channels])
        self.Y_gt = tf.placeholder("float", shape=[None, image_z, image_height, image_width, channels])
        self.lr = tf.placeholder('float')
        self.drop_conv = tf.placeholder('float')

        self.Y_pred = _create_conv_net(self.X, image_z, image_width, image_height, channels, self.drop_conv)
        self.cost = self.__get_cost(costname)

    def __get_cost(self, cost_name):
        if cost_name == "dice coefficient":
            smooth = 1e-7
            Z, H, W, C = self.Y_gt.get_shape().as_list()[1:]
            pred_flat = tf.reshape(self.Y_pred, [-1, H * W * Z])
            gt_flat = tf.reshape(self.Y_gt, [-1, H * W * Z])
            intersection = tf.reduce_sum(pred_flat * gt_flat, axis=1)
            denominator = tf.reduce_sum(gt_flat, axis=1) + tf.reduce_sum(pred_flat, axis=1)
            loss = (2.0 * intersection + smooth) / (denominator + smooth)
        return -tf.reduce_mean(loss)

    def train(self, train_images, train_lanbels, model_path, logs_path, learning_rate,
              dropout_conv=0.8, train_epochs=1000, batch_size=2):
        train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.all_variables())

        tf.summary.scalar("loss", self.cost)
        merged_summary_op = tf.summary.merge_all()
        sess = tf.InteractiveSession()
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        sess.run(init)

        DISPLAY_STEP = 1
        index_in_epoch = 0

        for i in range(train_epochs):
            # get new batch
            batch_xs_path, batch_ys_path, index_in_epoch = _next_batch(train_images, train_lanbels, batch_size,
                                                                       index_in_epoch)
            batch_xs = np.empty((len(batch_xs_path), self.image_z, self.image_height, self.image_width,
                                 self.channels))
            batch_ys = np.empty((len(batch_ys_path), self.image_z, self.image_height, self.image_width,
                                 self.channels))
            for num in range(len(batch_xs_path)):
                image = np.load(batch_xs_path[num])
                label = np.load(batch_ys_path[num])
                batch_xs[num, :, :, :, :] = np.reshape(image, (self.image_z, self.image_height, self.image_width,
                                                               self.channels))
                batch_ys[num, :, :, :, :] = np.reshape(label, (self.image_z, self.image_height, self.image_width,
                                                               self.channels))
            batch_xs = batch_xs.astype(np.float)
            batch_ys = batch_ys.astype(np.float)
            # Normalize from [0:255] => [0.0:1.0]
            batch_xs = np.multiply(batch_xs, 1.0 / 255.0)
            batch_ys = np.multiply(batch_ys, 1.0 / 255.0)
            # check progress on every 1st,2nd,...,10th,20th,...,100th... step
            if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:
                train_loss = self.cost.eval(feed_dict={self.X: batch_xs,
                                                       self.Y_gt: batch_ys,
                                                       self.lr: learning_rate,
                                                       self.drop_conv: dropout_conv})
                print('epochs %d training_loss => %.5f ' % (i, train_loss))

                if i % (DISPLAY_STEP * 10) == 0 and i:
                    DISPLAY_STEP *= 10

                    # train on batch
            _, summary = sess.run([train_op, merged_summary_op], feed_dict={self.X: batch_xs,
                                                                            self.Y_gt: batch_ys,
                                                                            self.lr: learning_rate,
                                                                            self.drop_conv: dropout_conv})
            summary_writer.add_summary(summary, i)
        summary_writer.close()

        save_path = saver.save(sess, model_path)
        print("Model saved in file:", save_path)

    def prediction(self, model_path, test_images):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(init)
        saver.restore(sess, model_path)
        # test_images size(imagez,imgewidth,imageheight,channels)
        test_images = np.reshape(test_images, (test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))
        y_dummy = test_images
        pred = sess.run(self.Y_pred, feed_dict={self.X: [test_images],
                                                self.Y_gt: [y_dummy],
                                                self.drop_conv: 1})
        result = pred.astype(np.float32) * 255.
        result = np.clip(result, 0, 255).astype('uint8')
        output = np.reshape(result, (test_images.shape[0], test_images.shape[1], test_images.shape[2]))
        return output
