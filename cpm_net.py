import tensorflow as tf
import numpy as np


class CPM_Model(object):
    def __init__(self, stages, joints):
        self.stages = stages
        self.stage_heatmap = []
        self.stage_loss = [0] * stages
        self.total_loss = 0
        self.image = None
        self.center_map = None
        self.gt_heatmap = None
        self.learning_rate = 0
        self.merged_summary = None
        self.joints = joints
        self.batch_size = 0

    def build_model(self, image, batch_size):
        self.batch_size = batch_size
        self.input_image = image
        with tf.variable_scope('sub_stages'):
            conv1_1 = tf.layers.conv2d(inputs=image,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv1_1')
            conv1_2 = tf.layers.conv2d(inputs=conv1_1,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv1_2')
            pool1_stage1 = tf.layers.max_pooling2d(inputs=conv1_2,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='same',
                                                name='pool1_stage1')
            conv2_1 = tf.layers.conv2d(inputs=pool1_stage1,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv2_1')
            conv2_2 = tf.layers.conv2d(inputs=conv2_1,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv2_2')
            pool2_stage1 = tf.layers.max_pooling2d(inputs=conv2_2,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='same',
                                                name='pool2_stage1')
            conv3_1 = tf.layers.conv2d(inputs=pool2_stage1,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv3_1')
            conv3_2 = tf.layers.conv2d(inputs=conv3_1,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv3_2')
            conv3_3 = tf.layers.conv2d(inputs=conv3_2,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv3_3')
            conv3_4 = tf.layers.conv2d(inputs=conv3_3,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv3_4')
            pool3_stage1 = tf.layers.max_pooling2d(inputs=conv3_4,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='same',
                                                name='pool3_stage1')
            conv4_1 = tf.layers.conv2d(inputs=pool3_stage1,
                                         filters=512,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv4_1')
            conv4_2 = tf.layers.conv2d(inputs=conv4_1,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='conv4_2')
            conv4_3 = tf.layers.conv2d(inputs=conv4_2,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='conv4_3')
            conv4_4 = tf.layers.conv2d(inputs=conv4_3,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='conv4_4')
            conv5_1 = tf.layers.conv2d(inputs=conv4_4,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='conv5_1')
            conv5_2 = tf.layers.conv2d(inputs=conv5_1,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='conv5_2')
            self.sub_stage_img_feature = tf.layers.conv2d(inputs=conv5_2,
                                                          filters=128,
                                                          kernel_size=[3, 3],
                                                          strides=[1, 1],
                                                          padding='same',
                                                          activation=tf.nn.relu,
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                          name='conv5_3_CPM')

        with tf.variable_scope('stage_1'):
            conv1 = tf.layers.conv2d(inputs=self.sub_stage_img_feature,
                                     filters=512,
                                     kernel_size=[1, 1],
                                     strides=[1, 1],
                                     padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv6_1_CPM')
            self.stage_heatmap.append(tf.layers.conv2d(inputs=conv1,
                                                       filters=self.joints,
                                                       kernel_size=[1, 1],
                                                       strides=[1, 1],
                                                       padding='same',
                                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                       name='conv6_2_CPM'))
        for stage in range(2, self.stages + 1):
            self._middle_conv(stage)

    def _middle_conv(self, stage):
        with tf.variable_scope('stage_' + str(stage)):
            self.current_featuremap = tf.concat([self.stage_heatmap[stage - 2],
                                                 self.sub_stage_img_feature,
                                                 ],
                                                axis=3)
            mid_conv1 = tf.layers.conv2d(inputs=self.current_featuremap,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv1_stage' + str(stage))
            mid_conv2 = tf.layers.conv2d(inputs=mid_conv1,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv2_stage' + str(stage))
            mid_conv3 = tf.layers.conv2d(inputs=mid_conv2,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv3_stage' + str(stage))
            mid_conv4 = tf.layers.conv2d(inputs=mid_conv3,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv4_stage' + str(stage))
            mid_conv5 = tf.layers.conv2d(inputs=mid_conv4,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv5_stage' + str(stage))
            mid_conv6 = tf.layers.conv2d(inputs=mid_conv5,
                                         filters=128,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv6_stage' + str(stage))
            self.current_heatmap = tf.layers.conv2d(inputs=mid_conv6,
                                                    filters=self.joints,
                                                    kernel_size=[1, 1],
                                                    strides=[1, 1],
                                                    padding='same',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name='Mconv7_stage' + str(stage))
            self.stage_heatmap.append(self.current_heatmap)

    def build_loss(self, gt_heatmap, lr, lr_decay_rate, lr_decay_step):
        self.gt_heatmap = gt_heatmap
        self.total_loss = 0
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step

        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap,
                                                       name='l2_loss') / self.batch_size
            tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]
            tf.summary.scalar('total loss', self.total_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.contrib.framework.get_or_create_global_step()

            self.lr = tf.train.exponential_decay(self.learning_rate,
                                                 global_step=self.global_step,
                                                 decay_rate=self.lr_decay_rate,
                                                 decay_steps=self.lr_decay_step)
            tf.summary.scalar('learning rate', self.lr)

            self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.lr,
                                                            optimizer='Adam')
        self.merged_summary = tf.summary.merge_all()

    def load_weights_from_file(self, weight_file_path, sess, finetune=True):
        weights = np.load(weight_file_path).item()

        with tf.variable_scope('', reuse=True):
            ## Pre stage conv
            # conv1
            for layer in range(1, 3):
                conv_kernel = tf.get_variable('sub_stages/conv1_' + str(layer) + '/kernel')
                conv_bias = tf.get_variable('sub_stages/conv1_' + str(layer) + '/bias')

                loaded_kernel = weights['conv1_' + str(layer)]['weights']
                loaded_bias = weights['conv1_' + str(layer)]['biases']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv2
            for layer in range(1, 3):
                conv_kernel = tf.get_variable('sub_stages/conv2_' + str(layer) + '/kernel')
                conv_bias = tf.get_variable('sub_stages/conv2_' + str(layer) + '/bias')

                loaded_kernel = weights['conv2_' + str(layer)]['weights']
                loaded_bias = weights['conv2_' + str(layer)]['biases']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv3
            for layer in range(1, 5):
                conv_kernel = tf.get_variable('sub_stages/conv3_' + str(layer) + '/kernel')
                conv_bias = tf.get_variable('sub_stages/conv3_' + str(layer) + '/bias')

                loaded_kernel = weights['conv3_' + str(layer)]['weights']
                loaded_bias = weights['conv3_' + str(layer)]['biases']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv4
            for layer in range(1, 5):
                conv_kernel = tf.get_variable('sub_stages/conv4_' + str(layer) + '/kernel')
                conv_bias = tf.get_variable('sub_stages/conv4_' + str(layer) + '/bias')

                loaded_kernel = weights['conv4_' + str(layer)]['weights']
                loaded_bias = weights['conv4_' + str(layer)]['biases']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv5
            for layer in range(1, 3):
                conv_kernel = tf.get_variable('sub_stages/conv5_' + str(layer) + '/kernel')
                conv_bias = tf.get_variable('sub_stages/conv5_' + str(layer) + '/bias')

                loaded_kernel = weights['conv5_' + str(layer)]['weights']
                loaded_bias = weights['conv5_' + str(layer)]['biases']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv5_3_CPM
            conv_kernel = tf.get_variable('sub_stages/sub_stage_img_feature/kernel')
            conv_bias = tf.get_variable('sub_stages/sub_stage_img_feature/bias')

            loaded_kernel = weights['conv5_3_CPM']['weights']
            loaded_bias = weights['conv5_3_CPM']['biases']

            sess.run(tf.assign(conv_kernel, loaded_kernel))
            sess.run(tf.assign(conv_bias, loaded_bias))

            ## stage 1
            conv_kernel = tf.get_variable('stage_1/conv1/kernel')
            conv_bias = tf.get_variable('stage_1/conv1/bias')

            loaded_kernel = weights['conv6_1_CPM']['weights']
            loaded_bias = weights['conv6_1_CPM']['biases']

            sess.run(tf.assign(conv_kernel, loaded_kernel))
            sess.run(tf.assign(conv_bias, loaded_bias))

            if finetune != True:
                conv_kernel = tf.get_variable('stage_1/stage_heatmap/kernel')
                conv_bias = tf.get_variable('stage_1/stage_heatmap/bias')

                loaded_kernel = weights['conv6_2_CPM']['weights']
                loaded_bias = weights['conv6_2_CPM']['biases']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

                ## stage 2 and behind
                for stage in range(2, self.stages + 1):
                    for layer in range(1, 8):
                        conv_kernel = tf.get_variable('stage_' + str(stage) + '/mid_conv' + str(layer) + '/kernel')
                        conv_bias = tf.get_variable('stage_' + str(stage) + '/mid_conv' + str(layer) + '/bias')

                        loaded_kernel = weights['Mconv' + str(layer) + '_stage' + str(stage)]['weights']
                        loaded_bias = weights['Mconv' + str(layer) + '_stage' + str(stage)]['biases']

                        sess.run(tf.assign(conv_kernel, loaded_kernel))
                        sess.run(tf.assign(conv_bias, loaded_bias))
