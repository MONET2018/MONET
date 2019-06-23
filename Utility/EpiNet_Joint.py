import tensorflow as tf
import numpy as np
# import cpm_net
from GeometryUtility import *
from Epi_class import *


def kl_divergence(q_logits, p_logits):
    q_logits = tf.reduce_sum(q_logits, -1)
    p_logits = tf.reduce_sum(p_logits, -1)
    q = tf.nn.softmax(q_logits)
    kl = tf.reduce_sum(tf.reduce_sum(q * (tf.nn.log_softmax(q_logits) - tf.nn.log_softmax(p_logits)), -1))
    return kl


def CrossLossTest(input1, input2, loss_type):
    if loss_type == 'L2':
        return tf.nn.l2_loss((input1 - input2))
    elif loss_type == 'KL':
        return kl_divergence(input1, input2) * 10.0
    elif loss_type == 'KL_swapped':
        return kl_divergence(input2, input1) * 10.0
    elif loss_type == 'JS':
        return (kl_divergence(input1, input2) + kl_divergence(input2, input1)) * 10.0


class EpiNet_pair(object):
    def __init__(self, stages, joints, lr, lr_decay_rate, lr_decay_step, input_size, label_size, ext):
        self.stages = stages
        self.model_ref = CPM_Model(stages, joints + 1)
        self.model_src = CPM_Model(stages, joints + 1)
        self.model_labeled = CPM_Model(stages, joints + 1)
        self.lr_labeled = lr
        self.lr_unlabeled = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.nJoints = joints
        self.input_size = input_size
        self.label_size = label_size
        self.ext = ext

    def BuildLabeledCPM(self, batch_size, image_labeled, label, reuse):
        self.batch_size_labeled = batch_size
        self.image = image_labeled
        self.model_labeled.build_model(image_labeled, self.batch_size_labeled, reuse)
        self.model_labeled.build_loss(label, self.lr_labeled, self.lr_decay_rate, self.lr_decay_step)

    def BuildUnlabeledCPM(self, batch_size, image_ref, image_src, H1, H01, a, b, heatmap, confidence):
        self.batch_size_unlabeled = batch_size

        self.H01_inv = tf.placeholder(dtype=tf.float32, shape=(batch_size, 3, 3))

        self.image_ref = image_ref
        self.image_src = image_src

        self.heatmap_unlabeled = heatmap
        self.confidence = confidence

        self.H1 = H1
        self.H01 = H01

        self.a = a
        self.b = b

        self.model_ref.build_model(image_ref, batch_size, True)
        self.model_src.build_model(image_src, batch_size, True)

        self.pred01 = EpipolarTransferHeatmap_ext(self.model_src.stage_heatmap[self.model_src.stages - 1],
                                                  self.H1,
                                                  self.H01_inv,
                                                  self.a,
                                                  self.b,
                                                  self.ext)

    def TotalLoss(self):
        self.stage_loss = [0] * self.stages
        self.l2_loss = [0] * self.stages
        self.l2_loss_l = [0] * self.stages
        self.l2_loss_unlabeled = [0] * self.stages
        self.unimodality = [0] * self.stages
        self.joint_modality = [0] * self.stages
        self.loss_cross = 0
        self.loss_modality = 0
        self.loss_confidence = 0
        self.total_loss = 0

        ################################################
        ## Joint loss
        ################################################
        out_src = EpipolarTransferHeatmap_siamese_src(self.model_src.stage_heatmap[self.stages - 1][:, :, :, :-1],
                                                      self.H1,
                                                      self.a,
                                                      self.b,
                                                      self.ext)

        # self.out_src = tf.nn.l2_normalize(out_src, [1])
        modality_weight = 0.5#0.5
        unlabeled_weight = 3
        confidence_weight = 1
        self.out_src = out_src

        for stage in range(self.stages):
            out_ref = EpipolarTransferHeatmap_siamese_ref(self.model_ref.stage_heatmap[stage][:, :, :, :-1],
                                                          self.H01,
                                                          self.ext)

            # out_ref = tf.nn.l2_normalize(out_ref, [1])
            self.out_ref = out_ref


            ###############################################################################################################
            ## Loss Type Selection for Cross Loss                           ###############################################
            ## L2: L2 Loss                                                  ###############################################
            ## KL: Kullback Leibler divergence                              ###############################################
            ## KL_swapped: Kullback Leibler divergence with swapped inputs  ###############################################
            ## JS: Jensen Shannon divergence                                ###############################################
            self.l2_loss[stage] = unlabeled_weight * CrossLossTest(out_ref, self.out_src, 'JS') / self.batch_size_unlabeled
            ###############################################################################################################

            unimodality = tf.reduce_max(out_ref, axis=[1])  # -tf.reduce_min(heatmap, axis=[1, 2])


            self.joint_modality = unimodality

            self.unimodality[stage] = -modality_weight * tf.nn.l2_loss(unimodality) / self.batch_size_unlabeled



            self.l2_loss_unlabeled[stage] = confidence_weight * tf.nn.l2_loss(
                (self.model_ref.stage_heatmap[stage] - self.heatmap_unlabeled)) / self.batch_size_unlabeled

        for stage in range(self.stages):
            self.loss_cross += self.l2_loss[stage]
            self.loss_modality += self.unimodality[stage]
            self.loss_confidence += self.l2_loss_unlabeled[stage]

        self.total_loss =  self.model_labeled.total_loss + self.loss_confidence +self.loss_cross# + self.loss_modality
        # self.total_loss = self.loss_cross + self.model_labeled.total_loss

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

    def build_model(self, image, batch_size, reuse):
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
                                         name='conv1_1',
                                        reuse=reuse)
            conv1_2 = tf.layers.conv2d(inputs=conv1_1,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv1_2',
                                        reuse=reuse)
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
                                         name='conv2_1',
                                        reuse=reuse)
            conv2_2 = tf.layers.conv2d(inputs=conv2_1,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv2_2',
                                        reuse=reuse)
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
                                         name='conv3_1',
                                        reuse=reuse)
            conv3_2 = tf.layers.conv2d(inputs=conv3_1,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv3_2',
                                        reuse=reuse)
            conv3_3 = tf.layers.conv2d(inputs=conv3_2,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv3_3',
                                        reuse=reuse)
            conv3_4 = tf.layers.conv2d(inputs=conv3_3,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv3_4',
                                        reuse=reuse)
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
                                         name='conv4_1',
                                        reuse=reuse)
            conv4_2 = tf.layers.conv2d(inputs=conv4_1,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='conv4_2',
                                        reuse=reuse)
            conv4_3 = tf.layers.conv2d(inputs=conv4_2,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='conv4_3',
                                        reuse=reuse)
            conv4_4 = tf.layers.conv2d(inputs=conv4_3,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='conv4_4',
                                        reuse=reuse)
            conv5_1 = tf.layers.conv2d(inputs=conv4_4,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='conv5_1',
                                        reuse=reuse)
            conv5_2 = tf.layers.conv2d(inputs=conv5_1,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='conv5_2',
                                        reuse=reuse)
            self.sub_stage_img_feature = tf.layers.conv2d(inputs=conv5_2,
                                                          filters=128,
                                                          kernel_size=[3, 3],
                                                          strides=[1, 1],
                                                          padding='same',
                                                          activation=tf.nn.relu,
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                          name='conv5_3_CPM',
                                        reuse=reuse)

        with tf.variable_scope('stage_1'):
            conv1 = tf.layers.conv2d(inputs=self.sub_stage_img_feature,
                                     filters=512,
                                     kernel_size=[1, 1],
                                     strides=[1, 1],
                                     padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv6_1_CPM',
                                        reuse=reuse)
            self.stage_heatmap.append(tf.layers.conv2d(inputs=conv1,
                                                       filters=self.joints,
                                                       kernel_size=[1, 1],
                                                       strides=[1, 1],
                                                       padding='same',
                                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                       name='conv6_2_CPM',
                                                       reuse=reuse))
        for stage in range(2, self.stages + 1):
            self._middle_conv(stage, reuse)

    def _middle_conv(self, stage, reuse):
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
                                         name='Mconv1_stage' + str(stage),
                                        reuse=reuse)
            mid_conv2 = tf.layers.conv2d(inputs=mid_conv1,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv2_stage' + str(stage),
                                        reuse=reuse)
            mid_conv3 = tf.layers.conv2d(inputs=mid_conv2,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv3_stage' + str(stage),
                                        reuse=reuse)
            mid_conv4 = tf.layers.conv2d(inputs=mid_conv3,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv4_stage' + str(stage),
                                        reuse=reuse)
            mid_conv5 = tf.layers.conv2d(inputs=mid_conv4,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv5_stage' + str(stage),
                                        reuse=reuse)
            mid_conv6 = tf.layers.conv2d(inputs=mid_conv5,
                                         filters=128,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv6_stage' + str(stage),
                                        reuse=reuse)
            self.current_heatmap = tf.layers.conv2d(inputs=mid_conv6,
                                                    filters=self.joints,
                                                    kernel_size=[1, 1],
                                                    strides=[1, 1],
                                                    padding='same',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name='Mconv7_stage' + str(stage),
                                        reuse=reuse)
            self.stage_heatmap.append(self.current_heatmap)

    def build_loss_unsupervised(self, gt_heatmap, lr, lr_decay_rate, lr_decay_step):
        self.gt_heatmap = gt_heatmap
        self.total_loss = 0
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step

        self.gt_heatmap = tf.nn.l2_normalize(self.gt_heatmap, [1, 2])

        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
                self.stage_heatmap[stage] = tf.nn.l2_normalize(self.stage_heatmap[stage], [1, 2])

                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap,
                                                       name='l2_loss') / self.batch_size
            tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]
            tf.summary.scalar('total loss', self.total_loss)


        self.merged_summary = tf.summary.merge_all()

    def train(self):
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

    def build_loss(self, gt_heatmap, lr, lr_decay_rate, lr_decay_step):
        self.gt_heatmap = gt_heatmap
        self.total_loss = 0
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step

        # self.gt_heatmap = tf.nn.l2_normalize(self.gt_heatmap, [1, 2])

        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
                # self.stage_heatmap[stage] = tf.nn.l2_normalize(self.stage_heatmap[stage], [1, 2])

                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap,
                                                       name='l2_loss') / self.batch_size
            tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]
            tf.summary.scalar('total loss', self.total_loss)

        # with tf.variable_scope('train'):
        #     self.global_step = tf.contrib.framework.get_or_create_global_step()
        #
        #     self.lr = tf.train.exponential_decay(self.learning_rate,
        #                                          global_step=self.global_step,
        #                                          decay_rate=self.lr_decay_rate,
        #                                          decay_steps=self.lr_decay_step)
        #     tf.summary.scalar('learning rate', self.lr)
        #
        #     self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
        #                                                     global_step=self.global_step,
        #                                                     learning_rate=self.lr,
        #                                                     optimizer='Adam')
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
