import tensorflow as tf
import numpy as np
# import cpm_net
from GeometryUtility import *
from Epi_class import *

class EpiNet_pair(object):
    def __init__(self, stages, joints, limbs, lr, lr_decay_rate, lr_decay_step, input_size, label_size, ext):
        self.stages = stages
        self.model_ref = CPM_Model(stages, joints + 1, limbs)
        self.model_src = CPM_Model(stages, joints + 1, limbs)
        self.model_labeled = CPM_Model(stages, joints + 1, limbs)
        self.lr_labeled = lr
        self.lr_unlabeled = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.nJoints = joints
        self.nLimbs = limbs
        self.input_size = input_size
        self.label_size = label_size
        self.ext = ext
        # self.step = tf.placeholder(dtype=tf.int32, shape=(1))

    def BuildLabeledCPM(self, batch_size, image_labeled, label, reuse):
        self.batch_size_labeled = batch_size
        self.image = image_labeled
        self.model_labeled.build_model(image_labeled, self.batch_size_labeled, reuse)
        self.model_labeled.build_loss(label, self.lr_labeled, self.lr_decay_rate, self.lr_decay_step)

    def BuildUnlabeledCPM(self, batch_size, image_ref, image_src, H1, H01, a, b):
        self.batch_size_unlabeled = batch_size

        self.H01_inv = tf.placeholder(dtype=tf.float32, shape=(batch_size, 3, 3))

        self.image_ref = image_ref
        self.image_src = image_src

        self.H1 = H1
        self.H01 = H01

        self.a = a
        self.b = b

        self.model_ref.build_model(image_ref, batch_size, True)
        self.model_src.build_model(image_src, batch_size, True)
        #
        # self.out_ref = EpipolarTransferHeatmap_siamese_ref(self.model_src1.stage_joint_heatmap[self.model_src1.stages - 1],
        #                                                   H_ref_placeholder,
        #                                                   heatmap_extension_length)
        #
        # self.out_src = EpipolarTransferHeatmap_siamese_src(heatmap_src_placeholder,
        #                                               H_src_placeholder,
        #                                               a_placeholder,
        #                                               b_placeholder,
        #                                               heatmap_extension_length)
        #
        #
        self.pred01 = EpipolarTransferHeatmap_ext(self.model_src.stage_joint_heatmap[self.model_src.stages - 1],
                                                  self.H1,
                                                  self.H01_inv,
                                                  self.a,
                                                  self.b,
                                                  self.ext)
        #
        # self.pred02 = EpipolarTransferHeatmap_ext(self.model_src2.stage_joint_heatmap[self.model_src2.stages - 1],
        #                                           self.H2,
        #                                           self.H02,
        #                                           self.a2,
        #                                           self.b2,
        #                                           self.ext)
        #
        # self.pred0 = self.pred01 * self.pred02

        # self.pred01_l = EpipolarTransferHeatmap_ext(self.model_src1.stage_limb_heatmap[self.model_src1.stages - 1],
        #                                           self.H1,
        #                                           self.H01,
        #                                           self.a1,
        #                                           self.b1,
        #                                           self.ext)
        #
        # self.pred02_l = EpipolarTransferHeatmap_ext(self.model_src2.stage_limb_heatmap[self.model_src1.stages - 1],
        #                                           self.H2,
        #                                           self.H02,
        #                                           self.a2,
        #                                           self.b2,
        #                                           self.ext)
        #
        # self.pred0_l = self.pred01_l * self.pred02_l

        # self.pred0_l = self.model_src1.stage_limb_heatmap[self.model_src1.stages - 1]

        # self.pred0 = tf.nn.l2_normalize(self.pred0, [1, 2])

        # for stage in range(self.model_ref.stages):
        #    with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
        #        self.model_ref.stage_heatmap[stage] = tf.nn.l2_normalize(self.model_ref.stage_heatmap[stage], [1, 2])


        # self.model_ref.joint_modality = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.nJoints))
        # self.confident_joint = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.nJoints))
        # self.model_ref.build_loss_unsupervised_slim(self.pred0, self.model_src1.stage_joint_heatmap, self.model_src2.stage_joint_heatmap, self.pred0_l, self.lr_unlabeled, self.lr_decay_rate, self.lr_decay_step)

        # self.model_ref.total_loss += tf.nn.l2_loss(self.model_ref.stage_limb_heatmap[4])
        # self.model_ref.total_loss += tf.nn.l2_loss(self.model_src1.stage_limb_heatmap[4])
        # self.model_ref.total_loss += tf.nn.l2_loss(self.model_src2.stage_limb_heatmap[4])

        # with tf.variable_scope('train'):
        #     self.global_step = tf.train.get_or_create_global_step()
        #     self.global_step0 = self.global_step

    def TotalLoss(self):
        self.stage_loss = [0] * self.stages
        self.l2_loss = [0] * self.stages
        self.l2_loss_l = [0] * self.stages
        self.l2_loss_unlabeled = [0] * self.stages
        self.unimodality = [0] * self.stages
        self.loss_cross = 0
        self.loss_modality = 0
        self.total_loss = 0

        ################################################
        ## Joint loss
        ################################################
        out_src = EpipolarTransferHeatmap_siamese_src(self.model_src.stage_joint_heatmap[self.stages-1][:, :, :, :-1],
                                                           self.H1,
                                                           self.a,
                                                           self.b,
                                                           self.ext)

        self.out_src = tf.nn.l2_normalize(out_src, [1])
        self.out_src = out_src
        for stage in range(self.stages):
            out_ref = EpipolarTransferHeatmap_siamese_ref(self.model_ref.stage_joint_heatmap[stage][:, :, :, :-1],
                                                          self.H01,
                                                          self.ext)

            out_ref = tf.nn.l2_normalize(out_ref, [1])
            self.out_ref = out_ref

            self.l2_loss[stage] = 2*tf.nn.l2_loss((out_ref - self.out_src)) / self.batch_size_unlabeled

            heatmap = tf.nn.l2_normalize(self.model_ref.stage_joint_heatmap[stage][:, :, :, :-1], [1, 2])
            unimodality = tf.reduce_max(heatmap, axis=[1, 2])  # -tf.reduce_min(heatmap, axis=[1, 2])
            unimodality = tf.nn.sigmoid(10 * (unimodality - 0.5))
            self.joint_modality = unimodality

            self.unimodality[stage] = -tf.nn.l2_loss(unimodality) / self.batch_size_unlabeled

            heatmap = tf.nn.l2_normalize(self.model_src.stage_joint_heatmap[stage][:, :, :, :-1], [1, 2])
            unimodality = tf.reduce_max(heatmap, axis=[1, 2])  # -tf.reduce_min(heatmap, axis=[1, 2])
            unimodality = tf.nn.sigmoid(10 * (unimodality - 0.5))

            self.unimodality[stage] += -tf.nn.l2_loss(unimodality) / self.batch_size_unlabeled
            # self.stage_loss[stage] = self.l2_loss[stage] + self.unimodality[stage]


        ################################################
        ## Limb loss
        ################################################
        out_src_l = EpipolarTransferHeatmap_siamese_src(self.model_src.stage_limb_heatmap[self.stages-1],
                                                           self.H1,
                                                           self.a,
                                                           self.b,
                                                           self.ext)

        # self.out_src = tf.nn.l2_normalize(out_src, [1])
        self.out_src_l = out_src_l
        for stage in range(self.stages):
            out_ref_l = EpipolarTransferHeatmap_siamese_ref(self.model_ref.stage_limb_heatmap[stage],
                                                          self.H01,
                                                          self.ext)

            # out_ref = tf.nn.l2_normalize(out_ref, [1])

            self.l2_loss_l[stage] = 0.5*tf.nn.l2_loss((out_ref_l - self.out_src_l)) / self.batch_size_unlabeled

        for stage in range(self.stages):
            self.loss_cross += self.l2_loss[stage] + self.l2_loss_l[stage]
            self.loss_modality += self.unimodality[stage]

        self.total_loss = self.loss_cross + self.loss_modality + self.model_labeled.total_loss
        # self.total_loss = self.loss_cross + self.model_labeled.total_loss


class EpiNet_multi(object):
    def __init__(self, stages, joints, limbs, lr, lr_decay_rate, lr_decay_step, input_size, label_size, ext):
        self.model_ref = CPM_Model(stages, joints + 1, limbs)
        self.model_src1 = CPM_Model(stages, joints + 1, limbs)
        self.model_src2 = CPM_Model(stages, joints + 1, limbs)
        self.model_labeled = CPM_Model(stages, joints + 1, limbs)
        self.lr_labeled = lr
        self.lr_unlabeled = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.nJoints = joints
        self.nLimbs = limbs
        self.input_size = input_size
        self.label_size = label_size
        self.ext = ext
        # self.step = tf.placeholder(dtype=tf.int32, shape=(1))

    def BuildLabeledCPM(self, batch_size, image_labeled, label, reuse):
        self.batch_size_labeled = batch_size
        self.image = image_labeled
        self.model_labeled.build_model(image_labeled, self.batch_size_labeled, reuse)
        self.model_labeled.build_loss(label, self.lr_labeled, self.lr_decay_rate, self.lr_decay_step)

    def BuildUnlabeledCPM(self, batch_size, image_ref, image_src1, image_src2, H1, H01, H2, H02, a1, b1, a2, b2):
        self.batch_size_unlabeled = batch_size

        # self.model_ref.build_model(image_ref, self.batch_size_labeled, True)

        # self.model_labeled.total_loss = tf.nn.l2_loss(self.model_labeled.stage_joint_heatmap[self.model_src1.stages - 1])

        # self.model_ref.total_loss =tf.nn.l2_loss(tf.concat([self.model_ref.stage_joint_heatmap[4], self.model_ref.stage_limb_heatmap[4]], 3))
        # return

        self.image_ref = image_ref
        self.image_src1 = image_src1
        self.image_src2 = image_src2

        self.H1 = H1
        self.H2 = H2
        self.H01 = H01
        self.H02 = H02

        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2

        # self.model_ref.build_model(image_ref, batch_size, False)
        # self.model_src1.build_model(image_src1, batch_size, False)
        # self.model_src2.build_model(image_src2, batch_size, False)

        self.model_ref.build_model(image_ref, batch_size, True)
        self.model_src1.build_model(image_src1, batch_size, True)
        self.model_src2.build_model(image_src2, batch_size, True)

        # self.model_ref.total_loss = tf.nn.l2_loss(self.model_ref.stage_joint_heatmap[self.model_src1.stages - 1])
        # return



        # self.confident_joint = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.nJoints))

        self.pred01 = EpipolarTransferHeatmap_ext(self.model_src1.stage_joint_heatmap[self.model_src1.stages - 1],
                                                  self.H1,
                                                  self.H01,
                                                  self.a1,
                                                  self.b1,
                                                  self.ext)

        self.pred02 = EpipolarTransferHeatmap_ext(self.model_src2.stage_joint_heatmap[self.model_src2.stages - 1],
                                                  self.H2,
                                                  self.H02,
                                                  self.a2,
                                                  self.b2,
                                                  self.ext)

        self.pred0 = self.pred01 * self.pred02

        # self.pred01_l = EpipolarTransferHeatmap_ext(self.model_src1.stage_limb_heatmap[self.model_src1.stages - 1],
        #                                           self.H1,
        #                                           self.H01,
        #                                           self.a1,
        #                                           self.b1,
        #                                           self.ext)
        #
        # self.pred02_l = EpipolarTransferHeatmap_ext(self.model_src2.stage_limb_heatmap[self.model_src1.stages - 1],
        #                                           self.H2,
        #                                           self.H02,
        #                                           self.a2,
        #                                           self.b2,
        #                                           self.ext)
        #
        # self.pred0_l = self.pred01_l * self.pred02_l

        self.pred0_l = self.model_src1.stage_limb_heatmap[self.model_src1.stages - 1]

        # self.pred0 = tf.nn.l2_normalize(self.pred0, [1, 2])

        # for stage in range(self.model_ref.stages):
        #    with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
        #        self.model_ref.stage_heatmap[stage] = tf.nn.l2_normalize(self.model_ref.stage_heatmap[stage], [1, 2])


        # self.model_ref.joint_modality = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.nJoints))
        # self.confident_joint = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.nJoints))
        self.model_ref.build_loss_unsupervised_slim(self.pred0, self.model_src1.stage_joint_heatmap, self.model_src2.stage_joint_heatmap, self.pred0_l, self.lr_unlabeled, self.lr_decay_rate, self.lr_decay_step)

        # self.model_ref.total_loss += tf.nn.l2_loss(self.model_ref.stage_limb_heatmap[4])
        # self.model_ref.total_loss += tf.nn.l2_loss(self.model_src1.stage_limb_heatmap[4])
        # self.model_ref.total_loss += tf.nn.l2_loss(self.model_src2.stage_limb_heatmap[4])

        # with tf.variable_scope('train'):
        #     self.global_step = tf.train.get_or_create_global_step()
        #     self.global_step0 = self.global_step

    def TotalLoss(self):
        self.total_loss = self.model_ref.total_loss + self.model_labeled.total_loss
        # self.total_loss = self.model_ref.total_loss
        # self.global_step = tf.train.get_or_create_global_step()
        # self.lr = tf.train.exponential_decay(self.lr_labeled,
        #                                      global_step=self.global_step,
        #                                      decay_rate=self.lr_decay_rate,
        #                                      decay_steps=self.lr_decay_step)
        # self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
        #                                                 global_step=self.global_step,
        #                                                 # tf.subtract(self.global_step,self.global_step0),
        #                                                 learning_rate=self.lr,
        #                                                 optimizer='RMSProp')


        # self.merged_summary = tf.summary.merge_all()

class EpiNet(object):
    def __init__(self, stages, joints, limbs, lr, lr_decay_rate, lr_decay_step, input_size, label_size, ext):
        self.model_ref = CPM_Model(stages, joints + 1, limbs)
        self.model_src1 = CPM_Model(stages, joints + 1, limbs)
        self.model_src2 = CPM_Model(stages, joints + 1, limbs)
        self.model_labeled = CPM_Model(stages, joints + 1, limbs)
        self.lr_labeled = lr
        self.lr_unlabeled = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.nJoints = joints
        self.nLimbs = limbs
        self.input_size = input_size
        self.label_size = label_size
        self.ext = ext
        self.step = tf.placeholder(dtype=tf.int32, shape=(1))

    def BuildLabeledCPM(self, batch_size):
        self.batch_size_labeled = batch_size

        self.image_labeled = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.input_size, self.input_size, 3))
        self.label = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.label_size, self.label_size, self.nJoints + 1 + 2*self.nLimbs))

        self.model_labeled.build_model(self.image_labeled, self.batch_size_labeled, False)
        self.model_labeled.build_loss(self.label, self.lr_labeled, self.lr_decay_rate, self.lr_decay_step)

    def BuildUnlabeledCPM(self, batch_size):
        self.batch_size_unlabeled = batch_size

        self.image_ref = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.input_size, self.input_size, 3))
        self.image_src1 = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.input_size, self.input_size, 3))
        self.image_src2 = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.input_size, self.input_size, 3))
        self.heatmap_unlabeled = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.label_size, self.label_size, self.nJoints + 1))

        self.model_ref.build_model(self.image_ref, batch_size, True)
        self.model_src1.build_model(self.image_src1, batch_size, True)
        self.model_src2.build_model(self.image_src2, batch_size, True)

        self.H1 = tf.placeholder(dtype=tf.float32, shape=(batch_size, 3, 3))
        self.H2 = tf.placeholder(dtype=tf.float32, shape=(batch_size, 3, 3))
        self.H01 = tf.placeholder(dtype=tf.float32, shape=(batch_size, 3, 3))
        self.H02 = tf.placeholder(dtype=tf.float32, shape=(batch_size, 3, 3))

        self.a1 = tf.placeholder(dtype=tf.float32, shape=(batch_size, 1))
        self.b1 = tf.placeholder(dtype=tf.float32, shape=(batch_size, 1))
        self.a2 = tf.placeholder(dtype=tf.float32, shape=(batch_size, 1))
        self.b2 = tf.placeholder(dtype=tf.float32, shape=(batch_size, 1))

        self.confident_joint = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.nJoints))

        self.pred01 = EpipolarTransferHeatmap_ext(self.model_src1.stage_joint_heatmap[self.model_src1.stages-1],
                                              self.H1,
                                              self.H01,
                                              self.a1,
                                              self.b1,
                                              self.ext)
        
        self.pred02 = EpipolarTransferHeatmap_ext(self.model_src2.stage_joint_heatmap[self.model_src2.stages - 1],
                                              self.H2,
                                              self.H02,
                                              self.a2,
                                              self.b2,
                                              self.ext)

        #self.pred01 = EpipolarTransferHeatmap_ext_softmax(self.model_src1.stage_heatmap[self.model_src1.stages-1],
        #                                      self.H1,
        #                                      self.H01,
        #                                      self.a1,
        #                                      self.b1,
        #                                      self.ext)

        #self.pred02 = EpipolarTransferHeatmap_ext_softmax(self.model_src2.stage_heatmap[self.model_src2.stages - 1],
        #                                      self.H2,
        #                                      self.H02,
        #                                      self.a2,
        #                                      self.b2,
        #                                      self.ext)
        self.pred0 = self.pred01*self.pred02

        #self.pred0 = tf.nn.l2_normalize(self.pred0, [1, 2])

        #for stage in range(self.model_ref.stages):
        #    with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
        #        self.model_ref.stage_heatmap[stage] = tf.nn.l2_normalize(self.model_ref.stage_heatmap[stage], [1, 2])


        self.model_ref.joint_modality = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.nJoints))
        self.model_ref.build_loss_unsupervised(self.pred0, self.heatmap_unlabeled, self.confident_joint, self.lr_unlabeled, self.lr_decay_rate, self.lr_decay_step)

        # with tf.variable_scope('train'):
        #     self.global_step = tf.train.get_or_create_global_step()
        #     self.global_step0 = self.global_step

    def TotalLoss(self):
        self.total_loss = self.model_ref.total_loss + self.model_labeled.total_loss
        self.global_step = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(self.lr_labeled,
                                            global_step = self.global_step,
                                            decay_rate = self.lr_decay_rate,
                                            decay_steps = self.lr_decay_step)
        self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step = self.global_step,#tf.subtract(self.global_step,self.global_step0),
                                                            learning_rate = self.lr,
                                                            optimizer = 'RMSProp')


        # self.merged_summary = tf.summary.merge_all()

class CPM_Model(object):
    def __init__(self, stages, joints, limbs):
        self.stages = stages
        self.stage_joint_heatmap = []
        self.stage_limb_heatmap = []
        self.stage_loss = [0] * stages
        self.l2_loss = [0] * stages
        self.l2_loss_l = [0] * stages

        self.l2_loss_unlabeled = [0] * stages

        self.unimodality = [0] * stages
        self.total_loss = 0

        self.image = None
        self.center_map = None
        self.gt_heatmap = None
        self.learning_rate = 0
        self.merged_summary = None
        self.joints = joints
        self.limbs = limbs
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
            conv4_3_CPM = tf.layers.conv2d(inputs=conv4_2,
                                           filters=256,
                                           kernel_size=[3, 3],
                                           strides=[1, 1],
                                           padding='same',
                                           activation=tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           name='conv4_3_CPM',
                                       reuse=reuse)
            self.sub_stage_img_feature = tf.layers.conv2d(inputs=conv4_3_CPM,
                                                          filters=128,
                                                          kernel_size=[3, 3],
                                                          strides=[1, 1],
                                                          padding='same',
                                                          activation=tf.nn.relu,
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                          name='conv4_4_CPM',
                                       reuse=reuse)
            conv5_1_CPM_L1 = tf.layers.conv2d(inputs=self.sub_stage_img_feature,
                                              filters=128,
                                              kernel_size=[3, 3],
                                              strides=[1, 1],
                                              padding='same',
                                              activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              name='conv5_1_CPM_L1',
                                       reuse=reuse)
            conv5_1_CPM_L2 = tf.layers.conv2d(inputs=self.sub_stage_img_feature,
                                              filters=128,
                                              kernel_size=[3, 3],
                                              strides=[1, 1],
                                              padding='same',
                                              activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              name='conv5_1_CPM_L2',
                                       reuse=reuse)
            conv5_2_CPM_L1 = tf.layers.conv2d(inputs=conv5_1_CPM_L1,
                                              filters=128,
                                              kernel_size=[3, 3],
                                              strides=[1, 1],
                                              padding='same',
                                              activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              name='conv5_2_CPM_L1',
                                       reuse=reuse)
            conv5_2_CPM_L2 = tf.layers.conv2d(inputs=conv5_1_CPM_L2,
                                              filters=128,
                                              kernel_size=[3, 3],
                                              strides=[1, 1],
                                              padding='same',
                                              activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              name='conv5_2_CPM_L2',
                                       reuse=reuse)
            self.sub_stage_img_feature_L1 = tf.layers.conv2d(inputs=conv5_2_CPM_L1,
                                                             filters=128,
                                                             kernel_size=[3, 3],
                                                             strides=[1, 1],
                                                             padding='same',
                                                             activation=tf.nn.relu,
                                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                             name='conv5_3_CPM_L1',
                                       reuse=reuse)
            self.sub_stage_img_feature_L2 = tf.layers.conv2d(inputs=conv5_2_CPM_L2,
                                                             filters=128,
                                                             kernel_size=[3, 3],
                                                             strides=[1, 1],
                                                             padding='same',
                                                             activation=tf.nn.relu,
                                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                             name='conv5_3_CPM_L2',
                                       reuse=reuse)

        with tf.variable_scope('stage_1'):
            conv5_4_CPM_L1 = tf.layers.conv2d(inputs=self.sub_stage_img_feature_L1,
                                              filters=512,
                                              kernel_size=[1, 1],
                                              strides=[1, 1],
                                              padding='same',
                                              activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              name='conv5_4_CPM_L1',
                                       reuse=reuse)
            conv5_4_CPM_L2 = tf.layers.conv2d(inputs=self.sub_stage_img_feature_L2,
                                              filters=512,
                                              kernel_size=[1, 1],
                                              strides=[1, 1],
                                              padding='same',
                                              activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              name='conv5_4_CPM_L2',
                                       reuse=reuse)

            self.stage_limb_heatmap.append(tf.layers.conv2d(inputs=conv5_4_CPM_L1,
                                                            filters=self.limbs * 2,
                                                            kernel_size=[1, 1],
                                                            strides=[1, 1],
                                                            padding='same',
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                            name='conv5_5_CPM_L1',
                                       reuse=reuse))
            self.stage_joint_heatmap.append(tf.layers.conv2d(inputs=conv5_4_CPM_L2,
                                                             filters=self.joints,
                                                             kernel_size=[1, 1],
                                                             strides=[1, 1],
                                                             padding='same',
                                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                             name='conv5_5_CPM_L2',
                                       reuse=reuse))
        for stage in range(2, self.stages + 1):
            self._middle_conv(stage, reuse)

        self._last_conv(self.stages + 1)

    def _middle_conv(self, stage, reuse):
        with tf.variable_scope('stage_' + str(stage)):
            self.current_featuremap = tf.concat([self.stage_joint_heatmap[stage - 2],
                                                 self.stage_limb_heatmap[stage - 2],
                                                 self.sub_stage_img_feature,
                                                 ],
                                                axis=3)
            mid_conv1_l1 = tf.layers.conv2d(inputs=self.current_featuremap,
                                            filters=128,
                                            kernel_size=[7, 7],
                                            strides=[1, 1],
                                            padding='same',
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name='Mconv1_stage' + str(stage) + 'L1',
                                       reuse=reuse)
            mid_conv1_l2 = tf.layers.conv2d(inputs=self.current_featuremap,
                                            filters=128,
                                            kernel_size=[7, 7],
                                            strides=[1, 1],
                                            padding='same',
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name='Mconv1_stage' + str(stage) + 'L2',
                                       reuse=reuse)
            mid_conv2_l1 = tf.layers.conv2d(inputs=mid_conv1_l1,
                                            filters=128,
                                            kernel_size=[7, 7],
                                            strides=[1, 1],
                                            padding='same',
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name='Mconv2_stage' + str(stage) + 'L1',
                                       reuse=reuse)
            mid_conv2_l2 = tf.layers.conv2d(inputs=mid_conv1_l2,
                                            filters=128,
                                            kernel_size=[7, 7],
                                            strides=[1, 1],
                                            padding='same',
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name='Mconv2_stage' + str(stage) + 'L2',
                                       reuse=reuse)
            mid_conv3_l1 = tf.layers.conv2d(inputs=mid_conv2_l1,
                                            filters=128,
                                            kernel_size=[7, 7],
                                            strides=[1, 1],
                                            padding='same',
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name='Mconv3_stage' + str(stage) + 'L1',
                                       reuse=reuse)
            mid_conv3_l2 = tf.layers.conv2d(inputs=mid_conv2_l2,
                                            filters=128,
                                            kernel_size=[7, 7],
                                            strides=[1, 1],
                                            padding='same',
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name='Mconv3_stage' + str(stage) + 'L2',
                                       reuse=reuse)
            mid_conv4_l1 = tf.layers.conv2d(inputs=mid_conv3_l1,
                                            filters=128,
                                            kernel_size=[7, 7],
                                            strides=[1, 1],
                                            padding='same',
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name='Mconv4_stage' + str(stage) + 'L1',
                                       reuse=reuse)
            mid_conv4_l2 = tf.layers.conv2d(inputs=mid_conv3_l2,
                                            filters=128,
                                            kernel_size=[7, 7],
                                            strides=[1, 1],
                                            padding='same',
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name='Mconv4_stage' + str(stage) + 'L2',
                                       reuse=reuse)
            mid_conv5_l1 = tf.layers.conv2d(inputs=mid_conv4_l1,
                                            filters=128,
                                            kernel_size=[7, 7],
                                            strides=[1, 1],
                                            padding='same',
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name='Mconv5_stage' + str(stage) + 'L1',
                                       reuse=reuse)
            mid_conv5_l2 = tf.layers.conv2d(inputs=mid_conv4_l2,
                                            filters=128,
                                            kernel_size=[7, 7],
                                            strides=[1, 1],
                                            padding='same',
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name='Mconv5_stage' + str(stage) + 'L2',
                                       reuse=reuse)
            mid_conv6_l1 = tf.layers.conv2d(inputs=mid_conv5_l1,
                                            filters=128,
                                            kernel_size=[1, 1],
                                            strides=[1, 1],
                                            padding='same',
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name='Mconv6_stage' + str(stage) + 'L1',
                                       reuse=reuse)
            mid_conv6_l2 = tf.layers.conv2d(inputs=mid_conv5_l2,
                                            filters=128,
                                            kernel_size=[1, 1],
                                            strides=[1, 1],
                                            padding='same',
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name='Mconv6_stage' + str(stage) + 'L2',
                                       reuse=reuse)
            self.current_heatmap_l1 = tf.layers.conv2d(inputs=mid_conv6_l1,
                                                       filters=self.limbs * 2,
                                                       kernel_size=[1, 1],
                                                       strides=[1, 1],
                                                       padding='same',
                                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                       name='Mconv7_stage' + str(stage) + 'L1',
                                       reuse=reuse)
            self.current_heatmap_l2 = tf.layers.conv2d(inputs=mid_conv6_l2,
                                                       filters=self.joints,
                                                       kernel_size=[1, 1],
                                                       strides=[1, 1],
                                                       padding='same',
                                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                       name='Mconv7_stage' + str(stage) + 'L2',
                                       reuse=reuse)
            self.stage_limb_heatmap.append(self.current_heatmap_l1)
            self.stage_joint_heatmap.append(self.current_heatmap_l2)

    def _last_conv(self, stage):
        with tf.variable_scope('stage_' + str(stage)):
            self.last_featuremap = tf.concat([self.stage_joint_heatmap[stage - 2],
                                              self.stage_limb_heatmap[stage - 2],
                                              ],
                                             axis=3)


    def build_loss_unsupervised_slim(self, gt_heatmap_joint, pred01, pred02, gt_heatmap_limb, lr, lr_decay_rate, lr_decay_step):
        # self.gt_heatmap = gt_heatmap_joint
        self.total_loss = 0

        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step

        # self.gt_heatmap = tf.nn.relu(gt_heatmap_joint)
        self.gt_heatmap = tf.nn.l2_normalize(gt_heatmap_joint, [1, 2])
        self.gt_heatmap_limb = tf.nn.l2_normalize(gt_heatmap_limb, [1, 2])

        for stage in range(self.stages):
            heatmap = tf.nn.l2_normalize(self.stage_joint_heatmap[stage], [1, 2])
            heatmap_l = tf.nn.l2_normalize(self.stage_limb_heatmap[stage], [1, 2])

            self.l2_loss_l[stage] = 0.01*tf.nn.l2_loss(
                (heatmap_l - self.gt_heatmap_limb)) / self.batch_size

            # self.l2_loss[stage] = 0.1*tf.nn.l2_loss(
            #     (heatmap[:, :, :, :-1] - self.gt_heatmap[:, :, :, :-1])) / self.batch_size

            self.l2_loss[stage] = tf.nn.l2_loss(
                (heatmap[:,:,:,:-1] - self.gt_heatmap[:,:,:,:-1])) / self.batch_size

            unimodality = tf.reduce_max(heatmap, axis=[1, 2])#-tf.reduce_min(heatmap, axis=[1, 2])
            unimodality = unimodality[:, :-1]
            unimodality = tf.nn.sigmoid(10 * (unimodality - 0.5))
            self.joint_modality = unimodality
            #
            # unimodality = unimodality[:, 3]


            self.unimodality[stage] = -tf.nn.l2_loss(unimodality) / self.batch_size



            # self.unimodality[stage] = -tf.nn.l2_loss(self.joint_modality) / self.batch_size

            # self.stage_loss[stage] = self.l2_loss[stage] + self.l2_loss_l[stage] + self.unimodality[stage]# + self.l2_loss_unlabeled[stage]
            self.stage_loss[stage] = self.l2_loss[stage] + self.unimodality[stage]
            # self.stage_loss[stage] = self.unimodality[stage]
            # tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])


            # # #
            # heatmap = tf.nn.relu(pred01)
            heatmap1 = tf.nn.l2_normalize(pred01[stage], [1, 2])
            unimodality1 = tf.reduce_max(heatmap1, axis=[1, 2])  # -tf.reduce_min(heatmap, axis=[1, 2])
            unimodality1 = unimodality1[:, :-1]
            # unimodality = unimodality[:, 3]
            unimodality1 = tf.nn.sigmoid(10 * (unimodality1 - 0.5))
            self.stage_loss[stage] += -tf.nn.l2_loss(unimodality1) / self.batch_size

            # heatmap = tf.nn.relu(pred02)
            heatmap2 = tf.nn.l2_normalize(pred02[stage], [1, 2])
            unimodality2 = tf.reduce_max(heatmap2, axis=[1, 2])  # -tf.reduce_min(heatmap, axis=[1, 2])
            unimodality2 = unimodality2[:, :-1]
            # unimodality = unimodality[:, 3]
            unimodality2 = tf.nn.sigmoid(10 * (unimodality2 - 0.5))
            self.stage_loss[stage] += -tf.nn.l2_loss(unimodality2) / self.batch_size


        # with tf.variable_scope('total_loss'):
        for stage in range(self.stages):
            self.total_loss += self.stage_loss[stage]
        # self.total_loss = self.stage_loss[self.stages-1]




        # tf.summary.scalar('total loss', self.total_loss)

    def build_loss_unsupervised(self, gt_heatmap, heatmap_unlabeled, confident_joint, lr, lr_decay_rate, lr_decay_step):
        self.gt_heatmap = gt_heatmap
        self.total_loss = 0
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step

        self.gt_heatmap = tf.nn.l2_normalize(self.gt_heatmap, [1, 2])

        # confident_joint1 = confident_joint

        confident_joint = tf.reshape(confident_joint, (self.batch_size, 1, 1, self.joints-1))
        heatmap_size = tf.shape(self.stage_joint_heatmap[0])
        confident_joint = tf.tile(confident_joint, (1, heatmap_size[1], heatmap_size[2], 1))

        unimodality_gt = tf.reduce_max(self.gt_heatmap, axis=[1, 2])
        unimodality_gt = unimodality_gt[:, :-1]
        unimodality_gt = tf.nn.sigmoid(30*(unimodality_gt-0.35))
        # self.joint_modality = unimodality_gt

        unimodality_gt = tf.reshape(unimodality_gt, (self.batch_size, 1, 1, self.joints - 1))
        heatmap_size = tf.shape(self.stage_joint_heatmap[0])
        unimodality_gt = tf.tile(unimodality_gt, (1, heatmap_size[1], heatmap_size[2], 1))#+0.01

        for stage in range(self.stages):
            self.stage_joint_heatmap[stage] = tf.nn.l2_normalize(self.stage_joint_heatmap[stage], [1, 2])
            unimodality = tf.reduce_max(self.stage_joint_heatmap[stage], axis=[1, 2])-tf.reduce_min(self.stage_joint_heatmap[stage], axis=[1, 2])
            # unimodality = confident_joint1 * unimodality[:,:-1]
            unimodality = unimodality[:,:-1]
            unimodality = tf.nn.sigmoid(30*(unimodality-0.45))
            self.joint_modality = unimodality

            self.l2_loss[stage] = tf.nn.l2_loss((self.stage_joint_heatmap[stage][:,:,:,:-1] - self.gt_heatmap[:,:,:,:-1])) / self.batch_size

            heatmap_unlabeled = tf.nn.l2_normalize(heatmap_unlabeled, [1, 2])

            self.l2_loss_unlabeled[stage] = tf.nn.l2_loss(
                confident_joint * (self.stage_joint_heatmap[stage][:, :, :, :-1] - heatmap_unlabeled[:, :, :, :-1])) / self.batch_size

            self.unimodality[stage] = -tf.nn.l2_loss(unimodality) / self.batch_size
            # self.unimodality[stage] = -tf.nn.l2_loss(self.joint_modality) / self.batch_size

            self.stage_loss[stage] = self.l2_loss[stage] + self.unimodality[stage]# + self.l2_loss_unlabeled[stage]
            # tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        # with tf.variable_scope('total_loss'):
        for stage in range(self.stages):
            self.total_loss += self.stage_loss[stage]
        # tf.summary.scalar('total loss', self.total_loss)

    def build_loss(self, gt_heatmap, lr, lr_decay_rate, lr_decay_step):
        self.gt_heatmap = gt_heatmap
        self.total_loss = 0
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step

        for stage in range(self.stages):
            # with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
            self.stage_loss[stage] = tf.nn.l2_loss(
                tf.concat([self.stage_joint_heatmap[stage], self.stage_limb_heatmap[stage]], 3) - self.gt_heatmap) / self.batch_size
            # tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        # with tf.variable_scope('total_loss'):
        for stage in range(self.stages):
            self.total_loss += self.stage_loss[stage]
        return self.total_loss
        # tf.summary.scalar('total loss', self.total_loss)

    def train(self):
        # with tf.variable_scope('train'):
        self.global_step = tf.train.get_or_create_global_step()

        self.lr = tf.train.exponential_decay(self.learning_rate,
                                             global_step=self.global_step,
                                             decay_rate=self.lr_decay_rate,
                                             decay_steps=self.lr_decay_step)
            # tf.summary.scalar('learning rate', self.lr)

        self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                        global_step=self.global_step,
                                                        learning_rate=self.lr,
                                                        optimizer='Adam')
        # self.merged_summary = tf.summary.merge_all()

# class CPM_Model(object):
#     def __init__(self, stages, joints):
#         self.stages = stages
#         self.stage_heatmap = []
#         self.stage_loss = [0] * stages
#         self.total_loss = 0
#         self.image = None
#         self.center_map = None
#         self.gt_heatmap = None
#         self.learning_rate = 0
#         self.merged_summary = None
#         self.joints = joints
#         self.batch_size = 0
#
#     def build_model(self, image, batch_size, reuse):
#         self.batch_size = batch_size
#         self.input_image = image
#         with tf.variable_scope('sub_stages'):
#             conv1_1 = tf.layers.conv2d(inputs=image,
#                                          filters=64,
#                                          kernel_size=[3, 3],
#                                          strides=[1, 1],
#                                          padding='same',
#                                          activation=tf.nn.relu,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                          name='conv1_1',
#                                         reuse=reuse)
#             conv1_2 = tf.layers.conv2d(inputs=conv1_1,
#                                          filters=64,
#                                          kernel_size=[3, 3],
#                                          strides=[1, 1],
#                                          padding='same',
#                                          activation=tf.nn.relu,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                          name='conv1_2',
#                                         reuse=reuse)
#             pool1_stage1 = tf.layers.max_pooling2d(inputs=conv1_2,
#                                                 pool_size=[2, 2],
#                                                 strides=2,
#                                                 padding='same',
#                                                 name='pool1_stage1')
#             conv2_1 = tf.layers.conv2d(inputs=pool1_stage1,
#                                          filters=128,
#                                          kernel_size=[3, 3],
#                                          strides=[1, 1],
#                                          padding='same',
#                                          activation=tf.nn.relu,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                          name='conv2_1',
#                                         reuse=reuse)
#             conv2_2 = tf.layers.conv2d(inputs=conv2_1,
#                                          filters=128,
#                                          kernel_size=[3, 3],
#                                          strides=[1, 1],
#                                          padding='same',
#                                          activation=tf.nn.relu,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                          name='conv2_2',
#                                         reuse=reuse)
#             pool2_stage1 = tf.layers.max_pooling2d(inputs=conv2_2,
#                                                 pool_size=[2, 2],
#                                                 strides=2,
#                                                 padding='same',
#                                                 name='pool2_stage1')
#             conv3_1 = tf.layers.conv2d(inputs=pool2_stage1,
#                                          filters=256,
#                                          kernel_size=[3, 3],
#                                          strides=[1, 1],
#                                          padding='same',
#                                          activation=tf.nn.relu,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                          name='conv3_1',
#                                         reuse=reuse)
#             conv3_2 = tf.layers.conv2d(inputs=conv3_1,
#                                          filters=256,
#                                          kernel_size=[3, 3],
#                                          strides=[1, 1],
#                                          padding='same',
#                                          activation=tf.nn.relu,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                          name='conv3_2',
#                                         reuse=reuse)
#             conv3_3 = tf.layers.conv2d(inputs=conv3_2,
#                                          filters=256,
#                                          kernel_size=[3, 3],
#                                          strides=[1, 1],
#                                          padding='same',
#                                          activation=tf.nn.relu,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                          name='conv3_3',
#                                         reuse=reuse)
#             conv3_4 = tf.layers.conv2d(inputs=conv3_3,
#                                          filters=256,
#                                          kernel_size=[3, 3],
#                                          strides=[1, 1],
#                                          padding='same',
#                                          activation=tf.nn.relu,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                          name='conv3_4',
#                                         reuse=reuse)
#             pool3_stage1 = tf.layers.max_pooling2d(inputs=conv3_4,
#                                                 pool_size=[2, 2],
#                                                 strides=2,
#                                                 padding='same',
#                                                 name='pool3_stage1')
#             conv4_1 = tf.layers.conv2d(inputs=pool3_stage1,
#                                          filters=512,
#                                          kernel_size=[3, 3],
#                                          strides=[1, 1],
#                                          padding='same',
#                                          activation=tf.nn.relu,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                          name='conv4_1',
#                                         reuse=reuse)
#             conv4_2 = tf.layers.conv2d(inputs=conv4_1,
#                                           filters=512,
#                                           kernel_size=[3, 3],
#                                           strides=[1, 1],
#                                           padding='same',
#                                           activation=tf.nn.relu,
#                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                           name='conv4_2',
#                                         reuse=reuse)
#             conv4_3 = tf.layers.conv2d(inputs=conv4_2,
#                                           filters=512,
#                                           kernel_size=[3, 3],
#                                           strides=[1, 1],
#                                           padding='same',
#                                           activation=tf.nn.relu,
#                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                           name='conv4_3',
#                                         reuse=reuse)
#             conv4_4 = tf.layers.conv2d(inputs=conv4_3,
#                                           filters=512,
#                                           kernel_size=[3, 3],
#                                           strides=[1, 1],
#                                           padding='same',
#                                           activation=tf.nn.relu,
#                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                           name='conv4_4',
#                                         reuse=reuse)
#             conv5_1 = tf.layers.conv2d(inputs=conv4_4,
#                                           filters=512,
#                                           kernel_size=[3, 3],
#                                           strides=[1, 1],
#                                           padding='same',
#                                           activation=tf.nn.relu,
#                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                           name='conv5_1',
#                                         reuse=reuse)
#             conv5_2 = tf.layers.conv2d(inputs=conv5_1,
#                                           filters=512,
#                                           kernel_size=[3, 3],
#                                           strides=[1, 1],
#                                           padding='same',
#                                           activation=tf.nn.relu,
#                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                           name='conv5_2',
#                                         reuse=reuse)
#             self.sub_stage_img_feature = tf.layers.conv2d(inputs=conv5_2,
#                                                           filters=128,
#                                                           kernel_size=[3, 3],
#                                                           strides=[1, 1],
#                                                           padding='same',
#                                                           activation=tf.nn.relu,
#                                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                                           name='conv5_3_CPM',
#                                         reuse=reuse)
#
#         with tf.variable_scope('stage_1'):
#             conv1 = tf.layers.conv2d(inputs=self.sub_stage_img_feature,
#                                      filters=512,
#                                      kernel_size=[1, 1],
#                                      strides=[1, 1],
#                                      padding='same',
#                                      activation=tf.nn.relu,
#                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                      name='conv6_1_CPM',
#                                         reuse=reuse)
#             self.stage_heatmap.append(tf.layers.conv2d(inputs=conv1,
#                                                        filters=self.joints,
#                                                        kernel_size=[1, 1],
#                                                        strides=[1, 1],
#                                                        padding='same',
#                                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                                        name='conv6_2_CPM',
#                                                        reuse=reuse))
#         for stage in range(2, self.stages + 1):
#             self._middle_conv(stage, reuse)
#
#     def _middle_conv(self, stage, reuse):
#         with tf.variable_scope('stage_' + str(stage)):
#             self.current_featuremap = tf.concat([self.stage_heatmap[stage - 2],
#                                                  self.sub_stage_img_feature,
#                                                  ],
#                                                 axis=3)
#             mid_conv1 = tf.layers.conv2d(inputs=self.current_featuremap,
#                                          filters=128,
#                                          kernel_size=[7, 7],
#                                          strides=[1, 1],
#                                          padding='same',
#                                          activation=tf.nn.relu,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                          name='Mconv1_stage' + str(stage),
#                                         reuse=reuse)
#             mid_conv2 = tf.layers.conv2d(inputs=mid_conv1,
#                                          filters=128,
#                                          kernel_size=[7, 7],
#                                          strides=[1, 1],
#                                          padding='same',
#                                          activation=tf.nn.relu,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                          name='Mconv2_stage' + str(stage),
#                                         reuse=reuse)
#             mid_conv3 = tf.layers.conv2d(inputs=mid_conv2,
#                                          filters=128,
#                                          kernel_size=[7, 7],
#                                          strides=[1, 1],
#                                          padding='same',
#                                          activation=tf.nn.relu,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                          name='Mconv3_stage' + str(stage),
#                                         reuse=reuse)
#             mid_conv4 = tf.layers.conv2d(inputs=mid_conv3,
#                                          filters=128,
#                                          kernel_size=[7, 7],
#                                          strides=[1, 1],
#                                          padding='same',
#                                          activation=tf.nn.relu,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                          name='Mconv4_stage' + str(stage),
#                                         reuse=reuse)
#             mid_conv5 = tf.layers.conv2d(inputs=mid_conv4,
#                                          filters=128,
#                                          kernel_size=[7, 7],
#                                          strides=[1, 1],
#                                          padding='same',
#                                          activation=tf.nn.relu,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                          name='Mconv5_stage' + str(stage),
#                                         reuse=reuse)
#             mid_conv6 = tf.layers.conv2d(inputs=mid_conv5,
#                                          filters=128,
#                                          kernel_size=[1, 1],
#                                          strides=[1, 1],
#                                          padding='same',
#                                          activation=tf.nn.relu,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                          name='Mconv6_stage' + str(stage),
#                                         reuse=reuse)
#             self.current_heatmap = tf.layers.conv2d(inputs=mid_conv6,
#                                                     filters=self.joints,
#                                                     kernel_size=[1, 1],
#                                                     strides=[1, 1],
#                                                     padding='same',
#                                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                                     name='Mconv7_stage' + str(stage),
#                                         reuse=reuse)
#             self.stage_heatmap.append(self.current_heatmap)
#
#     def build_loss_unsupervised(self, gt_heatmap, lr, lr_decay_rate, lr_decay_step):
#         self.gt_heatmap = gt_heatmap
#         self.total_loss = 0
#         self.learning_rate = lr
#         self.lr_decay_rate = lr_decay_rate
#         self.lr_decay_step = lr_decay_step
#
#         self.gt_heatmap = tf.nn.l2_normalize(self.gt_heatmap, [1, 2])
#
#         for stage in range(self.stages):
#             with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
#                 self.stage_heatmap[stage] = tf.nn.l2_normalize(self.stage_heatmap[stage], [1, 2])
#
#                 self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap,
#                                                        name='l2_loss') / self.batch_size
#             tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])
#
#         with tf.variable_scope('total_loss'):
#             for stage in range(self.stages):
#                 self.total_loss += self.stage_loss[stage]
#             tf.summary.scalar('total loss', self.total_loss)
#
#         # with tf.variable_scope('train'):
#         #     self.global_step = tf.contrib.framework.get_or_create_global_step()
#         #
#         #     self.lr = tf.train.exponential_decay(self.learning_rate,
#         #                                          global_step=self.global_step,
#         #                                          decay_rate=self.lr_decay_rate,
#         #                                          decay_steps=self.lr_decay_step)
#         #     tf.summary.scalar('learning rate', self.lr)
#         #
#         #     self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
#         #                                                     global_step=self.global_step,
#         #                                                     learning_rate=self.lr,
#         #                                                     optimizer='Adam')
#         self.merged_summary = tf.summary.merge_all()
#
#     def build_loss(self, gt_heatmap, lr, lr_decay_rate, lr_decay_step):
#         self.gt_heatmap = gt_heatmap
#         self.total_loss = 0
#         self.learning_rate = lr
#         self.lr_decay_rate = lr_decay_rate
#         self.lr_decay_step = lr_decay_step
#
#         # self.gt_heatmap = tf.nn.l2_normalize(self.gt_heatmap, [1, 2])
#
#         for stage in range(self.stages):
#             with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
#                 # self.stage_heatmap[stage] = tf.nn.l2_normalize(self.stage_heatmap[stage], [1, 2])
#
#                 self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap,
#                                                        name='l2_loss') / self.batch_size
#             tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])
#
#         with tf.variable_scope('total_loss'):
#             for stage in range(self.stages):
#                 self.total_loss += self.stage_loss[stage]
#             tf.summary.scalar('total loss', self.total_loss)
#
#         # with tf.variable_scope('train'):
#         #     self.global_step = tf.contrib.framework.get_or_create_global_step()
#         #
#         #     self.lr = tf.train.exponential_decay(self.learning_rate,
#         #                                          global_step=self.global_step,
#         #                                          decay_rate=self.lr_decay_rate,
#         #                                          decay_steps=self.lr_decay_step)
#         #     tf.summary.scalar('learning rate', self.lr)
#         #
#         #     self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
#         #                                                     global_step=self.global_step,
#         #                                                     learning_rate=self.lr,
#         #                                                     optimizer='Adam')
#         self.merged_summary = tf.summary.merge_all()
#
#
#
#     def load_weights_from_file(self, weight_file_path, sess, finetune=True):
#         weights = np.load(weight_file_path).item()
#
#         with tf.variable_scope('', reuse=True):
#             ## Pre stage conv
#             # conv1
#             for layer in range(1, 3):
#                 conv_kernel = tf.get_variable('sub_stages/conv1_' + str(layer) + '/kernel')
#                 conv_bias = tf.get_variable('sub_stages/conv1_' + str(layer) + '/bias')
#
#                 loaded_kernel = weights['conv1_' + str(layer)]['weights']
#                 loaded_bias = weights['conv1_' + str(layer)]['biases']
#
#                 sess.run(tf.assign(conv_kernel, loaded_kernel))
#                 sess.run(tf.assign(conv_bias, loaded_bias))
#
#             # conv2
#             for layer in range(1, 3):
#                 conv_kernel = tf.get_variable('sub_stages/conv2_' + str(layer) + '/kernel')
#                 conv_bias = tf.get_variable('sub_stages/conv2_' + str(layer) + '/bias')
#
#                 loaded_kernel = weights['conv2_' + str(layer)]['weights']
#                 loaded_bias = weights['conv2_' + str(layer)]['biases']
#
#                 sess.run(tf.assign(conv_kernel, loaded_kernel))
#                 sess.run(tf.assign(conv_bias, loaded_bias))
#
#             # conv3
#             for layer in range(1, 5):
#                 conv_kernel = tf.get_variable('sub_stages/conv3_' + str(layer) + '/kernel')
#                 conv_bias = tf.get_variable('sub_stages/conv3_' + str(layer) + '/bias')
#
#                 loaded_kernel = weights['conv3_' + str(layer)]['weights']
#                 loaded_bias = weights['conv3_' + str(layer)]['biases']
#
#                 sess.run(tf.assign(conv_kernel, loaded_kernel))
#                 sess.run(tf.assign(conv_bias, loaded_bias))
#
#             # conv4
#             for layer in range(1, 5):
#                 conv_kernel = tf.get_variable('sub_stages/conv4_' + str(layer) + '/kernel')
#                 conv_bias = tf.get_variable('sub_stages/conv4_' + str(layer) + '/bias')
#
#                 loaded_kernel = weights['conv4_' + str(layer)]['weights']
#                 loaded_bias = weights['conv4_' + str(layer)]['biases']
#
#                 sess.run(tf.assign(conv_kernel, loaded_kernel))
#                 sess.run(tf.assign(conv_bias, loaded_bias))
#
#             # conv5
#             for layer in range(1, 3):
#                 conv_kernel = tf.get_variable('sub_stages/conv5_' + str(layer) + '/kernel')
#                 conv_bias = tf.get_variable('sub_stages/conv5_' + str(layer) + '/bias')
#
#                 loaded_kernel = weights['conv5_' + str(layer)]['weights']
#                 loaded_bias = weights['conv5_' + str(layer)]['biases']
#
#                 sess.run(tf.assign(conv_kernel, loaded_kernel))
#                 sess.run(tf.assign(conv_bias, loaded_bias))
#
#             # conv5_3_CPM
#             conv_kernel = tf.get_variable('sub_stages/sub_stage_img_feature/kernel')
#             conv_bias = tf.get_variable('sub_stages/sub_stage_img_feature/bias')
#
#             loaded_kernel = weights['conv5_3_CPM']['weights']
#             loaded_bias = weights['conv5_3_CPM']['biases']
#
#             sess.run(tf.assign(conv_kernel, loaded_kernel))
#             sess.run(tf.assign(conv_bias, loaded_bias))
#
#             ## stage 1
#             conv_kernel = tf.get_variable('stage_1/conv1/kernel')
#             conv_bias = tf.get_variable('stage_1/conv1/bias')
#
#             loaded_kernel = weights['conv6_1_CPM']['weights']
#             loaded_bias = weights['conv6_1_CPM']['biases']
#
#             sess.run(tf.assign(conv_kernel, loaded_kernel))
#             sess.run(tf.assign(conv_bias, loaded_bias))
#
#             if finetune != True:
#                 conv_kernel = tf.get_variable('stage_1/stage_heatmap/kernel')
#                 conv_bias = tf.get_variable('stage_1/stage_heatmap/bias')
#
#                 loaded_kernel = weights['conv6_2_CPM']['weights']
#                 loaded_bias = weights['conv6_2_CPM']['biases']
#
#                 sess.run(tf.assign(conv_kernel, loaded_kernel))
#                 sess.run(tf.assign(conv_bias, loaded_bias))
#
#                 ## stage 2 and behind
#                 for stage in range(2, self.stages + 1):
#                     for layer in range(1, 8):
#                         conv_kernel = tf.get_variable('stage_' + str(stage) + '/mid_conv' + str(layer) + '/kernel')
#                         conv_bias = tf.get_variable('stage_' + str(stage) + '/mid_conv' + str(layer) + '/bias')
#
#                         loaded_kernel = weights['Mconv' + str(layer) + '_stage' + str(stage)]['weights']
#                         loaded_bias = weights['Mconv' + str(layer) + '_stage' + str(stage)]['biases']
#
#                         sess.run(tf.assign(conv_kernel, loaded_kernel))
#                         sess.run(tf.assign(conv_bias, loaded_bias))
