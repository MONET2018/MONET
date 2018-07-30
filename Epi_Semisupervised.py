import tensorflow as tf
import numpy as np
import cv2
from utils import cpm_utils, tf_utils

from Utility.EpiNet_Joint import *
import math
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from numpy import linalg as LA
from StructDefinition import *
from Utility.DataUtility import *
from Utility.GeometryUtility import *


def Image2Crop(x, para):
    scale = para[6]
    x_scaled = (x[0] - para[0]) * scale + para[4]
    y_scaled = (x[1] - para[1]) * scale + para[5]

    return [x_scaled, y_scaled]

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def main():
    gpu_id = [0]
    nGPUs = len(gpu_id)

    tfr_labeled_data_files = 'training_data.tfrecords'

    tfr_unlabeled_data_files = 'training_data_unlabeled_pair.tfrecords'

    input_size = 368
    heatmap_size = 46
    stages = 6
    num_of_joints = 19
    nBatch_supervised = 10
    nBatch_unsupervised = 12
    heatmap_extension_length = 20
    training_iterations = 10000000

    pretrained_model = "alg2.ckpt-2023"


    learning_rate = 1e-5
    lr_decay_rate = 0.5
    # lr_decay_step = 1000*30
    lr_decay_step = 2000
    color_channel = 'RGB'

    vPair = LoadPairData('pair.txt')

    validation_file = 'validation/val_list.txt'
    validation_frame = LoadValidationData(validation_file)
    print(len(validation_frame))


    output_path = 'vis/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    ################################################
    ## Set validation data
    ################################################
    vValidationImage = []
    vImage = []
    for iFrame in range(len(validation_frame)):
        validation_image = np.zeros((nBatch_unsupervised, input_size, input_size, 3))
        if iFrame == 0:
            for iImage in range(validation_frame[iFrame].shape[0]):
                filename = "validation/%07d_%07d.bmp" % (validation_frame[iFrame][iImage, 0], validation_frame[iFrame][iImage, 1])
                # print(filename)
                im = cv2.imread(filename).astype(float)/255
                vImage.append(im)
                if iImage%7 == 0:
                    validation_image[iImage/7,:,:,:] = im
                    print(filename)
        else:
            for iImage in range(0, validation_frame[iFrame].shape[0], 7):
                filename = "validation/%07d_%07d.bmp" % (validation_frame[iFrame][iImage, 0], validation_frame[iFrame][iImage, 1])
                print(filename)
                im = cv2.imread(filename).astype(float)/255
                validation_image[iImage/7,:,:,:] = im
        vValidationImage.append(validation_image)

        if (len(vValidationImage) > 2):
            break

    validation_image_ref = np.zeros((nBatch_unsupervised, input_size, input_size, 3))
    validation_image_src = np.zeros((nBatch_unsupervised, input_size, input_size, 3))
    validation_H1 = np.zeros((nBatch_unsupervised, 3, 3))
    validation_H01 = np.zeros((nBatch_unsupervised, 3, 3))
    validation_H01_inv = np.zeros((nBatch_unsupervised, 3, 3))
    validation_a = np.zeros((nBatch_unsupervised, 1))
    validation_b = np.zeros((nBatch_unsupervised, 1))
    validation_confidence = np.zeros((nBatch_unsupervised, num_of_joints))

    nVals = 0
    ref = -1
    vis_frame = 25
    idx_1 = 0
    print(len(vPair))
    for iBatch in range(len(vPair)):
        # print(vPair[iBatch].ref)
        if (vPair[iBatch].ref != vis_frame):
            continue
        nVals += 1
        print("%d %d %d" %(vPair[iBatch].time, vPair[iBatch].ref, vPair[iBatch].src))
        for iImage in range(validation_frame[0].shape[0]):
            if validation_frame[0][iImage, 1] == vPair[iBatch].ref:
                ref_idx = iImage
            if validation_frame[0][iImage, 1] == vPair[iBatch].src:
                src_idx = iImage

        # validation_image[iBatch, :, :, :] = vImage[src1_idx]
        ref = ref_idx

        validation_image_ref[idx_1,:,:,:] = vImage[ref_idx]
        validation_image_src[idx_1, :, :, :] = vImage[src_idx]
        validation_H1[idx_1,:,:] = vPair[iBatch].H1
        validation_H01[idx_1, :, :] = vPair[iBatch].H01
        validation_H01_inv[idx_1, :, :] = LA.inv(vPair[iBatch].H01)
        validation_a[idx_1, 0] = vPair[iBatch].a
        validation_b[idx_1, 0] = vPair[iBatch].b
        validation_confidence[idx_1, :] = vPair[iBatch].confidence

        idx_1 += 1
        if (idx_1 >= nBatch_unsupervised):
            break

    print(nVals)

    ################################################
    ## Set network
    ################################################

    with tf.device('/cpu:0'):

        image_placeholder = tf.placeholder(dtype=tf.float32, shape=(nBatch_unsupervised, heatmap_size, heatmap_size, 3),
                                           name='input_placeholer')
        H_ref_placeholder = tf.placeholder(dtype=tf.float32, shape=(nBatch_unsupervised, 3, 3),
                                           name='input_placeholer1')
        warped_image = IntensityWarping_ext(image_placeholder, H_ref_placeholder, heatmap_extension_length)


        image_b, heatmap_b, cropping_param_b, K_b, R_b, C_b, frame_b = ReadShuffledBatchData(
            tfr_labeled_data_files, input_size, heatmap_size, num_of_joints, num_of_limbs, nBatch_supervised)

        batch_queue_labeled = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [image_b, heatmap_b, cropping_param_b, K_b, R_b, C_b, frame_b], capacity=2 * nGPUs)

        image_ref_b, image_src_b, \
        H1_b, H01_b, \
        a_b, b_b, frame_unlabeled_b, \
        heatmap_unlabeled_b, confidence_b = ReadShuffledBatchDataUnlabeled_Pair(
            tfr_unlabeled_data_files, input_size, heatmap_size, num_of_joints, num_of_limbs, nBatch_unsupervised)

        batch_queue_unlabeled = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [image_ref_b, image_src_b, H1_b, H01_b, a_b, b_b, frame_unlabeled_b, heatmap_unlabeled_b, confidence_b], capacity= 2 * nGPUs)

        global_step = tf.train.get_or_create_global_step()

        lr = tf.train.exponential_decay(learning_rate,
                                             global_step=global_step,
                                             decay_rate=lr_decay_rate,
                                             decay_steps=lr_decay_step)
        opt = tf.train.AdamOptimizer(lr)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(nGPUs):
                with tf.device('/gpu:%d'%gpu_id[i]):
                    with tf.name_scope('net_%d'%gpu_id[i]) as scope:
                        image_batch, heatmap_batch, cropping_param_batch, K_batch, R_batch, C_batch, frame_batch = batch_queue_labeled.dequeue()

                        image_ref_batch, image_src_batch, \
                        H1_batch, H01_batch, \
                        a_batch, b_batch, frame_unlabeled_batch, \
                        heatmap_unlabeled_batch, confidence_batch  = batch_queue_unlabeled.dequeue()

                        epi_net = EpiNet_pair(stages, num_of_joints, lr, lr_decay_rate, lr_decay_step,
                                               input_size,
                                               heatmap_size, heatmap_extension_length)
                        if i == 0:
                            epi_net.BuildLabeledCPM(nBatch_supervised, image_batch, heatmap_batch[:,:,:,:num_of_joints+1], None)
                            model_val = epi_net
                        else:
                            epi_net.BuildLabeledCPM(nBatch_supervised, image_batch, heatmap_batch[:,:,:,:num_of_joints+1], True)

                        tf.get_variable_scope().reuse_variables()
                        epi_net.BuildUnlabeledCPM(nBatch_unsupervised, image_ref_batch, image_src_batch, \
                                                  H1_batch, H01_batch, a_batch, b_batch, heatmap_unlabeled_batch, confidence_batch)
                        epi_net.TotalLoss()

                        tf.get_variable_scope().reuse_variables()
                        loss = epi_net.total_loss
                        # loss = epi_net.model_labeled.total_loss
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

        # with tf.Session() as sess:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            init = tf.global_variables_initializer()
            sess.run(init)

            saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1000)
            saver.restore(sess, pretrained_model)
            tf.train.start_queue_runners(sess=sess)

            ################################################
            ## Train network
            ################################################

            nVals = 10
            scale = 2
            for step in range(training_iterations):
                if step % 50 == 0:
                    for j in range(len(vValidationImage)):
                        ref_heatmap \
                            = sess.run(
                            model_val.model_ref.stage_heatmap[stages - 1],
                            feed_dict={model_val.image_ref: vValidationImage[j]})

                        # print("validation_loss: %f %f %f" % (total_loss, np.sum(unimodal_loss), np.sum(cross_loss)))

                        # print(ref_heatmap.shape)

                        vis_all = []
                        for image_id in range(10):
                            v = VisualizeJointHeatmap(vValidationImage[j][image_id, :, :],
                                                                      ref_heatmap[image_id, :, :, :num_of_joints],
                                                                      heatmap_size * scale)
                            if (image_id == 0):
                                vis_all = v
                            else:
                                vis_all = np.concatenate((vis_all, v), axis=0)

                        cv2.imwrite("vis/prediction_confi%03d_%05d.jpg" % (j, step), vis_all)

                    val_ref, val_src, val_comb, val_out_src, val_out_ref, val_loss_cross, loss1\
                        = sess.run([model_val.model_ref.stage_heatmap[stages - 1],
                                    model_val.model_src.stage_heatmap[stages - 1],
                                    model_val.pred01,
                                    model_val.out_src,
                                    model_val.out_ref,
                                    model_val.loss_cross,
                                    model_val.l2_loss],
                                   feed_dict={model_val.image_ref: validation_image_ref,
                                              model_val.image_src: validation_image_src,
                                              model_val.H1: validation_H1,
                                              model_val.H01: validation_H01,
                                              model_val.H01_inv: validation_H01_inv,
                                              model_val.a: validation_a,
                                              model_val.b: validation_b
                                              })

                    # print(val_out_ref)
                    #
                    # print(loss1)
                    # print(loss2)
                    print("val_loss: %f %f" % (val_loss_cross, 0))

                    vis_all = VisualizeJointHeatmap_confident_joint(validation_image_ref[0, :, :], val_ref[0, :, :, :num_of_joints],
                                                    heatmap_size * scale, validation_confidence[0,:])

                    for j in range(nVals):
                        v = VisualizeJointHeatmap(validation_image_src[j, :, :], val_src[j, :, :, :num_of_joints],
                                                  heatmap_size * scale)
                        vis_all = np.concatenate((vis_all, v), axis=0)

                    cv2.imwrite("vis/cross%05d.jpg" % (step), vis_all)

                    vis_all = VisualizeJointHeatmap_confident_joint(validation_image_ref[0, :, :], val_ref[0, :, :, :num_of_joints],
                                                    heatmap_size * scale, validation_confidence[0,:])

                    com = np.zeros((heatmap_size, heatmap_size, num_of_joints))
                    for j in range(nVals):
                        com += val_comb[j, :, :, :num_of_joints]

                    v = VisualizeJointHeatmap(validation_image_ref[j, :, :], com,
                                              heatmap_size * scale)
                    vis_all = np.concatenate((vis_all, v), axis=0)

                    for j in range(nVals):
                        v = VisualizeJointHeatmap(validation_image_ref[j, :, :], val_comb[j, :, :, :num_of_joints],
                                                  heatmap_size * scale)
                        vis_all = np.concatenate((vis_all, v), axis=0)

                    cv2.imwrite("vis/cross_j%05d.jpg" % (step), vis_all)



                    ######################3
                    ## Test
                    #
                    image = cv2.resize(validation_image_ref[0, :, :], (heatmap_size, heatmap_size),
                                       interpolation=cv2.INTER_LANCZOS4)
                    image = np.expand_dims(image, axis=0)
                    image = np.tile(image, np.stack([nBatch_unsupervised, 1, 1, 1]))

                    warped_image_np = sess.run(warped_image,
                                           feed_dict={image_placeholder: image,
                                                      H_ref_placeholder: validation_H01})

                    out_ref_np = np.expand_dims(val_out_ref, axis=2)
                    out_ref_np = np.tile(out_ref_np, np.stack([1, 1, heatmap_size + 2 * heatmap_extension_length, 1]))

                    out_src_np = np.expand_dims(val_out_src, axis=2)
                    out_src_np = np.tile(out_src_np, np.stack([1, 1, heatmap_size + 2 * heatmap_extension_length, 1]))

                    vis_all = VisualizeJointHeatmap1(warped_image_np[0, :, :], out_ref_np[0, :, :, :num_of_joints],
                                                    heatmap_size * scale)
                    v = VisualizeJointHeatmap1(warped_image_np[0, :, :], out_src_np[0, :, :, :num_of_joints],
                                              heatmap_size * scale)
                    vis_all = np.concatenate((vis_all, v), axis=1)

                    for j in range(1,nVals):
                        v = VisualizeJointHeatmap1(warped_image_np[j, :, :], out_ref_np[j, :, :, :num_of_joints],
                                                        heatmap_size * scale)
                        vis_all = np.concatenate((vis_all, v), axis=1)
                        v = VisualizeJointHeatmap1(warped_image_np[j, :, :], out_src_np[j, :, :, :num_of_joints],
                                                  heatmap_size * scale)
                        vis_all = np.concatenate((vis_all, v), axis=1)

                    cv2.imwrite("vis/test%05d.jpg" % (step), vis_all)



                _, loss_value, model_labeled_loss, model_ref_l2_loss, model_ref_confidence, model_modality, \
                global_step_value = sess.run([train_op,
                                              model_val.total_loss,
                                              model_val.model_labeled.total_loss,
                                              model_val.loss_cross,
                                              model_val.loss_confidence,
                                              model_val.loss_modality,
                                              global_step])

                # loss_value, model_labeled_loss, model_ref_l2_loss, model_ref_confidence, \
                # global_step_value = sess.run([
                #                               model_val.total_loss,
                #                               model_val.model_labeled.total_loss,
                #                               model_val.loss_cross,
                #                               model_val.loss_confidence,
                #                               global_step])

                # cv2.imwrite("vis/aa.jpg", vis_all)

                # writer.close()
                print(
                    "%d total loss: %0.2f (labeled: %0.2f / multiview: %0.2f / uni: %0.2f / conf: %0.2f)" % (global_step_value,
                    np.sum(loss_value), np.sum(model_labeled_loss), np.sum(model_ref_l2_loss), np.sum(model_modality), np.sum(model_ref_confidence)))

                if step % 200 == 0:
                    saver.save(sess=sess, save_path="./model/alg4.ckpt", global_step=global_step_value)
                    print('\nModel checkpoint saved...\n')
                if np.sum(loss_value) < 50:
                    saver.save(sess=sess, save_path="./model/alg4.ckpt", global_step=global_step_value)
                    print('\nModel checkpoint saved...\n')
                    break
    print('Training done.')
    return


if __name__ == '__main__':
    main()




