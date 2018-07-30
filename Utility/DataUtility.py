import tensorflow as tf
import numpy as np
from Epi_class import *
import math
import cv2

def SaveJointData(filename, vJoint, confidence):
    print(filename)
    with open(filename, "w") as f:
        f.write("%d\n" % len(vJoint))
        for i in range(len(vJoint)):
            f.write("%d\n" % vJoint[i].shape[0])
            for j in range(vJoint[i].shape[0]):
                f.write("%f %f %f " % (vJoint[i][j,0], vJoint[i][j,1], confidence[i, j]))
            f.write("\n")
        f.close()

def SaveMatrixData(filename, M):
    print(filename)
    with open(filename, "w") as f:
        f.write("%d %d\n" % (M.shape[0], M.shape[1]))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                f.write("%f " % M[i,j])
            f.write("\n")
        f.close()

def LoadJointData(filename):
    with open(filename) as f:
        nJoints = next(f).split()

        vJJ = []
        for i in range(np.int(nJoints[0])):
            nFrames = next(f).split()
            nFrames = np.int(nFrames[0])
            dummy = next(f).split()
            jj = np.zeros((np.int(nFrames), 3))
            for j in range(np.int(nFrames)):
                jj[j, 0] = np.float(dummy[3*j])
                jj[j, 1] = np.float(dummy[3 * j+1])
                jj[j, 2] = np.float(dummy[3 * j + 2])
            vJJ.append(jj)

        f.close()
    return vJJ

def SaveSkeletonData(filename, vFilename, vFrame, vBB, vJoint):
    print(filename)
    with open(filename, "w") as f:
        for i in range(len(vFilename)):
            f.write("%s " % vFilename[i])
            f.write("%d %d %d %d %d %d " % (vFrame[i][0], vFrame[i][1], int(vBB[i][1]), int(vBB[i][0]), int(vBB[i][3]), int(vBB[i][2])))
            for j in range(vJoint[i].shape[0]):
                f.write("%d %d " % (int(vJoint[i][j,1]), int(vJoint[i][j,0])))
            f.write("\n")
        f.close()

def SavePairData(filename, pair):
    print(filename)
    with open(filename, "w") as f:
        f.write("%d\n"%len(pair))
        for i in range(len(pair)):
            f.write("%d %d %d\n" % (pair[i].time, pair[i].ref, pair[i].src))
            for j1 in range(3):
                for j2 in range(3):
                    f.write("%f " % (pair[i].H1[j1, j2]))
            for j1 in range(3):
                for j2 in range(3):
                    f.write("%f " % (pair[i].H01[j1, j2]))
            f.write("%f %f " % (pair[i].a, pair[i].b))

            for j in range(len(pair[i].confidence)):
                f.write("%d " % pair[i].confidence[j])
            f.write("\n")

        f.close()

def SaveTripleData(filename, triple):
    print(filename)
    with open(filename, "w") as f:
        f.write("%d\n"%len(triple))
        for i in range(len(triple)):
            f.write("%d %d %d %d\n" % (triple[i].time, triple[i].ref, triple[i].src1, triple[i].src2))
            for j1 in range(3):
                for j2 in range(3):
                    f.write("%f " % (triple[i].H1[j1, j2]))
            for j1 in range(3):
                for j2 in range(3):
                    f.write("%f " % (triple[i].H01[j1, j2]))
            for j1 in range(3):
                for j2 in range(3):
                    f.write("%f " % (triple[i].H2[j1, j2]))
            for j1 in range(3):
                for j2 in range(3):
                    f.write("%f " % (triple[i].H02[j1, j2]))
            f.write("%f %f %f %f\n" % (triple[i].a1, triple[i].b1, triple[i].a2, triple[i].b2))

        f.close()

def LoadPairData(filename):
    pair_set = []
    with open(filename) as f:
        n = next(f).split()
        for i in range(np.int(n[0])):
            pair = Pair()
            dummy = next(f).split()
            pair.time = np.int(dummy[0])
            pair.ref = np.int(dummy[1])
            pair.src = np.int(dummy[2])

            dummy = next(f).split()
            idx = 0
            for j1 in range(3):
                for j2 in range(3):
                    pair.H1[j1,j2] = np.float(dummy[idx])
                    idx += 1
            for j1 in range(3):
                for j2 in range(3):
                    pair.H01[j1,j2] = np.float(dummy[idx])
                    idx += 1

            pair.a = np.float(dummy[idx])
            idx += 1
            pair.b = np.float(dummy[idx])
            idx += 1

            m = len(dummy[idx:])
            pair.confidence = np.zeros(m)
            for j in range(m):
                pair.confidence[j] = np.float(dummy[idx])
                idx+=1

            pair_set.append(pair)

        f.close()
    return pair_set


def LoadTripleData(filename):
    triple_set = []
    with open(filename) as f:
        n = next(f).split()
        for i in range(np.int(n[0])):
            triple = Triple()
            dummy = next(f).split()
            triple.time = np.int(dummy[0])
            triple.ref = np.int(dummy[1])
            triple.src1 = np.int(dummy[2])
            triple.src2 = np.int(dummy[3])

            dummy = next(f).split()
            idx = 0
            for j1 in range(3):
                for j2 in range(3):
                    triple.H1[j1,j2] = np.float(dummy[idx])
                    idx += 1
            for j1 in range(3):
                for j2 in range(3):
                    triple.H01[j1,j2] = np.float(dummy[idx])
                    idx += 1

            for j1 in range(3):
                for j2 in range(3):
                    triple.H2[j1,j2] = np.float(dummy[idx])
                    idx += 1
            for j1 in range(3):
                for j2 in range(3):
                    triple.H02[j1,j2] = np.float(dummy[idx])
                    idx += 1

            triple.a1 = np.float(dummy[idx])
            idx += 1
            triple.b1 = np.float(dummy[idx])
            idx += 1
            triple.a2 = np.float(dummy[idx])
            idx += 1
            triple.b2 = np.float(dummy[idx])
            idx += 1
            triple_set.append(triple)

        f.close()
    return triple_set

def SaveValidationData(filename, vFrame):
    print(filename)
    with open(filename, "w") as f:
        f.write("nFrame: %d\n" % len(vFrame))
        for i in range(len(vFrame)):
            f.write("%d %d "%(vFrame[i][0][0], len(vFrame[i])))
            for j in range(len(vFrame[i])):
                f.write("%d " % vFrame[i][j][1])
            f.write("\n")
        f.close()

def LoadValidationData(filename):
    with open(filename) as f:
        dummy, n = next(f).split()
        vFrame = []
        print(n)
        for i in range(np.int(n)):
            dummy = next(f).split()
            m = np.int(dummy[1])
            frame = np.zeros((m,2))
            for j in range(m):
                frame[j,0] = np.int(dummy[0])
                frame[j,1] = np.int(dummy[2+j])
            vFrame.append(frame)
        f.close()
    return vFrame

def LoadLimbDefinitionData(filename):
    with open(filename) as f:
        dummy, n = next(f).split()
        limb_link = []
        for i in range(int(n)):
            idx1, idx2 = next(f).split()
            limb_link.append([int(idx1), int(idx2)])
        f.close()
    return limb_link

def Load3DSkeletonData(filename):

    with open(filename) as f:
        dummy, n = next(f).split()
        skeleton = np.zeros((int(n), 3))
        for i in range(int(n)):
            dummy = next(f).split()
            if abs(float(dummy[1])+1) < 1e-5:
                return False, skeleton

            skeleton[i, 0] = float(dummy[5])
            skeleton[i, 1] = float(dummy[6])
            skeleton[i, 2] = float(dummy[7])

        f.close()
    return True, skeleton


def LoadCameraData(filename):
    vCamera = []
    with open(filename) as f:
        next(f).split()
        next(f).split()
        dummy, n = next(f).split()

        for i in range(int(n)):
            camera = Camera()

            dummy, camera.frame = next(f).split()
            camera.frame = int(camera.frame)

            c = next(f).split()
            camera.C[0, 0] = float(c[0])
            camera.C[1, 0] = float(c[1])
            camera.C[2, 0] = float(c[2])

            for j in range(3):
                r = next(f).split()
                camera.R[j, 0] = float(r[0])
                camera.R[j, 1] = float(r[1])
                camera.R[j, 2] = float(r[2])
            vCamera.append(camera)

        f.close()
    return vCamera

def LoadCameraIntrinsicData(filename, vCamera):
    with open(filename) as f:
        next(f).split()
        next(f).split()
        dummy, n = next(f).split()

        for i in range(int(n)):
            dummy, frame = next(f).split()
            frame = int(frame)
            idx = -1
            for j in range(len(vCamera)):
                if vCamera[j].frame == frame:
                    idx = j
                    break

            for j in range(3):
                r = next(f).split()
                vCamera[idx].K[j, 0] = float(r[0])
                vCamera[idx].K[j, 1] = float(r[1])
                vCamera[idx].K[j, 2] = float(r[2])

        f.close()
    return vCamera

def ReadTFData(tfr_path, img_size, hmap_size, num_joints, num_epochs=None):

    tfr_queue = tf.train.string_input_producer([tfr_path], num_epochs=None, shuffle=True)
    queue_images, queue_labels, queue_cropping_param, queue_K, queue_R, queue_C, queue_frame = ReadTFRData(tfr_queue, img_size, hmap_size, num_joints)

    return queue_images, queue_labels, queue_cropping_param, queue_K, queue_R, queue_C, queue_frame


def ReadTFRData(tfr_queue, img_size, hmap_size, num_joints, num_limbs):
    tfr_reader = tf.TFRecordReader()
    _, serialized_example = tfr_reader.read(tfr_queue)

    queue_images = []
    queue_labels = []
    queue_cropping_param = []
    queue_K = []
    queue_R = []
    queue_C = []
    queue_frame = []


    for i in range(1):
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image': tf.FixedLenFeature([], tf.string),
                                               'heatmaps': tf.FixedLenFeature([int(hmap_size * hmap_size * (num_joints + 1))], tf.float32),
                                               'cropping_param': tf.FixedLenFeature([7], tf.float32),
                                               'K': tf.FixedLenFeature([9], tf.float32),
                                               'R': tf.FixedLenFeature([9], tf.float32),
                                               'C': tf.FixedLenFeature([3], tf.float32),
                                               'frame': tf.FixedLenFeature([2], tf.float32)
                                           })

        # img_size = 128
        # center_radius = 11
        img = tf.decode_raw(features['image'], tf.uint8)
        img = tf.reshape(img, [img_size, img_size, 3])
        img = tf.cast(img, tf.float32)

        #img = img[..., ::-1]

        heatmap = tf.reshape(features['heatmaps'], [hmap_size, hmap_size, (num_joints + 1)])

        cropping_param = tf.reshape(features['cropping_param'], [7])

        K = tf.reshape(features['K'], [3, 3])
        R = tf.reshape(features['R'], [3, 3])
        C = tf.reshape(features['C'], [3, 1])

        frame = tf.reshape(features['frame'], [2])


        img /= 255.0


        queue_images.append(img)
        queue_labels.append(heatmap)
        queue_cropping_param.append(cropping_param)
        queue_K.append(K)
        queue_R.append(R)
        queue_C.append(C)
        queue_frame.append(frame)

    return queue_images, queue_labels, queue_cropping_param, queue_K, queue_R, queue_C, queue_frame


def ReadTFRDataUnlabeled(tfr_queue, img_size, hmap_size, num_joints, num_limbs):
    tfr_reader = tf.TFRecordReader()
    _, serialized_example = tfr_reader.read(tfr_queue)

    queue_image_ref = []
    queue_image_src1 = []
    queue_image_src2 = []
    queue_heatmap = []
    queue_H1 = []
    queue_H01 = []
    queue_H2 = []
    queue_H02 = []
    queue_a1 = []
    queue_b1 = []
    queue_a2 = []
    queue_b2 = []
    queue_frame = []
    queue_confident_joint = []

    for i in range(1):
        # features = tf.parse_single_example(serialized_example,
        #                                    features={
        #                                        'image_ref': tf.FixedLenFeature([], tf.string),
        #                                        'image_src1': tf.FixedLenFeature([], tf.string),
        #                                        'image_src2': tf.FixedLenFeature([], tf.string),
        #                                        'heatmaps': tf.FixedLenFeature([int(hmap_size * hmap_size * (num_joints + 1 + 2*num_limbs))], tf.float32),
        #                                        'H1': tf.FixedLenFeature([9], tf.float32),
        #                                        'H01': tf.FixedLenFeature([9], tf.float32),
        #                                        'H2': tf.FixedLenFeature([9], tf.float32),
        #                                        'H02': tf.FixedLenFeature([9], tf.float32),
        #                                        'a1': tf.FixedLenFeature([1], tf.float32),
        #                                        'b1': tf.FixedLenFeature([1], tf.float32),
        #                                        'a2': tf.FixedLenFeature([1], tf.float32),
        #                                        'b2': tf.FixedLenFeature([1], tf.float32),
        #                                        'frame': tf.FixedLenFeature([2], tf.float32),
        #                                        'confident_joint': tf.FixedLenFeature([num_joints], tf.float32)
        #                                    })

        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image_ref': tf.FixedLenFeature([], tf.string),
                                               'image_src1': tf.FixedLenFeature([], tf.string),
                                               'image_src2': tf.FixedLenFeature([], tf.string),
                                               'heatmaps': tf.FixedLenFeature([int(hmap_size * hmap_size * (num_joints + 1))], tf.float32),
                                               'H1': tf.FixedLenFeature([9], tf.float32),
                                               'H01': tf.FixedLenFeature([9], tf.float32),
                                               'H2': tf.FixedLenFeature([9], tf.float32),
                                               'H02': tf.FixedLenFeature([9], tf.float32),
                                               'a1': tf.FixedLenFeature([1], tf.float32),
                                               'b1': tf.FixedLenFeature([1], tf.float32),
                                               'a2': tf.FixedLenFeature([1], tf.float32),
                                               'b2': tf.FixedLenFeature([1], tf.float32),
                                               'frame': tf.FixedLenFeature([2], tf.float32),
                                               'confident_joint': tf.FixedLenFeature([num_joints], tf.float32)
                                           })

        # img_size = 128
        # center_radius = 11
        img = tf.decode_raw(features['image_ref'], tf.uint8)
        img = tf.reshape(img, [img_size, img_size, 3])
        img_ref = tf.cast(img, tf.float32)
        #img_ref = img[..., ::-1]

        img = tf.decode_raw(features['image_src1'], tf.uint8)
        img = tf.reshape(img, [img_size, img_size, 3])
        img_src1 = tf.cast(img, tf.float32)
        #img_src1 = img[..., ::-1]

        img = tf.decode_raw(features['image_src2'], tf.uint8)
        img = tf.reshape(img, [img_size, img_size, 3])
        img_src2 = tf.cast(img, tf.float32)
        #img_src2 = img[..., ::-1]

        heatmap = tf.reshape(features['heatmaps'], [hmap_size, hmap_size, (num_joints + 1)])

        H1 = tf.reshape(features['H1'], [3, 3])
        H01 = tf.reshape(features['H01'], [3, 3])
        H2 = tf.reshape(features['H2'], [3, 3])
        H02 = tf.reshape(features['H02'], [3, 3])

        a1 = tf.reshape(features['a1'], [1])
        a2 = tf.reshape(features['a2'], [1])
        b1 = tf.reshape(features['b1'], [1])
        b2 = tf.reshape(features['b2'], [1])

        frame = tf.reshape(features['frame'], [2])

        confident_joint = tf.reshape(features['confident_joint'], [num_joints])

        img_ref /= 255.0
        img_src1 /= 255.0
        img_src2 /= 255.0

        queue_image_ref.append(img_ref)
        queue_image_src1.append(img_src1)
        queue_image_src2.append(img_src2)

        queue_heatmap.append(heatmap)
        queue_H1.append(H1)
        queue_H01.append(H01)
        queue_H2.append(H2)
        queue_H02.append(H02)

        queue_a1.append(a1)
        queue_b1.append(b1)

        queue_a2.append(a2)
        queue_b2.append(b2)

        queue_frame.append(frame)
        queue_confident_joint.append(confident_joint)

    return queue_image_ref, queue_image_src1, queue_image_src2, queue_heatmap, queue_H1, queue_H01, queue_H2, queue_H02, queue_a1, queue_b1, queue_a2, queue_b2, queue_frame, queue_confident_joint


def ReadTFRDataUnlabeled_Pair(tfr_queue, img_size, hmap_size, num_joints, num_limbs):
    tfr_reader = tf.TFRecordReader()
    _, serialized_example = tfr_reader.read(tfr_queue)

    queue_image_ref = []
    queue_image_src = []
    queue_H1 = []
    queue_H01 = []
    queue_a = []
    queue_b = []
    queue_frame = []
    queue_heatmap = []
    queue_confidence = []

    for i in range(1):
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image_ref': tf.FixedLenFeature([], tf.string),
                                               'image_src': tf.FixedLenFeature([], tf.string),
                                               'H1': tf.FixedLenFeature([9], tf.float32),
                                               'H01': tf.FixedLenFeature([9], tf.float32),
                                               'a': tf.FixedLenFeature([1], tf.float32),
                                               'b': tf.FixedLenFeature([1], tf.float32),
                                               'frame': tf.FixedLenFeature([2], tf.float32),
                                               'heatmaps': tf.FixedLenFeature([int(hmap_size * hmap_size * (num_joints + 1))], tf.float32),
                                               'confident_joint': tf.FixedLenFeature([num_joints], tf.float32)
                                           })

        img = tf.decode_raw(features['image_ref'], tf.uint8)
        img = tf.reshape(img, [img_size, img_size, 3])
        img_ref = tf.cast(img, tf.float32)
        #img_ref = img[..., ::-1]

        img = tf.decode_raw(features['image_src'], tf.uint8)
        img = tf.reshape(img, [img_size, img_size, 3])
        img_src = tf.cast(img, tf.float32)
        #img_src = img[..., ::-1]

        heatmap = tf.reshape(features['heatmaps'], [hmap_size, hmap_size, (num_joints + 1)])

        H1 = tf.reshape(features['H1'], [3, 3])
        H01 = tf.reshape(features['H01'], [3, 3])

        a = tf.reshape(features['a'], [1])
        b = tf.reshape(features['b'], [1])

        frame = tf.reshape(features['frame'], [2])

        confident_joint = tf.reshape(features['confident_joint'], [num_joints])

        img_ref /= 255.0
        img_src /= 255.0

        queue_image_ref.append(img_ref)
        queue_image_src.append(img_src)

        queue_H1.append(H1)
        queue_H01.append(H01)

        queue_a.append(a)
        queue_b.append(b)

        queue_heatmap.append(heatmap)
        queue_confidence.append(confident_joint)
        queue_frame.append(frame)

    return queue_image_ref, queue_image_src, queue_H1, queue_H01, queue_a, queue_b, queue_frame, queue_heatmap, queue_confidence


def ReadShuffledBatchData(tfr_path, img_size, hmap_size, num_joints, num_limbs, batch_size=20, num_epochs=None):
    with tf.name_scope('Batch_Inputs'):
        tfr_queue = tf.train.string_input_producer([tfr_path], num_epochs=None, shuffle=True)

        data_list = [ReadTFRData(tfr_queue, img_size, hmap_size, num_joints, num_limbs) for _ in range(2)]

        batch_images, batch_labels, batch_cropping_param, batch_K, batch_R, batch_C, batch_frame = tf.train.shuffle_batch_join(data_list,
                                                                                                   batch_size=batch_size,
                                                                                                   capacity=10 + 10 * batch_size,
                                                                                                   min_after_dequeue=2*batch_size,
                                                                                                   enqueue_many=True,
                                                                                                   name='batch_data_read')


    return batch_images, batch_labels, batch_cropping_param, batch_K, batch_R, batch_C, batch_frame


def ReadShuffledBatchDataUnlabeled(tfr_path, img_size, hmap_size, num_joints, num_limbs, batch_size=20, num_epochs=None):
    with tf.name_scope('Batch_Inputs'):
        tfr_queue = tf.train.string_input_producer([tfr_path], num_epochs=None, shuffle=True)

        data_list = [ReadTFRDataUnlabeled(tfr_queue, img_size, hmap_size, num_joints, num_limbs) for _ in range(2)]

        # batch_images, batch_labels, batch_cropping_param, batch_K, batch_R, batch_C, batch_frame = tf.train.batch_join(
        #     data_list,
        #     batch_size=batch_size,
        #     capacity=10 + 6 * batch_size,
        #     # min_after_dequeue=10,
        #     enqueue_many=True,
        #     name='batch_data_read')

        # queue_image_ref, queue_image_src1, queue_image_src2, queue_heatmap, \
        # queue_H1, queue_H01, queue_H2, queue_H02, \
        # queue_a1, queue_b1, queue_a2, queue_b2, queue_frame, \
        # queue_confident_joint                                        = tf.train.shuffle_batch_join(data_list,
        #                                                                                            batch_size=batch_size,
        #                                                                                            capacity=10 + 6 * batch_size,
        #                                                                                            min_after_dequeue=2*batch_size,
        #                                                                                            enqueue_many=True,
        #                                                                                            name='batch_data_read')


        queue_image_ref, queue_image_src1, queue_image_src2, queue_heatmap, \
        queue_H1, queue_H01, queue_H2, queue_H02, \
        queue_a1, queue_b1, queue_a2, queue_b2, queue_frame, queue_confident_joint                                        = tf.train.shuffle_batch_join(data_list,
                                                                                                   batch_size=batch_size,
                                                                                                   capacity=10 + 10 * batch_size,
                                                                                                   min_after_dequeue=2*batch_size,
                                                                                                   enqueue_many=True,
                                                                                                   name='batch_data_read')

    return queue_image_ref, queue_image_src1, queue_image_src2, queue_heatmap, queue_H1, queue_H01, queue_H2, queue_H02, queue_a1, queue_b1, queue_a2, queue_b2, queue_frame, queue_confident_joint


def ReadShuffledBatchDataUnlabeled_Pair(tfr_path, img_size, hmap_size, num_joints, num_limbs, batch_size=20, num_epochs=None):
    with tf.name_scope('Batch_Inputs'):
        tfr_queue = tf.train.string_input_producer([tfr_path], num_epochs=None, shuffle=True)

        data_list = [ReadTFRDataUnlabeled_Pair(tfr_queue, img_size, hmap_size, num_joints, num_limbs) for _ in range(2)]


        queue_image_ref, queue_image_src, \
        queue_H1, queue_H01, \
        queue_a, queue_b, queue_frame,\
        queue_heatmap, queue_confidence \
            = tf.train.shuffle_batch_join(data_list,
                                           batch_size=batch_size,
                                           capacity=10 + 10 * batch_size,
                                           min_after_dequeue=2*batch_size,
                                           enqueue_many=True,
                                           name='batch_data_read')

    return queue_image_ref, queue_image_src, queue_H1, queue_H01, queue_a, queue_b, queue_frame, queue_heatmap, queue_confidence



def ReadBatchData(tfr_path, img_size, hmap_size, num_joints, num_limbs, batch_size=20, num_epochs=None):
    with tf.name_scope('Batch_Inputs'):
        tfr_queue = tf.train.string_input_producer([tfr_path], num_epochs=None, shuffle=True)

        data_list = [ReadTFRData(tfr_queue, img_size, hmap_size, num_joints, num_limbs) for _ in range(2)]

        batch_images, batch_labels, batch_cropping_param, batch_K, batch_R, batch_C, batch_frame = tf.train.batch_join(
            data_list,
            batch_size=batch_size,
            capacity=10 + 10 * batch_size,
            # min_after_dequeue=10,
            enqueue_many=True,
            name='batch_data_read1')

        # batch_images, batch_labels, batch_cropping_param, batch_K, batch_R, batch_C, batch_frame = tf.train.shuffle_batch_join(data_list,
        #                                                                                            batch_size=batch_size,
        #                                                                                            capacity=10 + 6 * batch_size,
        #                                                                                            min_after_dequeue=10,
        #                                                                                            enqueue_many=True,
        #                                                                                            name='batch_data_read')



    return batch_images, batch_labels, batch_cropping_param, batch_K, batch_R, batch_C, batch_frame


