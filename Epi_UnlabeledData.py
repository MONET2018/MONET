import cv2
from utils import cpm_utils
from utils import tf_utils
import numpy as np
import math
import os

import tensorflow as tf
import time
import random
import matplotlib.pyplot as plt
from Utility.Epi_class import *
from Utility.EpiNet_Joint import *
from Utility.DataUtility import *
from Utility.GeometryUtility import *
from numpy import linalg as LA

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


tfr_file = 'js_training_data_unlabeled_pair.tfrecords'
tfr_file_ransac = 'js_training_data_unlabeled_ransac.tfrecords'
pretrained_model = "/media/yaoxx340/data/yaoxx340/cpm_ep/alg3_iter2.ckpt-595"
dataset_dir = ''

SHOW_INFO = False
img_size = 368
heatmap_size = 46
num_of_joints = 19
gaussian_radius = 1
heatmap_extension_length = 20
limb_threshold = 2
stages = 6
confidence_threshold = 0.4


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int32List(value=[value]))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

img_count = 0

vCamera = LoadCameraData('camera.txt') 
vCamera = LoadCameraIntrinsicData('intrinsic.txt', vCamera)
limb_idx = LoadLimbDefinitionData('limb_definition.txt')
num_of_limbs = len(limb_idx)
print(num_of_limbs)

pair_camera = GetPair(vCamera)

 

n = 0
for i in range(len(pair_camera)):
    n += len(pair_camera[i])
print(n)

val_path = 'validation/'
if not os.path.exists(val_path):
    os.makedirs(val_path)




gt_content = open('unlabel.txt', 'rb').readlines()
vCamera_new = []
time_instance = []
for idx, line in enumerate(gt_content):
    line = line.split()
    cur_img_path =  '/undis_img/' + line[0]
    cur_img = cv2.imread(cur_img_path)
    im_full = cur_img

    for iCamera in range(len(vCamera)):
        if vCamera[iCamera].frame == np.int(line[2]):
            camera_ref = vCamera[iCamera]

    print(cur_img_path)

    tmp = [float(x) for x in line[3:7]]
    cur_hand_bbox = [min([tmp[1], tmp[3]]), min([tmp[0], tmp[2]]), max([tmp[1], tmp[3]]), max([tmp[0], tmp[2]])]
    if cur_hand_bbox[0] < 0: cur_hand_bbox[0] = 0
    if cur_hand_bbox[1] < 0: cur_hand_bbox[1] = 0
    if cur_hand_bbox[2] > cur_img.shape[1]: cur_hand_bbox[2] = cur_img.shape[1]
    if cur_hand_bbox[3] > cur_img.shape[0]: cur_hand_bbox[3] = cur_img.shape[0]

    cur_hand_joints_y = [float(i) for i in line[7:60:2]]
    cur_hand_joints_x = [float(i) for i in line[8:60:2]]

    cur_img = cur_img[int(float(cur_hand_bbox[1])):int(float(cur_hand_bbox[3])),
              int(float(cur_hand_bbox[0])):int(float(cur_hand_bbox[2])), :]
    cur_hand_joints_x = [x - cur_hand_bbox[0] for x in cur_hand_joints_x]
    cur_hand_joints_y = [x - cur_hand_bbox[1] for x in cur_hand_joints_y]

    nInvisible = 0
    for i in range(len(cur_hand_joints_x)):
        x = cur_hand_joints_x[i]
        y = cur_hand_joints_y[i]
        if x < 0 or x > cur_img.shape[1] or y < 0 or y > cur_img.shape[0] :
            nInvisible +=1

    if (nInvisible > 5):
        continue

    output_image = np.ones(shape=(img_size, img_size, 3)) * 128

    if cur_img.shape[0] > cur_img.shape[1]:
        img_scale = img_size / (cur_img.shape[0] * 1.0)
        image = cv2.resize(cur_img, (0, 0), fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LANCZOS4)
        offset_x = int(img_size / 2 - math.floor(image.shape[1] / 2))
        offset_y = 0
    else:
        img_scale = img_size / (cur_img.shape[1] * 1.0)
        image = cv2.resize(cur_img, (0, 0), fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LANCZOS4)
        # heatmap_image = cv2.resize(cur_img, (0, 0), fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LANCZOS4)

        offset_x = 0
        offset_y = int(img_size / 2 - math.floor(image.shape[0] / 2))

    output_image[offset_y:offset_y+image.shape[0], offset_x:offset_x+image.shape[1],:] = image

    cropping_param = [int(float(cur_hand_bbox[0])),
                      int(float(cur_hand_bbox[1])),
                      int(float(cur_hand_bbox[2])),
                      int(float(cur_hand_bbox[3])),
                      offset_x,
                      offset_y,
                      img_scale
                      ]

    camera = Camera()
    camera.K = camera_ref.K
    camera.R = camera_ref.R
    camera.C = camera_ref.C
    camera.frame = camera_ref.frame
    camera.time = np.int(line[1])
    camera.image = np.uint8(output_image)
    camera.cropping_para = cropping_param
    vCamera_new.append(camera)

    isIn = False
    for i in range(len(time_instance)):
        if (time_instance[i] == camera.time):
            isIn = True
            break
    if (isIn == False):
        time_instance.append(camera.time)

vFrame = []
for iTime in range(len(time_instance)):
    vCamera_time = []
    frame = []
    for iCamera in range(len(vCamera_new)):
        if vCamera_new[iCamera].time == time_instance[iTime]:
            vCamera_time.append(vCamera_new[iCamera])
    for iImage in range(31):
        idx = -1
        for iCamera in range(len(vCamera_time)):
            if vCamera_time[iCamera].frame == iImage:
                idx = iCamera
                break
        if (idx < 0):
            continue

        camera = vCamera_time[iCamera]
        frame.append([camera.time, camera.frame])
        cv2.imwrite("validation/%07d_%07d.bmp" % (camera.time, camera.frame), camera.image)

    vFrame.append(frame)

SaveValidationData("validation/val_list.txt", vFrame)

gt_content = open('/media/yaoxx340/data/yaoxx340/panoptic-toolbox/scripts/171026_pose3/alg2.txt', 'rb').readlines()


for idx, line in enumerate(gt_content):
    line = line.split()
    cur_img_path = '/media/yaoxx340/data/yaoxx340/panoptic-toolbox/scripts/171026_pose3/undis_img/' + line[0]
    cur_img = cv2.imread(cur_img_path)
    im_full = cur_img

    for iCamera in range(len(vCamera)):
        if vCamera[iCamera].frame == np.int(line[2]):
            camera_ref = vCamera[iCamera]

    print(cur_img_path)

    tmp = [float(x) for x in line[3:7]]
    cur_hand_bbox = [min([tmp[1], tmp[3]]), min([tmp[0], tmp[2]]), max([tmp[1], tmp[3]]), max([tmp[0], tmp[2]])]
    if cur_hand_bbox[0] < 0: cur_hand_bbox[0] = 0
    if cur_hand_bbox[1] < 0: cur_hand_bbox[1] = 0
    if cur_hand_bbox[2] > cur_img.shape[1]: cur_hand_bbox[2] = cur_img.shape[1]
    if cur_hand_bbox[3] > cur_img.shape[0]: cur_hand_bbox[3] = cur_img.shape[0]

    cur_hand_joints_y = [float(i) for i in line[7:60:2]]
    cur_hand_joints_x = [float(i) for i in line[8:60:2]]

    cur_img = cur_img[int(float(cur_hand_bbox[1])):int(float(cur_hand_bbox[3])),
              int(float(cur_hand_bbox[0])):int(float(cur_hand_bbox[2])), :]
    cur_hand_joints_x = [x - cur_hand_bbox[0] for x in cur_hand_joints_x]
    cur_hand_joints_y = [x - cur_hand_bbox[1] for x in cur_hand_joints_y]

    nInvisible = 0
    for i in range(len(cur_hand_joints_x)):
        x = cur_hand_joints_x[i]
        y = cur_hand_joints_y[i]
        if x < 0 or x > cur_img.shape[1] or y < 0 or y > cur_img.shape[0]:
            nInvisible += 1

    if (nInvisible > 5):
        continue

    output_image = np.ones(shape=(img_size, img_size, 3)) * 128
    # heatmap_output_image = np.ones(shape=(heatmap_size, heatmap_size, 3)) * 128
    # output_heatmaps = np.zeros((heatmap_size, heatmap_size, num_of_joints))
    # output_limbs_heatmaps = np.zeros((heatmap_size, heatmap_size, num_of_limbs*2))

    if cur_img.shape[0] > cur_img.shape[1]:
        img_scale = img_size / (cur_img.shape[0] * 1.0)
        image = cv2.resize(cur_img, (0, 0), fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LANCZOS4)
        offset_x = int(img_size / 2 - math.floor(image.shape[1] / 2))
        offset_y = 0
    else:
        img_scale = img_size / (cur_img.shape[1] * 1.0)
        image = cv2.resize(cur_img, (0, 0), fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LANCZOS4)
        # heatmap_image = cv2.resize(cur_img, (0, 0), fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LANCZOS4)

        offset_x = 0
        offset_y = int(img_size / 2 - math.floor(image.shape[0] / 2))

    output_image[offset_y:offset_y + image.shape[0], offset_x:offset_x + image.shape[1], :] = image

    cropping_param = [int(float(cur_hand_bbox[0])),
                      int(float(cur_hand_bbox[1])),
                      int(float(cur_hand_bbox[2])),
                      int(float(cur_hand_bbox[3])),
                      offset_x,
                      offset_y,
                      img_scale
                      ]

    camera = Camera()
    camera.K = camera_ref.K
    camera.R = camera_ref.R
    camera.C = camera_ref.C
    camera.frame = camera_ref.frame
    camera.time = np.int(line[1])
    camera.image = np.uint8(output_image)
    # camera.im_full = im_full
    # camera.heatmap = output_heatmaps
    camera.cropping_para = cropping_param
    vCamera_new.append(camera)

    isIn = False
    for i in range(len(time_instance)):
        if (time_instance[i] == camera.time):
            isIn = True
            break
    if (isIn == False):
        time_instance.append(camera.time)


batch_size = 10
input_placeholder = tf.placeholder(dtype=tf.float32, shape=(batch_size, img_size, img_size, 3), name='input_placeholer')
# hmap_placeholder = tf.placeholder(dtype=tf.float32,
#                                   shape=(batch_size, heatmap_size, heatmap_size, num_of_joints + 1 + 2* num_of_limbs),
#                                   name='hmap_placeholder')
model = CPM_Model(stages, num_of_joints + 1)
model.build_model(input_placeholder, batch_size, False)

image_input = np.zeros((batch_size, img_size, img_size, 3))

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, pretrained_model)
    tf.train.start_queue_runners(sess=sess)

    n = 0
    for iBatch in range(0,len(vCamera_new),batch_size):
        for i in range(batch_size):
            if iBatch+i >= len(vCamera_new):
                continue
            im = vCamera_new[iBatch+i].image.astype(float)
            image_input[i,:,:,:] = im/255

        heatmap_np = sess.run(model.stage_heatmap[stages - 1], feed_dict={input_placeholder: image_input})
        # print(heatmap_np.shape)
        # heatmap_np = np.l2_normalize(heatmap_np, [1, 2])
        # print(heatmap_np.shape)
        for i in range(batch_size):
            if iBatch+i >= len(vCamera_new):
                continue

            vCamera_new[iBatch+i].conf = np.zeros((num_of_joints))
            vCamera_new[iBatch + i].heatmap1 = heatmap_np[i,:,:,:]

pair_ref = []
pair_src = []

print(time_instance)
# joint = LoadJointData('joint.txt')

for iTime in range(len(time_instance)):
    vCamera_time = []
    vP = []
    for iCamera in range(len(vCamera_new)):
        if vCamera_new[iCamera].time == time_instance[iTime]:
            vCamera_time.append(vCamera_new[iCamera])
            P = BuildCameraProjectionMatrix(vCamera_new[iCamera].K, vCamera_new[iCamera].R, vCamera_new[iCamera].C)
            vP.append(P)

    confident_joint = np.zeros((num_of_joints, len(vCamera_time)))
    confident_rank = np.zeros((num_of_joints, len(vCamera_time)))
    confident_rank.astype(int)
    joint = []
    for iJoint in range(num_of_joints):
        heatmap = np.zeros((len(vCamera_time), heatmap_size, heatmap_size))
        for iFrame in range(len(vCamera_time)):
            heatmap[iFrame,:,:] = vCamera_time[iFrame].heatmap1[:,:,iJoint]
        confident_joint[iJoint,:], confident_rank[iJoint,:], jj = GetConfidentView(heatmap, confidence_threshold)

        for iFrame in range(len(vCamera_time)):
            jj[iFrame, :] = Image2Heatmap(jj[iFrame, :], vCamera_time[iFrame].cropping_para[6], np.float(heatmap_size)/img_size, vCamera_time[iFrame].cropping_para[:2], vCamera_time[iFrame].cropping_para[4:6])
        joint.append(jj)

    Joint3D = []
    for iJoint in range(num_of_joints):

        set = []
        for iP in range(len(vP)): 
            # if joint[iJoint][iP,2] > 0.3:
            if confident_joint[iJoint,iP] > 0.4:
                set.append(iP)

        if len(set) < 2:
            max_conf = 0
            lst = []
            for iP in range(len(vP)): 
                lst.append(confident_joint[iJoint,iP])
            lst = np.array(lst)
            idx = lst.argsort()[-2:][::-1]
            for ii in idx:
                set.append(ii)

        vP_temp = []
        joint_temp = np.zeros((len(set),2))

        for iP in range(len(set)):
            vP_temp.append(vP[set[iP]])
            joint_temp[iP, :] = joint[iJoint][set[iP],:]


        X,n = RANSAC_Triangulation(vP_temp, joint_temp,200)
        Joint3D.append(X)

    # for iP in range(len(vP)):
    #     vCamera_time[iP].heatmap1 = np.zeros((heatmap_size, heatmap_size, num_of_joints))

    for iJoint in range(num_of_joints):
        for iP in range(len(vP)):
            x = Projection(vP[iP], Joint3D[iJoint])
            y = Heatmap2Image(x, vCamera_time[iP].cropping_para[6], np.float(heatmap_size) / img_size,
                              vCamera_time[iP].cropping_para[:2], vCamera_time[iP].cropping_para[4:6])
            ht = cpm_utils.make_gaussian(heatmap_size, float(gaussian_radius) * float(heatmap_size) / heatmap_size,
                                         y)

            vCamera_time[iP].heatmap1[:,:,iJoint] = ht

    for iP in range(len(vCamera_time)):
        vCamera_time[iP].heatmap1[:,:,-1] = np.ones((heatmap_size, heatmap_size)) - np.amax(vCamera_time[iP].heatmap1[:,:,:-1], axis=2)

    conf = np.sum(confident_joint, axis=0)
    print(conf)
    max_camera = np.argmax(conf)
    index_set = GetNearViewCamera(vCamera_time, max_camera, np.pi/5)
    # print("ref %d" % vCamera_time[max_camera].frame)
    # for i in range(len(index_set)):
    #     print(vCamera_time[index_set[i]].frame)

    for iJoint in range(num_of_joints):
        confident_joint[iJoint,:], confident_rank[iJoint,:] = RerankConfidentView(confident_joint[iJoint,:], index_set)

        for i in range(len(vCamera_time)-5):
            confident_joint[iJoint,np.int(confident_rank[iJoint,i])] = 0

    # print(confident_rank)
    # print(confident_joint)
    for iFrame in range(len(vCamera_time)):
        vCamera_time[iFrame].confident_joint = confident_joint[:,iFrame]

    # scale = 2
    # for iRank in range(len(vCamera_time)):
    #     image = np.zeros((img_size, img_size, 3, num_of_joints))
    #     ht = np.zeros((heatmap_size, heatmap_size, num_of_joints))
    #     conf = np.zeros((num_of_joints))
    #     for i in range(num_of_joints):
    #         # print(confident_rank[i, -iRank-1])
    #         image[:,:,:,i] = vCamera_time[np.int(confident_rank[i, -iRank-1])].image/255
    #         ht[:,:,i] = vCamera_time[np.int(confident_rank[i, -iRank-1])].heatmap1[:,:,i]
    #         conf[i] = vCamera_time[np.int(confident_rank[i, -iRank-1])].confident_joint[i]
    #
    #     v = VisualizeJointHeatmap_confident_joint_per_image(image, ht, heatmap_size * scale,
    #                                                         conf)
    #     cv2.imwrite("test%03d_%05d.jpg" % (iRank, iTime), v)

    for iFrame in range(len(vCamera_time)):
        idx_pair = -1
        for iPair in range(len(pair_camera)):
            # print(len(triple[iTriple]))
            # print(len(triple[iTriple]))
            # print(len(triple[iTriple][0]))
            if pair_camera[iPair][0][2] == vCamera_time[iFrame].frame:
                idx_pair = iPair
                break
        if idx_pair < 0:
            continue
        # nConfident_view = np.zeros((num_of_joints))
        for iPair in range(len(pair_camera[idx_pair])):
            frame1 = pair_camera[idx_pair][iPair][3]

            frame1_idx = -1
            for iFrame1 in range(len(vCamera_time)):
                if (vCamera_time[iFrame1].frame == frame1):
                    frame1_idx = iFrame1

            if frame1_idx == -1:
                continue
            pair_ref.append(vCamera_time[iFrame])
            pair_src.append(vCamera_time[frame1_idx])

tfr_writer = tf.python_io.TFRecordWriter(tfr_file)

img_count = 0
pair_save = []
for iPair in range(len(pair_ref)):

    camera_ref = pair_ref[iPair]
    camera_src = pair_src[iPair]

    es = EpiSiamese()
    es.SetParameter(camera_ref, camera_src, float(heatmap_size) / img_size, heatmap_extension_length)
    es.ComputeStereoRectification()

    pair = Pair()
    pair.time = camera_ref.time
    pair.ref = camera_ref.frame
    pair.src = camera_src.frame
    pair.H1 = es.H_stereo_rect1
    pair.H01 = es.H_stereo_rect01
    pair.a = es.a01.flatten()
    pair.b = es.b01.flatten()
    pair.confidence = camera_ref.confident_joint
    pair_save.append(pair)

    output_image_ref = es.camera_ref.image.astype(np.uint8).tostring()
    output_image_src = es.camera_src.image.astype(np.uint8).tostring()
    output_H1 = es.H_stereo_rect1.flatten().tolist()
    output_H01 = es.H_stereo_rect01.flatten().tolist()
    output_a = es.a01.flatten().tolist()
    output_b = es.b01.flatten().tolist()
    output_confident_joint = es.camera_ref.confident_joint.flatten().tolist()

 

    output_heatmap = es.camera_ref.heatmap1.flatten().tolist()
 
    raw_sample = tf.train.Example(features=tf.train.Features(
        feature={'image_ref': _bytes_feature(output_image_ref),
                 'image_src': _bytes_feature(output_image_src),
                 'H1': _float32_feature(output_H1),
                 'H01': _float32_feature(output_H01),
                 'a': _float32_feature(output_a),
                 'b': _float32_feature(output_b),
                 'frame': _float32_feature([es.camera_ref.time, es.camera_ref.frame]),
                 'heatmaps': _float32_feature(output_heatmap),
                 'confident_joint': _float32_feature(output_confident_joint)}))

    tfr_writer.write(raw_sample.SerializeToString())

    img_count += 1
print(img_count)

tfr_writer.close()
SavePairData("pair.txt", pair_save)



tfr_writer = tf.python_io.TFRecordWriter(tfr_file_ransac)
img_count = 0
for iTime in range(len(time_instance)):
    vCamera_time = []
    vP = []
    for iCamera in range(len(vCamera_new)):
        if vCamera_new[iCamera].time == time_instance[iTime]:
            vCamera_time.append(vCamera_new[iCamera])

    for iP in range(len(vCamera_time)):
        output_image_raw = vCamera_time[iP].image.astype(np.uint8).tostring()
        output_heatmaps_raw = vCamera_time[iP].heatmap1.flatten().tolist()
        # output_coords_raw = coords_set.flatten().tolist()
        output_cropping_param_raw = vCamera_time[iP].cropping_para
        output_R_raw = vCamera_time[iP].R.flatten().tolist()
        output_C_raw = vCamera_time[iP].C.flatten().tolist()
        output_K_raw = vCamera_time[iP].K.flatten().tolist()

        raw_sample = tf.train.Example(features=tf.train.Features(
            feature={'image': _bytes_feature(output_image_raw),
                     'heatmaps': _float32_feature(output_heatmaps_raw),
                     'cropping_param': _float32_feature(output_cropping_param_raw),
                     'K': _float32_feature(output_K_raw),
                     'R': _float32_feature(output_R_raw),
                     'C': _float32_feature(output_C_raw),
                     'frame': _float32_feature([np.float(line[1]), np.float(line[2])])}))

        tfr_writer.write(raw_sample.SerializeToString())

        img_count += 1
print(img_count)

tfr_writer.close()

