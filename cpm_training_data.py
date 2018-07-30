import cv2
from utils import cpm_utils
from utils import tf_utils
import numpy as np
import math
import tensorflow as tf
import time
import random
import os


tfr_file = 'alg3_iter2.tfrecords'
dataset_dir = ''

SHOW_INFO = False
img_size = 368
heatmap_size = 46
num_of_joints = 19
gaussian_radius = 1


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int32List(value=[value]))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))



tfr_writer = tf.python_io.TFRecordWriter(tfr_file)

img_count = 0

gt_content = open('/media/yaoxx340/data/yaoxx340/panoptic-toolbox/scripts/171204_pose3/alg3_iter2.txt', 'rb').readlines()

for idx, line in enumerate(gt_content):
    line = line.split()
    cur_img_path = '/media/yaoxx340/data/yaoxx340/panoptic-toolbox/scripts/171204_pose3/undis_img/' + line[0]
    cur_img = cv2.imread(cur_img_path)
    if os.path.isfile(cur_img_path) == False:
        continue
    
    print(cur_img_path)
    tmp = [float(x) for x in line[3:7]]
    cur_hand_bbox = [min([tmp[1], tmp[3]]), min([tmp[0], tmp[2]]),max([tmp[1], tmp[3]]),max([tmp[0], tmp[2]])]
    if cur_hand_bbox[0] < 0: cur_hand_bbox[0] = 0
    if cur_hand_bbox[1] < 0: cur_hand_bbox[1] = 0
    if cur_hand_bbox[2] > cur_img.shape[1]: cur_hand_bbox[2] = cur_img.shape[1]
    if cur_hand_bbox[3] > cur_img.shape[0]: cur_hand_bbox[3] = cur_img.shape[0]

    cur_hand_joints_y = [float(i) for i in line[7:60:2]]

    cur_hand_joints_x = [float(i) for i in line[8:60:2]]


    cur_img = cur_img[int(float(cur_hand_bbox[1])):int(float(cur_hand_bbox[3])),int(float(cur_hand_bbox[0])):int(float(cur_hand_bbox[2])),:]
    cur_hand_joints_x = [x - cur_hand_bbox[0] for x in cur_hand_joints_x]
    cur_hand_joints_y = [x - cur_hand_bbox[1] for x in cur_hand_joints_y]
    print(cur_img.shape)
    print(cur_hand_joints_x)


    output_image = np.ones(shape=(img_size, img_size, 3)) * 128
    heatmap_output_image = np.ones(shape=(heatmap_size, heatmap_size, 3)) * 128
    output_heatmaps = np.zeros((heatmap_size, heatmap_size, num_of_joints))

    # Resize and pad image to fit output image size
    # if h > w
    if cur_img.shape[0] > cur_img.shape[1]:
        img_scale = img_size / (cur_img.shape[0] * 1.0)
        heatmap_scale = heatmap_size / (cur_img.shape[0] * 1.0)

        # Relocalize points
        cur_hand_joints_x = map(lambda x: x * heatmap_scale, cur_hand_joints_x)
        cur_hand_joints_y = map(lambda x: x * heatmap_scale, cur_hand_joints_y)

        # Resize image
        image = cv2.resize(cur_img, (0, 0), fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LANCZOS4)
        offset = image.shape[1] % 2

        heatmap_image = cv2.resize(cur_img, (0, 0), fx=heatmap_scale, fy=heatmap_scale, interpolation=cv2.INTER_LANCZOS4)
        heatmap_offset = heatmap_image.shape[1] % 2

        output_image[:, int(img_size / 2 - math.floor(image.shape[1] / 2)): int(img_size / 2 + math.floor(image.shape[1] / 2) + offset), :] = image
        heatmap_output_image[:, int(heatmap_size / 2 - math.floor(heatmap_image.shape[1] / 2)): int(heatmap_size / 2 + math.floor(heatmap_image.shape[1] / 2) + heatmap_offset), :] = heatmap_image
        cur_hand_joints_x = map(lambda x: x + (heatmap_size / 2 - math.floor(heatmap_image.shape[1] / 2)),cur_hand_joints_x)

        cur_hand_joints_x = np.asarray(cur_hand_joints_x)
        cur_hand_joints_y = np.asarray(cur_hand_joints_y)

        if SHOW_INFO:
            hmap = np.zeros((heatmap_size, heatmap_size))
            # Plot joints
            for i in range(num_of_joints):
                cv2.circle(heatmap_output_image, (int(cur_hand_joints_x[i]), int(cur_hand_joints_y[i])), 1, (0, 255, 0), 2)

                # Generate joint gaussian map
                part_heatmap = cpm_utils.make_gaussian(heatmap_output_image.shape[0], gaussian_radius,[cur_hand_joints_x[i], cur_hand_joints_y[i]])
                hmap += part_heatmap * 50
        else:
            for i in range(num_of_joints):
                output_heatmaps[:, :, i] = cpm_utils.make_gaussian(heatmap_size, gaussian_radius,[cur_hand_joints_x[i], cur_hand_joints_y[i]])

    else:
        img_scale = img_size / (cur_img.shape[1] * 1.0)
        heatmap_scale = heatmap_size / (cur_img.shape[1] * 1.0)

        # Relocalize points
        cur_hand_joints_x = map(lambda x: x * heatmap_scale, cur_hand_joints_x)
        cur_hand_joints_y = map(lambda x: x * heatmap_scale, cur_hand_joints_y)

        # Resize image
        image = cv2.resize(cur_img, (0, 0), fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LANCZOS4)
        offset = image.shape[0] % 2

        heatmap_image = cv2.resize(cur_img, (0, 0), fx=heatmap_scale, fy=heatmap_scale, interpolation=cv2.INTER_LANCZOS4)
        heatmap_offset = heatmap_image.shape[0] % 2

        output_image[int(img_size / 2 - math.floor(image.shape[0] / 2)): int(img_size / 2 + math.floor(image.shape[0] / 2) + offset), :, :] = image
        heatmap_output_image[int(heatmap_size / 2 - math.floor(heatmap_image.shape[0] / 2)): int(heatmap_size / 2 + math.floor(heatmap_image.shape[0] / 2) + heatmap_offset), :, :] = heatmap_image
        cur_hand_joints_y = map(lambda x: x + (heatmap_size / 2 - math.floor(heatmap_image.shape[0] / 2)),cur_hand_joints_y)

        cur_hand_joints_x = np.asarray(cur_hand_joints_x)
        cur_hand_joints_y = np.asarray(cur_hand_joints_y)
        
        if SHOW_INFO:
            hmap = np.zeros((heatmap_size, heatmap_size))
            # Plot joints
            for i in range(num_of_joints):
                cv2.circle(heatmap_output_image, (int(cur_hand_joints_x[i]), int(cur_hand_joints_y[i])), 1, (0, 255, 0), 2)

                # Generate joint gaussian map
                part_heatmap = cpm_utils.make_gaussian(heatmap_output_image.shape[0], gaussian_radius,[cur_hand_joints_x[i], cur_hand_joints_y[i]])
                hmap += part_heatmap * 50
        else:
            for i in range(num_of_joints):
                output_heatmaps[:, :, i] = cpm_utils.make_gaussian(heatmap_size, gaussian_radius,[cur_hand_joints_x[i], cur_hand_joints_y[i]])
    if SHOW_INFO:
        #cv2.imshow('', hmap.astype(np.uint8))
        #cv2.imshow('i', output_image.astype(np.uint8))
        #cv2.waitKey(0)
        cv2.imwrite("training_data/"+str(img_count)+".png", output_image.astype(np.uint8))

    # Create background map
    output_background_map = np.ones((heatmap_size, heatmap_size)) - np.amax(output_heatmaps, axis=2)
    output_heatmaps = np.concatenate((output_heatmaps, output_background_map.reshape((heatmap_size, heatmap_size, 1))),axis=2)
    #print(output_heatmaps.shape)
    '''
    cv2.imshow('', (output_background_map*255).astype(np.uint8))
    cv2.imshow('h', (np.amax(output_heatmaps[:, :, 0:21], axis=2)*255).astype(np.uint8))
    cv2.waitKey(1000)
    '''


    coords_set = np.concatenate((np.reshape(cur_hand_joints_x, (num_of_joints, 1)),np.reshape(cur_hand_joints_y, (num_of_joints, 1))), axis=1)

    output_image_raw = output_image.astype(np.uint8).tostring()
    output_heatmaps_raw = output_heatmaps.flatten().tolist()
    output_coords_raw = coords_set.flatten().tolist()

    raw_sample = tf.train.Example(features=tf.train.Features(feature={'image': _bytes_feature(output_image_raw),'heatmaps': _float32_feature(output_heatmaps_raw)}))

    tfr_writer.write(raw_sample.SerializeToString())

    img_count += 1


tfr_writer.close()
