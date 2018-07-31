import tensorflow as tf
import utils.cpm_utils as cpm_utils


def read_and_decode_cpm(tfr_queue, img_size, hmap_size, num_joints):
    tfr_reader = tf.TFRecordReader()
    _, serialized_example = tfr_reader.read(tfr_queue)

    queue_images = []
    queue_labels = []


    for i in range(1):
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image': tf.FixedLenFeature([], tf.string),
                                               'heatmaps': tf.FixedLenFeature(
                                                   [int(hmap_size * hmap_size * (num_joints + 1))], tf.float32)
                                           })

        # img_size = 128
        # center_radius = 11
        img = tf.decode_raw(features['image'], tf.uint8)
        img = tf.reshape(img, [img_size, img_size, 3])
        img = tf.cast(img, tf.float32)

        #img = img[..., ::-1]

        heatmap = tf.reshape(features['heatmaps'], [hmap_size, hmap_size, (num_joints + 1)])



        img /= 255.0


        queue_images.append(img)
        queue_labels.append(heatmap)

    return queue_images, queue_labels


def read_batch_cpm(tfr_path, img_size, hmap_size, num_joints, batch_size=20, num_epochs=None):
    with tf.name_scope('Batch_Inputs'):
        tfr_queue = tf.train.string_input_producer(tfr_path, num_epochs=None, shuffle=True)

        data_list = [read_and_decode_cpm(tfr_queue, img_size, hmap_size, num_joints) for _ in range(2)]

        batch_images, batch_labels = tf.train.shuffle_batch_join(data_list,
                                                                                                   batch_size=batch_size,
                                                                                                   capacity=10 + 6 * batch_size,
                                                                                                   min_after_dequeue=10,
                                                                                                   enqueue_many=True,
                                                                                                   name='batch_data_read')



    return batch_images, batch_labels




