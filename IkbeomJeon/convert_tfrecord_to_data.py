##

#objectron_buckett = 'gs://objectron'
# Importing the necessary modules. We will run this notebook locally.

import tensorflow as tf
import glob
from IPython.core.display import display,HTML
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import cv2
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from objectron.schema import features
from objectron.dataset import box
from objectron.dataset import graphics


##

# Data pipeline for parsing the sequence examples. In this example, we grab a few frames from each sequence.

NUM_KEYPOINTS = 9
NUM_FRAMES = 4
NUM_PARALLEL_CALLS = 4

def parse_tfrecord(example):
    context, data = tf.io.parse_single_sequence_example(
        example,
        sequence_features=features.SEQUENCE_FEATURE_MAP,
        context_features=features.SEQUENCE_CONTEXT_MAP
    )

    # Number of frames in this video sequence
    num_examples = context['count']
    # The unique sequence id (class/batch-i/j) useful for visualization/debugging
    video_id = context['sequence_id']

    rand = tf.random.uniform([NUM_FRAMES], 0, num_examples, tf.int64)
    data['frame_ids'] = rand
    # Grab 4 random frames from the sequence and decode them
    for i in range(NUM_FRAMES):
        id = rand[i]
        image_tag = 'image-{}'.format(i)
        data[image_tag] = data[features.FEATURE_NAMES['IMAGE_ENCODED']][id]
        data[image_tag] = tf.image.decode_png(data[image_tag], channels=3)
        data[image_tag].set_shape([640, 480, 3])
    return context, data


# shards = tf.io.gfile.glob(objectron_buckett + '/v1/sequences/book/book_test*')
shards = "E:\\mobilepose\\records_shuffled\\book\\book_test-00000-of-00391"

dataset = tf.data.TFRecordDataset(shards)
dataset = dataset.map(parse_tfrecord)




##
# Data pipeline for parsing the sequence examples. In this example, we grab a few frames from each sequence.

NUM_KEYPOINTS = 9
NUM_FRAMES = 4


def parse_tfrecord(example):
    context, data = tf.io.parse_single_sequence_example(
        example,
        sequence_features=features.SEQUENCE_FEATURE_MAP,
        context_features=features.SEQUENCE_CONTEXT_MAP
    )

    # Number of frames in this video sequence
    num_examples = context['count']
    # The unique sequence id (class/batch-i/j) useful for visualization/debugging
    video_id = context['sequence_id']

    rand = tf.random.uniform([NUM_FRAMES], 0, num_examples, tf.int64)
    data['frame_ids'] = rand
    # Grab 4 random frames from the sequence and decode them
    for i in range(NUM_FRAMES):
        id = rand[i]
        image_tag = 'image-{}'.format(i)
        data[image_tag] = data[features.FEATURE_NAMES['IMAGE_ENCODED']][id]
        data[image_tag] = tf.image.decode_png(data[image_tag], channels=3)
        data[image_tag].set_shape([640, 480, 3])
    return context, data


# shards = tf.io.gfile.glob(objectron_buckett + '/v1/sequences/book/book_test*')
shards = "E:\\mobilepose\\sequences\\shoe\\shoe_test-00000-of-00424"

dataset = tf.data.TFRecordDataset(shards)
dataset = dataset.map(parse_tfrecord, num_parallel_calls = NUM_PARALLEL_CALLS)


##

##
num_rows = 100
for context, data in dataset.take(num_rows):
    fig, ax = plt.subplots(1, NUM_FRAMES, figsize=(12, 16))

    for i in range(NUM_FRAMES):
        num_frames = context['count']
        print(num_frames)
        id = data['frame_ids'][i]
        image = data['image-{}'.format(i)].numpy()
        num_instances = data[features.FEATURE_NAMES['INSTANCE_NUM']][id].numpy()[0]
        keypoints = data[features.FEATURE_NAMES['POINT_2D']].values.numpy().reshape(num_frames, num_instances,
                                                                                    NUM_KEYPOINTS, 3)
        for instance_id in range(num_instances):
            image = graphics.draw_annotation_on_image(image, keypoints[id, instance_id, :, :], [9])

        ax[i].grid(False)
        ax[i].imshow(image);
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)

    fig.tight_layout();
    plt.show()
