##
objectron_buckett = 'gs://objectron/v1/records_shuffled'
NUM_PARALLEL_CALLS = 4
WIDTH = 480
HEIGHT = 640
NUM_CHANNELS = 3
# The 3D bounding box has 9 vertices, 0: is the center, and the 8 vertices of the 3D box.
NUM_KEYPOINTS = 9
BATCH_SIZE = 4

##
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

def parse(example):
  """Parses a single tf.Example and decode the `png` string to an array."""
  data = tf.io.parse_single_example(example, features = features.FEATURE_MAP)
  data['image'] = tf.image.decode_png(data[features.FEATURE_NAMES['IMAGE_ENCODED']], channels=NUM_CHANNELS)
  data['image'].set_shape([HEIGHT, WIDTH, NUM_CHANNELS])
  print(data['image'].shape)
  return data

def augment(data):
  return data

def normalize(data):
  """Convert `image` from [0, 255] -> [-1., 1.] floats."""
  data['image'] = tf.cast(data['image'], tf.float32) * (2. / 255.) - 1.0
  return data

def load_tf_record(input_record):
  dataset = tf.data.TFRecordDataset(input_record)

  dataset = dataset.map(parse, num_parallel_calls = NUM_PARALLEL_CALLS)\
                   .map(augment, num_parallel_calls = NUM_PARALLEL_CALLS)\
                   .map(normalize, num_parallel_calls = NUM_PARALLEL_CALLS)
  # Our TF.records are shuffled in advance. If you re-generate the dataset from the video files, you'll need to
  # shuffle your examples. Keep in mind that you cannot shuffle the entire datasets using dataset.shuffle, since
  # it will be very slow.
  dataset = dataset.shuffle(100)\
                   .repeat()\
                   .batch(BATCH_SIZE)\
                   .prefetch(buffer_size=10)
  return dataset
#tf.load_library("f:\gcs_file_system.lo.lib")
#training_shards   = tf.io.gfile.glob(objectron_buckett + '/chair/chair_train*')
training_shards = "E:\\mobilepose\\records_shuffled\\book\\book_test-00000-of-00391"
dataset = load_tf_record(training_shards)

##

num_rows = 10
for data in dataset.take(num_rows):
    fig, ax = plt.subplots(1, BATCH_SIZE, figsize=(12, 16))
    # number_objects_batch is a tensor of shape (batch-size,) which tells the
    # number of objects in each batch slice.
    number_objects_batch = data[features.FEATURE_NAMES['INSTANCE_NUM']]
    num_obj_cumsum = np.sum(number_objects_batch)
    image_width = data[features.FEATURE_NAMES['IMAGE_WIDTH']]
    image_height = data[features.FEATURE_NAMES['IMAGE_HEIGHT']]
    # For debugging, Get the sample index from the IMAGE_FILENAME field.
    # print(data[features.FEATURE_NAMES['IMAGE_FILENAME']])
    keypoints = data[features.FEATURE_NAMES['POINT_2D']].values.numpy().reshape(np.sum(number_objects_batch),
                                                                                NUM_KEYPOINTS, 3)
    # The object annotation is a list of 3x1 keypoints for all the annotated
    # objects. The objects can have a varying number of keypoints. First we split
    # the list according to the number of keypoints for each object. This
    # also leaves an empty array at the end of the list.
    batch_keypoints = np.split(keypoints, np.array(np.cumsum(number_objects_batch)))
    # Visualize the first image/keypoint pair in the batch
    for id in range(BATCH_SIZE):
        w = image_width.numpy()[id][0]
        h = image_height.numpy()[id][0]
        # DeNormalize the image (for visualization purpose only)
        image = tf.cast((data['image'] + 1.0) / 2.0 * 255, tf.uint8).numpy()[id]
        num_instances = number_objects_batch[id].numpy()[0]
        keypoints_per_sample = batch_keypoints[id]
        for instance_id in range(num_instances):
            image = graphics.draw_annotation_on_image(image, keypoints_per_sample[instance_id, :, :], [9])

        ax[id].grid(False)
        ax[id].imshow(image);

    fig.tight_layout();
    plt.show()

