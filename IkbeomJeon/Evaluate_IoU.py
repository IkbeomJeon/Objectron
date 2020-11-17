##

import glob
import os
import subprocess

# import box as Box
import cv2
import numpy as np

from google.protobuf import text_format
#from IPython.core.display import display, HTML
import matplotlib.pyplot as plt

import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# The annotations are stored in protocol buffer format.
from objectron.schema import object_pb2 as object_protocol
from objectron.schema import annotation_data_pb2 as annotation_protocol
# The AR Metadata captured with each frame in the video
from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
from objectron.dataset import box as Box
from objectron.dataset import graphics
from objectron.dataset import iou
from objectron.dataset import box

##
def get_frame_annotation(sequence, frame_id):
  """Grab an annotated frame from the sequence."""
  data = sequence.frame_annotations[frame_id]
  object_id = 0
  object_keypoints_2d = []
  object_keypoints_3d = []
  object_rotations = []
  object_translations = []
  object_scale = []
  num_keypoints_per_object = []
  object_categories = []
  annotation_types = []
  # Get the camera for the current frame. We will use the camera to bring
  # the object from the world coordinate to the current camera coordinate.
  camera = np.array(data.camera.transform).reshape(4, 4)

  for obj in sequence.objects:
    rotation = np.array(obj.rotation).reshape(3, 3)
    translation = np.array(obj.translation)
    object_scale.append(np.array(obj.scale))
    transformation = np.identity(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = translation
    obj_cam = np.matmul(camera, transformation)
    object_translations.append(obj_cam[:3, 3])
    object_rotations.append(obj_cam[:3, :3])
    object_categories.append(obj.category)
    annotation_types.append(obj.type)

  keypoint_size_list = []
  for annotations in data.annotations:
    num_keypoints = len(annotations.keypoints)
    keypoint_size_list.append(num_keypoints)
    for keypoint_id in range(num_keypoints):
      keypoint = annotations.keypoints[keypoint_id]
      object_keypoints_2d.append(
          (keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth))
      object_keypoints_3d.append(
          (keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z))
    num_keypoints_per_object.append(num_keypoints)
    object_id += 1
  return (object_keypoints_2d, object_keypoints_3d, object_categories, keypoint_size_list,
          annotation_types)

def grab_video_and_fram_annotaiton_data(video_filename, annotation_file, skip_volume, show_window):
  sequence = annotation_protocol.Sequence()

  with open(annotation_file, 'rb') as pb:
    sequence.ParseFromString(pb.read())

  cap = cv2.VideoCapture(video_filename)

  frame_id = 0
  while(True):
    ret, frame2 = cap.read()
    if ret == False:
      break

    frame = cv2.flip(cv2.transpose(frame2), 1)
    if frame_id % skip_volume == 0:
      keypoints_2d, keypoints_3d, cat, num_keypoints, types = get_frame_annotation(sequence, frame_id)

      num_objects = len(num_keypoints)
      keypoints = np.split(keypoints_3d, np.array(np.cumsum(num_keypoints)))
      keypoints = [points.reshape(-1, 3) for points in keypoints]

      for object_id in range(num_objects):
        Evaluate_3DIOU(keypoints[object_id], keypoints[object_id], True)

      if show_window == True:
        result = graphics.draw_annotation_on_image(frame, keypoints_2d, num_keypoints)
        result = cv2.resize(result, (480, 640), interpolation=cv2.INTER_AREA)
        cv2.imshow("result", result)
        key = cv2.waitKey(1)

    frame_id+=1

  cap.release()
  cv2.destroyAllWindows()


def draw_boxes(boxes=[], clips=[], colors=['r', 'b', 'g', 'k']):
    """Draw a list of boxes.

        The boxes are defined as a list of vertices
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i, b in enumerate(boxes):
        x, y, z = b[:, 0], b[:, 1], b[:, 2]
        ax.scatter(x, y, z, c='r')
        for e in box.EDGES:
            ax.plot(x[e], y[e], z[e], linewidth=2, c=colors[i % len(colors)])

    if (len(clips)):
        points = np.array(clips)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=100, c='k')

    plt.gca().patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))

    # rotate the axes and update
    ax.view_init(30, 12)
    plt.draw()
    plt.show()

def Evaluate_3DIOU(v1, v2, bDrawbox):

    w1 = box.Box(vertices=v1)
    w2 = box.Box(vertices=v2)
    # Change the scale/position for testing
    #b1 = box.Box.from_transformation(np.array(w1.rotation), np.array(w1.translation), np.array([3., 1., 0.5]))
    #b2 = box.Box.from_transformation(np.array(w2.rotation), np.array(w2.translation), np.array([1.9, 1.6, 0.7]))
    b1 = w1
    b2 = w2
    # 0.3,
    loss = iou.IoU(b1, b2)
    print('iou = ', loss.iou())
    print('iou (via sampling)= ', loss.iou_sampling())
    intersection_points = loss.intersection_points

    if bDrawbox == True:
        draw_boxes([b1.vertices, b2.vertices], clips=loss.intersection_points)


def get_source_data_path(root_path, class_names):

  video_filepaths = []
  geometry_filepaths = []
  annotation_filepaths = []

  video_dirpath = f'{root_path}/videos'
  annotation_dirpath = f'{root_path}/annotations'

  for class_name in class_names:
    sub1 = f'{video_dirpath}/{class_name}'
    #for (paths, batch_names, files) in os.walk(target):
    batch_names = os.listdir(sub1)
    for batch_name in batch_names:
      sub2 = f'{sub1}/{batch_name}'
      idx_names = os.listdir(sub2)
      for idx_name in idx_names:
        sub3 = f'{sub2}/{idx_name}'

        video_filepaths.append(f'{sub3}/video.mov')
        geometry_filepaths.append(f'{sub3}/geometry.pbdata')
        annotation_filepaths.append(f'{annotation_dirpath}/{class_name}/{batch_name}/{idx_name}.pbdata')

  return video_filepaths, geometry_filepaths, annotation_filepaths

root_path = "../../Datasets/"
class_names = ['book', ]
skip_count = 10

video_filepaths, geometry_filepaths, annotation_filepaths = \
  get_source_data_path(root_path, class_names)

for i in range(len(video_filepaths)):
  grab_video_and_fram_annotaiton_data(video_filepaths[i], annotation_filepaths[i], skip_count, True)

