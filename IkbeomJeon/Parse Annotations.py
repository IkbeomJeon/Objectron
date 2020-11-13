
##

import glob
import os
import subprocess
from absl import app
from absl import flags

# import box as Box
import cv2
import numpy as np

from google.protobuf import text_format
from IPython.core.display import display, HTML
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
  return (object_keypoints_2d, object_categories, keypoint_size_list,
          annotation_types)

##

def grab_frame(video_file, frame_ids):
  """Grab an image frame from the video file."""
  frames = []
  capture = cv2.VideoCapture(video_file)
  height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
  capture.release()
  frame_size = width * height * 3

  for frame_id in frame_ids:
    frame_filter = r'select=\'eq(n\,{:d})\''.format(frame_id)
    command = [
        'ffmpeg', '-i', video_file, '-f', 'image2pipe', '-vf', frame_filter,
        '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-vsync', 'vfr', '-'
    ]
    pipe = subprocess.Popen(
        command, stdout=subprocess.PIPE, bufsize = 151 * frame_size)
    current_frame = np.fromstring(
        pipe.stdout.read(frame_size), dtype='uint8').reshape(width, height, 3)
    pipe.stdout.flush()

    frames.append(current_frame)
  return frames

def grab_frame_cv2(video_file, frame_id):
  """Grab an image frame from the video file."""
  capture = cv2.VideoCapture(video_file)
  height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

  capture.set(cv2.CAP_PROP_FRAME_COUNT, frame_id)
  ret, frame = capture.read()
  capture.release()

  return frame

##

def save_data(save_filename, video_filename, annotation_file, skip_volume, show_window):
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
      annotation, cat, num_keypoints, types = get_frame_annotation(sequence, frame_id)
      result = graphics.draw_annotation_on_image(frame, annotation, num_keypoints)

      result = cv2.resize(result, (480, 640), interpolation = cv2.INTER_AREA)

      if show_window == True:
        cv2.imshow("result", result)
        key = cv2.waitKey(1)
        if key == 'q':
          break

    frame_id+=1

  cap.release()
  cv2.destroyAllWindows()

  ##
root_path = "e:/mobilepose"
class_names = ['shoe', 'chair']

#batch_name = 'batch-1'
#video_id = 0
#annotation_file = f'{root_path}/annotations/{class_name}/{batch_name}/{video_id}.pbdata'
#video_filename = f'{root_path}/videos/{class_name}/{batch_name}/{video_id}/video.MOV'
#geometry_filename = f'{root_path}/videos/{class_name}/{batch_name}/{video_id}/geometry.pbdata'  # a.k.a. AR metadata
batch_id = 'batch-1/0'
for class_name in class_names:
  annotation_file = f'{root_path}/annotations/{class_name}/{batch_id}.pbdata'
  video_filename = f'{root_path}/videos/{class_name}/{batch_id}/video.MOV'
  geometry_filename = f'{root_path}/videos/{class_name}/{batch_id}/geometry.pbdata'  # a.k.a. AR metadata

  save_filename = "save.csv"
  save_data(save_filename, video_filename,annotation_file, 10, True)
