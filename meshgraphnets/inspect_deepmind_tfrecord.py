# python -m meshgraphnets.inspect_deepmind_tfrecord
# For filtering out trajectories that have too much global motion

import tensorflow.compat.v1 as tf

import os
import json
import numpy as np
import sys



from meshgraphnets import dataset
import tensorflow_graphics.geometry.transformation as tfg_transformation
from tensorflow_graphics.geometry.transformation import rotation_matrix_common
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import tfg_flags
tf.enable_eager_execution()


sys.path.append("..")
import data_creator_utils as dcu



def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(cells, node_type, node_mod, mesh_pos, world_pos, gripper_pos, tfn, stress, pd_stress, force, mesh_edges, world_edges):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
    'cells': _bytes_feature(cells), #int32
    'node_type': _bytes_feature(node_type), #int32
    'node_mod': _bytes_feature(node_mod), #float32
    'mesh_pos': _bytes_feature(mesh_pos), #float32
    'world_pos': _bytes_feature(world_pos), #float 32
    'gripper_pos': _bytes_feature(gripper_pos),
    'tfn': _bytes_feature(tfn),
    'stress': _bytes_feature(stress),
    'pd_stress': _bytes_feature(pd_stress),
    'force': _bytes_feature(force),
    'mesh_edges': _bytes_feature(mesh_edges),
    'world_edges': _bytes_feature(world_edges)

  }

  # Create a Features message using tf.train.Example.
  return tf.train.Example(features=tf.train.Features(feature=feature))




def evaluator_file_split():
  # Iterate through objects
  if os.environ["LOCAL_FLAG"] != "1": # If on NGC
    pass

  else:
    last_folder = '0-99_tfrecords/5e4_pd'
    total_folder = os.path.join('../../results_mgn_fresh/0-9999/', last_folder)
    total_files = [os.path.join(last_folder, k.split(".")[0]) for k in os.listdir(total_folder)]
    train_files = [k for k in total_files if '00000015' in k] # trapezoidal prism
    train_files = [k for k in total_files if '00000016' in k] # rectangular prism
    # train_files = [k for k in total_files if '00000007' in k] # circular disk

    test_files = train_files



  return train_files, test_files

write = False

files_folder = 'meshgraphnets/deforming_plate_data/deforming_plate'
tfrecord_files = [os.path.join(files_folder, k.split(".")[0]) for k in os.listdir(files_folder)]
print(tfrecord_files)
quit()

train_files, test_files = evaluator_file_split()

ds = dataset.load_dataset('../../results_mgn_fresh/0-9999', train_files, 1)


print("Inspecting data------\n")
counter = 0
# print(ds)

sorted_R_list = [29, 78, 75, 80, 98, 37, 35, 51, 20, 65, 24, 47, 8, 60, 19, 59, 14, 9, 46, 6, 96, 44, 64, 38, 91, 85, 87, 36, 12, 88, 49, 86, 10, 79, 45, 27, \
94, 55, 62, 90, 58, 61, 7, 89, 18, 13, 5, 28, 83, 22, 11, 50, 93, 25, 48, 54, 53, 41, 39, 42, 23, 97, 70, 34, 33, 1, 26, 31, 3, 77, 73, 52, 2, 56, 17, \
66, 72, 4, 30, 81, 15, 84, 76, 69, 74, 32, 68, 95, 63, 67, 82, 21, 57, 43, 0, 40, 16, 92, 71]
sorted_R_list = sorted_R_list[:int(len(sorted_R_list)/2)]


if write:
  writer = tf.python_io.TFRecordWriter(os.path.join('../../results_mgn_fresh/0-9999/0-99_tfrecords/5e4_pd_filtered', '00000016' + '.tfrecord'))


with open(os.path.join('../../results_mgn_fresh/0-9999', 'meta.json'), 'r') as fp:
    meta = json.loads(fp.read())
examples = []
for counter, example in enumerate(ds):

    print("Counter", counter)
    if counter not in sorted_R_list:
      continue
        
    '''
    print("-----")
    for k in example.keys():
      print(k, example[k].shape, type(k))
    '''
    



    cells = example['cells'].numpy()
    cells = cells[0,...]

    node_type = example['node_type'].numpy()
    node_type = node_type[0,...]

    node_mod = example['node_mod'].numpy()
    node_mod = node_mod[0,...]

    mesh_pos = example['mesh_pos'].numpy()
    mesh_pos = mesh_pos[0,...]

    world_pos = example['world_pos'].numpy()
    gripper_pos = example['gripper_pos'].numpy()

    tfn = example['tfn'].numpy()
    tfn = tfn[0,...]

    forces = example['force'].numpy()
    stresses = example['stress'].numpy()
    pd_stresses = example['pd_stress'].numpy()

    mesh_edges = example['mesh_edges'].numpy()
    mesh_edges = mesh_edges[0,...]

    world_edges = example['world_edges'].numpy()

    print(np.sum(forces))

    example = serialize_example(cells.tostring(), node_type.tostring(), node_mod.tostring(), mesh_pos.tostring(), world_pos.tostring(), gripper_pos.tostring(), \
      tfn.tostring(), stresses.tostring(), pd_stresses.tostring(), forces.tostring(),  mesh_edges.tostring(), world_edges.tostring())
    # print(example)
    if write:
      writer.write(example.SerializeToString())

if write:
  writer.close()
# Cells (6714, 4) int32
# Node type (2256, 1) int32
# Mesh_pos (2256, 3) float32
# World_pos (50, 2256, 3) float32
# Gripper pos (50, 1) float32
# Tfn (6, 1) float32
# stress (50, 2256, 1) float32
# forces (50, 1)
# mesh edges (10664, 2) int32
# World edges (50, 489, 2) int32
# node type (2256, 1) (6714, 4)