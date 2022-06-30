# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Commonly used data structures and functions."""
import os

import enum
# import tensorflow.compat.v1 as tf
import tensorflow as tf

# from tfdeterminism import patch
# patch()
SEED = 55
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)


import sys

from meshgraphnets import utils


class NodeType(enum.IntEnum):
  NORMAL = 0
  OBSTACLE = 1
  AIRFOIL = 2
  HANDLE = 3
  INFLOW = 4
  OUTFLOW = 5
  WALL_BOUNDARY = 6
  SIZE = 2



def sr_mesh_edges(mesh_edges):
  all_senders, all_receivers = tf.unstack(mesh_edges, axis=1)


  ### This should work once non-unique mesh edges are removed in the h5 to tf
  # return (tf.concat([all_senders, all_receivers], axis=0),
  #         tf.concat([all_receivers, all_senders], axis=0))

  # '''
  # Remove non-unique edges
  difference = tf.math.subtract(all_senders, all_receivers)
  unique_edge_idxs = tf.where(tf.not_equal(difference, 0))


  unique_mesh_edges = tf.gather(mesh_edges, unique_edge_idxs[:,0], axis=0)

  senders, receivers = tf.unstack(unique_mesh_edges, 2, axis=1)

  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))
  # '''

def sr_world_edges(world_edges):

  all_senders, all_receivers = tf.unstack(world_edges, axis=1)


  # JUST TO GET XLA TO WORK
  return (tf.concat([all_senders, all_receivers], axis=0),
          tf.concat([all_receivers, all_senders], axis=0))



  difference = tf.math.subtract(all_senders, all_receivers)
  unique_edges = tf.where(tf.not_equal(difference, 0))
  
  # print(world_edges.shape[0], unique_edges.shape[0])

  close_pair_idx = tf.gather(world_edges, unique_edges[:,0], axis=0)

  senders, receivers = tf.unstack(close_pair_idx, 2, axis=1)

  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))
  
# This is applicable only to flat meshes
def triangles_to_edges(faces):
  """Computes mesh edges from triangles."""
  # collect edges from triangles

  if faces.shape[1] == 4:
    edges = tf.concat([faces[:, 0:2], #0, 1
                     faces[:, 1:3], # 1, 2
                     faces[:, 2:4], # 2, 3
                     tf.stack([faces[:, 2], faces[:, 0]], axis=1), # 0, 2
                     tf.stack([faces[:, 3], faces[:, 0]], axis=1), # 0, 3
                     tf.stack([faces[:, 1], faces[:, 3]], axis=1)], axis=0)
  else:
    edges = tf.concat([faces[:, 0:2],
                       faces[:, 1:3],
                       tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)



  # those edges are sometimes duplicated (within the mesh) and sometimes
  # single (at the mesh boundary).
  # sort & pack edges as single tf.int64
  receivers = tf.reduce_min(edges, axis=1)
  senders = tf.reduce_max(edges, axis=1)
  packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
  # remove duplicates and unpack
  unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
  senders, receivers = tf.unstack(unique_edges, axis=1)
  # create two-way connectivity

  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))

def squared_dist(A, B):

  row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
  row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

  row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
  row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.



  return row_norms_A - 2 * tf.matmul(A, B, False, True) + row_norms_B




def squared_dist_point(point, others, thresh):
    dists = tf.reduce_sum(tf.square(point - others), axis=1)
    sq_thresh = thresh ** 2
    return tf.where(tf.math.less(dists, sq_thresh))


def construct_world_edges(world_pos, node_type, FLAGS):

  deformable_idx = tf.where(tf.not_equal(node_type[:, 0], NodeType.OBSTACLE))
  actuator_idx = tf.where(tf.equal(node_type[:, 0], NodeType.OBSTACLE))
  B = tf.squeeze(tf.gather(world_pos, deformable_idx))
  A = tf.squeeze(tf.gather(world_pos, actuator_idx))

  A = tf.cast(A, tf.float64)
  B = tf.cast(B, tf.float64)

  thresh = 0.003#0.005

  if utils.using_dm_dataset(FLAGS):
    thresh = 0.03

  # ''' Tried and true
  dists = squared_dist(A, B)


  # Compare to cdist

  rel_close_pair_idx = tf.where(tf.math.less(dists, thresh ** 2))


  close_pair_actuator = tf.gather(actuator_idx, rel_close_pair_idx[:,0])
  close_pair_def = tf.gather(deformable_idx, rel_close_pair_idx[:,1])
  close_pair_idx = tf.concat([close_pair_actuator, close_pair_def], 1)

  senders, receivers = tf.unstack(close_pair_idx, 2, axis=1)


  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))
  # '''

def construct_world_edges_min_dists(world_pos, node_type, FLAGS):

  deformable_idx = tf.where(tf.not_equal(node_type[:, 0], NodeType.OBSTACLE))
  actuator_idx = tf.where(tf.equal(node_type[:, 0], NodeType.OBSTACLE))
  B = tf.squeeze(tf.gather(world_pos, deformable_idx))
  A = tf.squeeze(tf.gather(world_pos, actuator_idx))

  A = tf.cast(A, tf.float64)
  B = tf.cast(B, tf.float64)

  thresh = 0.003#0.005

  if utils.using_dm_dataset(FLAGS):
    thresh = 0.03

  # ''' Tried and true
  dists = squared_dist(A, B)
  min_dists = tf.reduce_min(dists, axis=0)
  sigmoid_activated = tf.math.sigmoid(1e6 * (-1. * min_dists + thresh ** 2))

  '''
  print(min_dists.shape)
  print(sigmoid_activated)
  import matplotlib.pyplot as plt 
  plt.scatter(min_dists, sigmoid_activated)
  plt.scatter([thresh**2], [0.5])
  plt.show()
  quit()
  '''


  return tf.reduce_sum(sigmoid_activated)



