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

import enum
import tensorflow.compat.v1 as tf
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

  # Remove non-unique edges
  difference = tf.math.subtract(all_senders, all_receivers)
  unique_edge_idxs = tf.where(tf.not_equal(difference, 0))
  unique_mesh_edges = tf.gather(mesh_edges, unique_edge_idxs[:,0], axis=0)
  senders, receivers = tf.unstack(unique_mesh_edges, 2, axis=1)

  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))

def sr_world_edges(world_edges):
  # world_edges = tf.gather(world_edges, [0, 1], axis=1)

  all_senders, all_receivers = tf.unstack(world_edges, axis=1)
  difference = tf.math.subtract(all_senders, all_receivers)
  unique_edges = tf.where(tf.not_equal(difference, 0))
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

  return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B

def squared_dist_point(point, others, thresh):
    dists = tf.reduce_sum(tf.square(point - others), axis=1)
    sq_thresh = thresh ** 2
    return tf.where(tf.math.less(dists, sq_thresh))

def construct_world_edges(world_pos, node_type, FLAGS):

  deformable_idx = tf.where(tf.not_equal(node_type[:, 0], NodeType.OBSTACLE))
  actuator_idx = tf.where(tf.equal(node_type[:, 0], NodeType.OBSTACLE))
  B = tf.squeeze(tf.gather(world_pos, deformable_idx))
  A = tf.squeeze(tf.gather(world_pos, actuator_idx))

  thresh = 0.005

  if utils.using_dm_dataset(FLAGS):
    thresh = 0.03

  # ''' Tried and true
  dists = squared_dist(A, B)

  rel_close_pair_idx = tf.where(tf.math.less(dists, thresh ** 2))
  close_pair_actuator = tf.gather(actuator_idx, rel_close_pair_idx[:,0])
  close_pair_def = tf.gather(deformable_idx, rel_close_pair_idx[:,1])
  close_pair_idx = tf.concat([close_pair_actuator, close_pair_def], 1)
  senders, receivers = tf.unstack(close_pair_idx, 2, axis=1)

  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))
  # '''

####################################

  # Try broadcasting
  '''
  r = tf.reduce_sum(world_pos*world_pos, 1)
  r = tf.reshape(r, [-1, 1])
  dists = r - 2*tf.matmul(world_pos, tf.transpose(world_pos)) + tf.transpose(r)

  thresh = 0.005#  threshold, 0.005
  rel_close_pair_idx = tf.where(tf.math.less(dists, thresh ** 2))

  total_senders = rel_close_pair_idx[:,0]
  total_receivers = rel_close_pair_idx[:,1]

  node_types_of_pairs = tf.gather(tf.squeeze(node_type), rel_close_pair_idx)
  node_pairs_difference = tf.math.abs(tf.subtract(node_types_of_pairs[:,0], node_types_of_pairs[:,1]))
  different_type_connections = tf.where(tf.not_equal(node_pairs_difference, 0))
  senders = tf.gather(total_senders, different_type_connections)
  receivers = tf.gather(total_receivers, different_type_connections)
  return (senders[:,0], receivers[:,0])
  '''
  #################################
  # Try with while loop
  # '''
  num_B = tf.shape(B)[0]

  def body(i, outputs):
    random_vec = tf.random.uniform(shape=[1, 3])
    b = tf.gather(B, [i])
    # print(b.shape)
    a = squared_dist_point(b, A, thresh)
    num_matches = tf.shape(a)[0]
    bs = a*0 +  tf.cast(i, dtype=tf.int64)
    # bs = tf.repeat(tf.cast([i], dtype=tf.int64), num_matches)
    b_pairs = tf.concat([bs, a], axis=1)
    outputs = outputs.write(i, b_pairs)
    i += 1
    return i, outputs

  outputs = tf.TensorArray(dtype=tf.int64, infer_shape=False, size=1, element_shape=[None, 2], dynamic_size=True)
  _, outputs = tf.while_loop(lambda i, *_: tf.less(i, num_B), body, [0, outputs])
  outputs = outputs.concat()

  close_pair_def = tf.gather(deformable_idx, outputs[:,0])

  close_pair_actuator = tf.gather(actuator_idx, outputs[:,1])
  close_pair_idx = tf.concat([close_pair_actuator, close_pair_def], 1)
  senders, receivers = tf.unstack(close_pair_idx, 2, axis=1)

  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))
  # '''

  ###### Just set randomly
  '''
  senders, receivers = tf.range(6000).shape, tf.range(6000).shape

  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))
  '''

  ######## old stuff?

# def construct_world_edges(world_pos, node_type):
#   deformable_idx = tf.where(tf.not_equal(node_type[:, 0], NodeType.OBSTACLE))
#   actuator_idx = tf.where(tf.equal(node_type[:, 0], NodeType.OBSTACLE))

#   B = tf.squeeze(tf.gather(world_pos, deformable_idx))
#   A = tf.squeeze(tf.gather(world_pos, actuator_idx))

#   dists = squared_dist(A, B)
#   thresh = 0.03 ** 2 # squared threshold

#   rel_close_pair_idx = tf.where(tf.math.less(dists, thresh))

#   close_pair_actuator = tf.gather(actuator_idx, rel_close_pair_idx[:,0])
#   close_pair_def = tf.gather(deformable_idx, rel_close_pair_idx[:,1])

#   senders, receivers = close_pair_actuator, close_pair_def

#   return (tf.concat([senders, receivers], axis=0),
#           tf.concat([receivers, senders], axis=0))
