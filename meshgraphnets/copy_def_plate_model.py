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
"""Model for FlagSimple."""

import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_graphics.geometry.transformation as tfg_transformation


from meshgraphnets import common
from meshgraphnets import core_model
from meshgraphnets import normalization

import numpy as np
import os
import trimesh

def gripper_world_pos(inputs):

  # Load original finger positions
  finger1_path = os.path.join('meshgraphnets', 'assets', 'finger1_face_uniform' + '.stl')
  f1_trimesh = trimesh.load_mesh(finger1_path)
  f1_verts_original = tf.constant(f1_trimesh.vertices, dtype=tf.float32)

  finger2_path = os.path.join('meshgraphnets', 'assets', 'finger2_face_uniform' + '.stl')
  f2_trimesh = trimesh.load_mesh(finger2_path)
  f2_verts_original = tf.constant(f2_trimesh.vertices, dtype=tf.float32)
  f_pc_original = tf.concat((f1_verts_original, f2_verts_original), axis=0)

  # Load transformation params (euler and translation)
  euler, trans = tf.split(inputs['tfn'][:,0], 2, axis=0)
  tf_from_euler = tfg_transformation.rotation_matrix_3d.from_euler(euler)
  f_pc = tfg_transformation.rotation_matrix_3d.rotate(f_pc_original, tf_from_euler) + trans
  original_normal = tf.constant([1., 0., 0.], dtype=tf.float32)
  gripper_normal = tfg_transformation.rotation_matrix_3d.rotate(original_normal, tf_from_euler)
  f1_verts, f2_verts = tf.split(f_pc, 2, axis=0)

  # Apply gripper closings to the transformed fingers
  f1_verts_closed = f1_verts -  gripper_normal * (0.04 - inputs['gripper_pos'][0])
  f2_verts_closed = f2_verts + gripper_normal * (0.04 - inputs['gripper_pos'][1])
  f_verts = tf.concat((f1_verts_closed, f2_verts_closed), axis=0)

  f1_verts_next = f1_verts -  gripper_normal * (0.04 - inputs['target|gripper_pos'][0])
  f2_verts_next = f2_verts + gripper_normal * (0.04 - inputs['target|gripper_pos'][1])
  f_verts_next = tf.concat((f1_verts_next, f2_verts_next), axis=0)

  # Get velocity of each gripper
  num_verts_per_f = f1_verts_closed.shape[0]
  # f1_vel = tf.tile(tf.expand_dims(-1. * gripper_normal, axis=0), [num_verts_per_f, 1]) * (inputs['gripper_pos'][0] - inputs['target|gripper_pos'][0])
  # f2_vel = tf.tile(tf.expand_dims(gripper_normal, axis=0), [num_verts_per_f, 1]) * (inputs['gripper_pos'][1] - inputs['target|gripper_pos'][1])
  # f_vels = tf.concat((f1_vel, f2_vel), axis=0)

  f1_force_vecs = tf.tile(tf.expand_dims(-1. * gripper_normal, axis=0), [num_verts_per_f, 1]) * (inputs['target|force'] - inputs['force'])
  f2_force_vecs = tf.tile(tf.expand_dims(gripper_normal, axis=0), [num_verts_per_f, 1]) * (inputs['target|force'] - inputs['force'])
  f_force_vecs = tf.concat((f1_force_vecs, f2_force_vecs), axis=0)

  return f_verts, f_verts_next, f_force_vecs

  #########################

  # f_pc = np.concatenate((f1_verts_original, f2_verts_original))

class Model(snt.AbstractModule):
  """Model for static cloth simulation."""

  def __init__(self, learned_model, name='Model'):
    super(Model, self).__init__(name=name)
    with self._enter_variable_scope():
      self._learned_model = learned_model
      self._output_normalizer = normalization.Normalizer(
          size=1+3+1, name='output_normalizer') # closing_distance and velocity and stress
      self._node_normalizer = normalization.Normalizer(
          size=3+1+common.NodeType.SIZE, name='node_normalizer') # velocity + node_mod + node type

      self._edge_normalizer = normalization.Normalizer(
          size=8, name='edge_normalizer')  # 2*(3D coord  + length) = 8

      self._world_edge_normalizer = normalization.Normalizer(
          size=4, name='world_edge_normalizer')  # 3D coord  + length = 4

  def _build_graph(self, inputs, is_training):
    """Builds input graph."""
    # Apply transformations to f_pcs
    f_verts, f_verts_next, f_force_vecs = gripper_world_pos(inputs)
    f_vels = f_verts_next - f_verts
    num_f_verts_total = tf.shape(f_verts)[0]


    # construct graph nodes
    # velocity = inputs['target|world_pos'] - inputs['world_pos'] # This should apply only for nodes that are kinematic
    # zero_vel = velocity * 0

    # Mask way non-kinematic nodes
    actuator_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.OBSTACLE)
    # velocity = tf.where(actuator_mask, velocity, zero_vel)

    #########
    # Calculate force velocity from gripper_pos
    # '''
    zero_vel = tf.fill(tf.shape(inputs['mesh_pos']), 0.0)
    num_verts_total = tf.shape(inputs['mesh_pos'])[0]#.get_shape().as_list()[0]
    pad_diff = num_verts_total - num_f_verts_total

    paddings = [[0, pad_diff], [0, 0]]
    nonzero_vel = tf.pad(f_force_vecs, paddings, "CONSTANT")

    velocity = tf.where(actuator_mask, nonzero_vel, zero_vel)
    # '''

    ##########

    node_type = tf.one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)
    node_mod = inputs['node_mod'][:,:]

    node_features = tf.concat([velocity, node_mod, node_type], axis=-1)

    #######################################################################################
    # Calculate world pos partially from gripper_pos
    finger_world_pos = tf.pad(f_verts, paddings, "CONSTANT")
    world_pos = tf.where(actuator_mask, finger_world_pos, inputs['world_pos']) # This should be equal to inputs['world_pos'] anyway

    # Construct mesh graph edges
    senders, receivers = common.sr_mesh_edges(inputs['mesh_edges'])
      # senders, receivers = common.triangles_to_edges(inputs['cells'])


    relative_mesh_pos = (tf.gather(inputs['mesh_pos'], senders) -
                         tf.gather(inputs['mesh_pos'], receivers))

    relative_mesh_pos_world_edges = (tf.gather(world_pos, senders) -
                         tf.gather(world_pos, receivers))

    mesh_edge_features = tf.concat([
        relative_mesh_pos_world_edges,
        tf.norm(relative_mesh_pos_world_edges, axis=-1, keepdims=True),
        relative_mesh_pos,
        tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)


    mesh_edges = core_model.EdgeSet(
        name='mesh_edges',
        features=self._edge_normalizer(mesh_edge_features, is_training),
        receivers=receivers,
        senders=senders)

    # Construct graph world edges
    world_senders, world_receivers = common.sr_world_edges(inputs['world_edges'])
    # world_senders, world_receivers = common.construct_world_edges(world_pos, inputs['node_type'])

    relative_world_pos = (tf.gather(world_pos, world_senders) -
                          tf.gather(world_pos, world_receivers))

    world_edge_features = tf.concat([
        relative_world_pos,
        tf.norm(relative_world_pos, axis=-1, keepdims=True)], axis=-1)

    world_edges = core_model.EdgeSet(
        name='world_edges',
        features=self._world_edge_normalizer(world_edge_features, is_training),
        receivers=world_receivers,
        senders=world_senders)

    return core_model.MultiGraph(
        node_features=self._node_normalizer(node_features, is_training),
        edge_sets=[mesh_edges, world_edges])

  # Is this used? Yes in the background of AbstractModel
  def _build(self, inputs):
    graph = self._build_graph(inputs, is_training=False)
    per_node_network_output = self._learned_model(graph)
    return self._update(inputs, per_node_network_output)

  def print_debug(self, inputs):
    # a = common.construct_world_edges(inputs['world_pos'], inputs['node_type'])

    f_verts, _, f_vels = gripper_world_pos(inputs)
    num_f_verts_total = tf.shape(f_verts)[0]

    zero_vel = tf.fill(tf.shape(inputs['mesh_pos']), 0.0)
    num_verts_total = tf.shape(inputs['mesh_pos'])[0]#.get_shape().as_list()[0]
    pad_diff = num_verts_total - num_f_verts_total

    paddings = tf.Variable([[0, pad_diff], [0, 0]])
    nonzero_vel = tf.pad(f_vels, paddings, "CONSTANT")

    return tf.shape(nonzero_vel)

  @snt.reuse_variables
  def loss(self, inputs):

    """L2 loss on position."""
    graph = self._build_graph(inputs, is_training=True)
    network_output = self._learned_model(graph)
    # network_output, network_stress = tf.split(network_output, [3, 1], 1)


    # build target position change
    cur_position = inputs['world_pos']
    target_position = inputs['target|world_pos']
    target_stress = tf.math.log(inputs['stress'] + 1) # Use log stress for output
    target_position_change = target_position - cur_position

    # Normalize target position change and stress
    combined_target = tf.concat([target_position_change, target_stress], 1)
    target_normalized = self._output_normalizer(combined_target)

    # build loss
    loss_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL)
    error = tf.reduce_sum((target_normalized - network_output)**2, axis=1) 
    # loss = tf.reduce_mean(error[loss_mask]) # Take loss only for the normal nodes
    loss = tf.reduce_mean(error) # Take loss for gripper and object nodes
    return loss

  def _update(self, inputs, per_node_network_output):
    """Integrate model outputs."""
    f_verts, f_verts_next, _ = gripper_world_pos(inputs)

    num_f_verts_total = tf.shape(f_verts)[0]
    num_verts_total = tf.shape(inputs['mesh_pos'])[0]
    pad_diff = num_verts_total - num_f_verts_total
    paddings = [[0, pad_diff], [0, 0]]
    next_pos_gt = tf.pad(f_verts_next, paddings, "CONSTANT")

    position_change, curr_stress_pred = tf.split(self._output_normalizer.inverse(per_node_network_output), [3,1], 1)

    # integrate forward
    # next_pos_gt = inputs['target|world_pos']
    curr_stress_gt = inputs['stress']

    curr_position = inputs['world_pos']
    next_position_pred = curr_position + position_change

    mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL) # Only update normal nodes with predictions
    next_pos = tf.where(mask, next_position_pred, next_pos_gt) 
    
    actuator_mask = tf.not_equal(inputs['node_type'][:, 0], common.NodeType.OBSTACLE)
    zero_stress = curr_stress_pred * 0
    curr_stress_pred = tf.where(actuator_mask, curr_stress_pred, zero_stress)

    return next_position_pred, curr_stress_pred # For when predicting gripper movement too
    return next_pos, curr_stress_pred
