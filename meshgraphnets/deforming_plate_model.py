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
from meshgraphnets import utils

import numpy as np
import os
import trimesh

def open_gripper_at_pose(inputs):
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

  return f1_verts, f2_verts, gripper_normal

def gripper_world_pos(inputs):

  f1_verts, f2_verts, gripper_normal = open_gripper_at_pose(inputs)

  # Apply gripper closings to the transformed fingers
  f_verts = f_verts_at_pos(inputs, inputs['gripper_pos'])

  f_verts_next = f_verts_at_pos(inputs, inputs['target|gripper_pos'])

  # Get velocity of each gripper
  num_verts_per_f = f_verts.shape[0] // 2

  f1_force_vecs = tf.tile(tf.expand_dims(-1. * gripper_normal, axis=0), [num_verts_per_f, 1]) * (inputs['target|force'] - inputs['force'])
  f2_force_vecs = tf.tile(tf.expand_dims(gripper_normal, axis=0), [num_verts_per_f, 1]) * (inputs['target|force'] - inputs['force'])
  f_force_vecs = tf.concat((f1_force_vecs, f2_force_vecs), axis=0)

  return f_verts, f_verts_next, f_force_vecs

  #########################

  # f_pc = np.concatenate((f1_verts_original, f2_verts_original))

def f_verts_at_pos(inputs, gripper_pos):
  f1_verts, f2_verts, gripper_normal = open_gripper_at_pose(inputs)
  f1_verts_closed = f1_verts -  gripper_normal * (0.04 - gripper_pos)
  f2_verts_closed = f2_verts + gripper_normal * (0.04 - gripper_pos)
  f_verts = tf.concat((f1_verts_closed, f2_verts_closed), axis=0)
  return f_verts


class Model(snt.AbstractModule):
  """Model for static cloth simulation."""

  def __init__(self, learned_model, FLAGS, name='Model'):
    super(Model, self).__init__(name=name)
    with self._enter_variable_scope():
      self.FLAGS = FLAGS

      self._learned_model = learned_model

      ######### Output #########
      output_size = 4 # velocity and stress
      self._output_normalizer = normalization.Normalizer(
          size=output_size, name='output_normalizer') 

      ######### Auxiliary ########
      aux_output_size = 1
      self._aux_output_normalizer = normalization.Normalizer(
          size=1, name='aux_output_normalizer') 

      ######### Node #########
      node_feature_size = 3+1+common.NodeType.SIZE #  velocity + stress + node type
      if not utils.stress_t_as_node_feature(self.FLAGS):
        node_feature_size = 3+common.NodeType.SIZE # velocity + node type
      if self.FLAGS.force_label_node:
        node_feature_size += 1

      self._node_normalizer = normalization.Normalizer(
          size=node_feature_size, name='node_normalizer') 

      ######### Mesh edge #########
      self._edge_normalizer = normalization.Normalizer(
          size=8, name='edge_normalizer')  # 2*(3D coord  + length) = 8

      ######### World edge #########
      world_edge_feature_size = 3 + 1  # 3D coord  + length = 4
      if not self.FLAGS.force_label_node:
        world_edge_feature_size += 1 # force
      self._world_edge_normalizer = normalization.Normalizer(
          size=world_edge_feature_size, name='world_edge_normalizer') 

  def _build_graph(self, inputs, is_training):
    """Builds input graph."""

    f_verts, f_verts_next, f_force_vecs = gripper_world_pos(inputs) 
    num_f_verts_total = tf.shape(f_verts)[0] 


    # Mask way non-kinematic nodes
    actuator_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.OBSTACLE)



    #######################################################################################
    # Calculate world pos partially from gripper_pos
    # finger_world_pos = tf.pad(f_verts, paddings, "CONSTANT") bring back
    # world_pos = tf.where(actuator_mask, finger_world_pos, inputs['world_pos']) # This should be equal to inputs['world_pos'] anyway
    world_pos = inputs['world_pos']

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
    if self.FLAGS.compute_world_edges:
      world_senders, world_receivers = common.construct_world_edges(world_pos, inputs['node_type'])
    else:
      world_senders, world_receivers = common.sr_world_edges(inputs['world_edges'])

    # Force at nodes or edges
    if self.FLAGS.node_total_force_t:
      force_label = inputs['force'][0,0] # Label every node with total gripper force
    else:
      force_label = inputs['force'][0,0] / (0.5 * tf.to_float(tf.shape(world_senders)[0])) # Label every node with total gripper force / num contacts


    relative_world_pos = (tf.gather(world_pos, world_senders) -
                          tf.gather(world_pos, world_receivers))
    relative_world_norm = tf.norm(relative_world_pos, axis=-1, keepdims=True)

    if self.FLAGS.force_label_node:
      world_edge_features = tf.concat([
          relative_world_pos,
          relative_world_norm], axis=-1)
    else:
      world_edge_features = tf.concat([
          relative_world_pos,
          relative_world_norm,
          tf.fill(tf.shape(relative_world_norm), force_label)], axis=-1)


    world_edges = core_model.EdgeSet(
        name='world_edges',
        features=self._world_edge_normalizer(world_edge_features, is_training),
        receivers=world_receivers,
        senders=world_senders)

    #################### Node features
    #########
    # Calculate force velocity from gripper_pos
    # '''
    zero_vel = tf.fill(tf.shape(inputs['mesh_pos']), 0.0)
    num_verts_total = tf.shape(inputs['mesh_pos'])[0]#.get_shape().as_list()[0]
    pad_diff = num_verts_total - num_f_verts_total 
    paddings = [[0, pad_diff], [0, 0]] 
    

    # Use either change in force or change in gripper position as the velocity
    nonzero_vel = tf.pad(f_force_vecs, paddings, "CONSTANT")
    # nonzero_vel = tf.pad(f_verts_next - f_verts, paddings, "CONSTANT") 
    velocity = tf.where(actuator_mask, nonzero_vel, zero_vel) 
    # '''

    ##########
    node_type = tf.one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)
    node_mod = inputs['node_mod'][:,:]

    # Stress 
    if self.FLAGS.log_stress_t:
      stresses = tf.math.log(inputs['stress'] + 1)
    else:
      stresses = inputs['stress']


    if not utils.stress_t_as_node_feature(self.FLAGS):
      node_features = tf.concat([velocity, node_type], axis=-1)
    else:
      node_features = tf.concat([velocity, stresses, node_type], axis=-1)

    if self.FLAGS.force_label_node:
      node_force = tf.fill(tf.shape(inputs['stress']), force_label)

      node_features = tf.concat([node_force, node_features], axis=-1)


    return core_model.MultiGraph(
        node_features=self._node_normalizer(node_features, is_training),
        edge_sets=[mesh_edges, world_edges])

  # Is this used? Yes in the background of AbstractModel
  def _build(self, inputs, normalize=True, accumulate=False):
    graph = self._build_graph(inputs, is_training=False)
    per_node_network_output = self._learned_model(graph)
    return self._update(inputs, per_node_network_output, normalize, accumulate)

  def print_debug(self, inputs):
    """L2 loss on position."""
    return tf.reduce_mean(inputs['stress'][0])
    return self._node_normalizer._acc_count, self._output_normalizer._acc_count, self._edge_normalizer._acc_count, self._world_edge_normalizer._acc_count
    finger1_path = os.path.join('meshgraphnets', 'assets', 'finger1_face_uniform' + '.stl')
    f1_trimesh = trimesh.load_mesh(finger1_path)
    f1_verts_original = tf.constant(f1_trimesh.vertices, dtype=tf.float32)

    finger2_path = os.path.join('meshgraphnets', 'assets', 'finger2_face_uniform' + '.stl')
    f2_trimesh = trimesh.load_mesh(finger2_path)
    f2_verts_original = tf.constant(f2_trimesh.vertices, dtype=tf.float32)
    f_pc_original = tf.concat((f1_verts_original, f2_verts_original), axis=0)


    # inputs['f_original'] is (24920, 3)
    # Load transformation params (euler and translation)
    euler, trans = tf.split(inputs['tfn'], [3, 3], axis=1) # (24920, 3)
    tf_from_euler = tfg_transformation.rotation_matrix_3d.from_euler(euler) #(24920, 3, 3)

    # f_pc = tfg_transformation.rotation_matrix_3d.rotate(inputs['f_original'][5, ...], tf_from_euler[5, ...])# + trans #()
    pad_diff = tf.shape(inputs['node_type'])[0] - tf.shape(f_pc_original)[0]
    paddings = [[0, pad_diff], [0, 0]]
    f_pc_padded = tf.pad(f_pc_original, paddings, "CONSTANT")


    f_pc2 = tfg_transformation.rotation_matrix_3d.rotate(inputs['f_original'], tf_from_euler)# + trans #()

    return f_pc2, tf.shape(f_pc2)

    original_normal = tf.constant([1., 0., 0.], dtype=tf.float32)
    gripper_normal = tfg_transformation.rotation_matrix_3d.rotate(original_normal, tf_from_euler)
    f1_verts, f2_verts = tf.split(f_pc, 2, axis=0)

    return f1_verts, f2_verts, gripper_normal


  @snt.reuse_variables
  def loss(self, inputs, normalize=True, accumulate=False):
    """L2 loss on position."""
    graph = self._build_graph(inputs, is_training=accumulate) # is_training used to always be True -> accumulate always True
    network_output = self._learned_model(graph)


    if normalize:
      object_output, stress_output = tf.split(network_output, [3, 1], 1)
    else:
      object_output, stress_output = tf.split(self._output_normalizer.inverse(network_output), [3,1], 1)

    # Target gripper pos change
    num_verts_total = tf.shape(inputs['mesh_pos'])[0]

    # Target object position change
    cur_position = inputs['world_pos']
    target_position = inputs['target|world_pos']
    target_position_change = target_position - cur_position

    # Target stress
    if self.FLAGS.predict_log_stress_t_only:
      target_stress_change = tf.math.log(inputs['stress'] + 1)
      combined_target = tf.concat([target_position_change, target_stress_change], 1)
    elif self.FLAGS.predict_log_stress_t1 or self.FLAGS.predict_log_stress_t1_only:
      target_stress_change = tf.math.log(inputs['target|stress'] + 1)
      combined_target = tf.concat([target_position_change, target_stress_change], 1)
    elif self.FLAGS.predict_log_stress_change_t1 or self.FLAGS.predict_log_stress_change_only:
      target_stress_change = tf.math.log(inputs['target|stress'] + 1) - tf.math.log(inputs['stress'] + 1)
      combined_target = tf.concat([target_position_change, target_stress_change], 1)
    else:
      target_stress_change = inputs['target|stress']
      combined_target = tf.concat([target_position_change, target_stress_change], 1)

    # build loss
    object_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL)
    gripper_mask = tf.not_equal(inputs['node_type'][:, 0], common.NodeType.NORMAL)

    # Normalize target position change and stress first
    if normalize:
      target_normalized = self._output_normalizer(combined_target, accumulate)
    else:
      target_normalized = combined_target


    object_target_normalized, stress_target_normalized = tf.split(target_normalized, [3, 1], 1)
    object_error = tf.reduce_sum((object_target_normalized - object_output)**2, axis=1) 
    object_loss = tf.reduce_mean(object_error)
    stress_error = tf.reduce_sum((stress_target_normalized - stress_output)**2, axis=1) 
    stress_loss = tf.reduce_mean(stress_error[object_mask])
    if self.FLAGS.predict_log_stress_t_only or self.FLAGS.predict_log_stress_t1_only or self.FLAGS.predict_log_stress_change_only:
      loss = stress_loss
    elif self.FLAGS.predict_pos_change_only:
      loss = object_loss
    else:
      loss = object_loss + stress_loss


    if self.FLAGS.aux_mesh_edge_distance_change:
      normalized_pred_new_pos = cur_position + object_output

      # Get predicted new position (real, not normalized)
      real_pred_position_change, _ = tf.split(self._output_normalizer.inverse(network_output), [3,1], 1)
      real_pred_next_pos = cur_position + real_pred_position_change

      senders, receivers = common.sr_mesh_edges(inputs['mesh_edges'])

      current_relative_world_pos = (tf.gather(inputs['world_pos'], senders) -
                            tf.gather(inputs['world_pos'], receivers))
      current_relative_world_norm = tf.norm(current_relative_world_pos, axis=-1, keepdims=True)

      next_relative_world_pos = (tf.gather(inputs['target|world_pos'], senders) -
                            tf.gather(inputs['target|world_pos'], receivers))
      next_relative_world_norm = tf.norm(next_relative_world_pos, axis=-1, keepdims=True)

      pred_next_relative_world_pos = (tf.gather(real_pred_next_pos, senders) -
                            tf.gather(real_pred_next_pos, receivers))
      pred_next_relative_world_norm = tf.norm(pred_next_relative_world_pos, axis=-1, keepdims=True)


      gt_mesh_edge_norm_change = next_relative_world_norm - current_relative_world_norm
      pred_mesh_edge_norm_change = pred_next_relative_world_norm - current_relative_world_norm

      gt_mesh_edge_norm_change_normalized = self._aux_output_normalizer(gt_mesh_edge_norm_change, accumulate)
      pred_mesh_edge_norm_change_normalized = self._aux_output_normalizer(pred_mesh_edge_norm_change, False)


      mesh_distance_error = tf.reduce_sum((gt_mesh_edge_norm_change_normalized - pred_mesh_edge_norm_change_normalized)**2, axis=1) 
      mesh_distance_loss = tf.reduce_mean(mesh_distance_error)
      loss += mesh_distance_loss



    return loss


  def _update(self, inputs, per_node_network_output, normalize=True, accumulate=False):
    """Integrate model outputs."""
    actuator_mask = tf.not_equal(inputs['node_type'][:, 0], common.NodeType.OBSTACLE)
    mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL) # Only update normal nodes with predictions
    actuator_idx = tf.where(actuator_mask)
    object_idx = tf.where(mask)

    f_verts, f_verts_next, _ = gripper_world_pos(inputs)

    num_f_verts_total = tf.shape(f_verts)[0]
    num_verts_total = tf.shape(inputs['mesh_pos'])[0]
    pad_diff = num_verts_total - num_f_verts_total
    paddings = [[0, pad_diff], [0, 0]]
    next_pos_gt = tf.pad(f_verts_next, paddings, "CONSTANT")

    # Get predictions
    if self.FLAGS.predict_log_stress_t_only:
      _ , stress_change = tf.split(self._output_normalizer.inverse(per_node_network_output), [3,1], 1)
      position_change = inputs['target|world_pos'] - inputs['world_pos']
    elif self.FLAGS.predict_log_stress_t1_only:
      _, stress_change = tf.split(self._output_normalizer.inverse(per_node_network_output), [3,1], 1)
      position_change = inputs['target|world_pos'] - inputs['world_pos']
    elif self.FLAGS.predict_log_stress_change_only:
      _, stress_change = tf.split(self._output_normalizer.inverse(per_node_network_output), [3,1], 1)
      position_change = inputs['target|world_pos'] - inputs['world_pos']
    else:
      position_change, stress_change = tf.split(self._output_normalizer.inverse(per_node_network_output), [3,1], 1)

    # Integrate forward
    curr_stress_gt = inputs['stress']
    curr_position = inputs['world_pos']
    next_position_pred = curr_position + position_change


    zero_stress = stress_change * 0
    stress_change = tf.where(actuator_mask, stress_change, zero_stress)


    # next_stress_pred = stress_change  # If just predicting log next stress
    if self.FLAGS.predict_log_stress_t_only:
      next_stress_pred = tf.math.exp(stress_change) - 1
    elif self.FLAGS.predict_log_stress_t1 or self.FLAGS.predict_log_stress_t1_only:
      next_stress_pred = tf.math.exp(stress_change) - 1
    elif self.FLAGS.predict_log_stress_change_t1 or self.FLAGS.predict_log_stress_change_only:
      next_stress_pred = tf.math.exp(stress_change) * (inputs['stress'] + 1) - 1
    elif self.FLAGS.predict_pos_change_only:
      next_stress_pred = inputs['target|stress']
    else:
      next_stress_pred = stress_change

    next_stress_pred = tf.nn.relu(next_stress_pred)

    # round to nearest next_gripper_pos
    gripper_vert_pos_change = tf.gather(position_change, actuator_idx)

    # Get loss val
    loss_val = self.loss(inputs, normalize, accumulate)


    # return next_position_pred, curr_stress_pred # For when predicting gripper movement too
    return next_position_pred, next_stress_pred, loss_val
