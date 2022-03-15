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

  # euler = tf.zeros(3)
  # trans = tf.zeros(3)

  tf_from_euler = tfg_transformation.rotation_matrix_3d.from_euler(euler)
  f_pc = tfg_transformation.rotation_matrix_3d.rotate(f_pc_original, tf_from_euler) + trans
  original_normal = tf.constant([1., 0., 0.], dtype=tf.float32)
  gripper_normal = tfg_transformation.rotation_matrix_3d.rotate(original_normal, tf_from_euler)
  f1_verts, f2_verts = tf.split(f_pc, 2, axis=0)

  return f1_verts, f2_verts, gripper_normal

def f_verts_at_pos(inputs, gripper_pos):
  f1_verts, f2_verts, gripper_normal = open_gripper_at_pose(inputs)
  f1_verts_closed = f1_verts -  gripper_normal * (0.04 - gripper_pos)
  f2_verts_closed = f2_verts + gripper_normal * (0.04 - gripper_pos)
  f_verts = tf.concat((f1_verts_closed, f2_verts_closed), axis=0)
  return f_verts


def gripper_world_pos(inputs):

  f1_verts, f2_verts, gripper_normal = open_gripper_at_pose(inputs) # Might not need this line

  # Apply gripper closings to the transformed fingers
  f_verts = f_verts_at_pos(inputs, inputs['gripper_pos'])

  f_verts_next = f_verts_at_pos(inputs, inputs['target|gripper_pos'])

  # Get velocity of each gripper
  num_verts_per_f = f_verts.shape[0] // 2

  f1_force_vecs = tf.tile(tf.expand_dims(-1. * gripper_normal, axis=0), [num_verts_per_f, 1]) * (inputs['target|force'] - inputs['force'])
  f2_force_vecs = tf.tile(tf.expand_dims(gripper_normal, axis=0), [num_verts_per_f, 1]) * (inputs['target|force'] - inputs['force'])
  f_force_vecs = tf.concat((f1_force_vecs, f2_force_vecs), axis=0)

  unit_f1_force_vecs = tf.tile(tf.expand_dims(-1. * gripper_normal, axis=0), [num_verts_per_f, 1]) 
  unit_f2_force_vecs = tf.tile(tf.expand_dims(gripper_normal, axis=0), [num_verts_per_f, 1]) 
  unit_f_force_vecs = tf.concat((unit_f1_force_vecs, unit_f2_force_vecs), axis=0)

  return f_verts, f_verts_next, f_force_vecs, unit_f_force_vecs

  #########################

  # f_pc = np.concatenate((f1_verts_original, f2_verts_original))





class Model(snt.AbstractModule):
  """Model for static cloth simulation."""

  def __init__(self, learned_model, FLAGS, name='Model'):
    super(Model, self).__init__(name=name)
    with self._enter_variable_scope():
      self.FLAGS = FLAGS

      self._learned_model = learned_model

      ######### Output #########
      output_size = utils.get_output_size(self.FLAGS)

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

      if self.FLAGS.gripper_force_action_input and not self.FLAGS.force_label_node:
        world_edge_feature_size += 1 # force = 5

      if self.FLAGS.simplified_predict_stress_change_only:
        if self.FLAGS.predict_stress_t_only:
          world_edge_feature_size = 5 # 3D coord + length + curr force
        else:
          world_edge_feature_size = 6 # 3D coord + length + curr force, change in force

      self._world_edge_normalizer = normalization.Normalizer(
          size=world_edge_feature_size, name='world_edge_normalizer') 

  def _build_graph(self, inputs, is_training):
    """Builds input graph."""


    # Mask way non-kinematic nodes
    actuator_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.OBSTACLE)

    #######################################################################################
    # Calculate world pos partially from gripper_pos
    if not utils.using_dm_dataset(self.FLAGS):
      f_verts, f_verts_next, f_force_vecs, unit_f_force_vecs = gripper_world_pos(inputs) 
      num_f_verts_total = tf.shape(f_verts)[0]
      num_verts_total = tf.shape(inputs['mesh_pos'])[0]
      pad_diff = num_verts_total - num_f_verts_total
      paddings = [[0, pad_diff], [0, 0]]
      finger_world_pos = tf.pad(f_verts, paddings, "CONSTANT")
      world_pos = tf.where(actuator_mask, finger_world_pos, inputs['world_pos']) # This should be equal to inputs['world_pos'] anyway

    else:
    # If using full world pos directly
      world_pos = inputs['world_pos']
      quit()

    euler, trans = tf.split(inputs['tfn'][:,0], 2, axis=0)
    # world_pos = tf.tile(trans, [1, 2292])
    # world_pos = tf.where(actuator_mask, inputs['world_pos'] * trans[1], inputs['world_pos'])
    # world_pos = inputs['world_pos']
    # world_pos = inputs['mesh_pos']

    # transform world pos by tfn


    # tf_from_euler = tfg_transformation.rotation_matrix_3d.from_euler(euler)
    # world_pos = tfg_transformation.rotation_matrix_3d.rotate(tf.zeros([2292, 3]) + inputs['world_pos'], tf_from_euler) #+ trans

    ###################### Mesh edge features #########

    if utils.using_dm_dataset(self.FLAGS):
      senders, receivers = common.triangles_to_edges(inputs['cells'])
    else:
      senders, receivers = common.sr_mesh_edges(inputs['mesh_edges'])

    relative_mesh_pos = (tf.gather(inputs['mesh_pos'], senders) -
                         tf.gather(inputs['mesh_pos'], receivers))

    relative_world_pos_mesh_edges = (tf.gather(world_pos, senders) -
                         tf.gather(world_pos, receivers))

    mesh_edge_features = tf.concat([
        relative_world_pos_mesh_edges,
        tf.norm(relative_world_pos_mesh_edges, axis=-1, keepdims=True),
        relative_mesh_pos,
        tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)


    mesh_edges = core_model.EdgeSet(
        name='mesh_edges',
        features=self._edge_normalizer(mesh_edge_features, is_training),
        receivers=receivers,
        senders=senders)

    #################### World edge features #########

    if self.FLAGS.compute_world_edges:
      world_senders, world_receivers = common.construct_world_edges(world_pos, inputs['node_type'], self.FLAGS)
    else:
      world_senders, world_receivers = common.sr_world_edges(inputs['world_edges'])



    if self.FLAGS.gripper_force_action_input: # If we are working with force as inputs
      # Force at nodes or edges
      if self.FLAGS.node_total_force_t:
        force_label = inputs['force'][0,0] # Label every node/edge with total gripper force
      else:
        force_label = inputs['force'][0,0] / (0.5 * tf.to_float(tf.shape(world_senders)[0])) # Label every node/world edge with total gripper force / num contacts
        force_change_label = (inputs['target|force'][0,0] - inputs['force'][0,0]) / (0.5 * tf.to_float(tf.shape(world_senders)[0]))


    relative_world_pos = (tf.gather(world_pos, world_senders) -
                          tf.gather(world_pos, world_receivers))


    relative_world_norm = tf.norm(relative_world_pos, axis=-1, keepdims=True)


    if self.FLAGS.gripper_force_action_input and not self.FLAGS.force_label_node and not self.FLAGS.simplified_predict_stress_change_only:
      world_edge_features = tf.concat([
          relative_world_pos,
          relative_world_norm,
          tf.fill(tf.shape(relative_world_norm), force_label)], axis=-1)

    elif self.FLAGS.gripper_force_action_input and not self.FLAGS.force_label_node and self.FLAGS.simplified_predict_stress_change_only:
      if self.FLAGS.predict_stress_t_only:
        world_edge_features = tf.concat([
            relative_world_pos,
            relative_world_norm,
            tf.fill(tf.shape(relative_world_norm), force_label)], axis=-1)
      else:
        world_edge_features = tf.concat([
            relative_world_pos,
            relative_world_norm,
            tf.fill(tf.shape(relative_world_norm), force_label),
            tf.fill(tf.shape(relative_world_norm), force_change_label)], axis=-1)
    else:
      world_edge_features = tf.concat([
          relative_world_pos,
          relative_world_norm], axis=-1)


    world_edges = core_model.EdgeSet(
        name='world_edges',
        features=self._world_edge_normalizer(world_edge_features, is_training),
        receivers=world_receivers,
        senders=world_senders)

    #################### Node features #########
    # Calculate force velocity from gripper_pos
    # '''
  
    # Use either change in force or change in gripper position as the velocity
    zero_vel = tf.fill(tf.shape(inputs['mesh_pos']), 0.0)

    if self.FLAGS.gripper_force_action_input:
      num_f_verts_total = 1180# tf.shape(f_verts)[0] 
      num_verts_total = tf.shape(inputs['mesh_pos'])[0]#.get_shape().as_list()[0]
      pad_diff = num_verts_total - num_f_verts_total 
      paddings = [[0, pad_diff], [0, 0]] 
      if self.FLAGS.simplified_predict_stress_change_only:
        nonzero_vel = tf.pad(unit_f_force_vecs, paddings, "CONSTANT")

      else:
        nonzero_vel = tf.pad(f_force_vecs, paddings, "CONSTANT")

    else:
      nonzero_vel = inputs['target|world_pos'] - inputs['world_pos']

    velocity = tf.where(actuator_mask, nonzero_vel, zero_vel) 
    # '''

    ##########
    node_type = tf.one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)

    if not utils.using_dm_dataset(self.FLAGS):
      node_mod = inputs['node_mod'][:,:]

    # Stress 
    if self.FLAGS.log_stress_t: #TODO this should be false when predicting change in stress
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
        edge_sets=[mesh_edges, world_edges]), 2

  # Is this used? Yes in the background of AbstractModel
  def _build(self, inputs, normalize=True, accumulate=False):

    # For MLP
    ########
    # network_output = self._learned_model(inputs)
    # return self._update(inputs, network_output, normalize, accumulate)
    ###############


    graph, grad_debugs = self._build_graph(inputs, is_training=False)
    per_node_network_output = self._learned_model(graph)
    return self._update(inputs, per_node_network_output, normalize, accumulate)

  def print_debug(self, inputs):
    """L2 loss on position."""
    # inputs_stacked = tf.concat([inputs['gripper_pos'], inputs['force'], inputs['tfn']], axis=1)
    inputs_stacked = tf.squeeze(tf.concat([inputs['gripper_pos'], inputs['tfn'],  inputs['gripper_pos']], axis=1))
    return tf.shape(inputs_stacked)
    return tf.shape(inputs['node_type'])



  @snt.reuse_variables
  def loss(self, inputs, normalize=True, accumulate=False):
    """L2 loss on position."""

    #### MLP #####
    # network_output = self._learned_model(inputs)
    # loss = (tf.reduce_mean(inputs['stress']) - network_output) ** 2
    # return loss
    ######

    graph, debug_val = self._build_graph(inputs, is_training=accumulate) # is_training used to always be True -> accumulate always True
    # graph = self._build_graph(inputs, is_training=accumulate) # is_training used to always be True -> accumulate always True

    network_output = self._learned_model(graph)
    # return network_output 
    if normalize:
      if utils.predict_some_stress_only(self.FLAGS):
        stress_output = network_output
      else:
        object_output, stress_output = tf.split(network_output, [3, 1], 1)
    else:
      if utils.predict_some_stress_only(self.FLAGS):
        stress_output = self._output_normalizer.inverse(network_output)
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
    elif self.FLAGS.predict_stress_change_only:
      target_stress_change = inputs['target|stress'] - inputs['stress']
    elif self.FLAGS.predict_stress_t_only:
      target_stress_change = inputs['stress']
    elif self.FLAGS.predict_stress_change_t1:
      target_stress_change = inputs['target|stress'] - inputs['stress']
      combined_target = tf.concat([target_position_change, target_stress_change], 1)   
    else:
      target_stress_change = inputs['target|stress']
      combined_target = tf.concat([target_position_change, target_stress_change], 1)


    if utils.predict_some_stress_only(self.FLAGS):
      combined_target = target_stress_change



    # build loss
    if utils.using_dm_dataset(self.FLAGS): # If using DM dataset, we care about positions only for soft nodes
      object_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL)

    else: # Else if ABC, we want to care about errors for soft nodes and surface nodes (0 and 2)
        object_mask = tf.not_equal(inputs['node_type'][:, 0], common.NodeType.OBSTACLE)


    # Normalize target position change and stress first
    if normalize:
      target_normalized = self._output_normalizer(combined_target, accumulate)
    else:
      target_normalized = combined_target

    ### OBJECT POS LOSSES ######
    if not utils.predict_some_stress_only(self.FLAGS):
      object_target_normalized, stress_target_normalized = tf.split(target_normalized, [3, 1], 1)
      object_error = tf.reduce_sum((object_target_normalized - object_output)**2, axis=1) 

      if self.FLAGS.gripper_force_action_input: # Then we care about position change of gripper too
        object_loss = tf.reduce_mean(object_error)
      else:
        object_loss = tf.reduce_mean(object_error[object_mask])

    ### STRESS LOSSES #####
    else:
      stress_target_normalized = target_normalized

    stress_error = tf.reduce_sum((stress_target_normalized - stress_output)**2, axis=1) 
    stress_loss = tf.reduce_mean(stress_error[object_mask])


    if utils.predict_some_stress_only(self.FLAGS):
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


    # For simple MLP

    # next_position_pred = inputs['target|world_pos']
    # loss_val = self.loss(inputs, normalize, accumulate)
    # next_stress_pred = per_node_network_output # current mean stress pred
    # return next_position_pred, next_stress_pred, loss_val

    ############


    """Integrate model outputs."""
    actuator_mask = tf.not_equal(inputs['node_type'][:, 0], common.NodeType.OBSTACLE) # Where nodes are NOT actuators
    fixed_points = tf.equal(inputs['node_type'][:, 0], common.NodeType.HANDLE) # Where nodes ARE fixed
    actuator_idx = tf.where(actuator_mask)

    if not utils.using_dm_dataset(self.FLAGS):

      # '''
      f_verts, f_verts_next, _, _ = gripper_world_pos(inputs)
      num_f_verts_total = 1180 # tf.shape(f_verts)[0]
      num_verts_total = tf.shape(inputs['mesh_pos'])[0]
      pad_diff = num_verts_total - num_f_verts_total
      paddings = [[0, pad_diff], [0, 0]]
      finger_curr_pos_gt = tf.pad(f_verts, paddings, "CONSTANT")
      finger_next_pos_gt = tf.pad(f_verts_next, paddings, "CONSTANT")
      curr_pos_gt = tf.where(actuator_mask, inputs['world_pos'], finger_curr_pos_gt)
      next_pos_gt = tf.where(actuator_mask, inputs['target|world_pos'], finger_next_pos_gt)
      position_change_gt = next_pos_gt - curr_pos_gt
      # '''

    else:
      position_change_gt = inputs['target|world_pos'] - inputs['world_pos'] # If not calculating from gripper pos


    # Get predictions
    '''
    if self.FLAGS.predict_log_stress_t_only:
      _ , stress_change = tf.split(self._output_normalizer.inverse(per_node_network_output), [3,1], 1)
      position_change = position_change_gt
    elif self.FLAGS.predict_log_stress_t1_only:
      _, stress_change = tf.split(self._output_normalizer.inverse(per_node_network_output), [3,1], 1)
      position_change = position_change_gt
    elif self.FLAGS.predict_log_stress_change_only:
      _, stress_change = tf.split(self._output_normalizer.inverse(per_node_network_output), [3,1], 1)
      position_change = position_change_gt
    '''
    if utils.predict_some_stress_only(self.FLAGS): # In this case
      stress_change = self._output_normalizer.inverse(per_node_network_output)

      if self.FLAGS.predict_stress_change_only or self.FLAGS.predict_stress_t_only:
        position_change = inputs['world_pos'] - inputs['world_pos'] # Zero position change
        position_change = position_change_gt # JUST FOR NETWORK FOR GRIPPER_POS GRADIENTS
      else:
        position_change = position_change_gt
    else:
      position_change, stress_change = tf.split(self._output_normalizer.inverse(per_node_network_output), [3,1], 1)

    # If using change in gripper pos as action input, update to next gripper position exactly
    if not self.FLAGS.gripper_force_action_input:
      position_change = tf.where(actuator_mask, position_change, position_change_gt)

    # If there are fixed points, update to the same position. Applicable currently only to deep mind dataset where there are fp
    zero_position_change = position_change * 0
    position_change = tf.where(fixed_points, zero_position_change, position_change)


    # Integrate forward
    curr_stress_gt = inputs['stress']
    curr_position = inputs['world_pos']
    next_position_pred = curr_position + position_change

    zero_stress = stress_change * 0
    stress_change = tf.where(actuator_mask, stress_change, zero_stress)


    # next_stress_pred = stress_change  # If just predicting log next stress

    ## These are now all in units of raw stress
    if self.FLAGS.predict_log_stress_t_only:
      next_stress_pred = tf.math.exp(stress_change) - 1
    elif self.FLAGS.predict_stress_t_only:
      next_stress_pred = stress_change
    elif self.FLAGS.predict_log_stress_t1 or self.FLAGS.predict_log_stress_t1_only:
      next_stress_pred = tf.math.exp(stress_change) - 1
    elif self.FLAGS.predict_log_stress_change_t1 or self.FLAGS.predict_log_stress_change_only:
      next_stress_pred = tf.math.exp(stress_change) * (inputs['stress'] + 1) - 1
    elif self.FLAGS.predict_stress_change_t1 or self.FLAGS.predict_stress_change_only:

      next_stress_pred = stress_change + inputs['stress']
    elif self.FLAGS.predict_pos_change_only:
      next_stress_pred = inputs['target|stress']
    else:
      next_stress_pred = stress_change

    # Removed just for gradient simplification 
    # next_stress_pred = tf.nn.relu(next_stress_pred)


    # round to nearest next_gripper_pos
    gripper_vert_pos_change = tf.gather(position_change, actuator_idx)

    # Get loss val
    loss_val = self.loss(inputs, normalize, accumulate)


    # return next_position_pred, curr_stress_pred # For when predicting gripper movement too

    return next_position_pred, next_stress_pred, loss_val
