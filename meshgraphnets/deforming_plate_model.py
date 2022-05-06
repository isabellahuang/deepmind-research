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
import os

import sonnet as snt
import tensorflow as tf

from tfdeterminism import patch
# patch()
SEED = 55
os.environ['PYTHONHASHSEED'] = str(SEED)
# random.seed(SEED)
# tf.random.set_seed(SEED)


import tensorflow_graphics.geometry.transformation as tfg_transformation


from meshgraphnets import common
from meshgraphnets import core_model
from meshgraphnets import normalization
from meshgraphnets import utils

import numpy as np
import os
import trimesh

def open_gripper_at_pose(inputs, f_pc_original):
  ''' Transform original point cloud of the gripper to its pose.'''
  # Load transformation params (euler and translation)
  euler, trans = tf.split(inputs['tfn'][:,0], 2, axis=0)

  tf_from_euler = tfg_transformation.rotation_matrix_3d.from_euler(euler)
  f_pc = tfg_transformation.rotation_matrix_3d.rotate(f_pc_original, tf_from_euler) + trans
  original_normal = tf.constant([1., 0., 0.], dtype=tf.float32)
  gripper_normal = tfg_transformation.rotation_matrix_3d.rotate(original_normal, tf_from_euler)
  f1_verts, f2_verts = tf.split(f_pc, 2, axis=0)

  return f1_verts, f2_verts, gripper_normal

def f_verts_at_pos(inputs, gripper_pos, f_pc_original):
  ''' Transform original point cloud of the gripper to its pose, then apply gripper_pos'''

  # Get f_pc_original transformed to the pose, split into the two fingers
  f1_verts, f2_verts, gripper_normal = open_gripper_at_pose(inputs, f_pc_original)

  f1_verts_closed = f1_verts -  gripper_normal * (0.04 - gripper_pos)
  f2_verts_closed = f2_verts + gripper_normal * (0.04 - gripper_pos)
  f_verts = tf.concat((f1_verts_closed, f2_verts_closed), axis=0)

  return f_verts, gripper_normal



def gripper_world_pos(inputs, f_pc_original):

  # Apply gripper closings to the transformed fingers
  f_verts, gripper_normal = f_verts_at_pos(inputs, inputs['gripper_pos'], f_pc_original)
  f_verts_next, _ = f_verts_at_pos(inputs, inputs['target|gripper_pos'], f_pc_original)

  # Get velocity of each gripper
  num_verts_per_f = f_verts.shape[0] // 2

  f1_force_vecs = tf.tile(tf.expand_dims(-1. * gripper_normal, axis=0), [num_verts_per_f, 1]) * (inputs['target|force'] - inputs['force'])
  f2_force_vecs = tf.tile(tf.expand_dims(gripper_normal, axis=0), [num_verts_per_f, 1]) * (inputs['target|force'] - inputs['force'])
  f_force_vecs = tf.concat((f1_force_vecs, f2_force_vecs), axis=0)

  unit_f1_force_vecs = tf.tile(tf.expand_dims(-1. * gripper_normal, axis=0), [num_verts_per_f, 1]) 
  unit_f2_force_vecs = tf.tile(tf.expand_dims(gripper_normal, axis=0), [num_verts_per_f, 1]) 
  unit_f_force_vecs = tf.concat((unit_f1_force_vecs, unit_f2_force_vecs), axis=0)

  return f_verts, f_verts_next, f_force_vecs, unit_f_force_vecs


def gripper_world_pos_projection(inputs, f_pc_original):
  ''' Given initial gripper orientation, infer the initial gripper pos at contact '''
  f1_verts, f2_verts, gripper_normal = open_gripper_at_pose(inputs, f_pc_original) # Might not need this line

  # Apply gripper closings to the transformed fingers
  f_verts_open = f_verts_at_pos(inputs, 0.04, f_pc_original)

  f_verts_next = f_verts_at_pos(inputs, inputs['target|gripper_pos'], f_pc_original)



  def project_to_normal(normal, pos):
    perp_dists = np.dot(pos, normal)
    return pos - normal * perp_dists[:, None]


  def idx_in_rect(pos, corners):
    A, B, C, D  = corners


    AB = B - A 
    AD = D - A

    AM = pos - A
    inside_idx = np.where((np.dot(AM, AB) < np.dot(AB, AB)) & (0 < np.dot(AM, AB)) & (np.dot(AM, AD) < np.dot(AD, AD)) & (0 < np.dot(AM, AD)))
    # idx2 = np.where((np.dot(AM, AD) < np.dot(AD, AD)) & (0 < np.dot(AM, AD)))

    return inside_idx

  def gripper_pos_at_contact(points_in_rect, normal, p1, p2):
    # Closest to p1
    f1_closest_dist = np.min(np.dot(points_in_rect - p1, -1. * normal))
    f2_closest_dist = np.min(np.dot(points_in_rect - p2, normal))

    return 0.04 - min(f1_closest_dist, f2_closest_dist)




  ###############
  import matplotlib.pyplot as plt 
  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111, projection='3d')


  corner_idx1 = [43, 23, 8, 28] # Corners 
  p1_idx, p2_idx = 143, 1036
  corner_idx2 = [1167, 1147, 1132, 1152] # For face 2


  actuator_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.OBSTACLE)
  object_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.AIRFOIL)
  gt_gripper = tf.gather(inputs['world_pos'], tf.squeeze(tf.where(actuator_mask)))
  gt_mesh = tf.gather(inputs['world_pos'], tf.squeeze(tf.where(object_mask)))


  # Project points onto the gripper normal
  idx_in_rect = idx_in_rect(project_to_normal(gripper_normal, gt_mesh).numpy(), project_to_normal(gripper_normal, f_verts_open).numpy()[corner_idx1, :])
  points_in_rect = gt_mesh.numpy()[idx_in_rect]

  contact_gripper_pos = gripper_pos_at_contact(points_in_rect, gripper_normal, f_verts_open[p1_idx], f_verts_open[p2_idx])
  f_verts_at_contact = f_verts_at_pos(inputs, contact_gripper_pos)


  Xg, Yg, Zg = gt_gripper.numpy().T 
  Xm, Ym, Zm = gt_mesh.numpy().T #project_to_normal(gripper_normal, gt_mesh).numpy().T 
  X, Y, Z = f_verts_at_contact.numpy().T #project_to_normal(gripper_normal, f_verts).numpy().T #f_verts.numpy().T
  Xf, Yf, Zf = project_to_normal(gripper_normal, f_verts_open).numpy()[corner_idx1 + corner_idx2, :].T


  ax.scatter(Xm, Ym, Zm)
  ax.scatter(Xg, Yg, Zg)
  ax.scatter(X, Y, Z)
  ax.scatter(Xf, Yf, Zf, s = 100)

  max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

  mid_x = (X.max()+X.min()) * 0.5
  mid_y = (Y.max()+Y.min()) * 0.5
  mid_z = (Z.max()+Z.min()) * 0.5
  ax.set_xlim(mid_x - max_range, mid_x + max_range)
  ax.set_ylim(mid_y - max_range, mid_y + max_range)
  ax.set_zlim(mid_z - max_range, mid_z + max_range)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')

  plt.show()
  quit()
  #############


  return f_verts_at_contact

traj_len = None
signature_dict = {"cells": tf.TensorSpec(shape=[traj_len, None, 4], dtype=tf.int32, name="cells"),
                  "force": tf.TensorSpec(shape=[traj_len, 1, 1], dtype=tf.float32, name="force"),
                  "gripper_pos": tf.TensorSpec(shape=[traj_len, 1, 1], dtype=tf.float32, name="gripper_pos"),
                  "mesh_edges": tf.TensorSpec(shape=[traj_len, None, 2], dtype=tf.int32, name="mesh_edges"),
                  "mesh_pos": tf.TensorSpec(shape=[traj_len, None, 3], dtype=tf.float32, name="mesh_pos"),
                  "node_mod": tf.TensorSpec(shape=[traj_len, None, 1], dtype=tf.float32, name="node_mod"),
                  "node_type": tf.TensorSpec(shape=[traj_len, None, 1], dtype=tf.int32, name="node_type"),
                  "pd_stress": tf.TensorSpec(shape=[traj_len, None, 1], dtype=tf.float32, name="pd_stress"),
                  "sim_world_pos": tf.TensorSpec(shape=[traj_len, None, 3], dtype=tf.float32, name="sim_world_pos"),
                  "stress": tf.TensorSpec(shape=[traj_len, None, 1], dtype=tf.float32, name="stress"),
                  "target|force": tf.TensorSpec(shape=[traj_len, 1, 1], dtype=tf.float32, name="target|force"),
                  "target|gripper_pos": tf.TensorSpec(shape=[traj_len, 1, 1], dtype=tf.float32, name="target|force"),
                  "target|stress": tf.TensorSpec(shape=[traj_len, None, 1], dtype=tf.float32, name="target|stress"),
                  "target|world_pos": tf.TensorSpec(shape=[traj_len, None, 3], dtype=tf.float32, name="target|world_pos"),
                  "tfn": tf.TensorSpec(shape=[traj_len, 6, 1], dtype=tf.float32, name="tfn"),
                  "world_edges": tf.TensorSpec(shape=[traj_len, None, 2], dtype=tf.int32, name="world_edges"),
                  "world_pos": tf.TensorSpec(shape=[traj_len, None, 3], dtype=tf.float32, name="world_pos"),} 


# @snt.allow_empty_variables
class Model(snt.Module):
  """Model for static cloth simulation."""

  def __init__(self, learned_model, FLAGS, name='Model'):
    super().__init__(name=name)
    # super().__init__(name=name)

    ### DEBUGGIN SONNET
    ####

    # with self._enter_variable_scope():
    # with tf.variable_scope("foo"):
    self.FLAGS = FLAGS
    self._learned_model = learned_model
    self.input_signature_dict = signature_dict

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


    ### Get original finger positions
    # Load original finger positions
    finger1_path = os.path.join('meshgraphnets', 'assets', 'finger1_face_uniform' + '.stl')
    f1_trimesh = trimesh.load_mesh(finger1_path)
    f1_verts_original = tf.constant(f1_trimesh.vertices, dtype=tf.float32)

    finger2_path = os.path.join('meshgraphnets', 'assets', 'finger2_face_uniform' + '.stl')
    f2_trimesh = trimesh.load_mesh(finger2_path)
    f2_verts_original = tf.constant(f2_trimesh.vertices, dtype=tf.float32)
    self.f_pc_original = tf.concat((f1_verts_original, f2_verts_original), axis=0)





  @tf.function(input_signature=[signature_dict])
  def accumulate_stats(self, inputs):
    initial_state = {k: v[0] for k, v in inputs.items()}
    self._build_graph(initial_state, is_training=True) # is_training here is = accumulate

    # Accumulate raw targets
    combined_target = self.get_output_targets(initial_state)
    self._output_normalizer(combined_target, accumulate=True)


  def _build_graph(self, inputs, is_training):
    """Builds input graph."""

    # '''
    # Mask way non-kinematic nodes
    actuator_mask = tf.expand_dims(tf.equal(inputs['node_type'][:, 0], common.NodeType.OBSTACLE), -1)

    #######################################################################################
    # Calculate world pos partially from gripper_pos
    if not utils.using_dm_dataset(self.FLAGS):
      f_verts, f_verts_next, f_force_vecs, unit_f_force_vecs = gripper_world_pos(inputs, self.f_pc_original) 
      num_f_verts_total = tf.shape(f_verts)[0]
      num_verts_total = tf.shape(inputs['mesh_pos'])[0]
      pad_diff = num_verts_total - num_f_verts_total
      paddings = [[0, pad_diff], [0, 0]]
      finger_world_pos = tf.pad(f_verts, paddings, "CONSTANT")

      # world_pos = tf.where(actuator_mask, finger_world_pos, inputs['world_pos']) # This should be equal to inputs['world_pos'] anyway
      world_pos = tf.concat([f_verts, inputs['world_pos'][1180:, :]], axis=0)


    else:
    # If using full world pos directly
      world_pos = inputs['world_pos']
    # '''



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
        force_label = inputs['force'][0,0] / (0.5 * tf.cast(tf.shape(world_senders)[0], tf.float32)) # Label every node/world edge with total gripper force / num contacts
        force_change_label = (inputs['target|force'][0,0] - inputs['force'][0,0]) / (0.5 * tf.cast(tf.shape(world_senders)[0], tf.float32))


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

    # Use either change in force or change in gripper position as the velocity
    # '''
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

    # '''


    return core_model.MultiGraph(
        node_features=self._node_normalizer(node_features, is_training),
        edge_sets=[mesh_edges, world_edges])


  # Is this used? Yes in the background of AbstractModel
  def __call__(self, inputs, normalize=True, accumulate=False):

  # def _build(self, inputs, normalize=True, accumulate=False):

    # For MLP
    ########
    # network_output = self._learned_model(inputs)
    # return self._update(inputs, network_output, normalize, accumulate)
    ###############



    graph = self._build_graph(inputs, is_training=False)
    per_node_network_output = self._learned_model(graph)
    return self._update(inputs, per_node_network_output, normalize, accumulate)

  def print_debug(self, inputs):
    """L2 loss on position."""
    # inputs_stacked = tf.concat([inputs['gripper_pos'], inputs['force'], inputs['tfn']], axis=1)
    return tf.reduce_mean(inputs['world_pos'])
    inputs_stacked = tf.squeeze(tf.concat([inputs['gripper_pos'], inputs['tfn'],  inputs['gripper_pos']], axis=1))
    return tf.shape(inputs_stacked)
    return tf.shape(inputs['node_type'])


  # @tf.function works here!
  def get_output_targets(self, inputs):
    '''Get non-normalized targets from inputs'''
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

    return combined_target

  # @snt.reuse_variables
  def loss(self, inputs, normalize=True, accumulate=False):
    """L2 loss on position."""

    # with tf.variable_scope("foo", reuse=True):
    #### MLP #####
    # network_output = self._learned_model(inputs)
    # loss = (tf.reduce_mean(inputs['stress']) - network_output) ** 2
    # return loss
    ######

    # Get targets
    combined_target = self.get_output_targets(inputs)

    # Get prediction
    graph = self._build_graph(inputs, is_training=accumulate) # is_training used to always be True -> accumulate always True

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


    if self.FLAGS.loss_function.lower() == "mse":
      stress_error = tf.reduce_sum((stress_target_normalized - stress_output)**2, axis=1) 
    elif self.FLAGS.loss_function.lower() == "mae":
      stress_error = tf.reduce_sum(tf.math.abs(stress_target_normalized - stress_output), axis=1) 
    else:
      assert(False)
    stress_loss = tf.reduce_mean(stress_error[object_mask])


    if utils.predict_some_stress_only(self.FLAGS):
      loss = stress_loss
    elif self.FLAGS.predict_pos_change_only:
      loss = object_loss
    else:
      loss = object_loss + stress_loss


    return loss


  def _update(self, inputs, per_node_network_output, normalize=True, accumulate=False):


    # For simple MLP

    # next_position_pred = inputs['target|world_pos']
    # loss_val = self.loss(inputs, normalize, accumulate)
    # next_stress_pred = per_node_network_output # current mean stress pred
    # return next_position_pred, next_stress_pred, loss_val

    ############


    """Integrate model outputs."""
    actuator_mask = tf.expand_dims(tf.not_equal(inputs['node_type'][:, 0], common.NodeType.OBSTACLE), -1) # Where nodes are NOT actuators
    fixed_points = tf.expand_dims(tf.equal(inputs['node_type'][:, 0], common.NodeType.HANDLE), -1) # Where nodes ARE fixed
    actuator_idx = tf.where(actuator_mask)

    if not utils.using_dm_dataset(self.FLAGS):

      # '''
      f_verts, f_verts_next, _, _ = gripper_world_pos(inputs, self.f_pc_original)
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
    if utils.predict_some_stress_only(self.FLAGS): # In this case
      stress_change = self._output_normalizer.inverse(per_node_network_output)


      if self.FLAGS.predict_stress_change_only or self.FLAGS.predict_stress_t_only:
        position_change = inputs['world_pos'] - inputs['world_pos'] # Zero position change
        # position_change = position_change_gt # JUST FOR NETWORK FOR GRIPPER_POS GRADIENTS
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
    next_stress_pred = tf.nn.relu(next_stress_pred)

    # round to nearest next_gripper_pos
    gripper_vert_pos_change = tf.gather(position_change, actuator_idx)

    # Get loss val
    loss_val = self.loss(inputs, normalize, accumulate)


    return next_position_pred, next_stress_pred, loss_val
