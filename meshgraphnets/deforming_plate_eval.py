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
"""Functions to build evaluation metrics for cloth data."""
import os

import tensorflow as tf

from tfdeterminism import patch
# patch()
SEED = 55
os.environ['PYTHONHASHSEED'] = str(SEED)
# random.seed(SEED)
tf.random.set_seed(SEED)


import tensorflow_probability as tfp
from meshgraphnets.common import NodeType
from meshgraphnets import utils
from meshgraphnets import deforming_plate_model
from meshgraphnets import common



def _rollout(model, initial_state, inputs, num_steps, FLAGS, normalize, accumulate):
  """Rolls out a model trajectory."""
  mask = tf.equal(initial_state['node_type'][:, 0], NodeType.NORMAL)


  # def step_fn(step, cur_pos, curr_stress, inputs, trajectory, stress_trajectory, loss_trajectory, debug_trajectory):
  def step_fn(step, cur_pos, curr_stress, inputs, trajectory, stress_trajectory, loss_trajectory):


    curr_stress_feed = curr_stress
    if FLAGS.predict_log_stress_t_only or FLAGS.predict_stress_t_only: # it shouldnt have access to inputs['stress'][step] though
      curr_stress_feed = inputs['stress'][step]

    model_input_dict = {**initial_state,
                        'world_pos': cur_pos, 
                        'target|world_pos': inputs['target|world_pos'][step],
                        'stress': curr_stress_feed, #in raw units
                        'target|stress': inputs['target|stress'][step]}

    '''
    if not utils.using_dm_dataset(FLAGS):
      model_input_dict['gripper_pos'] = inputs['gripper_pos'][step]
      model_input_dict['target|gripper_pos'] = inputs['target|gripper_pos'][step]
      model_input_dict['tfn'] = inputs['tfn'][step]
    '''


    if not FLAGS.compute_world_edges:
    # if not utils.using_dm_dataset(FLAGS):
      model_input_dict['world_edges'] = inputs['world_edges'][step] # Use if not trained with world edges in graph

    if FLAGS.gripper_force_action_input:
      model_input_dict['force'] = inputs['force'][step]
      model_input_dict['target|force'] = inputs['target|force'][step]

    else:
      model_input_dict['gripper_pos'] = inputs['gripper_pos'][step]
      model_input_dict['target|gripper_pos'] = inputs['target|gripper_pos'][step]


    next_pos_pred, next_stress_pred, loss_val = model(model_input_dict, normalize, accumulate)




    trajectory = trajectory.write(step, cur_pos)

    if FLAGS.predict_log_stress_t_only or FLAGS.predict_stress_t_only:
      log_stress = tf.math.log(next_stress_pred + 1) # Next stress pred is actually stress now, at time t
      raw_stress = next_stress_pred
    else:
      log_stress = tf.math.log(curr_stress + 1)
      raw_stress = curr_stress

    stress_trajectory = stress_trajectory.write(step, raw_stress)

    # Save to loss trajectory
    loss_trajectory = loss_trajectory.write(step, loss_val)
    
    return step+1, next_pos_pred, next_stress_pred, inputs, trajectory, stress_trajectory, loss_trajectory


  _, _, _, _, output, stress_output, loss_output = tf.while_loop(
      cond=lambda step, cur, cur_gp, inputs, traj, stress_traj, loss_traj: tf.less(step, num_steps),
      body=step_fn,
      loop_vars=(0, initial_state['world_pos'], initial_state['stress'], inputs,
                 tf.TensorArray(tf.float32, num_steps), tf.TensorArray(tf.float32, num_steps), tf.TensorArray(tf.float32, num_steps)),
      parallel_iterations=1)

  return output.stack(), stress_output.stack(), loss_output.stack()#, debug_output.stack()


def get_real_median(v):
    v = tf.reshape(v, [-1])
    l = v.get_shape()[0]
    mid = l//2 + 1
    val = tf.nn.top_k(v, mid).values
    if l % 2 == 1:
        return val[-1]
    else:
        return 0.5 * (val[-1] + val[-2])


def refine_inputs(model, inputs, FLAGS, grad_data, custom_constant=0):

  grad = grad_data[0]
  # random_grad_data = tf.random.uniform([6], minval=-1, maxval=1)

  dot_tf = tf.expand_dims(tf.constant([0,0,0, 1, 1, 1], dtype=tf.float32), axis=-1)
  grad = tf.math.multiply(grad, dot_tf)

  # grad = tf.random.uniform([6], minval=-1, maxval=1) #this is not the way to do it, otherwise every constant is random

  '''
  euler, trans = tf.split(grad, 2, axis=0)
  trans_norm = tf.norm(trans)
  constant = 0.01 / trans_norm
  '''
  constant = custom_constant


  refined_tfn = inputs['tfn'][20] + custom_constant * grad
  # refined_tfn = inputs['force'][20] + custom_constant * grad
  # refined_tfn = inputs['gripper_pos'][20] + custom_constant * grad

  _, traj_ops_refined, _ = evaluate(model, inputs, FLAGS, tfn=refined_tfn)
  traj_ops_refined['refined_tfn'] = refined_tfn
  traj_ops_refined['grad_data'] = grad_data
  traj_ops_refined['grad'] = grad_data[0]
  traj_ops_refined['constant'] = constant
  traj_ops_refined['og_gripper_pos'] = inputs['gripper_pos'][20]

  return traj_ops_refined


def gripper_pos_at_first_contact(inputs, f_pc_original):
  # import numpy as np
  ''' Given initial gripper orientation, infer the initial gripper pos at contact '''

  # Apply gripper closings to the transformed fingers
  f_verts_open, gripper_normal = deforming_plate_model.f_verts_at_pos(inputs, 0.04, f_pc_original)

  def project_to_normal(normal, pos):

    # perp_dists = np.dot(pos, normal)
    perp_dists_tf = tf.tensordot(pos, normal, 1)

    return pos - normal * perp_dists_tf[:, None]



  def idx_in_rect_tf(pos, f_verts_open_proj):
    A, B, C, D = f_verts_open_proj[43], f_verts_open_proj[23], f_verts_open_proj[8], f_verts_open_proj[28]
    AB = B - A 
    AD = D - A
    AM = pos - A

    inside_idx = tf.squeeze(tf.where((tf.tensordot(AM, AB, 1) < tf.tensordot(AB, AB, 1)) & (0 < tf.tensordot(AM, AB, 1)) & (tf.tensordot(AM, AD, 1) < tf.tensordot(AD, AD, 1)) & (0 < tf.tensordot(AM, AD, 1))))

    return inside_idx


  def gripper_pos_at_contact_tf(points_in_rect, normal, p1, p2):
    f1_closest_dist = tf.reduce_min(tf.tensordot(points_in_rect - p1, -1. * normal, 1))

    f2_closest_dist = tf.reduce_min(tf.tensordot(points_in_rect - p2, normal,1))
    return 0.04 - tf.reduce_min(tf.stack([f1_closest_dist, f2_closest_dist])) #- 0.0005 # little bit of tolerance at the end

  corner_idx1 = [43, 23, 8, 28] # Corners 
  p1_idx, p2_idx = 143, 1036
  corner_idx2 = [1167, 1147, 1132, 1152] # For face 2


  actuator_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.OBSTACLE)
  object_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.AIRFOIL)
  gt_gripper = inputs['world_pos'][:1180, :] #tf.gather(inputs['world_pos'], tf.squeeze(tf.where(actuator_mask)))
  gt_mesh = inputs['world_pos'][1180:, :] #tf.gather(inputs['world_pos'], tf.squeeze(tf.where(object_mask)))


  # Project points onto the gripper normal
  idx_in_rect_tf = idx_in_rect_tf(project_to_normal(gripper_normal, gt_mesh), project_to_normal(gripper_normal, f_verts_open))
  points_in_rect_tf = tf.gather(gt_mesh, idx_in_rect_tf)
  contact_gripper_pos_tf = gripper_pos_at_contact_tf(points_in_rect_tf, gripper_normal, f_verts_open[p1_idx], f_verts_open[p2_idx])

  contact_gripper_pos_tf = tf.expand_dims(contact_gripper_pos_tf, axis=0)
  contact_gripper_pos_tf = tf.expand_dims(contact_gripper_pos_tf, axis=0)


  return contact_gripper_pos_tf



def evaluate(model, inputs, FLAGS, num_steps=None, normalize=True, accumulate=False, tfn=0, eval_step=0, force=0):
  """Performs model rollouts and create stats."""

  # For gradients

  ####

  initial_state = {k: v[0] for k, v in inputs.items()}
  initial_state['gripper_pos'] = gripper_pos_at_first_contact(initial_state, model.f_pc_original)


  if not num_steps:
    num_steps = inputs['cells'].shape[0] # Length of trajectory



  # ##### FOR MEMORY SAKE. DELETE AFTER
  # num_steps = 3

  pos_prediction, stress_prediction, rollout_losses = _rollout(model, initial_state, inputs, num_steps, FLAGS, normalize, accumulate)


  # stress_prediction is in log
  log_gt_stress = tf.math.log(inputs['stress'] + 1)

  pos_error = tf.reduce_mean(tf.reduce_sum(((pos_prediction - inputs['world_pos'][:num_steps]) ** 2), axis=-1), axis=-1) 
  baseline_pos_error = tf.reduce_mean(tf.reduce_sum(((inputs['world_pos'][:num_steps] - inputs['world_pos'][0]) ** 2), axis=-1), axis=-1)

  # Raw stress errors
  stress_error = tf.reduce_mean(tf.reduce_sum(((stress_prediction - inputs['stress'][:num_steps]) ** 2), axis=-1), axis=-1) # new way
  baseline_stress_error = tf.reduce_mean(tf.reduce_sum(((inputs['stress'][:num_steps] - inputs['stress'][0]) ** 2), axis=-1), axis=-1) # new way
  # stress_error = baseline_stress_error # for simple mlp only


  scalars = {}
  
  log_actual_stress = tf.math.log(inputs['stress'][-1] + 1)

  scalars['actual_final_stress'] = tf.reduce_mean(log_gt_stress[-1])
  scalars['pred_final_stress'] = tf.reduce_mean(stress_prediction[-1])
  scalars['stress_error'] = stress_error
  scalars['stress_mean_error'] = tf.reduce_mean(stress_error)
  scalars['stress_final_error'] = stress_error[-1]
  scalars['pos_error'] = pos_error
  scalars['pos_mean_error'] = tf.reduce_mean(pos_error)
  scalars['pos_final_error'] = pos_error[-1]
  scalars['pred_final_pos'] = pos_prediction[-1]
  scalars['rollout_losses'] = rollout_losses
  scalars['baseline_pos_mean_error'] = tf.reduce_mean(baseline_pos_error)
  scalars['baseline_pos_final_error'] = baseline_pos_error[-1]
  scalars['baseline_stress_mean_error'] = tf.reduce_mean(baseline_stress_error)
  scalars['baseline_stress_final_error'] = baseline_stress_error[-1]


  traj_ops = {
      'faces': inputs['cells'],
      'node_type': inputs['node_type'],
      'mesh_pos': inputs['mesh_pos'],
      'gt_pos': inputs['world_pos'],
      'gt_stress': inputs['stress'],
      'pred_pos': pos_prediction,
      'pred_stress': stress_prediction,
      'sim_world_pos': inputs['sim_world_pos'] if 'sim_world_pos' in inputs.keys() else inputs['world_pos'],
      'mean_pred_stress': tf.reduce_mean(stress_prediction),
      # 'avg_gt_stress': tf.shape(log_gt_stress[:num_steps]), #(n_horizon, n_nodes, 1)
      # 'avg_pred_stress': tf.shape(stress_prediction) #(n_horizon, n_nodes, 1)
      'avg_gt_stress': tf.reduce_mean(tf.squeeze(log_gt_stress[:num_steps], axis=-1), axis=-1), #(n_horizon, n_nodes, 1)
      'avg_pred_stress': tf.reduce_mean(tf.squeeze(stress_prediction, axis=-1), axis=-1), #(n_horizon, n_nodes, 1)
      # 'pred_gripper_pos': gripper_pos_prediction,
      "pos_error": pos_error,
      'baseline_pos_error': baseline_pos_error,
      'rollout_losses': rollout_losses,
      'stress_error': stress_error,
      'tfn': initial_state['tfn'],
      'force': inputs['force'],
      # 'gripper_normal': gripper_normal
      'gripper_pos': inputs['gripper_pos'],
      # 'world_edges': inputs['world_edges'],
      'node_mod': inputs['node_mod'],
  }




  if not utils.using_dm_dataset(FLAGS):
    traj_ops['gt_pd_stress'] = inputs['pd_stress']
    traj_ops['gt_gripper_pos'] = inputs['gripper_pos']
    traj_ops['gt_force'] = inputs['force']
    # traj_ops['world_edges'] = inputs['world_edges'] # got rid of world_edges from inputs to reduce function retracing



  return tf.reduce_mean(rollout_losses), traj_ops, scalars # The correct one
