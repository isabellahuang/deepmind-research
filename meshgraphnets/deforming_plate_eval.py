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

# from tfdeterminism import patch
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
    if FLAGS.predict_log_stress_t_only or FLAGS.predict_stress_t_only: # it shouldnt have access to inputs['stress'][step] though, but needed to calculate loss
      curr_stress_feed = inputs['stress'][step]

    if FLAGS.predict_pos_change_from_initial_only:
      cur_pos = inputs['world_pos'][step]

    model_input_dict = {**initial_state,
                        'world_pos': cur_pos, 
                        'target|world_pos': inputs['target|world_pos'][step],
                        'stress': curr_stress_feed, #in raw units
                        'target|stress': inputs['target|stress'][step],
                        'sim_world_pos': inputs['sim_world_pos'][step]}

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

    use_precomputed = False
    next_pos_pred, next_stress_pred, loss_val = model(model_input_dict, use_precomputed, normalize, accumulate)


    if FLAGS.predict_log_stress_t_only or FLAGS.predict_stress_t_only:
      log_stress = tf.math.log(next_stress_pred + 1) # Next stress pred is actually stress now, at time t
      raw_stress = next_stress_pred
    else:
      log_stress = tf.math.log(curr_stress + 1)
      raw_stress = curr_stress


    stress_trajectory = stress_trajectory.write(step, raw_stress)

    if FLAGS.predict_pos_change_from_initial_only:

      trajectory = trajectory.write(step, next_pos_pred)
    else:
      trajectory = trajectory.write(step, cur_pos)

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

  f_verts_open, gripper_normal = deforming_plate_model.f_verts_at_pos(inputs, [0.04, 0.04], f_pc_original)


  def project_to_normal(normal, pos):
    perp_dists_tf = tf.tensordot(pos, tf.transpose(normal), 1) #tensordot with transpose of normal instead of normal

    return pos - perp_dists_tf * normal
    # return pos - normal * perp_dists_tf[:, None]


  def idx_in_rect_tf(pos, f_verts_open_proj):
    A, B, C, D = f_verts_open_proj[43], f_verts_open_proj[23], f_verts_open_proj[8], f_verts_open_proj[28]
    AB = B - A 
    AD = D - A
    AM = pos - A

    inside_idx = tf.squeeze(tf.where((tf.tensordot(AM, AB, 1) < tf.tensordot(AB, AB, 1)) & (0 < tf.tensordot(AM, AB, 1)) & (tf.tensordot(AM, AD, 1) < tf.tensordot(AD, AD, 1)) & (0 < tf.tensordot(AM, AD, 1))))

    return inside_idx


  def gripper_pos_at_contact_tf(points_in_rect, normal, p1, p2):

    f1_closest_dist = tf.reduce_min(tf.tensordot(points_in_rect - p1, -1. * tf.transpose(normal), 1))

    f2_closest_dist = tf.reduce_min(tf.tensordot(points_in_rect - p2, tf.transpose(normal),1))
    return tf.stack([0.04 - f1_closest_dist + 0.0005, 0.04 - f2_closest_dist + 0.0005])

  corner_idx1 = [43, 23, 8, 28] # Corners 
  p1_idx, p2_idx = 143, 1036
  corner_idx2 = [1167, 1147, 1132, 1152] # For face 2


  actuator_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.OBSTACLE)
  object_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.AIRFOIL)
  gt_gripper = tf.gather(inputs['world_pos'], tf.squeeze(tf.where(actuator_mask)))
  gt_mesh = tf.gather(inputs['world_pos'], tf.squeeze(tf.where(object_mask)))


  # Project points onto the gripper normal
  idx_in_rect_tf = idx_in_rect_tf(project_to_normal(gripper_normal, gt_mesh), project_to_normal(gripper_normal, f_verts_open))
  points_in_rect_tf = tf.gather(gt_mesh, idx_in_rect_tf)
  contact_gripper_pos_tf = gripper_pos_at_contact_tf(points_in_rect_tf, gripper_normal, f_verts_open[p1_idx], f_verts_open[p2_idx])

  contact_gripper_pos_tf = tf.expand_dims(contact_gripper_pos_tf, axis=0)

  return contact_gripper_pos_tf

def evaluate_one_step(model, inputs, FLAGS, num_steps=None, normalize=True, accumulate=False, tfn=0, eval_step=0, force=0):

  # This should be replaced later, move this to the dataset processing   
  initial_state = {k: v[0] for k, v in inputs.items()}

  # initial_state['gripper_pos'] = gripper_pos_at_first_contact(initial_state, model.f_pc_original)

  use_precomputed = True
  next_pos_pred, next_stress_pred, loss_val = model(initial_state, use_precomputed, normalize, accumulate)
  return loss_val



def evaluate(model, inputs, FLAGS, num_steps=None, normalize=True, accumulate=False, tfn=0, eval_step=0, force=0):
  """Performs model rollouts and create stats."""

  # For gradients

  ####

  # Use partial trajectory
  inputs = {k: v[:30] for k, v in inputs.items()}

  initial_state = {k: v[0] for k, v in inputs.items()}

  # Use this only when calculating gradients for grasp refinement
  # initial_state['gripper_pos'] = gripper_pos_at_first_contact(initial_state, model.f_pc_original)

  if not num_steps:
    num_steps = inputs['node_type'].shape[0] # Length of trajectory


  pos_prediction, stress_prediction, rollout_losses = _rollout(model, initial_state, inputs, num_steps, FLAGS, normalize, accumulate)
  stress_cutoff = 500.0



  # stress_prediction is in log
  log_gt_stress = tf.math.log(inputs['stress'] + 1)

  # pos_error = tf.reduce_mean(tf.reduce_sum(((pos_prediction - inputs['sim_world_pos'][:num_steps]) ** 2), axis=-1)[:, 1180:], axis=-1) 
  # baseline_pos_error = tf.reduce_mean(tf.reduce_sum(((inputs['sim_world_pos'][:num_steps] - inputs['sim_world_pos'][0]) ** 2), axis=-1)[:, 1180:], axis=-1)

  # MAE instead
  pos_difference_norm = tf.norm(pos_prediction - inputs['sim_world_pos'][:num_steps], axis=-1) #(48, num_nodes)
  baseline_difference_norm = tf.norm(inputs['sim_world_pos'][:num_steps] - inputs['sim_world_pos'][0], axis=-1) #(48, num_nodes)

  # Set difference to zero when stress is under threshold
  pos_difference_norm = tf.where(tf.squeeze(inputs['stress'], -1) > stress_cutoff, pos_difference_norm, tf.zeros_like(pos_difference_norm))[:, 1180:] # (48, num_nodes - 1180)
  baseline_difference_norm = tf.where(tf.squeeze(inputs['stress'], -1) > stress_cutoff, baseline_difference_norm, tf.zeros_like(baseline_difference_norm))[:, 1180:]

  pos_error = tf.reduce_mean(pos_difference_norm, axis=-1) # (48,)
  baseline_pos_error = tf.reduce_mean(baseline_difference_norm, axis=-1) # (48,)

  # Raw stress errors, in real units
  # stress_error = tf.reduce_mean(tf.reduce_sum(((stress_prediction - inputs['stress'][:num_steps]) ** 2), axis=-1)[:, 1180:], axis=-1) # don't include the grippers
  # baseline_stress_error = tf.reduce_mean(tf.reduce_sum(((inputs['stress'][:num_steps] - inputs['stress'][0]) ** 2), axis=-1)[:, 1180:], axis=-1) # don't inculde the grippers

  # MAE instead
  stress_error = tf.reduce_mean(tf.reduce_sum((tf.math.abs(stress_prediction - inputs['stress'][:num_steps])), axis=-1)[:, 1180:], axis=-1) # don't include the grippers
  baseline_stress_error = tf.reduce_mean(tf.reduce_sum((tf.math.abs(inputs['stress'][:num_steps] - inputs['stress'][0])), axis=-1)[:, 1180:], axis=-1) # don't inculde the grippers

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

  deformations = pos_prediction - inputs['world_pos']
  actual_deformations = inputs['sim_world_pos'] - inputs['world_pos']
  deformation_norms = tf.norm(deformations, axis=-1)
  actual_deformation_norms = tf.norm(actual_deformations, axis=-1)

  # Get def percent errors

  difference_in_deformation = actual_deformations - deformations
  difference_in_deformation_norms = tf.norm(difference_in_deformation, axis=-1)
  eps = 1e-10
  actual_deformation_norms_no_zeros = tf.where(tf.abs(actual_deformation_norms) < eps, eps * tf.ones_like(actual_deformation_norms), actual_deformation_norms)
  deformation_norms_percent_error = tf.math.divide(difference_in_deformation_norms, actual_deformation_norms_no_zeros) # (start at 2: otherwise lots of zero entries)

  # Mask out deformation when object not in contact (e.g. swaying in the wind)
  deformation_norms_percent_error = tf.where(tf.squeeze(inputs['stress'], -1) > stress_cutoff, deformation_norms_percent_error, tf.zeros_like(deformation_norms_percent_error))
  deformation_norms_percent_error = deformation_norms_percent_error[2:, 1180:]

  # Get stress percent errors
  stress_prediction_abs_error = tf.math.abs(stress_prediction - inputs['stress'])
  actual_stress_no_zeros = tf.where(tf.abs(inputs['stress']) < eps, eps * tf.ones_like(inputs['stress']), inputs['stress'])
  stress_percent_error = tf.math.divide(stress_prediction_abs_error, actual_stress_no_zeros)
  stress_percent_error = tf.where(inputs['stress'] > stress_cutoff, stress_percent_error, tf.zeros_like(stress_percent_error))
  stress_percent_error = stress_percent_error[2:, 1180:]

  nodes_under_stress = tf.where(tf.squeeze(inputs['stress'], -1) > stress_cutoff)
  '''
  print(nodes_under_stress.shape)
  print(tf.gather(tf.squeeze(inputs['stress'], -1), nodes_under_stress).shape)
  print(tf.gather(deformation_norms, nodes_under_stress).shape)
  print(tf.where(tf.squeeze(inputs['stress'], -1) > stress_cutoff, actual_deformation_norms_no_zeros, tf.zeros_like(actual_deformation_norms_no_zeros)).shape)
  print(stress_prediction.shape)
  '''
  # (19402, 2)
  # (19402, 2, 2332)
  # (19402, 2, 2332)
  # (48, 2332)
  # (48, 2332, 1)

  ### Deformation norms under stress only
  squeezed_stress = tf.squeeze(inputs['stress'], -1)
  squeezed_stress_prediction = tf.squeeze(stress_prediction, -1)

  nodes_under_stress_only = tf.where(squeezed_stress > stress_cutoff, tf.ones_like(deformation_norms), tf.zeros_like(deformation_norms))
  deformation_norms_under_stress_only = tf.where(squeezed_stress > stress_cutoff, deformation_norms, tf.zeros_like(deformation_norms))
  actual_deformation_norms_under_stress_only = tf.where(tf.squeeze(inputs['stress'], -1) > stress_cutoff, actual_deformation_norms, tf.zeros_like(actual_deformation_norms))

  sum_nodes_under_stress = tf.reduce_sum(nodes_under_stress_only, axis=-1)
  sum_nodes_under_stress = tf.where(tf.equal(sum_nodes_under_stress, 0), tf.ones_like(sum_nodes_under_stress), sum_nodes_under_stress)

  mean_deformation_under_stress_only = tf.math.divide(tf.reduce_sum(deformation_norms_under_stress_only, axis=-1), sum_nodes_under_stress)
  mean_actual_deformation_under_stress_only = tf.math.divide(tf.reduce_sum(actual_deformation_norms_under_stress_only, axis=-1), sum_nodes_under_stress)

  ### Stress under stress only
  stress_under_stress_only = tf.where(squeezed_stress > stress_cutoff, squeezed_stress_prediction, tf.zeros_like(squeezed_stress_prediction))
  actual_stress_under_stress_only = tf.where(squeezed_stress > stress_cutoff, squeezed_stress, tf.zeros_like(squeezed_stress))

  mean_stress_under_stress_only = tf.math.divide(tf.reduce_sum(stress_under_stress_only, axis=-1), sum_nodes_under_stress)
  actual_mean_stress_under_stress_only = tf.math.divide(tf.reduce_sum(actual_stress_under_stress_only, axis=-1), sum_nodes_under_stress)

  traj_ops = {
      'faces': inputs['cells'],
      'name': inputs['name'],
      'node_type': inputs['node_type'],
      'mesh_pos': inputs['mesh_pos'],
      'gt_pos': inputs['world_pos'],
      'gt_stress': inputs['stress'],
      'pred_pos': pos_prediction,
      'pred_stress': stress_prediction,
      'sim_world_pos': inputs['sim_world_pos'] if 'sim_world_pos' in inputs.keys() else inputs['world_pos'],
      'mean_pred_stress': tf.reduce_mean(stress_prediction[:, 1180:]),
      'max_pred_stress': tf.reduce_max(stress_prediction[:, 1180:]),

      'mean_actual_stress': tf.reduce_mean(inputs['stress'][:num_steps, 1180:]),
      'max_actual_stress': tf.reduce_max(inputs['stress'][:num_steps, 1180:]),

      'mean_actual_stress_under_stress_only': tf.reduce_mean(actual_mean_stress_under_stress_only),
      'mean_pred_stress_under_stress_only': tf.reduce_mean(mean_stress_under_stress_only),

      'mean_deformation_norm': tf.reduce_mean(tf.where(tf.squeeze(inputs['stress'], -1) > stress_cutoff, deformation_norms, tf.zeros_like(deformation_norms))[:, 1180:]),
      'max_deformation_norm': tf.reduce_max(tf.where(tf.squeeze(inputs['stress'], -1) > stress_cutoff, deformation_norms, tf.zeros_like(deformation_norms))[:, 1180:]),

      'mean_actual_deformation_norm': tf.reduce_mean(tf.where(tf.squeeze(inputs['stress'], -1) > stress_cutoff, actual_deformation_norms_no_zeros, tf.zeros_like(actual_deformation_norms_no_zeros))[:, 1180:]),
      'max_actual_deformation_norm': tf.reduce_max(tf.where(tf.squeeze(inputs['stress'], -1) > stress_cutoff, actual_deformation_norms_no_zeros, tf.zeros_like(actual_deformation_norms_no_zeros))[:, 1180:]),
      
      'mean_deformation_norm_under_stress_only': tf.reduce_mean(mean_deformation_under_stress_only),
      'mean_actual_deformation_norm_under_stress_only': tf.reduce_mean(mean_actual_deformation_under_stress_only),


      'mean_deformation_percent_error': tf.reduce_mean(deformation_norms_percent_error),
      'mean_stress_percent_error': tf.reduce_mean(stress_percent_error),
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

  # Just for debugging whether we can optimize over increasing contact points
  # world_senders, world_receivers = common.construct_world_edges(world_pos, inputs['node_type'], self.FLAGS)




  if not utils.using_dm_dataset(FLAGS):
    traj_ops['gt_pd_stress'] = inputs['pd_stress']
    traj_ops['gt_gripper_pos'] = inputs['gripper_pos']
    traj_ops['gt_force'] = inputs['force']
    # traj_ops['world_edges'] = inputs['world_edges'] # got rid of world_edges from inputs to reduce function retracing



  return tf.reduce_mean(rollout_losses), traj_ops, scalars # The correct one
