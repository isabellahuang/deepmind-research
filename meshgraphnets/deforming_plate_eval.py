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

import tensorflow.compat.v1 as tf

from tfdeterminism import patch
# patch()
SEED = 55
os.environ['PYTHONHASHSEED'] = str(SEED)
# random.seed(SEED)
tf.set_random_seed(SEED)


import tensorflow_probability as tfp
from meshgraphnets.common import NodeType
from meshgraphnets import utils
from meshgraphnets import deforming_plate_model



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

    if not utils.using_dm_dataset(FLAGS):
      model_input_dict['gripper_pos'] = inputs['gripper_pos'][step]
      model_input_dict['target|gripper_pos'] = inputs['target|gripper_pos'][step]
      model_input_dict['tfn'] = inputs['tfn'][step]


    if not FLAGS.compute_world_edges:
    # if not utils.using_dm_dataset(FLAGS):
      model_input_dict['world_edges'] = inputs['world_edges'][step] # Use if not trained with world edges in graph

    if FLAGS.gripper_force_action_input:
      model_input_dict['force'] = inputs['force'][step]
      model_input_dict['target|force'] = inputs['target|force'][step]



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



def evaluate(model, inputs, FLAGS, num_steps=None, normalize=True, accumulate=False, tfn=0, eval_step=0):
  """Performs model rollouts and create stats."""
  # return model.loss(inputs), tf.constant(5, dtype=tf.float32)
  initial_state = {k: v[0] for k, v in inputs.items()}

  ################### WE HAVE SET INITIAL STATE TO STEP 20

  if not num_steps:
    num_steps = inputs['cells'].shape[0] # Length of trajectory


  ############################################
  # Instead of doing a whole rollout, do just one prediction for gradient debugging
  '''

  # initial_state['force'] = inputs['force'][20] 
  # initial_state['gripper_pos'] = inputs['gripper_pos'][20] 
  # initial_state['world_pos'] = inputs['world_pos'][20]

  if tfn != 0:
    initial_state['tfn'] = initial_state['tfn'] * 0 + tfn
    # initial_state['force'] = initial_state['force'] * 0 + tfn
    # initial_state['gripper_pos'] = initial_state['gripper_pos'] * 0 + tfn


  next_pos_pred, next_stress_pred, loss_val = model(initial_state, normalize, accumulate)

  scalars, traj_ops = {}, {}

  # Full trajectory

  traj_ops['mean_pred_stress'] = tf.reduce_mean(next_stress_pred)
  traj_ops['pred_stress'] = next_stress_pred

  traj_ops['loss_val'] = loss_val
  traj_ops['pred_pos'] = next_pos_pred
  traj_ops['world_pos'] = initial_state['world_pos']
  traj_ops['gripper_pos'] = initial_state['gripper_pos']
  traj_ops['mesh_edges'] = initial_state['mesh_edges']
  traj_ops['force'] = initial_state['force']
  traj_ops['tfn'] = initial_state['tfn']
  traj_ops['world_edges'] = initial_state['world_edges']
  traj_ops['inputs'] = inputs
  traj_ops['inside_grad'] = tf.gradients(traj_ops['mean_pred_stress'], traj_ops['gripper_pos'], stop_gradients=[traj_ops['world_pos']])

  return loss_val, traj_ops, scalars
  '''

  ##############################################


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
      'world_edges': inputs['world_edges'],
      'node_mod': inputs['node_mod']

  }


  if not utils.using_dm_dataset(FLAGS):
    traj_ops['gt_pd_stress'] = inputs['pd_stress']
    traj_ops['gt_gripper_pos'] = inputs['gripper_pos']
    traj_ops['gt_force'] = inputs['force']
    traj_ops['world_edges'] = inputs['world_edges']



  # return tf.reduce_mean(error), tf.reduce_mean(rollout_losses), scalars
  return tf.reduce_mean(rollout_losses), traj_ops, scalars
