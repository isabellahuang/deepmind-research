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

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from meshgraphnets.common import NodeType
from meshgraphnets import utils

def _rollout(model, initial_state, inputs, num_steps, FLAGS, normalize, accumulate):
  """Rolls out a model trajectory."""
  mask = tf.equal(initial_state['node_type'][:, 0], NodeType.NORMAL)


  def step_fn(step, cur_pos, curr_stress, inputs, trajectory, stress_trajectory, loss_trajectory):
    curr_stress_feed = curr_stress
    if FLAGS.predict_log_stress_t_only:
      curr_stress_feed = inputs['stress'][step]


    next_pos_pred, next_stress_pred, loss_val = model({**initial_state,
                        'world_pos': cur_pos, ### JUST FOR DEBUGGING used inputs['world_pos'][step]
                        'target|world_pos': inputs['target|world_pos'][step],
                        'stress': curr_stress_feed, #in raw units
                        'target|stress': inputs['target|stress'][step],
                        'world_edges': inputs['world_edges'][step], # Use if not trained with world edges in graph
                        'force': inputs['force'][step],
                        'target|force': inputs['target|force'][step]}, normalize, accumulate)

                        
    # gripper_pos_trajectory = gripper_pos_trajectory.write(step, curr_gripper_pos)
    trajectory = trajectory.write(step, cur_pos)

    if FLAGS.predict_log_stress_t_only:
      log_stress = tf.math.log(next_stress_pred + 1)
    else:
      log_stress = tf.math.log(curr_stress + 1)
    stress_trajectory = stress_trajectory.write(step, log_stress)

    # Save to loss trajectory
    loss_trajectory = loss_trajectory.write(step, loss_val)
    
    return step+1, next_pos_pred, next_stress_pred, inputs, trajectory, stress_trajectory, loss_trajectory

  _, _, _, _, output, stress_output, loss_output = tf.while_loop(
      cond=lambda step, cur, cur_gp, inputs, traj, stress_traj, loss_traj: tf.less(step, num_steps),
      body=step_fn,
      loop_vars=(0, initial_state['world_pos'], initial_state['stress'], inputs,
                 tf.TensorArray(tf.float32, num_steps), tf.TensorArray(tf.float32, num_steps), tf.TensorArray(tf.float32, num_steps)),
      parallel_iterations=1)
  return output.stack(), stress_output.stack(), loss_output.stack()


def get_real_median(v):
    v = tf.reshape(v, [-1])
    l = v.get_shape()[0]
    mid = l//2 + 1
    val = tf.nn.top_k(v, mid).values
    if l % 2 == 1:
        return val[-1]
    else:
        return 0.5 * (val[-1] + val[-2])


def evaluate(model, inputs, FLAGS, num_steps=None, normalize=True, accumulate=False):
  """Performs model rollouts and create stats."""
  # return model.loss(inputs), tf.constant(5, dtype=tf.float32)
  initial_state = {k: v[0] for k, v in inputs.items()}


  if not num_steps:
    num_steps = inputs['cells'].shape[0] # Length of trajectory

  pos_prediction, stress_prediction, rollout_losses = _rollout(model, initial_state, inputs, num_steps, FLAGS, normalize, accumulate)

  # stress_prediction is in log
  log_gt_stress = tf.math.log(inputs['stress'] + 1)



  pos_error = tf.reduce_mean(tf.reduce_sum(((pos_prediction - inputs['world_pos'][:num_steps]) ** 2), axis=1), axis=-1)

  stress_error = tf.reduce_mean((tf.squeeze(stress_prediction - log_gt_stress[:num_steps]))**2, axis=-1)


  # scalars = {'mse_%d_steps' % horizon: tf.reduce_mean(error[1:horizon+1])
  #            for horizon in [1, 10, 20, 50, 100, 200]}

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
  scalars['all_losses'] = rollout_losses



  traj_ops = {
      'faces': inputs['cells'],
      'mesh_pos': inputs['mesh_pos'],
      'gt_pos': inputs['world_pos'],
      'gt_pd_stress': inputs['pd_stress'],
      'gt_stress': log_gt_stress,
      'gt_gripper_pos': inputs['gripper_pos'],
      'pred_pos': pos_prediction,
      'pred_stress': stress_prediction,
      # 'avg_gt_stress': tf.shape(log_gt_stress[:num_steps]), #(n_horizon, n_nodes, 1)
      # 'avg_pred_stress': tf.shape(stress_prediction) #(n_horizon, n_nodes, 1)
      'avg_gt_stress': tf.reduce_mean(tf.squeeze(log_gt_stress[:num_steps], axis=-1), axis=-1), #(n_horizon, n_nodes, 1)
      'avg_pred_stress': tf.reduce_mean(tf.squeeze(stress_prediction, axis=-1), axis=-1) #(n_horizon, n_nodes, 1)
      # 'pred_gripper_pos': gripper_pos_prediction,
  }

  # return tf.reduce_mean(error), tf.reduce_mean(rollout_losses), scalars
  return tf.reduce_mean(rollout_losses), traj_ops, scalars
