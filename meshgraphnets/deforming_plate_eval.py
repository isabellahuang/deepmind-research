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
"""Functions to build evaluation metrics for deforming plate data."""
import os

import tensorflow as tf
import tensorflow_probability as tfp
from meshgraphnets.common import NodeType
from meshgraphnets import utils
from meshgraphnets import deforming_plate_model
from meshgraphnets import common


def _rollout(model, initial_state, inputs, num_steps, FLAGS, normalize,
             accumulate):
    """Rolls out a model trajectory."""
    mask = tf.equal(initial_state['node_type'][:, 0], NodeType.NORMAL)

    def step_fn(step, cur_pos, curr_stress, inputs, trajectory,
                stress_trajectory, loss_trajectory):

        # Prediction shouldnt have access to inputs['stress'][step] though, but save it to rollout as needed needed to calculate loss
        curr_stress_feed = inputs['stress'][step]

        cur_pos = inputs['world_pos'][step]

        model_input_dict = {
            **initial_state,
            'world_pos': cur_pos,
            'target|world_pos': inputs['target|world_pos'][step],
            'stress': curr_stress_feed,  #in raw units
            'target|stress': inputs['target|stress'][step],
            'sim_world_pos': inputs['sim_world_pos'][step]
        }

        if not FLAGS.compute_world_edges:
            model_input_dict['world_edges'] = inputs['world_edges'][step]

        model_input_dict['force'] = inputs['force'][step]
        model_input_dict['target|force'] = inputs['target|force'][step]

        use_precomputed = False
        next_pos_pred, next_stress_pred, loss_val = model(
            model_input_dict, use_precomputed, normalize, accumulate)

        log_stress = tf.math.log(
            next_stress_pred +
            1)  # Next stress pred is actually stress now, at time t
        raw_stress = next_stress_pred

        stress_trajectory = stress_trajectory.write(step, raw_stress)

        trajectory = trajectory.write(step, next_pos_pred)

        # Save to loss trajectory
        loss_trajectory = loss_trajectory.write(step, loss_val)

        return step + 1, next_pos_pred, next_stress_pred, inputs, trajectory, stress_trajectory, loss_trajectory

    _, _, _, _, output, stress_output, loss_output = tf.while_loop(
        cond=lambda step, cur, cur_gp, inputs, traj, stress_traj, loss_traj: tf
        .less(step, num_steps),
        body=step_fn,
        loop_vars=(0, initial_state['world_pos'], initial_state['stress'],
                   inputs, tf.TensorArray(tf.float32, num_steps),
                   tf.TensorArray(tf.float32, num_steps),
                   tf.TensorArray(tf.float32, num_steps)),
        parallel_iterations=1)

    return output.stack(), stress_output.stack(), loss_output.stack(
    )  #, debug_output.stack()


def gripper_pos_at_first_contact(inputs, f_pc_original):
    # import numpy as np
    ''' Given initial gripper orientation, infer the initial gripper pos at contact '''

    # Apply gripper closings to the transformed fingers

    f_verts_open, gripper_normal = deforming_plate_model.f_verts_at_pos(
        inputs, [0.04, 0.04], f_pc_original)

    def project_to_normal(normal, pos):
        perp_dists_tf = tf.tensordot(
            pos, tf.transpose(normal),
            1)  #tensordot with transpose of normal instead of normal

        return pos - perp_dists_tf * normal
        # return pos - normal * perp_dists_tf[:, None]

    def idx_in_rect_tf(pos, f_verts_open_proj):
        A, B, C, D = f_verts_open_proj[43], f_verts_open_proj[
            23], f_verts_open_proj[8], f_verts_open_proj[28]
        AB = B - A
        AD = D - A
        AM = pos - A

        inside_idx = tf.squeeze(
            tf.where((tf.tensordot(AM, AB, 1) < tf.tensordot(AB, AB, 1))
                     & (0 < tf.tensordot(AM, AB, 1))
                     & (tf.tensordot(AM, AD, 1) < tf.tensordot(AD, AD, 1))
                     & (0 < tf.tensordot(AM, AD, 1))))

        return inside_idx

    def gripper_pos_at_contact_tf(points_in_rect, normal, p1, p2):

        f1_closest_dist = tf.reduce_min(
            tf.tensordot(points_in_rect - p1, -1. * tf.transpose(normal), 1))

        f2_closest_dist = tf.reduce_min(
            tf.tensordot(points_in_rect - p2, tf.transpose(normal), 1))
        return tf.stack(
            [0.04 - f1_closest_dist + 0.0005, 0.04 - f2_closest_dist + 0.0005])

    corner_idx1 = [43, 23, 8, 28]  # Corners
    p1_idx, p2_idx = 143, 1036
    corner_idx2 = [1167, 1147, 1132, 1152]  # For face 2

    actuator_mask = tf.equal(inputs['node_type'][:, 0],
                             common.NodeType.OBSTACLE)
    object_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.AIRFOIL)
    gt_gripper = tf.gather(inputs['world_pos'],
                           tf.squeeze(tf.where(actuator_mask)))
    gt_mesh = tf.gather(inputs['world_pos'], tf.squeeze(tf.where(object_mask)))

    # Project points onto the gripper normal
    idx_in_rect_tf = idx_in_rect_tf(
        project_to_normal(gripper_normal, gt_mesh),
        project_to_normal(gripper_normal, f_verts_open))
    points_in_rect_tf = tf.gather(gt_mesh, idx_in_rect_tf)
    contact_gripper_pos_tf = gripper_pos_at_contact_tf(points_in_rect_tf,
                                                       gripper_normal,
                                                       f_verts_open[p1_idx],
                                                       f_verts_open[p2_idx])

    contact_gripper_pos_tf = tf.expand_dims(contact_gripper_pos_tf, axis=0)

    return contact_gripper_pos_tf


def evaluate_one_step(model,
                      inputs,
                      FLAGS,
                      num_steps=None,
                      normalize=True,
                      accumulate=False,
                      tfn=0,
                      eval_step=0,
                      force=0):

    initial_state = {k: v[0] for k, v in inputs.items()}

    # If initial gripper pos is unknown, can also calculate geometrically
    # initial_state['gripper_pos'] = gripper_pos_at_first_contact(initial_state, model.f_pc_original)

    use_precomputed = True
    next_pos_pred, next_stress_pred, loss_val = model(initial_state,
                                                      use_precomputed,
                                                      normalize, accumulate)
    return loss_val


def evaluate(model,
             inputs,
             FLAGS,
             num_steps=None,
             normalize=True,
             accumulate=False,
             tfn=0,
             eval_step=0,
             force=0):
    """Performs model rollouts and create stats."""

    # Use partial trajectory
    inputs = {k: v for k, v in inputs.items()}

    initial_state = {k: v[0] for k, v in inputs.items()}

    # If initial gripper pos is unknown, can also calculate geometrically
    # initial_state['gripper_pos'] = gripper_pos_at_first_contact(initial_state, model.f_pc_original)

    if not num_steps:
        num_steps = inputs['node_type'].shape[0]  # Length of trajectory

    pos_prediction, stress_prediction, rollout_losses = _rollout(
        model, initial_state, inputs, num_steps, FLAGS, normalize, accumulate)
    stress_cutoff = 500.0

    # if stress_prediction is in log
    log_gt_stress = tf.math.log(inputs['stress'] + 1)

    # MAE instead
    pos_difference_norm = tf.norm(pos_prediction -
                                  inputs['sim_world_pos'][:num_steps],
                                  axis=-1)  #(48, num_nodes)
    baseline_difference_norm = tf.norm(inputs['sim_world_pos'][:num_steps] -
                                       inputs['sim_world_pos'][0],
                                       axis=-1)  #(48, num_nodes)

    # Set difference to zero when stress is under threshold
    pos_difference_norm = tf.where(
        tf.squeeze(inputs['stress'], -1) > stress_cutoff, pos_difference_norm,
        tf.zeros_like(pos_difference_norm))[:, 1180:]  # (48, num_nodes - 1180)
    baseline_difference_norm = tf.where(
        tf.squeeze(inputs['stress'], -1) > stress_cutoff,
        baseline_difference_norm,
        tf.zeros_like(baseline_difference_norm))[:, 1180:]

    pos_error = tf.reduce_mean(pos_difference_norm, axis=-1)  # (48,)
    baseline_pos_error = tf.reduce_mean(baseline_difference_norm,
                                        axis=-1)  # (48,)

    # MAE instead
    stress_error = tf.reduce_mean(tf.reduce_sum(
        (tf.math.abs(stress_prediction - inputs['stress'][:num_steps])),
        axis=-1)[:, 1180:],
                                  axis=-1)  # don't include the grippers
    baseline_stress_error = tf.reduce_mean(
        tf.reduce_sum(
            (tf.math.abs(inputs['stress'][:num_steps] - inputs['stress'][0])),
            axis=-1)[:, 1180:],
        axis=-1)  # don't inculde the grippers

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
    scalars['baseline_stress_mean_error'] = tf.reduce_mean(
        baseline_stress_error)
    scalars['baseline_stress_final_error'] = baseline_stress_error[-1]


    traj_ops = {
        'faces': inputs['cells'],
        'name': inputs['name'],
        'node_type': inputs['node_type'],
        'mesh_pos': inputs['mesh_pos'],
        'gt_pos': inputs['world_pos'],
        'gt_stress': inputs['stress'],
        'stress_error': stress_error,
        'pred_pos': pos_prediction,
        'pred_stress': stress_prediction,
        'sim_world_pos': inputs['sim_world_pos'] if 'sim_world_pos' in inputs.keys() else inputs['world_pos']
    }

    if not utils.using_dm_dataset(FLAGS):
        traj_ops['gt_pd_stress'] = inputs['pd_stress']
        traj_ops['gt_gripper_pos'] = inputs['gripper_pos']
        traj_ops['gt_force'] = inputs['force']
        # traj_ops['world_edges'] = inputs['world_edges'] # got rid of world_edges from inputs to reduce function retracing

    return tf.reduce_mean(rollout_losses), traj_ops, scalars  # The correct one

