
import pickle

from absl import app
from absl import flags

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from meshgraphnets import utils

FLAGS = flags.FLAGS
flags.DEFINE_string('rollout_path', None, 'Path to rollout pickle file')

import sys
sys.path.append("..")
import data_creator_utils as dcu



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


def construct_world_edges(world_pos, node_type):

  deformable_idx = tf.where(tf.equal(node_type[:, 0], 1))
  actuator_idx = tf.where(tf.equal(node_type[:, 0], 0))
  B = tf.squeeze(tf.gather(world_pos, deformable_idx))
  A = tf.squeeze(tf.gather(world_pos, actuator_idx))

  thresh = 0.005


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

def main(unused_argv):
  with open(FLAGS.rollout_path, 'rb') as fp:
    rollout_data = pickle.load(fp)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111, projection='3d')
  skip = 5
  num_steps = rollout_data[0]['gt_pos'].shape[0]
  num_frames = len(rollout_data) * num_steps // skip


  # compute bounds
  bounds = []
  for trajectory in rollout_data:
    bb_min = trajectory['gt_pos'].min(axis=(0, 1))
    bb_max = trajectory['gt_pos'].max(axis=(0, 1))
    bounds.append((bb_min, bb_max))

  E, v = 5e4, 0.3

  # Prism
  sorted_R_list = [29, 78, 75, 80, 98, 37, 35, 51, 20, 65, 24, 47, 8, 60, 19, 59, 14, 9, 46, 6, 96, 44, 64, 38, 91, 85, 87, 36, 12, 88, 49, 86, 10, 79, 45, 27, \
  94, 55, 62, 90, 58, 61, 7, 89, 18, 13, 5, 28, 83, 22, 11, 50, 93, 25, 48, 54, 53, 41, 39, 42, 23, 97, 70, 34, 33, 1, 26, 31, 3, 77, 73, 52, 2, 56, 17, \
  66, 72, 4, 30, 81, 15, 84, 76, 69, 74, 32, 68, 95, 63, 67, 82, 21, 57, 43, 0, 40, 16, 92, 71]
  sorted_R_list = sorted_R_list[:int(len(sorted_R_list)/2)]

  def animate(num):


    step = (num*skip) % num_steps
    traj = (num*skip) // num_steps

    # If plotting sorted total rotation
    # traj = sorted_R_list[traj]


    ax.cla()
    bound = bounds[traj]
    x_min, x_max = [bound[0][0], bound[1][0]]
    y_min, y_max = [bound[0][1], bound[1][1]]
    z_min, z_max = [bound[0][2], bound[1][2]]


    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    dm_dataset = 'gt_force' not in rollout_data[traj].keys()


    pos = np.copy(rollout_data[traj]['pred_pos'][step])
    stress = np.copy(rollout_data[traj]['pred_stress'][step])

    gt_pos = np.copy(rollout_data[traj]['gt_pos'][step])
    gt_stress = np.copy(rollout_data[traj]['gt_stress'][step])

    if not dm_dataset:
      gt_force = np.copy(rollout_data[traj]['gt_force'][step])
      gt_pd_stress = np.copy(rollout_data[traj]['gt_pd_stress'][step])
      world_edges = np.copy(rollout_data[traj]['world_edges'][step])



    # print(step, np.min(stress), np.max(stress), np.max(gt_stress))


    # Convert back to real stress
    # '''
    positive_idx = np.argwhere(stress > 0)
    stress[positive_idx] = np.exp(stress[positive_idx]) - 1
    gt_positive_idx = np.argwhere(gt_stress > 0)
    gt_stress[gt_positive_idx] = np.exp(gt_stress[gt_positive_idx]) - 1
    # '''


    faces = rollout_data[traj]['faces'][step]

    vmax = np.max(stress)
    vmin = np.min(stress)

    gt_vmax = np.exp(np.max(rollout_data[traj]['gt_stress'])) - 1
    gt_vmin = np.min(rollout_data[traj]['gt_stress'])

    # '''
    #######################
    # Calculate stress from positions
    # '''

    gt_final_pos = np.copy(rollout_data[traj]['gt_pos'][-1])
    pred_final_pos = np.copy(rollout_data[traj]['pred_pos'][-1])
    pos_error = np.copy(rollout_data[traj]['pos_error'])
    baseline_pos_error = np.copy(rollout_data[traj]['baseline_pos_error'])
    print(gt_final_pos.shape, pred_final_pos.shape)
    mse = np.square(gt_final_pos - pred_final_pos).mean()
    print("MSE", mse)
    print("Sum  gt final pos", np.sum(gt_final_pos))
    print("Sum  pred final pos", np.sum(pred_final_pos))
    # print(pos_error)

    all_gt_pos = np.copy(rollout_data[traj]['gt_pos'])
    all_pred_pos = np.copy(rollout_data[traj]['pred_pos'])
    our_pos_error = np.mean(np.sum(np.square(all_gt_pos - all_pred_pos), axis=-1), axis=-1)

    print(np.mean(np.sqrt(our_pos_error)))



    faces = np.copy(rollout_data[traj]['faces'][0])
    faces_tf = tf.convert_to_tensor(faces)

    node_type = np.copy(rollout_data[traj]['node_type'][0])

    # mesh_senders, mesh_receivers = construct_world_edges(tf.convert_to_tensor(gt_pos), tf.convert_to_tensor(node_type))



    tets = np.copy(rollout_data[traj]['faces'][step])
    mesh_pos = np.copy(rollout_data[traj]['gt_pos'][0])
    curr_pos = np.copy(rollout_data[traj]['pred_pos'][step])


    tet_object = dcu.TetObject(E, v, tets, mesh_pos)
    pd_stress = tet_object.get_pd_vertex_stresses(curr_pos)
    # gt_pd_stress = tet_object.get_pd_vertex_stresses(gt_pos)



    if not dm_dataset:
      # print("*******", np.mean(gt_pd_stress), np.mean(pd_stress), np.mean(gt_stress))      
      gripper_normal = dcu.get_gripper_normal(mesh_pos, np.copy(rollout_data[traj]['gt_pos'][-1]))

      # pd_force = tet_object.get_total_surface_force(curr_pos, world_edges, gripper_normal)
      gt_pd_force = tet_object.get_total_surface_force(gt_pos, world_edges, gripper_normal)

      # vertex_forces = tet_object.vertex_forces_from_tris(curr_pos, world_edges, gripper_normal)
      gt_vertex_forces = tet_object.vertex_forces_from_tris(gt_pos, world_edges, gripper_normal)

      print(gt_pd_force, gt_force)
      # '''

    ######################



    # ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=stress[:], vmin=vmin, vmax=vmax)
    # ax.scatter(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], c=gt_stress[:], vmin=gt_vmin, vmax=gt_vmax)
    # ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=pd_stress[:])
    # ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=gt_pd_stress[:])

    ##################
    # ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=gt_vertex_forces[:])

    # ax.scatter(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], c=pd_stress[:])
    X, Y, Z  = pos[:, 0], pos[:, 1], pos[:, 2]
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=pd_stress[:])

    print(node_type.shape)
    # Plot mesh edges
    # for p, q in zip(mesh_senders, mesh_receivers):
      # ax.plot([X[p], X[q]], [Y[p], Y[q]], [Z[p], Z[q]], 'ro-')

    ax.set_title('Trajectory %d Step %d' % (traj, step))
    return fig,

  _ = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
  plt.show(block=True)

  dm_dataset = 'gt_force' not in trajectory.keys()

  if dm_dataset:
    quit()

  # Print out final deformation and final stress predictions
  # plt.figure()
  gt_final_pd_stresses, final_pd_stresses = [], []
  final_stresses = []
  gt_final_mean_defs, final_mean_defs = [], []
  gt_final_max_defs, final_max_defs = [], []
  gt_final_pd_forces, final_pd_forces = [], []
  gripper_displacements = []


 
  for t_idx, trajectory in enumerate(rollout_data[:]):
    print(t_idx)
    tets = np.copy(trajectory['faces'][0])
    mesh_pos = np.copy(trajectory['gt_pos'][0])
    gt_final_pos = np.copy(trajectory['gt_pos'][-1])
    first_pos = np.copy(trajectory['pred_pos'][0])
    final_pos = np.copy(trajectory['pred_pos'][-1])
    gt_final_force = np.copy(trajectory['gt_force'][-1][0][0])
    final_world_edges = np.copy(trajectory['world_edges'][-1])

    first_gripper_pos = np.copy(trajectory['gt_gripper_pos'][0])
    final_gripper_pos = np.copy(trajectory['gt_gripper_pos'][-1])
    gripper_displacements.append(first_gripper_pos[0] - final_gripper_pos[0])

    tet_object = dcu.TetObject(E, v, tets, mesh_pos)

    ####### Final stress comparison
    gt_final_pd_stress = np.copy(trajectory['gt_pd_stress'][-1])
    ### Maybe final stress is different when calculated
    gt_final_pd_stress = tet_object.get_pd_vertex_stresses(gt_final_pos)

    gt_final_pd_stresses.append(gt_final_pd_stress)
    final_pd_stress = tet_object.get_pd_vertex_stresses(final_pos)
    final_pd_stresses.append(final_pd_stress)
    final_stress = np.copy(trajectory['pred_stress'][-1])
    # Convert back to real stress
    # '''
    positive_idx = np.argwhere(final_stress > 0)
    final_stress[positive_idx] = np.exp(final_stress[positive_idx]) - 1
    # '''
    final_stresses.append(final_stress)

    ###### Final deformation comparison
    gt_final_mean_def, gt_final_max_def, _ = utils.get_global_deformation_metrics(first_pos, gt_final_pos)
    final_mean_def, final_max_def, _ = utils.get_global_deformation_metrics(first_pos, final_pos)
    gt_final_mean_defs.append(gt_final_mean_def)
    final_mean_defs.append(final_mean_def)
    gt_final_max_defs.append(gt_final_max_def)
    final_max_defs.append(final_max_def)


    ###### Final pd force comparison
    

    gripper_normal = dcu.get_gripper_normal(mesh_pos, gt_final_pos)
    # gt_final_pd_force = dcu.get_total_surface_force(E, v, mesh_pos, gt_final_pos, tets, final_world_edges, gripper_normal)
    gt_final_pd_force = tet_object.get_total_surface_force(gt_final_pos, final_world_edges, gripper_normal)
    gt_final_pd_forces.append(gt_final_force)

    
    # final_pd_force = dcu.get_total_surface_force(E, v, mesh_pos, final_pos, tets, final_world_edges, gripper_normal)
    final_pd_force = tet_object.get_total_surface_force(final_pos, final_world_edges, gripper_normal)
    final_pd_forces.append(gt_final_pd_force)

    ### Histograms of stress
    # plt.hist(final_pd_stress, alpha=0.5)
    # plt.hist(gt_final_pd_stress, label="gt", alpha=0.5)
    # plt.legend()
    # plt.show()

    ### Histograms of def



  gt_final_pd_stresses_means = [np.mean(k) for k in gt_final_pd_stresses]
  final_pd_stresses_means = [np.mean(k) for k in final_pd_stresses]
  gt_final_pd_stresses_maxes = [np.max(k) for k in gt_final_pd_stresses]
  final_pd_stresses_maxes = [np.max(k) for k in final_pd_stresses]
  final_stresses_means = [np.mean(k) for k in final_stresses]
  final_stresses_maxes = [np.max(k) for k in final_stresses]
  n_train = 40


  fig, axs = plt.subplots(2, 4)

  axs[0,1].scatter(gt_final_pd_stresses_means[:n_train], final_pd_stresses_means[:n_train], label="Training set")
  axs[0,1].scatter(gt_final_pd_stresses_means[n_train:], final_pd_stresses_means[n_train:], label="Validation set")
  axs[0,1].legend()
  axs[0,1].plot(gt_final_pd_stresses_means, gt_final_pd_stresses_means, '-')
  axs[0,1].set_xlabel("Ground truth")
  axs[0,1].set_ylabel("Predicted")
  axs[0,1].set_title("Gt vs. predicted final mean stress (analytical)")


  axs[0,2].scatter(gt_final_pd_stresses_means[:n_train], final_stresses_means[:n_train], label="Training set")
  axs[0,2].scatter(gt_final_pd_stresses_means[n_train:], final_stresses_means[n_train:], label="Validation set")
  axs[0,2].legend()
  axs[0,2].plot(gt_final_pd_stresses_means, gt_final_pd_stresses_means, '-')
  axs[0,2].set_xlabel("Ground truth")
  axs[0,2].set_ylabel("Predicted")
  axs[0,2].set_title("Gt vs. predicted final mean stress (direct prediction)")


  axs[1,1].scatter(gt_final_pd_stresses_maxes[:n_train], final_pd_stresses_maxes[:n_train], label="Training set")
  axs[1,1].scatter(gt_final_pd_stresses_maxes[n_train:], final_pd_stresses_maxes[n_train:], label="Validation set")
  axs[1,1].legend()
  axs[1,1].plot(gt_final_pd_stresses_maxes, gt_final_pd_stresses_maxes, '-')
  axs[1,1].set_xlabel("Ground truth")
  axs[1,1].set_ylabel("Predicted")
  axs[1,1].set_title("Gt vs. predicted final max stress (analytical)")


  axs[1,2].scatter(gt_final_pd_stresses_maxes[:n_train], final_stresses_maxes[:n_train], label="Training set")
  axs[1,2].scatter(gt_final_pd_stresses_maxes[n_train:], final_stresses_maxes[n_train:], label="Validation set")
  axs[1,2].legend()
  axs[1,2].plot(gt_final_pd_stresses_maxes, gt_final_pd_stresses_maxes, '-')
  axs[1,2].set_xlabel("Ground truth")
  axs[1,2].set_ylabel("Predicted")
  axs[1,2].set_title("Gt vs. predicted final max stress (direct prediction)")


  # plt.figure()
  axs[0,0].scatter(gt_final_mean_defs[:n_train], final_mean_defs[:n_train], label="Training set")
  axs[0,0].scatter(gt_final_mean_defs[n_train:], final_mean_defs[n_train:], label="Validation set")
  axs[0,0].legend()
  axs[0,0].plot(gt_final_mean_defs, gt_final_mean_defs, '-')
  axs[0,0].set_xlabel("Ground truth")
  axs[0,0].set_ylabel("Predicted")
  axs[0,0].set_title("Gt vs. predicted final mean def")

  # axs[1,0].figure()
  axs[1,0].scatter(gt_final_max_defs[:n_train], final_max_defs[:n_train], label="Training set")
  axs[1,0].scatter(gt_final_max_defs[n_train:], final_max_defs[n_train:], label="Validation set")
  axs[1,0].legend()
  axs[1,0].plot(gt_final_max_defs, gt_final_max_defs, '-')
  axs[1,0].set_xlabel("Ground truth")
  axs[1,0].set_ylabel("Predicted")
  axs[1,0].set_title("Gt vs. predicted final max def")


  ###### Gt final pd force ve pred final pd force
  axs[0,3].scatter(gt_final_pd_forces[:n_train], final_pd_forces[:n_train], label="Training set")
  axs[0,3].scatter(gt_final_pd_forces[n_train:], final_pd_forces[n_train:], label="Validation set")
  axs[0,3].legend()
  axs[0,3].plot(gt_final_pd_forces, gt_final_pd_forces, '-')
  axs[0,3].set_xlabel("Ground truth")
  axs[0,3].set_ylabel("Predicted")
  axs[0,3].set_title("Gt vs. predicted final pd force")


  # plt.figure()
  # plt.title("Gripper displacement vs gt final and mean def")
  # plt.scatter(gripper_displacements, gt_final_max_defs, label="vs. final def")
  # plt.scatter(gripper_displacements, gt_final_mean_defs, label="vs. mean def")
  # plt.legend()

  plt.show()


if __name__ == '__main__':
  app.run(main)
