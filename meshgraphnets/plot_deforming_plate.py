
import pickle

from absl import app
from absl import flags

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from meshgraphnets import utils

FLAGS = flags.FLAGS
flags.DEFINE_string('rollout_path', None, 'Path to rollout pickle file')

import sys
sys.path.append("..")
import data_creator_utils as dcu


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



  def animate(num):
    step = (num*skip) % num_steps
    traj = (num*skip) // num_steps
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


    pos = np.copy(rollout_data[traj]['pred_pos'][step])
    stress = np.copy(rollout_data[traj]['pred_stress'][step])

    gt_pos = np.copy(rollout_data[traj]['gt_pos'][step])
    gt_stress = np.copy(rollout_data[traj]['gt_stress'][step])
    gt_pd_stress = np.copy(rollout_data[traj]['gt_pd_stress'][step])

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
    tets = np.copy(rollout_data[traj]['faces'][step])
    mesh_pos = np.copy(rollout_data[traj]['gt_pos'][0])
    curr_pos = np.copy(rollout_data[traj]['pred_pos'][step])

    pd_stress = dcu.get_pd_vertex_stresses(E, v, mesh_pos, curr_pos, tets)

    print("*******", np.mean(gt_pd_stress), np.mean(pd_stress), np.mean(gt_stress))      

    ######################



    # ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=stress[:], vmin=vmin, vmax=vmax)
    # ax.scatter(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], c=gt_stress[:], vmin=gt_vmin, vmax=gt_vmax)
    # ax.scatter(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], c=vertex_stresses[:])
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=pd_stress[:])


    ax.set_title('Trajectory %d Step %d' % (traj, step))
    return fig,

  _ = animation.FuncAnimation(fig, animate, frames=num_frames, interval=1)
  plt.show(block=True)



  # Print out final deformation and final stress predictions
  plt.figure()
  gt_final_pd_stresses, final_pd_stresses = [], []
  gt_final_mean_defs, final_mean_defs = [], []
  gt_final_max_defs, final_max_defs = [], []
 
  for t_idx, trajectory in enumerate(rollout_data):
    print(t_idx)
    tets = np.copy(trajectory['faces'][0])
    mesh_pos = np.copy(trajectory['gt_pos'][0])
    gt_final_pos = np.copy(trajectory['gt_pos'][-1])
    first_pos = np.copy(trajectory['pred_pos'][0])
    final_pos = np.copy(trajectory['pred_pos'][-1])

    ####### Final stress comparison
    gt_final_pd_stress = np.copy(trajectory['gt_pd_stress'][-1])
    gt_final_pd_stresses.append(gt_final_pd_stress)
    final_pd_stress = dcu.get_pd_vertex_stresses(E, v, mesh_pos, final_pos, tets)
    final_pd_stresses.append(final_pd_stress)

    ###### Final deformation comparison
    gt_final_mean_def, gt_final_max_def, _ = utils.get_global_deformation_metrics(first_pos, gt_final_pos)
    final_mean_def, final_max_def, _ = utils.get_global_deformation_metrics(first_pos, final_pos)
    gt_final_mean_defs.append(gt_final_mean_def)
    final_mean_defs.append(final_mean_def)
    gt_final_max_defs.append(gt_final_max_def)
    final_max_defs.append(final_max_def)


  gt_final_pd_stresses_means = [np.mean(k) for k in gt_final_pd_stresses]
  final_pd_stresses_means = [np.mean(k) for k in final_pd_stresses]

  plt.scatter(gt_final_pd_stresses_means, final_pd_stresses_means)
  plt.plot(gt_final_pd_stresses_means, gt_final_pd_stresses_means, '-')
  plt.xlabel("Ground truth")
  plt.ylabel("Predicted")
  plt.title("Gt vs. predicted final mean stress")


  plt.figure()
  plt.scatter(gt_final_mean_defs, final_mean_defs)
  plt.plot(gt_final_mean_defs, gt_final_mean_defs, '-')
  plt.xlabel("Ground truth")
  plt.ylabel("Predicted")
  plt.title("Gt vs. predicted final mean def")

  plt.figure()
  plt.scatter(gt_final_max_defs, final_max_defs)
  plt.plot(gt_final_max_defs, gt_final_max_defs, '-')
  plt.xlabel("Ground truth")
  plt.ylabel("Predicted")
  plt.title("Gt vs. predicted final max def")

  plt.show()


if __name__ == '__main__':
  app.run(main)
