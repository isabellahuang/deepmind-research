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


def main(unused_argv):
    with open(FLAGS.rollout_path, 'rb') as fp:
        rollout_data = pickle.load(fp)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    skip = 5
    num_steps = rollout_data[0]['gt_pos'].shape[0]
    num_frames = len(rollout_data) * num_steps // skip

    # Convert everything to numpy
    for ri in range(len(rollout_data)):
        for key in rollout_data[ri]:
            try:
                rollout_data[ri][key] = rollout_data[ri][key].numpy()
            except:
                rollout_data[ri][key] = rollout_data[ri][key]

    # compute bounds
    bounds = []
    for trajectory in rollout_data:
        bb_min = trajectory['gt_pos'].min(axis=(0, 1))
        bb_max = trajectory['gt_pos'].max(axis=(0, 1))
        bounds.append((bb_min, bb_max))

    def animate(num):

        step = (num * skip) % num_steps
        traj = (num * skip) // num_steps

        ax.cla()
        bound = bounds[traj]
        x_min, x_max = [bound[0][0], bound[1][0]]
        y_min, y_max = [bound[0][1], bound[1][1]]
        z_min, z_max = [bound[0][2], bound[1][2]]

        max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min
                              ]).max() / 2.0
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
        gt_sim_pos = np.copy(rollout_data[traj]['sim_world_pos'][step])

        gt_stress = np.copy(rollout_data[traj]['gt_stress'][step])

        faces = rollout_data[traj]['faces'][step]

        vmax = np.max(rollout_data[traj]['pred_stress'][:, 1180:])
        vmin = np.min(rollout_data[traj]['pred_stress'][:, 1180:])

        gt_vmax = np.max(rollout_data[traj]['gt_stress'])
        gt_vmin = np.min(rollout_data[traj]['gt_stress'])

        ax.scatter(pos[:, 0],
                   pos[:, 1],
                   pos[:, 2],
                   c=stress[:],
                   vmin=vmin,
                   vmax=vmax)

        # Uncomment the following line to visualize the ground truth
        # ax.scatter(gt_sim_pos[:, 0], gt_sim_pos[:, 1], gt_sim_pos[:, 2], c=gt_stress[:], vmin=vmin, vmax=gt_vmax)

        ax.set_title('Trajectory %d Step %d' % (traj, step))
        return fig,

    _ = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
    plt.show(block=True)

    dm_dataset = 'gt_force' not in trajectory.keys()

    if dm_dataset:
        quit()


if __name__ == '__main__':
    app.run(main)
