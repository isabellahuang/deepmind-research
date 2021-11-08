# python -m meshgraphnets.polar_decomposition --rollout_path=meshgraphnets/data/chk/pos_change_only_10traj/prism_rollout.pkl

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
  num_traj = len(rollout_data)

  # compute bounds
  bounds = []
  for trajectory in rollout_data:
    bb_min = trajectory['gt_pos'].min(axis=(0, 1))
    bb_max = trajectory['gt_pos'].max(axis=(0, 1))
    bounds.append((bb_min, bb_max))

  E, v = 5e4, 0.3


  sum_angles = []
  for traj in range(num_traj):
    if traj % 10 == 0:
      print("Traj", traj)
    tets = np.copy(rollout_data[traj]['faces'][0])
    mesh_pos = np.copy(rollout_data[traj]['gt_pos'][0])
    last_gt_pos = np.copy(rollout_data[traj]['gt_pos'][-1])

    sum_angle = 0
    for ti, tet_idxs in enumerate(tets):
      if len(np.unique(tet_idxs)) < 4:
        continue

      R, U = dcu.polar_decomposition(mesh_pos, last_gt_pos, tet_idxs)
      sum_angle += dcu.R_to_angle(R)
    sum_angles.append(sum_angle)


  print("Sorted by total R angles:")
  print(list(np.argsort(sum_angles)))
  print(len(sum_angles))

if __name__ == '__main__':
  app.run(main)
