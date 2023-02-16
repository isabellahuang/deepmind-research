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
"""Runs the learner/evaluator."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_CPP_VMODULE'] = 'asm_compiler=2'


import pickle
import sys
import re
import h5py

import sonnet as snt
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import timeit
from scipy import stats


# Try to run with more gpu usage in TF 2
devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)


# from tfdeterminism import patch
SEED = 55


# Hardcoded paths to different datasets
ABC_DATASET_FOLDER = os.path.join("..", "..", "abc_dataset", "abc_dataset", "0-9999")
SIMPLE_GEOMETRIC_DATASET_FOLDER = os.path.join("..", "..", "deformable_object_grasping", "simple_geometric_dataset")
CONTAINER_DATASET_FOLDER = os.path.join("..", "..", "deformable_object_grasping", "container_dataset")
REAL_FRUITS_DATASET_FOLDER = os.path.join("..", "..", "deformable_object_grasping", "real_fruits_dataset")

DATASET_FOLDER = REAL_FRUITS_DATASET_FOLDER

os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from meshgraphnets import common
from meshgraphnets import cfd_eval
from meshgraphnets import cfd_model
from meshgraphnets import cloth_eval
from meshgraphnets import cloth_model
from meshgraphnets import deforming_plate_model
from meshgraphnets import deforming_plate_eval
from meshgraphnets import core_model
from meshgraphnets import dataset
from meshgraphnets import utils



FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval', 'sample'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', None, ['cfd', 'cloth', 'deforming_plate'],
                  'Select model to run.')
flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoint')
flags.DEFINE_string('dataset_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('rollout_path', None,
                    'Pickle file to save eval trajectories')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('num_rollouts', 20, 'No. of rollout trajectories')
flags.DEFINE_integer('num_training_steps', int(10e6), 'No. of training steps')

flags.DEFINE_integer('batch_size', 5, 'Batch size')
flags.DEFINE_integer('num_epochs', 50, 'Num epochs')
flags.DEFINE_integer('num_epochs_per_validation', 5, 'Num epochs to train before validating')
flags.DEFINE_integer('num_training_trajectories', int(10e6), 'No. of training trajectories')
flags.DEFINE_integer('num_testing_trajectories', int(10e6), 'No. of trajectories to train on')


flags.DEFINE_integer('latent_size', 128, 'Latent size')
flags.DEFINE_integer('num_layers', 2, 'Num layers')
flags.DEFINE_integer('message_passing_steps', 15, 'Message passing steps')
flags.DEFINE_float('learning_rate', 1e-4, 'Message passing steps')
flags.DEFINE_float('noise_scale', 3e-3, 'Noise scale on world pos')

flags.DEFINE_bool('noise_on_adjacent_step', True, 'Add perturbation to both t and t + 1 step. If False, add only to t step.')
flags.DEFINE_bool('layernorm', False, 'Use layer norm')

flags.DEFINE_bool('gripper_force_action_input', True, 'Change in gripper force as action input')
flags.DEFINE_enum('mod', 'none', ['none', 'node', 'edge'],
                  'How to featurize node_mod.')

flags.DEFINE_bool('log_stress_t', False, 'Take the log of stress at time t as feature')

flags.DEFINE_bool('force_label_node', False, 'Whether each node should be labeled with a force-related signal. If False, mesh edges are labeled instead')
flags.DEFINE_bool('node_total_force_t', False, 'Whether each node should be labeled with the total gripper force')
# The default is currently labelling mesh edges with normalized force over nodes in contact
flags.DEFINE_bool('compute_world_edges', False, 'Whether to compute world edges')
flags.DEFINE_bool('use_cpu', False, 'Use CPU rather than GPU')
flags.DEFINE_bool('eager', False, 'Use eager execution')


# Exactly one of these must be set to True
flags.DEFINE_bool("incremental", False, "Predict step changes")
flags.DEFINE_bool('predict_log_stress_t1', False, 'Predict the log stress at t1')

flags.DEFINE_bool('predict_log_stress_change_t1', False, 'Predict the log of stress change from t to t1')
flags.DEFINE_bool('predict_stress_change_t1', False, 'Predict the stress change from t to t1')

flags.DEFINE_bool('predict_incremental', False,' Predict change in stress and in pos, with action as change in force')
flags.DEFINE_bool('predict_log_stress_t_only', False, 'Do not make deformation predictions. Only predict stress at time t')
flags.DEFINE_bool('predict_log_stress_t1_only', False, 'Do not make deformation predictions. Only predict stress at time t1')
flags.DEFINE_bool('predict_log_stress_change_only', False, 'Do not make deformation predictions. Only predict stress at time t1')
flags.DEFINE_bool('predict_pos_change_only', False, 'Do not make stress predictions. Only predict change in pos between time t and t1')
flags.DEFINE_bool('predict_pos_change_from_initial_only', False, 'Do not make stress predictions. Only predict change in pos between time 0 and t') # No rollout

flags.DEFINE_bool('predict_stress_change_only', False, 'Do not make pos predictions. Only predict change in raw stress between time t and t1')
flags.DEFINE_bool('predict_stress_t_only', False, 'Do not make pos predictions. Only predict raw stress at time t')

flags.DEFINE_bool('simplified_predict_stress_change_only', False, 'Simplified network for predicting stress only. No worries about position')


flags.DEFINE_bool('aux_mesh_edge_distance_change', False, 'Add auxiliary loss term: mesh edge distances')
flags.DEFINE_bool('use_pd_stress', True, 'Use pd_stress rather than stress for inputs["stress"]')


flags.DEFINE_integer('num_objects', 1, 'No. of objects to train on')
flags.DEFINE_integer('n_horizon', 1, 'No. of steps in training horizon')

flags.DEFINE_string('loss_function', 'MSE',
                    'which loss function to use')

PARAMETERS = {
    'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=cfd_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval),
    'deforming_plate': dict(noise=3e-3, gamma=0.1, field=['world_pos', 'gripper_pos', 'force', 'stress'], noise_field='world_pos', history=False,
                  size=4, batch=16, model=deforming_plate_model, evaluator=deforming_plate_eval),
}

LEN_TRAJ = 0 



def get_flattened_dataset(ds, params, n_horizon=None, do_cutoff=False, batch=False, takeall=False):
  ds = dataset.add_targets(ds, FLAGS, params['field'] if type(params['field']) is list else [params['field']], add_history=params['history'])

  test = not n_horizon
  if test:
    n_horizon = LEN_TRAJ

  noise_field = params['noise_field']
  noise_scale = params['noise']

  # ds = ds.take(10)
  '''
  if not test:
    ds = ds.skip(20)
    ds = ds.take(FLAGS.num_training_trajectories) # This is the number of trajectories seen during 
  else:
    # ds = ds.skip(FLAGS.num_training_trajectories)
    ds = ds.take(20) # ds.take(FLAGS.num_testing_trajectories)
  '''

  '''

  if not takeall:
    if not test:
      ds = ds.skip(10)
      ds = ds.take(40) #ds.take(FLAGS.num_training_trajectories) # This is the number of trajectories seen during 
    else:
      # ds = ds.skip(FLAGS.num_training_trajectories)
      ds = ds.take(10) # ds.take(FLAGS.num_testing_trajectories)
  '''

  # Filter trajectories for outliers
  def filter_func(elem):
      """ return True if  element is to be kept """
      cutoff = elem["cutoffs"][0][0][1]  # Cutoffs are percentiles [95, 96, 97, 98, 99]
      max_last_stress = tf.reduce_max(elem["target|stress"])
      max_force = tf.reduce_max(elem['target|force'])
      # return max_force <= 16 and max_force >= 10
      return max_last_stress <= cutoff and max_force <= 16 and max_force >= 10


  if do_cutoff:
    ds = ds.filter(lambda ff: filter_func(ff))

  # Shuffle trajectories selected for training
  ds = ds.shuffle(5000)


  def repeat_and_shift(trajectory):
    out = {}
    for key, val in trajectory.items():
      shifted_lists = []
      for i in range(LEN_TRAJ - n_horizon + 1):
        shifted_list = tf.roll(val, shift=-i, axis=0)[:n_horizon]
        shifted_lists.append(shifted_list)

      out[key] = shifted_lists
    return tf.data.Dataset.from_tensor_slices(out)


  def world_edges_processing_map(frame):

    # Remove non-unique edges
    val = frame['world_edges']
    sum_across_row = tf.reduce_sum(val, axis=-1)
    unique_edges = tf.where(tf.not_equal(sum_across_row, 0))
    close_pair_idx = tf.gather(val, unique_edges[:,1], axis=1)
    frame['world_edges'] = close_pair_idx  


    # Assign distributed force to each edge
    num_unique_world_edges = tf.shape(frame['world_edges'])[1]
    if FLAGS.node_total_force_t:
      force_features = tf.fill(tf.shape(frame['world_edges']), frame['force'][0][0][0])
    else:
      force_features = tf.fill(tf.shape(frame['world_edges']), frame['force'][0][0][0] / tf.cast(num_unique_world_edges, dtype=tf.float32))

    if FLAGS.incremental:
      force_features = tf.fill(tf.shape(frame['world_edges']), (frame['target|force'][0][0][0] - frame['force'][0][0][0]) / tf.cast(num_unique_world_edges, dtype=tf.float32))

    senders, receivers = tf.unstack(force_features, axis=-1)
    frame['world_edges_over_force'] = tf.expand_dims(tf.concat([senders, receivers], axis=-1), axis=-1)

    return frame


  def use_pd_stress(frame):
    frame['stress'] = frame['pd_stress']
    return frame

  def filter_stress_outliers(frame):
    ceilings = tfp.stats.percentile(frame['stress'][:, 1180:, :], 99.8, axis=1)
    ceilings = tf.expand_dims(ceilings, axis=-1)
    frame['stress'] = tf.clip_by_value(frame['stress'], 0, ceilings)
    return frame


  def add_noise(frame):
    noise = tf.random.normal(tf.shape(frame[noise_field]),
                             stddev=FLAGS.noise_scale, dtype=tf.float32)
    # don't apply noise to boundary nodes
    mask = tf.equal(frame['node_type'], common.NodeType.NORMAL)
    mask = tf.repeat(mask, repeats=3, axis=-1)

    noise = tf.where(mask, noise, tf.zeros_like(noise))
    frame[noise_field] += noise
    if FLAGS.noise_on_adjacent_step:
      frame['target|'+noise_field] += noise
    return frame

  if FLAGS.use_pd_stress:
    ds = ds.map(use_pd_stress)

  ds = ds.map(filter_stress_outliers)

  if not test:
    ds =  ds.flat_map(repeat_and_shift)

    # Remove zero world edges
    ds = ds.map(world_edges_processing_map, num_parallel_calls=tf.data.AUTOTUNE)

    # Add noise
    ds = ds.map(add_noise, num_parallel_calls=tf.data.AUTOTUNE) #Used to be 8


  ds = ds.shuffle(1000) # Used to both be 500

  if batch and FLAGS.batch_size > 1:
    ds = dataset.batch_dataset(ds, FLAGS.batch_size)

  return ds.prefetch(1)


def filter_for_good_objects(total_files, params):
  import trimesh

  # First, filter out total_files that are too large (i.e., any dimension of bounding box greater than 20 cm)
  # Extents are l, w, and h of bounding box
  # Then, sort by complexity and take simpler ones
  total_files_within_bb_size = []
  num_original_vertices = []
  for f in total_files:
    obj_name = f.split("/")[-1].split(".")[0]
    object_model_folder = os.path.join(SIMPLE_GEOMETRIC_DATASET_FOLDER, obj_name)
    processed_stl_file = os.path.join(object_model_folder, [k for k in os.listdir(object_model_folder) if "_processed.stl" in k][0])
    original_stl_file = os.path.join(object_model_folder, [k for k in os.listdir(object_model_folder) if ".stl" in k and "processed" not in k][0])

    processed_obj_mesh = trimesh.load(processed_stl_file)
    original_obj_mesh = trimesh.load(original_stl_file)

    if all(trimesh.bounds.oriented_bounds(processed_obj_mesh)[1] < 0.2):
      total_files_within_bb_size.append(f)
      num_original_vertices.append(processed_obj_mesh.vertices.shape[0])

  sorted_indices_by_num_original_vertices = np.argsort(num_original_vertices)
  geometrically_simpler_object_files = list(np.array(total_files_within_bb_size)[sorted_indices_by_num_original_vertices][:50])



  # Now, filter out objects whose max stresses are too high
  total_ds = dataset.load_dataset(FLAGS.dataset_dir, geometrically_simpler_object_files)
  total_ds = get_flattened_dataset(total_ds, params, do_cutoff=True,  batch=False, takeall=True)
  iterator = iter(total_ds)

  traj_count = 0
  final_stresses_dict = dict()
  try:
    while True:
      inputs = iterator.get_next()
      kk = tf.reduce_max(inputs["stress"][-1])
      object_name = inputs['name'].numpy()[0][0][0].decode()

      if object_name not in final_stresses_dict.keys():
        final_stresses_dict[object_name] = []
      final_stresses_dict[object_name].append(kk)

      traj_count += 1
  except tf.errors.OutOfRangeError:
    print("This many trajectories seen", traj_count)

  all_k_names, all_max_stresses = [], []
  for k_name in final_stresses_dict.keys():
    all_max_stresses.append(max(final_stresses_dict[k_name]))
    all_k_names.append(k_name)
  sorted_indices = np.argsort(all_max_stresses) # Trim off ends
  sorted_indices = sorted_indices[:int(0.97*len(sorted_indices))]
  return list(np.array(geometrically_simpler_object_files)[sorted_indices])


def evaluator_file_split(params):
  total_folder = FLAGS.dataset_dir
  total_files = sorted([os.path.join(total_folder, k) for k in os.listdir(total_folder)])


  # Set conditions for splitting test and train sets, eg: 
  train_files = [k for k in total_files if "test" not in k]
  test_files = [k for k in total_files if "test" in k]
  return train_files, test_files

def remove_zero_world_edges(inputs):
  world_edges = inputs["world_edges"]
  all_senders, all_receivers = tf.unstack(world_edges, axis=-1)
  difference = tf.math.subtract(all_senders, all_receivers)

  unique_edges = tf.where(tf.not_equal(difference, 0))

  close_pair_idx = tf.gather(world_edges, unique_edges[:,1], axis=1)
  inputs["world_edges"] = close_pair_idx
  return inputs


# @tf.function
def learner(model, params):
  """Run a learner job."""

  # Set up tensorboard writer
  import datetime
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  train_log_dir = FLAGS.checkpoint_dir
  # train_summary_writer = tf.summary.create_file_writer(train_log_dir)


  n_horizon = FLAGS.n_horizon
  train_files, test_files = evaluator_file_split(params)


  ############### Checkpointing
  losses = []
  lowest_val_errors = [sys.maxsize]


  ### Use normal saver
  num_best_models = 2


  # Try to do checkpointing in TF 2
  checkpoint = tf.train.Checkpoint(module=model)
  manager = tf.train.CheckpointManager(checkpoint, FLAGS.checkpoint_dir, max_to_keep=2)

  checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "checkpoint_name")
  latest = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
  ################


  ### Load datasets by files in folders 

  train_ds_og = dataset.load_dataset(FLAGS.dataset_dir, train_files)
  test_ds = dataset.load_dataset(FLAGS.dataset_dir, test_files)

  train_ds = get_flattened_dataset(train_ds_og, params, n_horizon, do_cutoff=True, batch=True) 
  single_train_ds = get_flattened_dataset(train_ds_og, params, n_horizon, do_cutoff=True, batch=False) 
  test_ds = get_flattened_dataset(test_ds, params, do_cutoff=True,  batch=False)


  ##### TF 2 optimizer
  optimizer = snt.optimizers.Adam(learning_rate=FLAGS.learning_rate)
  @tf.function(input_signature=[model.input_signature_dict],  jit_compile=False)
  def train_step(inputs):
    with tf.GradientTape() as tape:
      loss_op = params['evaluator'].evaluate_one_step(model, inputs, FLAGS, num_steps=n_horizon, normalize=True)

    grads = tape.gradient(loss_op, model.trainable_variables)
    optimizer.apply(grads, model.trainable_variables)

    return loss_op

  @tf.function(input_signature=[model.validation_signature_dict],  jit_compile=False)
  def validation_step(inputs):
    test_loss, test_traj_data, test_scalar_data = params['evaluator'].evaluate(model, inputs, FLAGS, normalize=True)
    return test_loss, test_traj_data, test_scalar_data


  if latest is not None:
    print("Restoring previous checkpoint")
    checkpoint.restore(latest)
  else:
    print("No existing checkpoint")



  ########### Take in all training data for normalization stats

  t0_accumulate = timeit.default_timer()

  if latest is None: # Accumulate stats anyway
    iterator = iter(single_train_ds)

    num_train_dp = 0
    print("Accumulating stats without training")

    try:
      while True:

        inputs = iterator.get_next()
        model.accumulate_stats(inputs)



        num_train_dp += 1
        if num_train_dp % 1000 == 0:
          print("Accumulated", num_train_dp)

    except tf.errors.OutOfRangeError:
      print("Accumulated stats. Total number of train points:", num_train_dp)
      tnext = timeit.default_timer()
      print("Time taken for accumulation", timeit.default_timer() - t0_accumulate)
      pass

  # print(model.accumulate_stats.pretty_printed_concrete_signatures()) 
  # quit()

  ########### Get distribution of all stresses seen during training (for debugging)
  '''
  iterator = iter(test_ds)

  traj_count = 0
  final_stresses_dict = dict()
  try:
    while True:
      inputs = iterator.get_next()
      kk = tf.reduce_max(inputs["stress"][-1])
      object_name = inputs['name'].numpy()[0][0][0].decode()

      if object_name not in final_stresses_dict.keys():
        final_stresses_dict[object_name] = []
      final_stresses_dict[object_name].append(kk)

      traj_count += 1

  except tf.errors.OutOfRangeError:
    print("This many trajectories seen", traj_count)

  import matplotlib.pyplot as plt 
  all_k_names, all_max_stresses = [], []
  for k_name in final_stresses_dict.keys():
    all_max_stresses.append(max(final_stresses_dict[k_name]))
    all_k_names.append(k_name)
    plt.plot(final_stresses_dict[k_name], label=k_name)
  sorted_indices = np.argsort(all_max_stresses)
  for si in sorted_indices:
    print(all_k_names[si], all_max_stresses[si])
  plt.legend()
  plt.show()
  '''
  #####################

  step = 0
  epoch_time = 0


  # tf.profiler.experimental.start(FLAGS.checkpoint_dir)
  for i in range(FLAGS.num_epochs):

    iterator = iter(train_ds)
    train_losses = []

    # '''
    try:
      t0 = timeit.default_timer()
      start_time = timeit.default_timer()
      while True:
        
        # with tf.profiler.experimental.Trace('my_train_step', step_num=step, _r=1):
        inputs = iterator.get_next()


        step += 1

        train_loss = train_step(inputs)
        train_losses.append(train_loss)
        if step % 500 == 0:
          logging.info('Epoch %d, Step %d: Avg Loss %g', i, step, np.mean(train_losses))
          print(timeit.default_timer() - start_time)
          start_time = timeit.default_timer()


    except tf.errors.OutOfRangeError:

      tnext = timeit.default_timer()
      epoch_time = tnext - t0
      print("------Time taken for epoch", i, ":", epoch_time, np.mean(train_losses))
    # '''
    ################
    # '''
    if i % FLAGS.num_epochs_per_validation == 0:
      print("Validating")

      iterator = iter(test_ds)

      test_losses, test_pos_mean_errors, test_pos_final_errors = [], [], []
      test_stress_mean_errors, test_stress_final_errors = [], []
      baseline_pos_mean_errors, baseline_pos_final_errors = [], []
      baseline_stress_mean_errors, baseline_stress_final_errors = [], []
      actual_mean_stresses, pred_mean_stresses = [], []
      actual_max_stresses, pred_max_stresses = [], []
      actual_mean_defs, pred_mean_defs = [], []
      mean_def_percent_errors, mean_stress_percent_errors = [], []

      try:
        validation_counter = 0
        validation_t0 = timeit.default_timer()
        while True:
          inputs = iterator.get_next()

          # '''
          validation_counter += 1

          # Make tf function
          test_loss, test_traj_data, test_scalar_data = validation_step(inputs) # Full trajectory. NORMALIZE SHOULD BE TRUE. 

          # actual_mean_stresses.append(test_traj_data['mean_actual_stress'])
          actual_mean_stresses.append(test_traj_data['mean_actual_stress_under_stress_only'])
          actual_max_stresses.append(test_traj_data['max_actual_stress'])

          # pred_mean_stresses.append(test_traj_data['mean_pred_stress'])
          pred_mean_stresses.append(test_traj_data['mean_pred_stress_under_stress_only'])
          pred_max_stresses.append(test_traj_data['max_pred_stress'])

          # As a test
          pred_mean_defs.append(test_traj_data['median_deformation_norm_opt'])
          actual_mean_defs.append(test_traj_data['median_actual_deformation_norm'])

          test_losses.append(test_loss)
          test_pos_mean_errors.append(test_scalar_data["pos_mean_error"])
          test_pos_final_errors.append(test_scalar_data["pos_final_error"])
          baseline_pos_mean_errors.append(test_scalar_data['baseline_pos_mean_error'])
          baseline_pos_final_errors.append(test_scalar_data['baseline_pos_final_error'])

          test_stress_mean_errors.append(test_scalar_data["stress_mean_error"])
          test_stress_final_errors.append(test_scalar_data["stress_final_error"])
          baseline_stress_mean_errors.append(test_scalar_data['baseline_stress_mean_error'])
          baseline_stress_final_errors.append(test_scalar_data['baseline_stress_final_error'])

          mean_def_percent_errors.append(test_traj_data['mean_deformation_percent_error'])
          mean_stress_percent_errors.append(test_traj_data['mean_stress_percent_error'])


          # '''

      except tf.errors.OutOfRangeError:
        validation_tnext = timeit.default_timer()
        validation_time = validation_tnext - validation_t0
        print("****** Time taken for validation", i, ":", validation_time, "with counter at", validation_counter)
        pass

      '''
      import matplotlib.pyplot as plt
      # plt.plot(actual_mean_defs)
      plt.scatter(actual_mean_stresses, actual_mean_stresses)
      plt.scatter(actual_mean_stresses, pred_mean_stresses)
      plt.show()
      '''
      # Save only if validation loss is good
      if np.mean(test_losses) <= lowest_val_errors[-1]:
        lowest_val_errors.append(np.mean(test_losses))
        lowest_val_errors.sort()
        lowest_val_errors = lowest_val_errors[:num_best_models]
        print("Saving checkpoint. Lowest validation errors updated:", lowest_val_errors)
        if FLAGS.checkpoint_dir:
          print("Saving")
          save_path = manager.save()
          print("Saved to", save_path)
    # '''

    print("~~~~~TEST VS RAW:", np.mean(test_losses), np.mean(test_pos_mean_errors))



    tau_stress, _ = stats.kendalltau(actual_mean_stresses, pred_mean_stresses)
    tau_def, _ = stats.kendalltau(actual_mean_defs, pred_mean_defs)
    tau_max_stress, _ = stats.kendalltau(actual_max_stresses, pred_max_stresses)
    # print(actual_mean_defs)
    
    print("MAE def stress", np.mean(test_pos_mean_errors), np.mean(test_stress_mean_errors))# print(pred_mean_defs)
    print("Kendall's taus:", tau_def, tau_stress, tau_max_stress)
    # Record losses in text file
    logging.info('Epoch %d, Step %d: Train Loss %g, Test Errors %g %g', i, step, np.mean(train_losses), np.mean(test_pos_mean_errors), np.mean(test_stress_mean_errors))

    
    if FLAGS.checkpoint_dir:
      file = open(os.path.join(FLAGS.checkpoint_dir, "losses.txt"), "a")
      log_line = "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n" % (step, np.mean(train_losses), np.mean(test_losses), np.mean(test_pos_mean_errors), \
        np.mean(test_pos_final_errors), np.mean(baseline_pos_mean_errors), np.mean(baseline_pos_final_errors), \
        np.mean(test_stress_mean_errors), np.mean(test_stress_final_errors), np.mean(baseline_stress_mean_errors), np.mean(baseline_stress_final_errors), \
        tau_stress, tau_def, np.mean(mean_def_percent_errors), np.mean(mean_stress_percent_errors), tau_max_stress, epoch_time)

      file.write(log_line)
      file.close()

  logging.info('Training complete.')
  # tf.profiler.experimental.stop()

  # train()




#########################3
def evaluator(model, params):
  """Run a model rollout trajectory."""
  train_files, test_files = evaluator_file_split(params)

  trajectories = []
  scalars = []

  all_object_actual_final_stresses = dict()
  all_object_pred_final_stresses = dict()

  all_object_actual_final_deformations = dict()
  all_object_pred_final_deformations = dict()

  stresses_accuracy_running = dict()
  deformations_accuracy_running = dict()

  for k in ['mean', 'max', 'median']:
    all_object_actual_final_stresses[k], all_object_pred_final_stresses[k] = [], []
    all_object_actual_final_deformations[k], all_object_pred_final_deformations[k] = [], []
    stresses_accuracy_running[k] = [0, 0]
    deformations_accuracy_running[k] = [0, 0]



  # Restore checkpoint
  checkpoint = tf.train.Checkpoint(module=model)
  latest = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

  if latest is not None:
    print("Restoring previous checkpoint")
    checkpoint.restore(latest)



  for ot, obj_file in enumerate(test_files):
    print("===", obj_file, ot, "of", len(test_files))
    obj_name = obj_file.split("/")[-1].split(".")[0]
    obj_name = "strawberry02"
    ds = dataset.load_dataset(FLAGS.dataset_dir, [obj_file])
    ds = dataset.add_targets(ds, FLAGS, params['field'] if type(params['field']) is list else [params['field']], add_history=params['history'])
    iterator = iter(ds)


    traj_idx = 0

    # Sort predicted and final stresses
    actual_final_stresses = dict()
    pred_final_stresses = dict()

    actual_final_deformations = dict()
    pred_final_deformations = dict()
    for k in ['mean', 'max', 'median']:
      actual_final_stresses[k], pred_final_stresses[k] = [], []

      actual_final_deformations[k], pred_final_deformations[k] = [], []
      

    try:
      
      while True:

        inputs = iterator.get_next()

        # Calculate gradients
        quantity_to_optimize = "max_pred_stress" #"mean_pred_stress"max_deformation_norm_opt


        direct_output, traj_data, scalar_data = params['evaluator'].evaluate(model, inputs, FLAGS)


        # Record rollout trajectory

        logging.info('Rollout trajectory %d', traj_idx)
        traj_idx += 1


        trajectories.append(traj_data)
        scalars.append(scalar_data)

        actual_final_stresses['mean'].append(np.mean(traj_data['gt_stress'][-1]))
        actual_final_stresses['max'].append(np.max(traj_data['gt_stress'][-1]))

        pred_final_stresses['mean'].append(np.mean(traj_data['pred_stress'][-1]))
        pred_final_stresses['max'].append(np.max(traj_data['pred_stress'][-1]))

        mean_a, max_a, median_a = utils.get_global_deformation_metrics(traj_data['gt_pos'][0].numpy(), traj_data['gt_pos'][-1].numpy())
        mean_p, max_p, median_p = utils.get_global_deformation_metrics(traj_data['pred_pos'][0].numpy(), traj_data['pred_pos'][-1].numpy())

        actual_final_deformations['mean'].append(mean_a)
        actual_final_deformations['max'].append(max_a)
        actual_final_deformations['median'].append(median_a)

        pred_final_deformations['mean'].append(mean_p)
        pred_final_deformations['max'].append(max_p)
        pred_final_deformations['median'].append(median_p)

        print("Final error", scalar_data['stress_error'][-1], scalar_data['rollout_losses'][-1])

        # '''

    except tf.errors.OutOfRangeError:
      pass


  with open(os.path.join(FLAGS.checkpoint_dir, 'rollout.pkl'), 'wb') as fp:
    pickle.dump(trajectories, fp)



def main(argv):
  del argv
  # tf.enable_resource_variables()

  # tf.config.optimizer.set_jit("autoclustering")
  tf.config.run_functions_eagerly(FLAGS.eager)


  global LEN_TRAJ
  utils.check_consistencies(FLAGS)

  if utils.using_dm_dataset(FLAGS): # if default deepmind dataset
    LEN_TRAJ = 400 - 2
  else: # If defgraspnets dataset
    LEN_TRAJ = 50 - 2
    LEN_TRAJ = 30

  # Use CPU
  if FLAGS.use_cpu:
    tf.config.set_visible_devices([], 'GPU')


  params = PARAMETERS[FLAGS.model]

  # Write flags to file
  
  if FLAGS.checkpoint_dir:
    flags_file = os.path.join(FLAGS.checkpoint_dir, 'flags.txt')
    os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)
    FLAGS.append_flags_into_file(flags_file)


  output_size = utils.get_output_size(FLAGS)


  #### Set up mixed precision
  # support_modes = snt.mixed_precision.modes([tf.float32, tf.float16])
  # core_model.EncodeProcessDecode.__call__ = support_modes(core_model.EncodeProcessDecode.__call__)
  # snt.mixed_precision.enable(tf.float16)



  learned_model = core_model.EncodeProcessDecode(
      output_size=output_size,
      latent_size=FLAGS.latent_size,
      num_layers=FLAGS.num_layers,
      use_layernorm=FLAGS.layernorm,
      message_passing_steps=FLAGS.message_passing_steps
      )


  model = params['model'].Model(learned_model, FLAGS)




  if FLAGS.mode == 'train':
    learner(model, params)
  elif FLAGS.mode == 'eval':
    evaluator(model, params)




if __name__ == '__main__':
  app.run(main)
