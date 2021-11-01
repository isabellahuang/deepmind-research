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

import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from meshgraphnets import cfd_eval
from meshgraphnets import cfd_model
from meshgraphnets import cloth_eval
from meshgraphnets import cloth_model
from meshgraphnets import deforming_plate_model
from meshgraphnets import deforming_plate_eval
from meshgraphnets import core_model
from meshgraphnets import dataset
from meshgraphnets import utils

import os


FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval'],
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

flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_integer('num_epochs', 50, 'Num epochs')
flags.DEFINE_integer('num_epochs_per_validation', 5, 'Num epochs to train before validating')
flags.DEFINE_integer('num_training_trajectories', 100, 'No. of training trajectories')


flags.DEFINE_integer('latent_size', 128, 'Latent size')
flags.DEFINE_integer('num_layers', 2, 'Num layers')
flags.DEFINE_integer('message_passing_steps', 15, 'Message passing steps')
flags.DEFINE_float('learning_rate', 1e-4, 'Message passing steps')

flags.DEFINE_bool('log_stress_t', True, 'Take the log of stress at time t as feature')

flags.DEFINE_bool('force_label_node', False, 'Whether each node should be labeled with a force-related signal. If False, mesh edges are labeled instead')
flags.DEFINE_bool('node_total_force_t', False, 'Whether each node should be labeled with the total gripper force')
flags.DEFINE_bool('compute_world_edges', False, 'Whether to compute world edges')


# Exactly one of these must be set to True
flags.DEFINE_bool('predict_log_stress_t1', False, 'Predict the log stress at t1')
flags.DEFINE_bool('predict_log_stress_change_t1', False, 'Predict the log of stress change from t to t1')

flags.DEFINE_bool('predict_log_stress_t_only', False, 'Do not make deformation predictions. Only predict stress at time t')
flags.DEFINE_bool('predict_log_stress_t1_only', False, 'Do not make deformation predictions. Only predict stress at time t1')
flags.DEFINE_bool('predict_log_stress_change_only', False, 'Do not make deformation predictions. Only predict stress at time t1')
flags.DEFINE_bool('predict_pos_change_only', False, 'Do not make stress predictions. Only predict change in pos between time t and t1')

flags.DEFINE_bool('aux_mesh_edge_distance_change', False, 'Add auxiliary loss term: mesh edge distances')


flags.DEFINE_integer('num_objects', 1, 'No. of objects to train on')
flags.DEFINE_integer('n_horizon', 1, 'No. of steps in training horizon')



PARAMETERS = {
    'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=cfd_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval),
    'deforming_plate': dict(noise=3e-3, gamma=0.1, field=['world_pos', 'gripper_pos', 'force', 'stress'], noise_field='world_pos', history=False,
                  size=4, batch=16, model=deforming_plate_model, evaluator=deforming_plate_eval),
}

LEN_TRAJ = 50 - 2

def classification_accuracy_ranking(predicted, actual, percentile):
  actual_threshold = np.percentile(actual, percentile)
  pred_threshold = np.percentile(predicted, percentile)
  num_g = len(predicted)
  assert(num_g == len(actual))

  actual_top = [k for k in range(num_g) if actual[k] >= actual_threshold]
  actual_bottom = [k for k in range(num_g) if actual[k] < actual_threshold]

  predicted_top = [k for k in range(num_g) if predicted[k] >= pred_threshold]
  predicted_bottom = [k for k in range(num_g) if predicted[k] < pred_threshold]

  predicted_bottom_correct = [k for k in predicted_bottom if k in actual_bottom]
  predicted_top_correct = [k for k in predicted_top if k in actual_top]

  # print("Actual", np.min(actual), threshold, np.max(actual))
  # print("Predicted", np.min(predicted), threshold, np.max(predicted))

  return len(predicted_bottom_correct) + len(predicted_top_correct), len(predicted)

def classification_accuracy_threshold(predicted, actual, percentile):

  threshold = np.percentile(actual, percentile)
  num_g = len(predicted)
  assert(num_g == len(actual))

  actual_top = [k for k in range(num_g) if actual[k] >= threshold]
  actual_bottom = [k for k in range(num_g) if actual[k] < threshold]

  predicted_top = [k for k in range(num_g) if predicted[k] >= threshold]
  predicted_bottom = [k for k in range(num_g) if predicted[k] < threshold]

  predicted_bottom_correct = [k for k in predicted_bottom if k in actual_bottom]
  predicted_top_correct = [k for k in predicted_top if k in actual_top]

  # print("Actual", np.min(actual), threshold, np.max(actual))
  # print("Predicted", np.min(predicted), threshold, np.max(predicted))

  return len(predicted_bottom_correct) + len(predicted_top_correct), len(predicted)




def get_flattened_dataset(ds, params, n_horizon=LEN_TRAJ, n_training_trajectories=100):
  # ds = dataset.load_dataset(FLAGS.dataset_dir, 'train', FLAGS.num_objects)
  ds = dataset.add_targets(ds, params['field'] if type(params['field']) is list else [params['field']], add_history=params['history'])

  # if flatten:
  #   ds = dataset.split_and_preprocess(ds, num_epochs=1, noise_field=params['noise_field'],
  #                                     noise_scale=params['noise'],
  #                                     noise_gamma=params['gamma'])


  # if FLAGS.batch_size > 1:
  #   ds = dataset.batch_dataset(ds, FLAGS.batch_size)
  # if os.environ["LOCAL_FLAG"] == "1": # If not on NGC
  #   ds = ds.take(1)
    # pass

  if n_training_trajectories != 100:
    ds = ds.take(n_training_trajectories)

  # Pre shuffle buffer
  ds = ds.shuffle(100)

  def repeat_and_shift(trajectory):
    out = {}
    for key, val in trajectory.items():
      shifted_lists = []

      for i in range(LEN_TRAJ - n_horizon + 1):
        shifted_list = tf.roll(val, shift=-i, axis=0)
        shifted_lists.append(shifted_list)
      out[key] = shifted_lists

    return tf.data.Dataset.from_tensor_slices(out)

  if n_horizon < LEN_TRAJ:
    ds =  ds.flat_map(repeat_and_shift)

  ds = ds.shuffle(100)
  return ds


def evaluator_file_split():
  # Iterate through objects
  if os.environ["LOCAL_FLAG"] != "1": # If on NGC
    train_folder = os.path.join(FLAGS.dataset_dir, '0-99_tfrecords', '5e4')
    total_folder = train_folder
    train_files = [os.path.join('0-99_tfrecords', '5e4', k.split(".")[0]) for k in os.listdir(train_folder)]
    test_files = train_files

  else:
    last_folder = '0-99_tfrecords/5e4_pd'
    total_folder = os.path.join(FLAGS.dataset_dir, last_folder)
    total_files = [os.path.join(last_folder, k.split(".")[0]) for k in os.listdir(total_folder)]
    train_files = [k for k in total_files if '00000015' in k]
    test_files = train_files

    # train_folder = os.path.join(FLAGS.dataset_dir, '0-99_tfrecords/5e4_pd')
    # train_files = [os.path.join('0-99_tfrecords/5e4', k.split(".")[0]) for k in os.listdir(train_folder)]
    # test_files = [os.path.join('0-99_tfrecords/ngc', k.split(".")[0]) for k in os.listdir(total_folder) if k not in os.listdir(train_folder)]

  return train_files, test_files


def learner(model, params):
  """Run a learner job."""
  global_step = tf.train.create_global_step()

  n_horizon = FLAGS.n_horizon

  train_files, test_files = evaluator_file_split()

  ######## For training
  if os.environ["LOCAL_FLAG"] != "1": # If on NGC
    obj_15 = [k for k in train_files if '00000015' in k]
    train_ds_og = dataset.load_dataset(FLAGS.dataset_dir, obj_15, FLAGS.num_objects)
    test_ds = dataset.load_dataset(FLAGS.dataset_dir, obj_15, max(1, int(0.3 * FLAGS.num_objects)))
    # test_ds = dataset.load_dataset(FLAGS.dataset_dir, 'test', max(1, int(0.3 * FLAGS.num_objects)))
  else:
    # train_ds_og = dataset.load_dataset(FLAGS.dataset_dir, 'train', FLAGS.num_objects)
    # test_files = [k for k in train_files if '00000007' in k]
    # test_ds = dataset.load_dataset(FLAGS.dataset_dir, 'train', max(1, int(0.3 * FLAGS.num_objects)))

    ### Load datasets by files in folders 
    train_ds_og = dataset.load_dataset(FLAGS.dataset_dir, train_files, FLAGS.num_objects)
    test_ds = dataset.load_dataset(FLAGS.dataset_dir, test_files, max(1, int(0.3 * FLAGS.num_objects)))



  train_ds = get_flattened_dataset(train_ds_og, params, n_horizon, n_training_trajectories=FLAGS.num_training_trajectories)
  single_train_ds = get_flattened_dataset(train_ds_og, params, n_horizon, n_training_trajectories=FLAGS.num_training_trajectories)


  test_ds = get_flattened_dataset(test_ds, params)

  iterator = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)

  inputs = iterator.get_next()
  single_train_init_op = iterator.make_initializer(single_train_ds)
  train_init_op = iterator.make_initializer(train_ds)
  test_init_op = iterator.make_initializer(test_ds)


  loss_op, traj_op, scalar_op = params['evaluator'].evaluate(model, inputs, FLAGS, num_steps=n_horizon, normalize=True)
  test_loss_op, test_traj_op, test_scalar_op = params['evaluator'].evaluate(model, inputs, FLAGS, normalize=True) # Full trajectory

  debug_op = model.print_debug(inputs)

  lr = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate, # Used to be 1e-4
                                  global_step=global_step,
                                  decay_steps=int(5e6),
                                  decay_rate=0.1) + 1e-6
  optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  train_op = optimizer.minimize(loss_op, global_step=global_step) # Passing in global step increments it every time a batch finishes

  # Don't train for the first few steps, just accumulate normalization stats. Where does this happen? 
  # accumulate_op = model.loss(inputs, accumulate=True, normalize=True)
  accumulate_op, _ , _ = params['evaluator'].evaluate(model, inputs, FLAGS, num_steps=1, normalize=True, accumulate=True)
  # train_op = tf.cond(tf.less(global_step, 100),
  #                    lambda: tf.group(tf.assign_add(global_step, 1)), # If global step < 1000
  #                    lambda: tf.group(train_op)) # If global step > 1000

  losses = []


  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.train.MonitoredTrainingSession(
      hooks=[tf.train.StopAtStepHook(last_step=FLAGS.num_training_steps)],
      config=config,
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_steps=1000) as sess:

    # Take in all training data for normalization stats
    # '''
    sess.run(single_train_init_op)
    num_train_dp = 0
    # print("Accuulating stats")
    try:
      while True:
        _, a = sess.run([accumulate_op, debug_op]) # currently doesnt use input, so infinite loop
        num_train_dp += 1

    except tf.errors.OutOfRangeError:
      print("Accumulated stats. Total number of train points:", num_train_dp)
      pass
    # '''

    step = 0
    for i in range(FLAGS.num_epochs):
      sess.run(train_init_op)
      train_losses = []

      ##########
      # '''
      try:
        while True:

          # _, step, train_loss, a, traj_opt_data, scalar_op_data = sess.run([train_op, global_step, loss_op, debug_op, traj_op, scalar_op]) 
          _, step, train_loss, a, traj_opt_data = sess.run([train_op, global_step, loss_op, debug_op, traj_op]) 
          # _, step = sess.run([train_op, global_step])
          # train_loss, a = sess.run([loss_op, debug_op])
          ## Note to self: look at gradients
          # loss, traj_data = sess.run([scalar_op, traj_ops])
          # final_error = sess.run([scalar_op])
          # print(final_error)
          # print("Train", step, traj_opt_data['avg_gt_stress'], traj_opt_data['avg_pred_stress'])
          # print(scalar_op_data['error'])

          train_losses.append(train_loss)

          # print("training", step, a)
          if step % 1000 == 0:
            logging.info('Epoch %d, Step %d: Avg Loss %g', i, step, np.mean(train_losses))


      except tf.errors.OutOfRangeError:
        pass
      # '''


      if i % FLAGS.num_epochs_per_validation == 0:
        print("Validating")
        sess.run(test_init_op)
        test_losses, test_mean_errors, test_final_errors = [], [], []


        # '''
        try:
          while True:

            test_loss, test_scalar_data, test_traj_data, a = sess.run([test_loss_op, test_scalar_op, test_traj_op, debug_op]) # used to be raw_loss_op
            test_losses.append(test_loss)
            test_mean_errors.append(test_scalar_data["pos_mean_error"])
            test_final_errors.append(test_scalar_data["pos_final_error"])


            if step % 1000 == 0:
              logging.info('Step %d: Avg Loss %g', step, np.mean(test_mean_errors))


        except tf.errors.OutOfRangeError:
          pass
        # '''

      # Record losses in text file
      logging.info('Epoch %d, Step %d: Train Loss %g, Test Errors %g %g %g', i, step, np.mean(train_losses), np.mean(test_losses), np.mean(test_mean_errors), np.mean(test_final_errors))
      if FLAGS.checkpoint_dir:
        file = open(os.path.join(FLAGS.checkpoint_dir, "losses.txt"), "a")
        log_line = "%s, %s, %s, %s, %s\n" % (step, np.mean(train_losses), np.mean(test_losses), np.mean(test_mean_errors), np.mean(test_final_errors))
        file.write(log_line)
        file.close()

    logging.info('Training complete.')


def evaluator(model, params):
  """Run a model rollout trajectory."""
  train_files, test_files = evaluator_file_split()


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


  # train_objects = test_files[:1] # object 15
  # test_objects = [k for k in train_files if '00000007' in k]
  test_objects = test_files 


  for ot, obj_file in enumerate(test_objects):
    print("===", obj_file, ot, "of", len(test_files))

    # ds = dataset.load_dataset(FLAGS.dataset_dir, FLAGS.rollout_split)
    ds = dataset.load_dataset(FLAGS.dataset_dir, obj_file, FLAGS.num_objects)
    ds = dataset.add_targets(ds, params['field'] if type(params['field']) is list else [params['field']], add_history=params['history'])
    inputs = tf.data.make_one_shot_iterator(ds).get_next()
    _, traj_ops, scalar_op = params['evaluator'].evaluate(model, inputs, FLAGS)


    # Try to get gradients
    pred_stress_output = traj_ops['pred_stress']
    # grad_op = tf.gradients(pred_stress_output, inputs['world_pos'])
    # grad_op = tf.gradients(scalar_op['actual_final_stress'], inputs['stress'], unconnected_gradients='zero') # This one for gradients

    try:
      tf.train.create_global_step()
    except:
      pass

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.checkpoint_dir,
        config=config,
        save_checkpoint_secs=None,
        save_checkpoint_steps=None) as sess:


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
        # for traj_idx in range(FLAGS.num_rollouts):
        # for traj_idx in range(5):
          logging.info('Rollout trajectory %d', traj_idx)
          traj_idx += 1
          scalar_data, traj_data = sess.run([scalar_op, traj_ops])

          trajectories.append(traj_data)
          scalars.append(scalar_data)

          actual_final_stresses['mean'].append(np.mean(traj_data['gt_stress'][-1]))
          actual_final_stresses['max'].append(np.max(traj_data['gt_stress'][-1]))

          pred_final_stresses['mean'].append(np.mean(traj_data['pred_stress'][-1]))
          pred_final_stresses['max'].append(np.max(traj_data['pred_stress'][-1]))

          # print(pred_final_stresses['mean'][-1], actual_final_stresses['mean'][-1])
          # '''
          mean_a, max_a, median_a = utils.get_global_deformation_metrics(traj_data['gt_pos'][0], traj_data['gt_pos'][-1])
          mean_p, max_p, median_p = utils.get_global_deformation_metrics(traj_data['pred_pos'][0], traj_data['pred_pos'][-1])

          actual_final_deformations['mean'].append(mean_a)
          actual_final_deformations['max'].append(max_a)
          actual_final_deformations['median'].append(median_a)

          pred_final_deformations['mean'].append(mean_p)
          pred_final_deformations['max'].append(max_p)
          pred_final_deformations['median'].append(median_p)


          # '''

      except tf.errors.OutOfRangeError:
        pass

      ################
      '''
      import matplotlib.pyplot as plt
      plt.figure()
      plt.hist(pred_final_stresses['mean'], alpha=0.5, label="pred")
      plt.hist(actual_final_stresses['mean'], alpha=0.5, label="gt")
      plt.legend()

      plt.figure()
      plt.scatter(actual_final_stresses['mean'], pred_final_stresses['mean'], label="pred")
      plt.scatter(actual_final_stresses['mean'], actual_final_stresses['mean'], label="gt")
      plt.legend()
      
      plt.show()
      '''

      ####################

      for k in ['mean', 'max', 'median']:
        all_object_actual_final_stresses[k].extend(actual_final_stresses[k])
        all_object_pred_final_stresses[k].extend(pred_final_stresses[k])
        all_object_actual_final_deformations[k].extend(actual_final_deformations[k])
        all_object_pred_final_deformations[k].extend(pred_final_deformations[k])


      indices = list(range(len(actual_final_stresses)))

      # Classify by threshold per object
      preds = [pred_final_stresses, pred_final_deformations]
      actuals = [actual_final_stresses, actual_final_deformations]
      runnings = [stresses_accuracy_running, deformations_accuracy_running]
      metric_names = ["Stress", "Deformation"]
      for ind, (metric_name, pred, actual) in enumerate(zip(metric_names, preds, actuals)):
        print("======Metric:", metric_name)
        for m_type in ['mean', 'max']:
          num_correct, num_total = classification_accuracy_threshold(pred[m_type], actual[m_type], 50)
          # num_correct, num_total = classification_accuracy_ranking(pred[m_type], actual[m_type], 50)

          print(m_type, "accuracy:", num_correct / num_total, "of", num_total)
          runnings[ind][m_type][0] += num_correct
          runnings[ind][m_type][1] += num_total
          print(m_type, "Running accuracy", runnings[ind][m_type][0]/runnings[ind][m_type][1], "of", runnings[ind][m_type][1])


  all_preds = [all_object_pred_final_stresses, all_object_pred_final_deformations]
  all_actuals = [all_object_actual_final_stresses, all_object_actual_final_deformations]
  metric_names = ["Stress", "Deformation"]
  print("\nAll object statistics")
  for metric_name, all_pred, all_actual in zip(metric_names, all_preds, all_actuals):
    print("======Metric:", metric_name)
    for m_type in ['mean', 'max']:
      num_correct, num_total = classification_accuracy_threshold(all_pred[m_type], all_actual[m_type], 50)
      # num_correct, num_total = classification_accuracy_ranking(all_pred[m_type], all_actual[m_type], 50)
      print(m_type, "Accuracy:", num_correct / num_total, "of", num_total)


    # for key in scalars[0]:
    #   logging.info('%s: %g', key, np.mean([x[key] for x in scalars]))

  with open(FLAGS.rollout_path, 'wb') as fp:
    pickle.dump(trajectories, fp)


def main(argv):
  del argv
  tf.enable_resource_variables()
  tf.disable_eager_execution()
  params = PARAMETERS[FLAGS.model]

  output_size = params['size']

  learned_model = core_model.EncodeProcessDecode(
      output_size=output_size,
      latent_size=FLAGS.latent_size,
      num_layers=FLAGS.num_layers,
      message_passing_steps=FLAGS.message_passing_steps)
  model = params['model'].Model(learned_model, FLAGS)

  if FLAGS.mode == 'train':
    learner(model, params)
  elif FLAGS.mode == 'eval':
    evaluator(model, params)

if __name__ == '__main__':
  app.run(main)
