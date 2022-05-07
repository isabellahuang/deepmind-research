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
import pickle
import sys

import sonnet as snt
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import timeit


# Try to run with more gpu usage in TF 2
devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)


from tfdeterminism import patch
SEED = 55
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
flags.DEFINE_integer('num_testing_trajectories', 0, 'No. of trajectories to train on')


flags.DEFINE_integer('latent_size', 128, 'Latent size')
flags.DEFINE_integer('num_layers', 2, 'Num layers')
flags.DEFINE_integer('message_passing_steps', 15, 'Message passing steps')
flags.DEFINE_float('learning_rate', 1e-4, 'Message passing steps')
flags.DEFINE_float('noise_scale', 3e-3, 'Noise scale on world pos')

flags.DEFINE_bool('noise_on_adjacent_step', True, 'Add perturbation to both t and t + 1 step. If False, add only to t step.')

flags.DEFINE_bool('gripper_force_action_input', True, 'Change in gripper force as action input')


flags.DEFINE_bool('log_stress_t', False, 'Take the log of stress at time t as feature')

flags.DEFINE_bool('force_label_node', False, 'Whether each node should be labeled with a force-related signal. If False, mesh edges are labeled instead')
flags.DEFINE_bool('node_total_force_t', False, 'Whether each node should be labeled with the total gripper force')
# The default is currently labelling mesh edges with normalized force over nodes in contact
flags.DEFINE_bool('compute_world_edges', False, 'Whether to compute world edges')
flags.DEFINE_bool('use_cpu', False, 'Use CPU rather than GPU')
flags.DEFINE_bool('eager', False, 'Use eager execution')


# Exactly one of these must be set to True
flags.DEFINE_bool('predict_log_stress_t1', False, 'Predict the log stress at t1')

flags.DEFINE_bool('predict_log_stress_change_t1', False, 'Predict the log of stress change from t to t1')
flags.DEFINE_bool('predict_stress_change_t1', False, 'Predict the stress change from t to t1')


flags.DEFINE_bool('predict_log_stress_t_only', False, 'Do not make deformation predictions. Only predict stress at time t')
flags.DEFINE_bool('predict_log_stress_t1_only', False, 'Do not make deformation predictions. Only predict stress at time t1')
flags.DEFINE_bool('predict_log_stress_change_only', False, 'Do not make deformation predictions. Only predict stress at time t1')
flags.DEFINE_bool('predict_pos_change_only', False, 'Do not make stress predictions. Only predict change in pos between time t and t1')

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








def get_flattened_dataset(ds, params, n_horizon=None, cutoff=None):
  # ds = dataset.load_dataset(FLAGS.dataset_dir, 'train', FLAGS.num_objects)
  ds = dataset.add_targets(ds, FLAGS, params['field'] if type(params['field']) is list else [params['field']], add_history=params['history'])
  
  test = not n_horizon
  if test:
    n_horizon = LEN_TRAJ

  # if FLAGS.batch_size > 1:
  #   ds = dataset.batch_dataset(ds, FLAGS.batch_size)
  # if os.environ["LOCAL_FLAG"] == "1": # If not on NGC
  #   ds = ds.take(1)
    # pass

  noise_field = params['noise_field']
  noise_scale = params['noise']


  if not test:
    ds = ds.take(FLAGS.num_training_trajectories) 
  else:
    # ds = ds.skip(FLAGS.num_training_trajectories)
    ds = ds.take(FLAGS.num_testing_trajectories)

  # Filter trajectories for outliers
  def filter_func(elem, cutoff):
      """ return True if the element is to be kept """
      last_stress = elem["stress"]
      max_last_stress = tf.reduce_max(last_stress)
      return max_last_stress <= cutoff
  if cutoff:
    ds = ds.filter(lambda ff: filter_func(ff, cutoff))

  # Shuffle trajectories selected for training
  ds = ds.shuffle(500)

  def repeat_and_shift(trajectory):
    out = {}
    for key, val in trajectory.items():
      shifted_lists = []
      for i in range(LEN_TRAJ - n_horizon + 1):
        shifted_list = tf.roll(val, shift=-i, axis=0)
        # shifted_lists.append(shifted_list)
        shifted_lists.append(shifted_list[:n_horizon])


      out[key] = shifted_lists


    return tf.data.Dataset.from_tensor_slices(out)


  def use_pd_stress(frame):
    frame['stress'] = frame['pd_stress']
    return frame


  def add_noise(frame):
    # for key, val in frame.items():
    #   print(key)

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

  if not test:
    ds =  ds.flat_map(repeat_and_shift)

    # Add noise
    ds = ds.map(add_noise, num_parallel_calls=8) #Used to be 8


  ds = ds.shuffle(500)

  # ds = dataset.batch_dataset(ds, 5)

  return ds.prefetch(tf.data.AUTOTUNE)


def evaluator_file_split():
  total_folder = FLAGS.dataset_dir
  total_files = [os.path.join(total_folder, k) for k in os.listdir(total_folder)]
  train_files = [k for k in total_files if '00000015' in k] # trapezoidal prism
  train_files = [k for k in total_files if '00000007' in k] # circular disk
  train_files = [k for k in total_files if '00000016' in k] # rectangular prism
  train_files = [k for k in total_files if 'rectangle' in k] # rectangle dense grasps

  if utils.using_dm_dataset(FLAGS):
    train_files = [k for k in total_files if 'valid' in k] # MGN dataset

  test_files = train_files

  return train_files, test_files


def learner(model, params):
  """Run a learner job."""

  # Set up tensorboard writer
  import datetime
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  train_log_dir = FLAGS.checkpoint_dir
  # train_summary_writer = tf.summary.create_file_writer(train_log_dir)


  n_horizon = FLAGS.n_horizon
  train_files, test_files = evaluator_file_split()


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
  train_ds_og = dataset.load_dataset(FLAGS.dataset_dir, train_files, FLAGS.num_objects)
  test_ds = dataset.load_dataset(FLAGS.dataset_dir, test_files, max(1, int(0.3 * FLAGS.num_objects)))

  train_ds_with_outliers = get_flattened_dataset(train_ds_og, params, cutoff=None)

  # Scan training data for outliers in high stresses
  # '''
  if latest is None:
    print("Filtering out outliers")

    iterator = iter(train_ds_with_outliers)
    final_stresses = []
    traj_count = 0
    try:
      while True:
        traj_count += 1
        inputs = iterator.get_next()

        final_stresses.append(tf.reduce_max(inputs["stress"][-1])  )

    except tf.errors.OutOfRangeError:
      pass
  # '''
  # Save this to dict per object name
  percentile_cutoff = tfp.stats.percentile(final_stresses, 97, interpolation='linear')



 
  train_ds = get_flattened_dataset(train_ds_og, params, n_horizon, cutoff=percentile_cutoff) #percentile_cutoff
  single_train_ds = get_flattened_dataset(train_ds_og, params, n_horizon, cutoff=percentile_cutoff) #percentile_cutoff
  test_ds = get_flattened_dataset(test_ds, params, cutoff=percentile_cutoff)




  ##### TF 2 optimizer
  optimizer = snt.optimizers.Adam(learning_rate=FLAGS.learning_rate)
  @tf.function(input_signature=[model.input_signature_dict])
  def train_step(inputs):
    with tf.GradientTape() as tape:
      loss_op, traj_op, scalar_op = params['evaluator'].evaluate(model, inputs, FLAGS, num_steps=n_horizon, normalize=True)

    grads = tape.gradient(loss_op, model.trainable_variables)
    optimizer.apply(grads, model.trainable_variables)

    return loss_op




  if latest is not None:
    print("Restoring previous checkpoint")
    checkpoint.restore(latest)
  else:
    print("No existing checkpoint")



  # Take in all training data for normalization stats

  # '''
  t0_accumulate = timeit.default_timer()
  if latest is None:
    iterator = iter(single_train_ds)

    num_train_dp = 0
    print("Accumulating stats without training")

    try:
      while True:
        inputs = iterator.get_next()

        # print(inputs['world_pos'])

        model.accumulate_stats(inputs)

        num_train_dp += 1
        if num_train_dp % 1000 == 0:
          print("Accumulated", num_train_dp)

    except tf.errors.OutOfRangeError:
      print("Accumulated stats. Total number of train points:", num_train_dp)
      tnext = timeit.default_timer()
      print("Time taken for accumulation", timeit.default_timer() - t0_accumulate)
      pass
  # '''
  # print(model.accumulate_stats.pretty_printed_concrete_signatures())  
  # quit()
  ########### Get distribution of all stresses seen during training
  '''
  iterator = iter(test_ds)

  traj_count = 0
  final_stresses = []
  try:
    while True:
      inputs = iterator.get_next()
      kk = tf.reduce_max(inputs["stress"][-1])
      print(traj_count, kk)
 
      final_stresses.append(kk)
      traj_count += 1

  except tf.errors.OutOfRangeError:
    print("This many trajectories seen", traj_count)

  print(max(final_stresses), np.argmax(final_stresses))
  import matplotlib.pyplot as plt 
  plt.plot(final_stresses)
  plt.show()

  quit()
  '''
  #####################

  step = 0

  # tf.profiler.experimental.start(FLAGS.checkpoint_dir)
  for i in range(FLAGS.num_epochs):
    iterator = iter(train_ds)

    train_losses = []


    try:
      t0 = timeit.default_timer()
      while True:
      # with tf.profiler.experimental.Trace('my_train_step', step_num=step, _r=1):
        inputs = iterator.get_next()
        step += 1

        train_loss = train_step(inputs)
        train_losses.append(train_loss)

        if step % 100 == 0:
          logging.info('Epoch %d, Step %d: Avg Loss %g', i, step, np.mean(train_losses))

    except tf.errors.OutOfRangeError:
      # with train_summary_writer.as_default():
        # tf.summary.scalar('train_loss', np.mean(train_losses), step=i)
      tnext = timeit.default_timer()
      print("------Time taken for epoch", i, ":", tnext - t0)


    # print(train_step.pretty_printed_concrete_signatures())  
    continue


    ################
    if i % FLAGS.num_epochs_per_validation == 0:
      print("Validating")

      iterator = iter(test_ds)

      test_losses, test_pos_mean_errors, test_pos_final_errors = [], [], []
      test_stress_mean_errors, test_stress_final_errors = [], []
      baseline_pos_mean_errors, baseline_pos_final_errors = [], []
      baseline_stress_mean_errors, baseline_stress_final_errors = [], []

      try:
        validation_counter = 0
        while True:
          inputs = iterator.get_next()

          validation_counter += 1
          test_loss, test_traj_data, test_scalar_data = params['evaluator'].evaluate(model, inputs, FLAGS, normalize=True) # Full trajectory. NORMALIZE SHOULD BE TRUE. 


          test_losses.append(test_loss)
          test_pos_mean_errors.append(test_scalar_data["pos_mean_error"])
          test_pos_final_errors.append(test_scalar_data["pos_final_error"])
          baseline_pos_mean_errors.append(test_scalar_data['baseline_pos_mean_error'])
          baseline_pos_final_errors.append(test_scalar_data['baseline_pos_final_error'])

          test_stress_mean_errors.append(test_scalar_data["stress_mean_error"])
          test_stress_final_errors.append(test_scalar_data["stress_final_error"])
          baseline_stress_mean_errors.append(test_scalar_data['baseline_stress_mean_error'])
          baseline_stress_final_errors.append(test_scalar_data['baseline_stress_final_error'])


      except tf.errors.OutOfRangeError:
        pass


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

    # Record losses in text file
    logging.info('Epoch %d, Step %d: Train Loss %g, Test Errors %g %g', i, step, np.mean(train_losses), np.mean(test_pos_mean_errors), np.mean(test_stress_mean_errors))

    if FLAGS.checkpoint_dir:
      file = open(os.path.join(FLAGS.checkpoint_dir, "losses.txt"), "a")
      log_line = "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n" % (step, np.mean(train_losses), np.mean(test_losses), np.mean(test_pos_mean_errors), \
        np.mean(test_pos_final_errors), np.mean(baseline_pos_mean_errors), np.mean(baseline_pos_final_errors), \
        np.mean(test_stress_mean_errors), np.mean(test_stress_final_errors), np.mean(baseline_stress_mean_errors), np.mean(baseline_stress_final_errors))

      file.write(log_line)
      file.close()

  logging.info('Training complete.')
  # tf.profiler.experimental.stop()



# Not yet migrated to TF 2
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

  # Restore checkpoint
  checkpoint = tf.train.Checkpoint(module=model)
  latest = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

  if latest is not None:
    print("Restoring previous checkpoint")
    checkpoint.restore(latest)


  # ds = dataset.load_dataset(FLAGS.dataset_dir, test_files, max(1, int(0.3 * FLAGS.num_objects)))
  # ds = get_flattened_dataset(ds, params)



  for ot, obj_file in enumerate(test_objects):
    print("===", obj_file, ot, "of", len(test_files))
    ds = dataset.load_dataset(FLAGS.dataset_dir, [obj_file], FLAGS.num_objects)
    ds = dataset.add_targets(ds, FLAGS, params['field'] if type(params['field']) is list else [params['field']], add_history=params['history'])
    iterator = iter(ds)





    traj_idx = 0

    # writer = tf.summary.FileWriter("output", sess.graph)


    # Sort predicted and final stresses
    actual_final_stresses = dict()
    pred_final_stresses = dict()

    actual_final_deformations = dict()
    pred_final_deformations = dict()
    for k in ['mean', 'max', 'median']:
      actual_final_stresses[k], pred_final_stresses[k] = [], []

      actual_final_deformations[k], pred_final_deformations[k] = [], []
      

    num_decrease, num_increase = 0, 0
    try:
      # while True:
      for traj_idx in range(20): # should be 48
        inputs = iterator.get_next()

        # if traj_idx < 200:
        #   continue



        # Calculate gradients
        calculate_gradients = False
        if calculate_gradients:
          optimizer = snt.optimizers.Adam(learning_rate=5e-4) #1e-7
          inputs = {k: v[20:21] for k, v in inputs.items()}
          inputs['tfn'] = tf.Variable(inputs['tfn'])


          def refine_step(inputs):
            with tf.GradientTape() as tape:
              tape.watch(inputs['tfn'])
              original_tfn_np = inputs['tfn'].numpy()
              original_tfn = inputs['tfn']

              direct_output, traj_data, scalar_data = params['evaluator'].evaluate(model, inputs, FLAGS) # Remove normalize field

            input_grad = -1. * tape.gradient(traj_data['mean_pred_stress'], inputs['tfn'])

            # Optimize over input
            optimizer.apply(input_grad, [inputs['tfn']])

            # Keep the euler part the same
            input_grad_np = input_grad.numpy()
            updated_input_np = inputs['tfn'].numpy()
            updated_input_np[:, :3, :] = original_tfn_np[:, :3, :]

            inputs['tfn'] = tf.Variable(tf.convert_to_tensor(updated_input_np))
            return traj_data['mean_pred_stress'], inputs

            # print("Original mean pred stress", traj_data['mean_pred_stress'].numpy())

          original_k = 0

          import matplotlib
          matplotlib.use('TkAgg')
          from matplotlib import animation
          import matplotlib.pyplot as plt 


          # Optimize over inputs using optimizer
          # '''
          refined_stresses = []
          world_pos_traj= []
          tfn_traj = []
          f_verts_traj = []
          for k in range(50):
            mean_pred_stress, inputs = refine_step(inputs)
            print(mean_pred_stress)
            refined_stresses.append(mean_pred_stress)
            world_pos_traj.append(inputs['world_pos'].numpy())
            tfn_traj.append(inputs['tfn'].numpy())

            # Infer gripper pos at first contact
            initial_state = {k: v[0] for k, v in inputs.items()}
            # print(initial_state['tfn'])
            initial_state['gripper_pos'] = deforming_plate_eval.gripper_pos_at_first_contact(initial_state, model.f_pc_original)
            g_pos = np.random.rand()
            f_verts, _ = deforming_plate_model.f_verts_at_pos(initial_state, initial_state['gripper_pos'][0], model.f_pc_original)
            f_verts_traj.append(f_verts.numpy())

          # '''
          fig2 = plt.figure()
          plt.plot(refined_stresses)
          plt.xlabel("Iteration step #")
          plt.ylabel("Predicted mean stress")
          # plt.show()

          fig = plt.figure(figsize=(8, 8))
          ax = fig.add_subplot(111, projection='3d')

          def animate(frame_num):
            ax.cla()

            world_pos = world_pos_traj[frame_num]
            f_verts_pos = f_verts_traj[frame_num]
            X, Y, Z = world_pos.T
            Xf, Yf, Zf = f_verts_pos.T
            ax.scatter(X, Y, Z)
            ax.scatter(Xf, Yf, Zf)
            return fig,



          # Plot the refined grasp pose + object
          _ = animation.FuncAnimation(fig, animate, frames=50, repeat=False)
          plt.show(block=True)
          # quit()
          continue



        else:
          direct_output, traj_data, scalar_data = params['evaluator'].evaluate(model, inputs, FLAGS)





        logging.info('Rollout trajectory %d', traj_idx)
        traj_idx += 1






        trajectories.append(traj_data)
        scalars.append(scalar_data)

        actual_final_stresses['mean'].append(np.mean(traj_data['gt_stress'][-1]))
        actual_final_stresses['max'].append(np.max(traj_data['gt_stress'][-1]))

        pred_final_stresses['mean'].append(np.mean(traj_data['pred_stress'][-1]))
        pred_final_stresses['max'].append(np.max(traj_data['pred_stress'][-1]))

        # print(pred_final_stresses['mean'][-1], actual_final_stresses['mean'][-1])
        # '''



        mean_a, max_a, median_a = utils.get_global_deformation_metrics(traj_data['gt_pos'][0].numpy(), traj_data['gt_pos'][-1].numpy())
        mean_p, max_p, median_p = utils.get_global_deformation_metrics(traj_data['pred_pos'][0].numpy(), traj_data['pred_pos'][-1].numpy())



        actual_final_deformations['mean'].append(mean_a)
        actual_final_deformations['max'].append(max_a)
        actual_final_deformations['median'].append(median_a)

        pred_final_deformations['mean'].append(mean_p)
        pred_final_deformations['max'].append(max_p)
        pred_final_deformations['median'].append(median_p)

        print("Final error", scalar_data['stress_error'][-1], scalar_data['rollout_losses'][-1])
        #print("Baseline final error", scalar_data['baseline_pos_final_error'])
        #print("Sum of gt final pos", np.sum(traj_data['gt_pos'][-1]))
        #print("Sum of pred final pos", np.sum(traj_data['pred_pos'][-1]))
        # quit()


        # '''

    except tf.errors.OutOfRangeError:
      pass

    # plt.show()

    '''
    num_total = num_increase + num_decrease

    print("Num increase", num_increase, num_increase/num_total)
    print("Num decrease", num_decrease, num_decrease/num_total)
    print("Total", num_total)
    quit()
    '''
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
        num_correct, num_total = utils.classification_accuracy_threshold(pred[m_type], actual[m_type], 50)
        # num_correct, num_total = utils.classification_accuracy_ranking(pred[m_type], actual[m_type], 50)

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
      num_correct, num_total = utils.classification_accuracy_threshold(all_pred[m_type], all_actual[m_type], 50)
      # num_correct, num_total = utils.classification_accuracy_ranking(all_pred[m_type], all_actual[m_type], 50)
      print(m_type, "Accuracy:", num_correct / num_total, "of", num_total)


    # for key in scalars[0]:
    #   logging.info('%s: %g', key, np.mean([x[key] for x in scalars]))

  with open(os.path.join(FLAGS.checkpoint_dir, 'rollout.pkl'), 'wb') as fp:
    pickle.dump(trajectories, fp)

###################3
class MyMLP(snt.Module):
  def __init__(self, name=None):
    super(MyMLP, self).__init__(name=name)
    self.hidden1 = snt.Linear(1024, name="hidden1")
    self.output = snt.Linear(10, name="output")

  def __call__(self, x):
    x = self.hidden1(x)
    x = tf.nn.relu(x)
    x = self.output(x)
    return x
  ##########################

def main(argv):
  del argv
  # tf.enable_resource_variables()

  # tf.config.optimizer.set_jit("autoclustering")
  tf.config.run_functions_eagerly(FLAGS.eager)


  global LEN_TRAJ
  utils.check_consistencies(FLAGS)

  if utils.using_dm_dataset(FLAGS):
    LEN_TRAJ = 400 - 2
  else:
    LEN_TRAJ = 50 - 2

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


  # learned_model = core_model.SimplifiedNetwork() # for mlp

  #### Set up mixed precision
  # support_modes = snt.mixed_precision.modes([tf.float32, tf.float16])
  # core_model.EncodeProcessDecode.__call__ = support_modes(core_model.EncodeProcessDecode.__call__)
  # snt.mixed_precision.enable(tf.float16)




  learned_model = core_model.EncodeProcessDecode(
      output_size=output_size,
      latent_size=FLAGS.latent_size,
      num_layers=FLAGS.num_layers,
      message_passing_steps=FLAGS.message_passing_steps)


  model = params['model'].Model(learned_model, FLAGS)


  # Try to make summary writer to visualize graph on tensorboard

  # @tf.function 
  # def my_fun(x):
  #   return 2 * x

  # a = tf.constant(10, tf.float32)
  # writer = tf.summary.create_file_writer(FLAGS.checkpoint_dir)
  # with writer.as_default():
  #   tf.summary.graph(learned_model.__call__.get_concrete_function().graph)
  #   # my_fun_graph = my_fun.get_concrete_function(a).graph
  #   # tf.summary.graph(my_fun_graph)



  if FLAGS.mode == 'train':
    learner(model, params)
  elif FLAGS.mode == 'eval':
    evaluator(model, params)





if __name__ == '__main__':



  app.run(main)
