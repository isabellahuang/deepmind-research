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
"""Utility functions for reading the datasets."""

import functools
import json
import os
import trimesh
import tensorflow.compat.v1 as tf

from meshgraphnets.common import NodeType


def _parse(proto, meta):
  """Parses a trajectory from tf.Example."""
  feature_lists = {k: tf.io.VarLenFeature(tf.string)
                   for k in meta['field_names']}
  features = tf.io.parse_single_example(proto, feature_lists)
  out = {}
  for key, field in meta['features'].items():
    print("=====", key, field)
    data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
    data = tf.reshape(data, field['shape'])
    if field['type'] == 'static':
      data = tf.tile(data, [meta['trajectory_length'], 1, 1])

    elif field['type'] == 'dynamic_varlen':
      length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
      length = tf.reshape(length, [-1])
      data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
    elif field['type'] != 'dynamic':
      raise ValueError('invalid data format')

    out[key] = data

  '''
  out['tfn'] = tf.transpose(out['tfn'], [0, 2, 1])
  num_nodes = tf.shape(out['stress'])[1]
  for tile_key in ['gripper_pos', 'force', 'tfn']:
    out[tile_key] = tf.tile(out[tile_key], [1, num_nodes, 1])

  f_size = 590
  f1_sign = tf.ones([meta['trajectory_length'], f_size, 1]) * -1
  f2_sign = tf.ones([meta['trajectory_length'], f_size, 1])
  paddings = [[0, 0], [0, num_nodes - 2*f_size], [0, 0]]
  
  out['f_sign'] = tf.pad(tf.concat([f1_sign, f2_sign], axis=1), paddings, "CONSTANT")

  finger1_path = os.path.join('meshgraphnets', 'assets', 'finger1_face_uniform' + '.stl')
  f1_trimesh = trimesh.load_mesh(finger1_path)
  f1_verts_original = tf.constant(f1_trimesh.vertices, dtype=tf.float32)

  finger2_path = os.path.join('meshgraphnets', 'assets', 'finger2_face_uniform' + '.stl')
  f2_trimesh = trimesh.load_mesh(finger2_path)
  f2_verts_original = tf.constant(f2_trimesh.vertices, dtype=tf.float32)
  f_pc_original = tf.concat((f1_verts_original, f2_verts_original), axis=0)
  f_pc_original_pad = tf.pad(f_pc_original, paddings[1:], "CONSTANT")

  f_pc_original_expand = tf.expand_dims(f_pc_original_pad, axis=0)
  f_pc_original_tile = tf.tile(f_pc_original_expand, [meta['trajectory_length'], 1, 1])
  out['f_original'] = f_pc_original_tile
  '''
  return out


def load_dataset(path, split, num_objects):
  """Load dataset. Path contains all the .tfrecord files, split is a list of paths to .tfrecords"""


  # Search for meta.json file 
  meta_path = path 
  meta_path_search_counter = 0
  while 'meta.json' not in os.listdir(meta_path):
    meta_path_search_counter += 1 
    if meta_path_search_counter > 5:
      print("No meta.json file found from", path)
      quit()
    meta_path = os.path.join(meta_path, '..')

  with open(os.path.join(meta_path, 'meta.json'), 'r') as fp:
    meta = json.loads(fp.read())


  assert(type(split) is list)

  tfrecords_files = sorted(split)
  print("Tfrecords_files", tfrecords_files)
  ds = tf.data.TFRecordDataset(tfrecords_files)

  ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
  ds = ds.prefetch(1)
  return ds

def add_targets(ds, fields, add_history):
  """Adds target and optionally history fields to dataframe."""
  def fn(trajectory):
    out = {}
    for key, val in trajectory.items():
      if "stress" in key:
        val = tf.nn.relu(val)
      out[key] = val[1:-1]
      if not add_history:
        out[key] = val[1:-1] # either [:-1] for full traj or [1:-1]
      if key in fields:
        if add_history:
          out['prev|'+key] = val[0:-2]
          out['target|'+key] = val[2:]
        else:
          out['target|'+key] = val[2:] # Either [1:] for full traj or [2:]
    return out
  return ds.map(fn, num_parallel_calls=8)


def split_and_preprocess(ds, num_epochs, noise_field, noise_scale, noise_gamma):
  """Splits trajectories into frames, and adds training noise."""
  def add_noise(frame):
    noise = tf.random.normal(tf.shape(frame[noise_field]),
                             stddev=noise_scale, dtype=tf.float32)
    # don't apply noise to boundary nodes
    mask = tf.equal(frame['node_type'], NodeType.NORMAL)[:, 0]
    noise = tf.where(mask, noise, tf.zeros_like(noise))
    frame[noise_field] += noise
    # frame['target|'+noise_field] += (1.0 - noise_gamma) * noise
    frame['target|'+noise_field] += noise
    return frame

  ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
  ds = ds.map(add_noise, num_parallel_calls=8)

  ds = ds.repeat(num_epochs)
  ds = ds.shuffle(5000) # This is the size of the shuffle buffer


  return ds.prefetch(10)


def batch_dataset(ds, batch_size):
  """Batches input datasets."""
  shapes = ds.output_shapes
  types = ds.output_types

  def renumber(buffer, frame):
    nodes, cells = buffer
    new_nodes, new_cells = frame
    return nodes + new_nodes, tf.concat([cells, new_cells+nodes], axis=0)

  def batch_accumulate(ds_window):
    out = {}
    for key, ds_val in ds_window.items():
      initial = tf.zeros((0, shapes[key][1]), dtype=types[key])
      if key in ['cells', 'mesh_edges', 'world_edges']:
        # renumber node indices in cells
        num_nodes = ds_window['node_type'].map(lambda x: tf.shape(x)[0])
        cells = tf.data.Dataset.zip((num_nodes, ds_val))
        initial = (tf.constant(0, tf.int32), initial)
        _, out[key] = cells.reduce(initial, renumber)
      else:
        merge = lambda prev, cur: tf.concat([prev, cur], axis=0)
        out[key] = ds_val.reduce(initial, merge)
    return out
  if batch_size > 1:
    ds = ds.window(batch_size, drop_remainder=True)
    ds = ds.map(batch_accumulate, num_parallel_calls=8)
  return ds