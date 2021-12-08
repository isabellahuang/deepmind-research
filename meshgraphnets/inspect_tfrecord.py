import tensorflow.compat.v1 as tf
import numpy as np
import os
import json
import functools
import enum
import matplotlib.pyplot as plt
import pickle
import trimesh
import sys
sys.path.append("../..")
from meshgraphnets import dataset
import tensorflow_graphics.geometry.transformation as tfg_transformation
from tensorflow_graphics.geometry.transformation import rotation_matrix_common
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import tfg_flags
tf.enable_eager_execution()

class NodeType(enum.IntEnum):
  NORMAL = 0
  OBSTACLE = 1
  AIRFOIL = 2
  HANDLE = 3
  INFLOW = 4
  OUTFLOW = 5
  WALL_BOUNDARY = 6
  SIZE = 9

# for str_rec in tf.python_io.tf_record_iterator('test.tfrecord'):
#     example = tf.train.Example()
#     example.ParseFromString(str_rec)
#     print(sorted(dict(example.features.feature).keys()))
#     string = example.features.feature['node_type'].bytes_list.value[0]
    # output = np.fromstring(string, 'int')

# Dict keys ['cells', 'mesh_pos', 'node_type', 'stress', 'world_pos']


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


  out['tfn'] = tf.transpose(out['tfn'], [0, 2, 1])
  num_nodes = tf.shape(out['stress'])[1]
  for tile_key in ['gripper_pos', 'force', 'tfn']:
    out[tile_key] = tf.tile(out[tile_key], [1, num_nodes, 1])

  f_size = 590
  f1_sign = tf.ones([meta['trajectory_length'], f_size, 1]) * -1
  f2_sign = tf.ones([meta['trajectory_length'], f_size, 1])
  paddings = [[0, 0], [0, num_nodes - 2*f_size], [0, 0]]
  
  out['f_sign'] = tf.pad(tf.concat([f1_sign, f2_sign], axis=1), paddings, "CONSTANT")


  finger1_path = os.path.join('..', 'assets', 'finger1_face_uniform' + '.stl')
  f1_trimesh = trimesh.load_mesh(finger1_path)
  f1_verts_original = tf.constant(f1_trimesh.vertices, dtype=tf.float32)

  finger2_path = os.path.join('..', 'assets', 'finger2_face_uniform' + '.stl')
  f2_trimesh = trimesh.load_mesh(finger2_path)
  f2_verts_original = tf.constant(f2_trimesh.vertices, dtype=tf.float32)
  f_pc_original = tf.concat((f1_verts_original, f2_verts_original), axis=0)
  f_pc_original_pad = tf.pad(f_pc_original, paddings[1:], "CONSTANT")

  f_pc_original_expand = tf.expand_dims(f_pc_original_pad, axis=0)
  f_pc_original_tile = tf.tile(f_pc_original_expand, [meta['trajectory_length'], 1, 1])

  # Rotate and translate f_pc original
  # euler, trans = tf.split(out['tfn'], [3,3], axis=-1) 
  # tf_from_euler = tfg_transformation.rotation_matrix_3d.from_euler(euler)
  # f_pc = rotate(f_pc_original_tile, tf_from_euler)# + trans #()
  # out['tf_from_euler'] = tf_from_euler
  # out['f_pc_original'] = f_pc_original_tile
  # out['f_pc'] = f_pc
  return out


def rotate(point,
           matrix,
           name="rotation_matrix_3d_rotate"):
  """Rotate a point using a rotation matrix 3d.
  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.
  Args:
    point: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a 3d point.
    matrix: A tensor of shape `[A1, ..., An, 3,3]`, where the last dimension
      represents a 3d rotation matrix.
    name: A name for this op that defaults to "rotation_matrix_3d_rotate".
  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    a 3d point.
  Raises:
    ValueError: If the shape of `point` or `rotation_matrix_3d` is not
    supported.
  """
  with tf.name_scope(name):
    point = tf.convert_to_tensor(value=point)
    matrix = tf.convert_to_tensor(value=matrix)

    shape.check_static(
        tensor=point, tensor_name="point", has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=matrix,
        tensor_name="matrix",
        has_rank_greater_than=1,
        has_dim_equals=((-2, 3), (-1, 3)))
    shape.compare_batch_dimensions(
        tensors=(point, matrix),
        tensor_names=("point", "matrix"),
        last_axes=(-2, -3),
        broadcast_compatible=True)

    point = tf.expand_dims(point, axis=-1)
    print(point.shape)
    print(matrix.shape)

    common_batch_shape = shape.get_broadcasted_shape(point.shape[:-2],
                                                     matrix.shape[:-2])


    def dim_value(dim):
      return -1 if dim is None else tf.compat.dimension_value(dim)

    common_batch_shape = [dim_value(dim) for dim in common_batch_shape]

    point = tf.broadcast_to(point, common_batch_shape + [3, 1])
    matrix = tf.broadcast_to(matrix, common_batch_shape + [3, 3])
    rotated_point = tf.matmul(matrix, point)
    return tf.squeeze(rotated_point, axis=-1)


def load_dataset(path, split):
  """Load dataset."""
  with open(os.path.join(path, 'meta_pb.json'), 'r') as fp:
    meta = json.loads(fp.read())

  ds = tf.data.TFRecordDataset(os.path.join(path, split+'.tfrecord'))
  ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
  ds = ds.prefetch(1)
  return ds

def add_targets(ds, fields, add_history):
  """Adds target and optionally history fields to dataframe."""
  def fn(trajectory):
    out = {}
    for key, val in trajectory.items():
      print("`````",key, val.shape)
      out[key] = val[1:-1]
      print(val)
      print(val[1:-1])
      if key in fields:
        if add_history:
          out['prev|'+key] = val[0:-2]
        out['target|'+key] = val[2:]
    print(out.keys())

    return out
  return ds.map(fn, num_parallel_calls=8)

def split_and_preprocess(ds, noise_field, noise_scale, noise_gamma):
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
  ds = ds.shuffle(10000) # This is the sie of the shuffle buffer

  # ds = ds.repeat(None)


  return ds.prefetch(10)


# if __name__ == "__main__":
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

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
  return senders, receivers 

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  # return tf.train.Feature(bytes_list=tf.train.BytesList(value=value.flatten()))


def squared_dist(A, B):
  row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
  row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

  row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
  row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

  return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B

def squared_dist_point(point, others, thresh):
    dists = tf.reduce_sum(tf.square(point - others), axis=1)
    sq_thresh = thresh ** 2
    return tf.where(tf.math.less(dists, sq_thresh))

def construct_world_edges(world_pos, node_type, world_edges, stress):
  print("World pos shape", world_pos.shape)

  g = 0
  indices = tf.range(start=0, limit=tf.shape(world_pos)[0], dtype=tf.int32)
  shuffled_indices = tf.random.shuffle(indices)

  # world_pos = tf.gather(world_pos, shuffled_indices)
  # node_type = tf.gather(node_type, shuffled_indices)

  deformable_idx = tf.where(tf.not_equal(node_type[:, 0], NodeType.OBSTACLE))
  actuator_idx = tf.where(tf.equal(node_type[:, 0], NodeType.OBSTACLE))




  B = tf.squeeze(tf.gather(world_pos, deformable_idx))
  A = tf.squeeze(tf.gather(world_pos, actuator_idx))
  thresh = 0.005
  thresh = 0.003

  '''
  dists = squared_dist(A, B)

  r = tf.reduce_sum(world_pos*world_pos, 1)
  r = tf.reshape(r, [-1, 1])
  dists = r - 2*tf.matmul(world_pos, tf.transpose(world_pos)) + tf.transpose(r)


  rel_close_pair_idx = tf.where(tf.math.less(dists, thresh ** 2))


  total_senders = rel_close_pair_idx[:,0]
  total_receivers = rel_close_pair_idx[:,1]

  node_types_of_pairs = tf.gather(tf.squeeze(node_type), rel_close_pair_idx)
  node_pairs_difference = tf.math.abs(tf.subtract(node_types_of_pairs[:,0], node_types_of_pairs[:,1]))
  different_type_connections = tf.where(tf.not_equal(node_pairs_difference, 0))
  senders = tf.gather(total_senders, different_type_connections)
  receivers = tf.gather(total_receivers, different_type_connections)
  print(senders[:,0].shape)
  close_pair_idx = tf.concat([senders, receivers], 1)
  '''

  # dists_bool = tf.map_fn(lambda x: squared_dist_point(x, A, thresh), B)
  # rel_close_pair_idx = tf.where(dists_bool)
  # close_pair_actuator = tf.gather(actuator_idx, rel_close_pair_idx[:,1])
  # close_pair_def = tf.gather(deformable_idx, rel_close_pair_idx[:,0])
  # close_pair_idx = tf.concat([close_pair_actuator, close_pair_def], 1)
  # senders, receivers = close_pair_actuator, close_pair_def

  '''
  dists = squared_dist(A, B)

  rel_close_pair_idx = tf.where(tf.math.less(dists, thresh ** 2))

  close_pair_actuator = tf.gather(actuator_idx, rel_close_pair_idx[:,0])
  close_pair_def = tf.gather(deformable_idx, rel_close_pair_idx[:,1])
  close_pair_idx = tf.concat([close_pair_actuator, close_pair_def], 1)
  senders, receivers = close_pair_actuator, close_pair_def
  print(senders.shape)
  '''
  # quit()

  ### Try with while loop

  num_B = tf.shape(B)[0]

  def body(i, outputs):
    random_vec = tf.random.uniform(shape=[1, 3])
    b = tf.gather(B, [i])
    # print(b.shape)
    a = squared_dist_point(b, A, thresh)
    num_matches = tf.shape(a)[0]
    bs = a * 0 + tf.cast(i, dtype=tf.int64)
    # bs = tf.repeat(tf.cast([i], dtype=tf.int64), num_matches)
    b_pairs = tf.concat([bs, a], axis=1)
    outputs = outputs.write(i, b_pairs)
    i += 1
    return i, outputs

  outputs = tf.TensorArray(dtype=tf.int64, infer_shape=False, size=1, element_shape=[None, 2], dynamic_size=True)
  _, outputs = tf.while_loop(lambda i, *_: tf.less(i, num_B), body, [0, outputs])
  outputs = outputs.concat()

  close_pair_def = tf.gather(deformable_idx, outputs[:,0])
  close_pair_actuator = tf.gather(actuator_idx, outputs[:,1])
  close_pair_idx = tf.concat([close_pair_actuator, close_pair_def], 1)
  senders, receivers = tf.unstack(close_pair_idx, 2, axis=1)

  # quit()

  ###### If world edges are written in the tfrecord
  # '''
  world_edges_difference = tf.math.subtract(world_edges[:,0], world_edges[:,1])
  unique_edges = tf.where(tf.not_equal(world_edges_difference, 0))
  close_pair_idx = tf.gather(world_edges, unique_edges[:,0], axis=0)
  print("Close pair idx", close_pair_idx.shape)
  # '''


  ###################


  # '''
  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111, projection='3d')

  X, Y, Z = world_pos[:,0], world_pos[:,1], world_pos[:,2]

  max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

  mid_x = (X.max()+X.min()) * 0.5
  mid_y = (Y.max()+Y.min()) * 0.5
  mid_z = (Z.max()+Z.min()) * 0.5
  ax.set_xlim(mid_x - max_range, mid_x + max_range)
  ax.set_ylim(mid_y - max_range, mid_y + max_range)
  ax.set_zlim(mid_z - max_range, mid_z + max_range)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')

  # ax.plot_trisurf(X, Y, Z, triangles=dcu.get_surface_tris(dcu.tets_to_tris(total_tets)))
  arg_neg = np.argwhere(stress < 0)
  clipped_stress = np.clip(stress, a_min=0, a_max=None)
  ax.scatter(X, Y, Z, c=clipped_stress)
  ax.scatter(X[arg_neg], Y[arg_neg], Z[arg_neg], c='r', s=500)
  for p in close_pair_idx.numpy():
    ax.plot([X[p[0]], X[p[1]]], [Y[p[0]], Y[p[1]]], [Z[p[0]], Z[p[1]]], 'ro-')

  plt.show()
  # '''
  quit()
  

# Loading a DS
# '''

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
    print("====== about to batch")
    for key, ds_val in ds_window.items():
      print(key, types[key], shapes[key])
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



inspect_data = True
save_traj = True
if inspect_data:

    # ds = load_dataset('../abc/0-9999', '100-199_tfrecords/00000111')
    # ds = load_dataset('../abc/0-9999', 'debug_tfrecords/1e9_00000004coarse')
    # ds = load_dataset('abc/0-9999', 'train_subset/00000018')
    files_folder = '../../../../results_mgn_fresh/0-9999/0-99_tfrecords/pd'
    # print(os.listdir(files_folder))
    tfrecord_files = [os.path.join('0-99_tfrecords/pd', k.split(".")[0]) for k in os.listdir(files_folder)]
    for tfrecord_file in tfrecord_files[:]:
      # if "00000012" not in tfrecord_file:
        # continue

      ds = load_dataset('../../../../results_mgn_fresh/0-9999', tfrecord_file)
      # ds = dataset.add_targets(ds, ['world_pos', 'gripper_pos', 'force', 'stress'], add_history=False)
      # ds = split_and_preprocess(ds, noise_field='world_pos',
                                        # noise_scale=3e-3,
                                        # noise_gamma=0.1)
      # ds = ds.take(100)
      # ds = batch_dataset(ds, 3)
      # ds = ds.repeat(None)

      print("Inspecting data------\n", tfrecord_file)
      counter = 0
      # print(ds)

      with open(os.path.join('../../../../results_mgn_fresh/0-9999', 'meta_pb.json'), 'r') as fp:
          meta = json.loads(fp.read())
      examples = []
      for example in ds:

          if save_traj:
              examples.append(example)
          counter += 1
              
          '''
          print("-----")
          for k in example.keys():
            print(k, example[k].shape)
          '''
          # print(example['stress'].shape, example['mesh_edges'].shape, example['world_edges'].shape, np.max(example['world_edges']))
          print(example['node_type'].shape, example['pd_stress'].shape)
          print(np.unique(example['node_type']))

          # euler, trans = tf.split(example['tfn'], [3,3], axis=-1) 
          # print(euler.shape, trans.shape)
          # print(example['force'])
          # tf_from_euler = tfg_transformation.rotation_matrix_3d.from_euler(euler)
          # f_pc = tfg_transformation.rotation_matrix_3d.rotate(example['f_original'], tf_from_euler) 
          # print(euler.shape, tf_from_euler.shape, f_pc.shape)




          # construct_world_edges(example['world_pos'].numpy(), example["node_type"], example['mesh_edges'], example["stress"])
      print(counter)
      if save_traj:
          with open('debug.pkl', 'wb') as fp:
              pickle.dump(examples, fp, protocol=pickle.HIGHEST_PROTOCOL)
 
# '''