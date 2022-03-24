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
"""Core learned graph net model."""
import os

import collections
import functools
import sonnet as snt
# import tensorflow.compat.v1 as tf
import tensorflow as tf

from tfdeterminism import patch
# patch()
SEED = 55
os.environ['PYTHONHASHSEED'] = str(SEED)
# random.seed(SEED)
tf.random.set_seed(SEED)



EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])
from meshgraphnets import deforming_plate_model
from meshgraphnets import common

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# class GraphNetBlock(snt.AbstractModule):
class GraphNetBlock(snt.Module):

  """Multi-Edge Interaction Network with residual connections."""

  def __init__(self, model_fn, name='GraphNetBlock'):
    # super(self).__init__(name=name)
    super().__init__(name=name)
    self._model_fn = model_fn

  @snt.once
  def _initialize(self):
    self._update_edge_features_fn = self._model_fn()
    self._update_node_features_fn = self._model_fn()

  def _update_edge_features(self, node_features, edge_set):
    """Aggregrates node features, and applies edge function."""
    sender_features = tf.gather(node_features, edge_set.senders)
    receiver_features = tf.gather(node_features, edge_set.receivers)
    features = [sender_features, receiver_features, edge_set.features]
    # with tf.variable_scope(edge_set.name+'_edge_fn'):
    # return self._model_fn()(tf.concat(features, axis=-1))
    return self._update_edge_features_fn(tf.concat(features, axis=-1))

  def _update_node_features(self, node_features, edge_sets):
    """Aggregrates edge features, and applies node function."""
    num_nodes = tf.shape(node_features)[0]
    features = [node_features]
    for edge_set in edge_sets:
      features.append(tf.math.unsorted_segment_sum(edge_set.features,
                                                   edge_set.receivers,
                                                   num_nodes))
    # with tf.variable_scope('node_fn'):
    # return self._model_fn()(tf.concat(features, axis=-1))
    return self._update_node_features_fn(tf.concat(features, axis=-1))


  # @tf.function
  def __call__(self, graph):
  # def _build(self, graph):
    """Applies GraphNetBlock and returns updated MultiGraph."""
    self._initialize()

    # apply edge functions
    new_edge_sets = []
    for edge_set in graph.edge_sets: # There are two, world_edges and mesh_edges
      updated_features = self._update_edge_features(graph.node_features,
                                                    edge_set)
      new_edge_sets.append(edge_set._replace(features=updated_features))

    # apply node function
    new_node_features = self._update_node_features(graph.node_features,
                                                   new_edge_sets)

    # add residual connections
    new_node_features += graph.node_features
    new_edge_sets = [es._replace(features=es.features + old_es.features)
                     for es, old_es in zip(new_edge_sets, graph.edge_sets)]
    return MultiGraph(new_node_features, new_edge_sets)


class MyMLP():
  def __init__(self, output_sizes, activation=tf.nn.relu, activate_final=False):

    self.model = Sequential()
    num_layers = len(output_sizes)
    for index, output_size in enumerate(output_sizes):
      if (index < num_layers - 1) or activate_final:
        self.model.add(Dense(output_size, activation=activation))
      else:
        self.model.add(Dense(output_size))

  def __call__(self, inputs):
    return self.model(inputs)

# class EncodeProcessDecode(snt.AbstractModule):
class EncodeProcessDecode(snt.Module):
  """Encode-Process-Decode GraphNet model."""

  def __init__(self,
               output_size,
               latent_size,
               num_layers,
               message_passing_steps,
               name='EncodeProcessDecode'):
    super().__init__(name=name)

    self._latent_size = latent_size
    self._output_size = output_size
    self._num_layers = num_layers
    self._message_passing_steps = message_passing_steps

  
  @snt.once # this might be necessary to fix what used to be controlled by variable scopes
  def _initialize(self, graph):
    '''Initialize all the MLPs'''
    self._decoder_fn =  self._make_mlp(self._output_size, layer_norm=False)
    self._node_latents_fn = self._make_mlp(self._latent_size)

    self._edge_latent_fns = []
    for k in range(len(graph.edge_sets)):
      self._edge_latent_fns.append(self._make_mlp(self._latent_size))

    model_fn = functools.partial(self._make_mlp, output_size=self._latent_size)
    self.gnb = GraphNetBlock(model_fn)


  def _make_mlp(self, output_size, layer_norm=False): ## For gradient debugging, set layer_norm=False. It's typically = True
    """Builds an MLP."""
    widths = [self._latent_size] * self._num_layers + [output_size]
    network = snt.nets.MLP(widths, activate_final=False)
    # network = MyMLP(widths, activate_final=False)

    if layer_norm:
      network = snt.Sequential([network, snt.LayerNorm()])
    return network

  def _encoder(self, graph):
    """Encodes node and edge features into latent features."""
    # with tf.variable_scope('encoder'):
    # node_latents = self._make_mlp(self._latent_size)(graph.node_features)
    node_latents = self._node_latents_fn(graph.node_features)

    new_edges_sets = []
    for edge_latent_fn, edge_set in zip(self._edge_latent_fns, graph.edge_sets):  # There are two, world_edges and mesh_edges
      # latent = self._make_mlp(self._latent_size)(edge_set.features)
      latent = edge_latent_fn(edge_set.features)

      new_edges_sets.append(edge_set._replace(features=latent))

    return MultiGraph(node_latents, new_edges_sets)

  def _decoder(self, graph):
    """Decodes node features from graph."""
    '''
    with tf.variable_scope('decoder'):
      decoder = self._make_mlp(self._output_size, layer_norm=False)
      return decoder(graph.node_features)
    '''
    return self._decoder_fn(graph.node_features)

  # @tf.function
  def __call__(self, graph):
  # def _build(self, graph):
    """Encodes and processes a multigraph, and returns node features."""
    self._initialize(graph)

    # model_fn = functools.partial(self._make_mlp, output_size=self._latent_size)
    latent_graph = self._encoder(graph)

    # self.gnb = GraphNetBlock(model_fn)
    for _ in range(self._message_passing_steps):
      # latent_graph = GraphNetBlock(model_fn)(latent_graph)

      latent_graph = self.gnb(latent_graph)
      # latent_graph = GraphNetBlock(self.model_fn)(latent_graph)
      # latent_graph = self._GNB(latent_graph)
    return self._decoder(latent_graph) # returns per-node output



'''
class SimplifiedNetwork(snt.AbstractModule):

  def __init__(self,
               name='SimplifiedNetwork'):
    super(SimplifiedNetwork, self).__init__(name=name)
    self._latent_size = 2
    self._num_layers = 10

  def _make_mlp(self, output_size, layer_norm=False):
    """Builds an MLP."""
    # network = snt.nets.MLP([8, 16, 32, 8, 1], activate_final=False) # Used when input was just grippe rpos, force, and tfn
    network = snt.nets.MLP([2000, 500, 30, 1], activate_final=False)
    if layer_norm:
      network = snt.Sequential([network, snt.LayerNorm()])
    return network
    # self.hidden1 = snt.Linear(8)
    # self.hidden2 = snt.Linear(16)
    # self.hidden3 = snt.Linear(32)
    # self.hidden4


  def _build(self, inputs):
    simplified_mlp = self._make_mlp(1)
    f_verts, _, _, _ = deforming_plate_model.gripper_world_pos(inputs)
    f_verts_flattened = tf.reshape(f_verts, [-1, 1])

    num_f_verts_total = 1180# tf.shape(f_verts)[0]
    num_verts_total = tf.shape(inputs['mesh_pos'])[0]
    pad_diff = num_verts_total - num_f_verts_total
    paddings = [[0, pad_diff], [0, 0]]
    finger_world_pos = tf.pad(f_verts, paddings, "CONSTANT")
    actuator_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.OBSTACLE)
    world_pos = tf.where(actuator_mask, finger_world_pos, inputs['world_pos']) # This should be equal to inputs['world_pos'] anyway
    world_pos_flattened = tf.reshape(world_pos, [-1, 1])


    # mlp_inputs = tf.transpose(tf.concat([inputs['gripper_pos'], inputs['force'], inputs['tfn']], axis=0))
    # mlp_inputs = tf.ensure_shape(mlp_inputs, [1, 1 + 1 + 6])


    mlp_inputs = tf.transpose(tf.concat([inputs['force'], world_pos_flattened], axis=0))
    mlp_inputs = tf.ensure_shape(mlp_inputs, [1, 1 + 2292*3])

    return simplified_mlp(mlp_inputs)
'''