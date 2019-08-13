# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from nn_skeleton import ModelSkeleton

class SqueezeDet(ModelSkeleton):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      self._add_forward_graph()
      self._add_interpretation_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()

  def _add_forward_graph(self):
    """NN architecture."""

    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

    conv1 = self._conv_layer(
        'conv1', self.image_input, filters=64, size=3, stride=2,
        padding='SAME', freeze=True)
    pool1 = self._pooling_layer(
        'pool1', conv1, size=3, stride=2, padding='SAME')

    with tf.variable_scope('fire23') as scope:
        with tf.variable_scope('fire2'):
            sq1x1 = self._conv_layer(
                'fire2'+'/squeeze1x1', pool1, filters=16, size=1, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            ex1x1 = self._conv_layer(
                'fire2'+'/expand1x1', sq1x1, filters=64, size=1, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            ex3x3 = self._conv_layer(
                'fire2'+'/expand3x3', sq1x1, filters=64, size=3, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            fire2 = tf.concat([ex1x1, ex3x3], 3, name='fire2'+'/concat')
        with tf.variable_scope('fire3'):
            sq1x1 = self._conv_layer(
                'fire3'+'/squeeze1x1', fire2, filters=16, size=1, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            ex1x1 = self._conv_layer(
                'fire3'+'/expand1x1', sq1x1, filters=64, size=1, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            ex3x3 = self._conv_layer(
                'fire3'+'/expand3x3', sq1x1, filters=64, size=3, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            fire3 = tf.concat([ex1x1, ex3x3], 3, name='fire3'+'/concat')
        fire3 = tf.nn.relu(fire2 + fire3, 'relu')
        pool3 = self._pooling_layer(
        'pool3', fire3, size=3, stride=2, padding='SAME')

    with tf.variable_scope('fire45') as scope:
        with tf.variable_scope('fire4'):
            sq1x1 = self._conv_layer(
                'fire4'+'/squeeze1x1', pool3, filters=32, size=1, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            ex1x1 = self._conv_layer(
                'fire4'+'/expand1x1', sq1x1, filters=128, size=1, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            ex3x3 = self._conv_layer(
                'fire4'+'/expand3x3', sq1x1, filters=128, size=3, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            fire4 = tf.concat([ex1x1, ex3x3], 3, name='fire4'+'/concat')
        with tf.variable_scope('fire5'):
            sq1x1 = self._conv_layer(
                'fire5'+'/squeeze1x1', fire4, filters=32, size=1, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            ex1x1 = self._conv_layer(
                'fire5'+'/expand1x1', sq1x1, filters=128, size=1, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            ex3x3 = self._conv_layer(
                'fire5'+'/expand3x3', sq1x1, filters=128, size=3, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            fire5 = tf.concat([ex1x1, ex3x3], 3, name='fire5'+'/concat')
        fire5 = tf.nn.relu(fire4 + fire5, 'relu')
        pool5 = self._pooling_layer(
        'pool5', fire5, size=3, stride=2, padding='SAME')

    with tf.variable_scope('fire678910') as scope:
        with tf.variable_scope('fire6'):
            sq1x1 = self._conv_layer(
                'fire6'+'/squeeze1x1', pool5, filters=48, size=1, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            ex1x1 = self._conv_layer(
                'fire6'+'/expand1x1', sq1x1, filters=192, size=1, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            ex3x3 = self._conv_layer(
                'fire6'+'/expand3x3', sq1x1, filters=192, size=3, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            fire6 = tf.concat([ex1x1, ex3x3], 3, name='fire6'+'/concat')
        with tf.variable_scope('fire7'):
            sq1x1 = self._conv_layer(
                'fire7'+'/squeeze1x1', fire6, filters=48, size=1, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            ex1x1 = self._conv_layer(
                'fire7'+'/expand1x1', sq1x1, filters=192, size=1, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            ex3x3 = self._conv_layer(
                'fire7'+'/expand3x3', sq1x1, filters=192, size=3, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            fire7 = tf.concat([ex1x1, ex3x3], 3, name='fire7'+'/concat')
            fire7 = tf.concat([fire6,fire7],3)
        with tf.variable_scope('fire8'):
            sq1x1 = self._conv_layer(
                'fire8'+'/squeeze1x1', fire7, filters=64, size=1, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            ex1x1 = self._conv_layer(
                'fire8'+'/expand1x1', sq1x1, filters=256, size=1, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            ex3x3 = self._conv_layer(
                'fire8'+'/expand3x3', sq1x1, filters=256, size=3, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            fire8 = tf.concat([ex1x1, ex3x3], 3, name='fire8'+'/concat')
            fire8 = tf.concat([fire6,fire7,fire8],3)
        with tf.variable_scope('fire9'):
            sq1x1 = self._conv_layer(
                'fire9'+'/squeeze1x1', fire8, filters=64, size=1, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            ex1x1 = self._conv_layer(
                'fire9'+'/expand1x1', sq1x1, filters=256, size=1, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            ex3x3 = self._conv_layer(
                'fire9'+'/expand3x3', sq1x1, filters=256, size=3, stride=1,
                padding='SAME', stddev=0.01, freeze=False)
            fire9 = tf.concat([ex1x1, ex3x3], 3, name='fire9'+'/concat')
            fire9 = tf.concat([fire6,fire7,fire8,fire9],3)

    fire10 = self._fire_layer(
        'fire10', fire9, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    fire11 = self._fire_layer(
        'fire11', fire10, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    dropout11 = tf.nn.dropout(fire11, self.keep_prob, name='drop11')

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    self.preds = self._conv_layer(
        'conv12', dropout11, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001)

  def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01,
      freeze=False):
    """Fire layer constructor.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """

    sq1x1 = self._conv_layer(
        layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex1x1 = self._conv_layer(
        layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex3x3 = self._conv_layer(
        layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)

    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')
