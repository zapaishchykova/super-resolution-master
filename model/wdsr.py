from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_addons as tfa
#import tensorflow as tf
from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model

import warnings
import tensorflow.compat.v2 as tf
from model.common import normalize, denormalize, pixel_shuffle


def wdsr_a(scale, num_filters=32, num_res_blocks=8, res_block_expansion=4, res_block_scaling=None):
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_a)


def wdsr_b(scale, num_filters=32, num_res_blocks=8, res_block_expansion=6, res_block_scaling=None):
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_b)


def wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block):
    x_in = Input(shape=(None, None, 2))
    depth, s1 = tf.split(x_in, num_or_size_splits=2, axis=3)
    x = Lambda(normalize)(depth)

    # main branch
    m = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)
        #m = res_block(m, num_filters)
        m = tf.keras.layers.BatchNormalization()(m)
        #m = tf.keras.layers.Dropout(0.2)(m)
        #m = tf.keras.layers.experimental.InstanceNormalization()(m)
    m = conv2d_weightnorm(1 * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(m)
    #m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    s = conv2d_weightnorm(1 * scale ** 2, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(x)
    #s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    #x = Add()([depth, x])
    x = Lambda(denormalize)(x)

    return Model(x_in, x, name="wdsr")

def res_block(x_in, num_filters, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = Add()([x_in, x])
    return x

def res_block_a(x_in, num_filters, expansion, kernel_size, scaling):
    x = conv2d_weightnorm(num_filters * expansion, kernel_size, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def res_block_b(x_in, num_filters, expansion, kernel_size, scaling):
    linear = 0.8
    x = conv2d_weightnorm(num_filters * expansion, 1, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(int(num_filters * linear), 1, padding='same')(x)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    #return tfa.layers.WeightNormalization(Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)
    #return tfa.layers.InstanceNormalization(Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs))
    return Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs)