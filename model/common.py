import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

#DIV2K_RGB_MEAN = np.array([1 ,127.5, 127.5, 127.5])
DIV2K_RGB_MEAN = np.array([500.])#, 127.5])
#DIV2K_RGB_MEAN = np.array([0.4488*10,0.4488 , 0.4371 , 0.4040 ]) * 255


def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    #print(lr_batch)

    sr_batch = model(lr_batch)
    #sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.clip_by_value(sr_batch, 0, 2500)
    sr_batch = tf.round(sr_batch)
    #sr_batch = tf.cast(sr_batch, tf.float32)
    #print(sr_batch)
   # sr_batch = tf.cast(sr_batch, tf.uint16)
    return sr_batch

def depth_refine(init_depth_map):
    """ refine depth image with the image """
    batch_size = 1
    depth_num= 8
    depth_start = tf.reshape(tf.slice(init_depth_map, [ 0, 1, 3, 0], [batch_size, 1, 1, 1]), [batch_size])
    depth_interval = tf.reshape(tf.slice(init_depth_map, [0, 1, 3, 1], [batch_size, 1, 1, 1]), [batch_size])

    depth_interval = depth_interval
    # normalization parameters
    depth_shape = tf.shape(init_depth_map)
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    depth_start_mat = tf.tile(tf.reshape(
        depth_start, [depth_shape[0], 1, 1, 1]), [1, depth_shape[1], depth_shape[2], 1])
    depth_end_mat = tf.tile(tf.reshape(
        depth_end, [depth_shape[0], 1, 1, 1]), [1, depth_shape[1], depth_shape[2], 1])
    depth_scale_mat = depth_end_mat - depth_start_mat

    # normalize depth map (to 0~1)
    init_norm_depth_map = tf.compat.v1.div(init_depth_map - depth_start_mat, depth_scale_mat)
    return init_norm_depth_map

def evaluate(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        #psnr_value = scale_invariant_error(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


# ---------------------------------------
#  Normalization
# ---------------------------------------


def normalize(x, rgb_mean=DIV2K_RGB_MEAN):

    #depth, s1, s2, s3 = tf.split(x, num_or_size_splits=4, axis=3)
    #color=tf.concat((s1, s2, s3), axis=3, name='concat')
    #color = tf.image.per_image_standardization(color)
    #depth = tf.image.per_image_standardization(depth)
    #depth = depth_refine(depth)
    #print(color)
    #print(depth)
    #x = tf.concat((depth, color), axis=3, name='concat')
    #print((x/DIV2K_RGB_MEAN))
    return (x)#/DIV2K_RGB_MEAN)


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    '''
    batch_size = 1
    depth_num = 8
    depth_start = tf.reshape(tf.slice(x, [ 0, 1, 3, 0], [batch_size,  1, 1, 1]), [batch_size])
    depth_interval = tf.reshape(tf.slice(x, [ 0, 1, 3, 1], [batch_size,  1, 1, 1]), [batch_size])

    depth_interval = depth_interval
    # normalization parameters
    depth_shape = tf.shape(x)
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    depth_start_mat = tf.tile(tf.reshape(
        depth_start, [depth_shape[0], 1, 1, 1]), [1, depth_shape[1], depth_shape[2], 1])
    depth_end_mat = tf.tile(tf.reshape(
        depth_end, [depth_shape[0], 1, 1, 1]), [1, depth_shape[1], depth_shape[2], 1])
    depth_scale_mat = depth_end_mat - depth_start_mat

    refined_depth_map = tf.multiply(x, depth_scale_mat) + depth_start_mat
    '''
    return x #* 1000.#*1200.#+ 0.18


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x #/ 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=2500.0)
    #return tf.image.ssim(x1, x2)

# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------3


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


