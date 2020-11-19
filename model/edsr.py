from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, UpSampling2D, MaxPooling2D, Dropout,concatenate
from tensorflow.python.keras.models import Model
import tensorflow as tf
from model.common import normalize, denormalize, pixel_shuffle
import tensorflow_addons as tfa


def edsr_a(scale, num_filters=32, num_res_blocks=8, res_block_expansion=4, res_block_scaling=None):
    return edsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_a)


def edsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block):
    x_in = Input(shape=(None, None, 2))
    depth, s1 = tf.split(x_in, num_or_size_splits=2, axis=3)
    x = Lambda(normalize)(depth)

    # main branch
    m = conv2d_weightnorm(num_filters, 2, padding='same')(x)
    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)

    m = conv2d_weightnorm(1 * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(m)
    m = conv2d_weightnorm(3 * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(m)
    #m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    s = conv2d_weightnorm(3 * scale ** 2, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(x)
    s = conv2d_weightnorm(1 * scale ** 2, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(s)
    #s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    x = Add()([x, depth])
    x = Lambda(denormalize)(x)

    return Model(x_in, x)

'''
def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    x_in = Input(shape=(None, None, 2))
    #depth, s1, s2 ,s3 = tf.split(x_in, num_or_size_splits=4, axis=3)
    depth, s1 = tf.split(x_in, num_or_size_splits=2, axis=3)
    x = Lambda(normalize)(depth)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    
    for i in range(num_res_blocks):
        b  = res_block(b, num_filters, res_block_scaling)
   
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([b, x])

    #x = upsample(x, scale, num_filters)

    #x = Conv2D(16, 3, padding='same')(x)
    #x = Conv2D(4, 3, padding='same')(x)
    x = Conv2D(1, 3, padding='same')(x)

    #d = upsample(depth, scale, 1)
    x = Lambda(denormalize)(x)
    x = Add()([depth, x])
    return Model(x_in, x, name="edsr")
'''

def edsr_b(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    x_in = Input(shape=(None, None, 4))
    depth, s1, s2 ,s3 = tf.split(x_in, num_or_size_splits=4, axis=3)
    #x = Lambda(normalize)(x_in)

    #x = b = Conv2D(num_filters, 3, padding='same')(x)

    conv1 = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x_in)
    conv1 = Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(num_filters*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(num_filters*2, 3,  padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(num_filters*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(num_filters*4, 3, padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(num_filters*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(num_filters*8, 3, padding='same', kernel_initializer='he_normal')(conv4)
    #drop4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(conv5)
    #drop5 = Dropout(0.3)(conv5)

    up6 = Conv2D(num_filters*8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    #merge6 = concatenate([drop4, up6], axis=3)
    merge6 = Add()([conv4, up6])
    conv6 = Conv2D(num_filters*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(num_filters*8, 3,  padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(num_filters*4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    #merge7 = concatenate([conv3, up7], axis=3)
    merge7 = Add()([conv3, up7])
    conv7 = Conv2D(num_filters*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(num_filters*4, 3,padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(num_filters*2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    #merge8 = concatenate([conv2, up8], axis=3)
    merge8 = Add()([conv2, up8])
    conv8 = Conv2D(num_filters*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(num_filters*2, 3,  padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(num_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    #merge9 = concatenate([conv1, up9], axis=3)
    merge9 = Add()([conv1, up9])
    conv9 = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(num_filters, 3,  padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, padding='same')(conv9)

    #for i in range(num_res_blocks):
    #    b = res_block(b, num_filters, res_block_scaling)
    #b = Conv2D(num_filters, 3, padding='same')(b4)
    #x = Add()([b1, x])

    #x = upsample(x, scale, num_filters)
    #x = Conv2D(1, 3, padding='same')(x)

    #d = upsample(depth, scale, 1)
    x = Add()([depth, conv10])
    x = Conv2D(1, 1, padding='same')(x)

    #x = Lambda(denormalize)(x)
    return Model(x_in, x, name="edsr")


def res_block_a(x_in, num_filters, expansion=4, kernel_size=3, scaling=1):
    x = conv2d_weightnorm(num_filters * expansion, kernel_size, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x

def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return tfa.layers.WeightNormalization(Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)

def res_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2),3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2)#, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3)#, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2)#, name='conv2d_1_scale_2')
        x = upsample_1(x, 2)#, name='conv2d_2_scale_2')

    return x
