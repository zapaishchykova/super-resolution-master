import time
import tensorflow as tf

from model import evaluate
from model import srgan

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_dir='./ckpt/wdsr'):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(0.0),
                                              optimizer=Adam(learning_rate),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        self.restore()

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()
        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()

            loss = self.train_step(lr, hr)
            loss_mean(loss)

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                # Compute PSNR on validation dataset
                psnr_value = self.evaluate(valid_dataset)

                duration = time.perf_counter() - self.now
                print(f'{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')

                if save_best_only and psnr_value >= ckpt.psnr:
                    self.now = time.perf_counter()
                    # skip saving checkpoint, no PSNR improvement
                    continue

                ckpt.psnr = psnr_value
                ckpt_mgr.save()

                self.now = time.perf_counter()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)

            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')


class EdsrTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=PiecewiseConstantDecay(boundaries=[1000], values=[1e-4, 5e-6])):
        super().__init__(model, loss=MeanSquaredError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)


    def _with_flat_batch(self,flat_batch_fn):
        def fn(x, *args, **kwargs):
            shape = tf.shape(x)
            flat_batch_x = tf.reshape(x, tf.concat([[-1], shape[-3:]], axis=0))
            flat_batch_r = flat_batch_fn(flat_batch_x, *args, **kwargs)
            r = nest.map_structure(lambda x: tf.reshape(x, tf.concat([shape[:-3], x.shape[1:]], axis=0)),
                                   flat_batch_r)
            return r

        return fn

    def structural_similarity(self, X, Y, K1=0.01, K2=0.03, win_size=7,data_range=10000.0, use_sample_covariance=True):
        """
        Structural SIMilarity (SSIM) index between two images
        Args:
            X: A tensor of shape `[..., in_height, in_width, in_channels]`.
            Y: A tensor of shape `[..., in_height, in_width, in_channels]`.
        Returns:
            The SSIM between images X and Y.
        Reference:
            https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/_structural_similarity.py
        Broadcasting is supported.
        """
        X = tf.convert_to_tensor(X)
        Y = tf.convert_to_tensor(Y)

        ndim = 2  # number of spatial dimensions
        nch = tf.shape(X)[-1]

        filter_func = self._with_flat_batch(tf.nn.depthwise_conv2d)
        kernel = tf.cast(tf.fill([win_size, win_size, nch, 1], 1 / win_size ** 2), X.dtype)
        filter_args = {'filter': kernel, 'strides': [1] * 4, 'padding': 'VALID'}

        NP = win_size ** ndim

        # filter has already normalized by NP
        if use_sample_covariance:
            cov_norm = NP / (NP - 1)  # sample covariance
        else:
            cov_norm = 1.0  # population covariance to match Wang et. al. 2004

        # compute means
        ux = filter_func(X, **filter_args)
        uy = filter_func(Y, **filter_args)

        # compute variances and covariances
        uxx = filter_func(X * X, **filter_args)
        uyy = filter_func(Y * Y, **filter_args)
        uxy = filter_func(X * Y, **filter_args)
        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)

        R = data_range
        C1 = (K1 * R) ** 2
        C2 = (K2 * R) ** 2

        A1, A2, B1, B2 = ((2 * ux * uy + C1,
                           2 * vxy + C2,
                           ux ** 2 + uy ** 2 + C1,
                           vx + vy + C2))
        D = B1 * B2
        S = (A1 * A2) / D

        ssim = tf.reduce_mean(S, axis=[-3, -2, -1])
        l_ssim = 1 - ssim * 0.5
        return 1-l_ssim

    def loss_DSSIM_theano(self, y_true, y_pred):
        # expected net output is of shape [batch_size, row, col, image_channels]
        # e.g. [10, 480, 640, 3] for a batch of 10 640x480 RGB images
        # We need to shuffle this to [Batch_size, image_channels, row, col]


        u_true = K.mean(y_true, axis=-1)
        u_pred = K.mean(y_pred, axis=-1)
        var_true = K.var(y_true, axis=-1)
        var_pred = K.var(y_pred, axis=-1)
        std_true = K.sqrt(var_true)
        std_pred = K.sqrt(var_pred)
        c1 = 0.01# ** 2
        c2 = 0.03# ** 2
        ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
        denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred ** 2  + var_true ** 2  + c2)

        ssim /= K.clip(denom, K.epsilon(), np.inf)
        # ssim = tf.select(tf.is_nan(ssim), K.zeros_like(ssim), ssim)

        return K.mean((1.0 - ssim) / 2.0)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)

    def maskedMSEloss(self, y_true, y_pred):
        diff = y_true-y_pred
        l = tf.reduce_mean(diff**2)
        return l

    def scale_invariant_loss(self, y_true, y_pred):
        max_val = 2500.0
        first_log = K.log(K.clip(y_pred, K.epsilon(), max_val) + 1.)
        second_log = K.log(K.clip(y_true, K.epsilon(), max_val) + 1.)
        return K.mean(K.square(first_log - second_log), axis=-1) - 0.5 * K.square(K.mean(first_log - second_log, axis=-1))

    def log10(self, x):
        numerator = K.log(x)
        denominator = K.log(K.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    def scale_invariant_loss_log10(self, y_true, y_pred):
        max_val = 2500.0
        first_log = self.log10(K.clip(y_pred, K.epsilon(), max_val))
        second_log = self.log10(K.clip(y_true, K.epsilon(), max_val))
        diff = K.mean(K.square(first_log - second_log), axis=-1) - 1. * K.square(K.mean(first_log - second_log, axis=-1))

        return diff#+ self.grad_loss(first_log - second_log)


    def avg_log10(self, depth1, depth2):
        """
        Computes average log_10 error (Liu, Neural Fields, 2015).
        Takes preprocessed depths (no nans, infs and non-positive values)
        depth1:  one depth map
        depth2:  another depth map
        Returns:
            abs_relative_distance
        """
        max_val = 2500.0
        log_diff = self.log10(K.clip(depth1, K.epsilon(), max_val) + 1.) - self.log10(K.clip(depth2, K.epsilon(), max_val) + 1.)

        # Edges
        dy_true, dx_true = tf.image.image_gradients(depth1)
        dy_pred, dx_pred = tf.image.image_gradients(depth2)
        l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

        return K.mean(K.abs(log_diff)) + .5 * self.compute_smooth_loss(depth1, depth2) + .5 * K.mean(l_edges)

    def _tf_fspecial_gauss(self, size, sigma):
        """Function to mimic the 'fspecial' gaussian MATLAB function
        """
        x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / tf.reduce_sum(g)

    def tf_ssim(self, img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
        window = self._tf_fspecial_gauss(size, sigma)  # window shape [size, size]
        K1 = 0.01
        K2 = 0.03
        L = 1000.0  # depth of image (255 in case the image has a differnt scale)
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        mu1 = tf.nn.avg_pool2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
        mu2 = tf.nn.avg_pool2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = tf.nn.avg_pool2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
        sigma2_sq = tf.nn.avg_pool2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
        sigma12 = tf.nn.avg_pool2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
        if cs_map:
            value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                  (sigma1_sq + sigma2_sq + C2)),
                     (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        else:
            value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                 (sigma1_sq + sigma2_sq + C2))

        if mean_metric:
            value = tf.reduce_mean(value)
        return value

    def depth_loss_function(self, y_true, y_pred):
        # Point-wise depth

        max_val = 2000.0
        l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

        # Edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

        # Structural similarity (SSIM) index
        l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, max_val)) * 0.5, 0, 1)


        # Weights
        w1 = 1.0
        w2 = 1.0
        w3 = .1 #0.3

        return  (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth)) + .1 * self.compute_smooth_loss(y_true, y_pred)
                #+ .01 * self.sobel_gradient_loss(y_true, y_pred)

    def scale_invariant_loss_masked(self, y_true, y_pred):
        maskValue = 0

        mask_true = K.cast(K.not_equal(y_true, maskValue), K.floatx())
        mask_pred = K.cast(K.not_equal(y_pred, maskValue), K.floatx())

        combined_mask = K.cast(K.equal(mask_pred, mask_true), K.floatx())

        first_log = K.log(K.clip(y_pred * combined_mask, K.epsilon(), np.inf) + 1.)

        second_log = K.log(K.clip(y_true * combined_mask, K.epsilon(), np.inf) + 1.)
        return K.mean(K.square(first_log - second_log), axis=-1) - 0.5 * K.square(
            K.mean(first_log - second_log, axis=-1))

    def sobel_edges(self,img):
        ch = img.get_shape().as_list()[3]
        kerx = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
        kery = tf.constant([[-1, -2, 1], [0, 0, 0], [1, 2, 1]], dtype='float32')
        kerx = tf.expand_dims(kerx, 2)
        kerx = tf.expand_dims(kerx, 3)
        kerx = tf.tile(kerx, [1, 1, ch, 1])
        kery = tf.expand_dims(kery, 2)
        kery = tf.expand_dims(kery, 3)
        kery = tf.tile(kery, [1, 1, ch, 1])
        gx = tf.compat.v1.nn.depthwise_conv2d_native(img, kerx, strides=[1, 1, 1, 1], padding="VALID")
        gy = tf.compat.v1.nn.depthwise_conv2d_native(img, kery, strides=[1, 1, 1, 1], padding="VALID")
        return tf.concat([gx, gy], 3)

    def sobel_gradient_loss(self, guess, truth):
        g1 = self.sobel_edges(guess)
        g2 = self.sobel_edges(truth)
        return tf.reduce_mean(tf.pow(tf.abs(g1 - g2), 1))

    def compute_smooth_loss(self, disp, img):
        def _gradient(pred):
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            return D_dx, D_dy

        disp_gradients_x, disp_gradients_y = _gradient(disp)
        image_gradients_x, image_gradients_y = _gradient(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))



class WdsrTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 #learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4])):
                 learning_rate=PiecewiseConstantDecay(boundaries=[2000], values=[1e-4, 5e-5])):
        #super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)
        super().__init__(model, loss=self.depth_loss_function, learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def scale_invariant_loss(self, y_true, y_pred):
        first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
        second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
        return K.mean(K.square(first_log - second_log), axis=-1) - 0.5 * K.square(K.mean(first_log - second_log, axis=-1)) #+ self.grad_loss(first_log - second_log)

    def smooth_L1_loss(self,y_true, y_pred):
        return tf.compat.v1.losses.huber_loss(y_true, y_pred)

    def ssim_loss(self,y_true,y_pred):
       return -1 * tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1000.0))

    def scale_invariant_loss_masked(self, y_true, y_pred):
        maskValue = 0

        mask_true = K.cast(K.not_equal(y_true, maskValue), K.floatx())
        mask_pred = K.cast(K.not_equal(y_pred, maskValue), K.floatx())

        combined_mask = K.cast(K.equal(mask_pred, mask_true), K.floatx())

        first_log = K.log(K.clip(y_pred * combined_mask, K.epsilon(), np.inf) + 1.)

        second_log = K.log(K.clip(y_true * combined_mask, K.epsilon(), np.inf) + 1.)
        return K.mean(K.square(first_log - second_log), axis=-1) - 0.4 * K.square(
            K.mean(first_log - second_log, axis=-1))

    def depth_loss_function(self, y_true, y_pred):
        # Point-wise depth

        max_val = 1000.0
        l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

        # Edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

        # Structural similarity (SSIM) index
        l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, max_val)) * 0.5, 0, 1)

        # Weights
        w1 = 1.0
        w2 = 1.0
        w3 = .1  # 0.3

        return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))
    '''
    def scale_invariant_loss_masked(self, y_true, y_pred):
        first_log = K.log(K.clip(y_pred[y_pred != 0], K.epsilon(), np.inf) + 1.)
        second_log = K.log(K.clip(y_true[y_true != 0], K.epsilon(), np.inf) + 1.)
        return K.mean(K.square(first_log - second_log), axis=-1) - 0.5 * K.square(K.mean(first_log - second_log, axis=-1)) #+ self.grad_loss(first_log - second_log)
    '''
    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class SrganGeneratorTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=1e-4):
        super().__init__(model, loss=MeanSquaredError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=1000000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class SrganTrainer:
    #
    # TODO: model and optimizer checkpoints
    #
    def __init__(self,
                 generator,
                 discriminator,
                 content_loss='VGG54',
                 learning_rate=PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])):

        if content_loss == 'VGG22':
            self.vgg = srgan.vgg_22()
        elif content_loss == 'VGG54':
            self.vgg = srgan.vgg_54()
        else:
            raise ValueError("content_loss must be either 'VGG22' or 'VGG54'")

        self.content_loss = content_loss
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = Adam(learning_rate=learning_rate)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate)

        self.binary_cross_entropy = BinaryCrossentropy(from_logits=False)
        self.mean_squared_error = MeanSquaredError()

    def train(self, train_dataset, steps=200000):
        pls_metric = Mean()
        dls_metric = Mean()
        step = 0

        for lr, hr in train_dataset.take(steps):
            step += 1

            pl, dl = self.train_step(lr, hr)
            pls_metric(pl)
            dls_metric(dl)

            if step % 50 == 0:
                print(f'{step}/{steps}, perceptual loss = {pls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}')
                pls_metric.reset_states()
                dls_metric.reset_states()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.generator(lr, training=True)

            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)

            con_loss = self._content_loss(hr, sr)
            gen_loss = self._generator_loss(sr_output)
            perc_loss = con_loss + 0.001 * gen_loss
            disc_loss = self._discriminator_loss(hr_output, sr_output)

        gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return perc_loss, disc_loss

    @tf.function
    def _content_loss(self, hr, sr):
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)

    def _generator_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def _discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss
