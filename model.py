import tensorflow as tf
import numpy as np

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.random_normal_initializer(stddev=0.02)) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.constant_initializer(0.0))
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])

#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.

def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, np.int32(s[1]), np.int32(s[2]), 1, np.int32(s[3]), 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, np.int32(s[1]), np.int32(s[2]) * factor, np.int32(s[3]) * factor])
        return x

#----------------------------------------------------------------------------
# Fused upscale2d + conv2d.
# Faster and uses less memory than performing the operations separately.

def upscale2d_conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, np.int32(x.shape[2] * 2), np.int32(x.shape[3] * 2)]
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Box filter downscaling layer.

def downscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale2D'):
        ksize = [1, 1, factor, factor]
        return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

#----------------------------------------------------------------------------
# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.

def conv2d_downscale2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

def bn(x, epsilon=1e-5, decay=0.9, name='batch_norm'):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(x, decay=decay, epsilon=epsilon, scale=True)

def generator_x(input_z, reuse=False):

    s0 = 4
    nf0 = 512
    with tf.variable_scope('generator_x', reuse=reuse):

        with tf.variable_scope('fc'):
            fc = bn(dense(input_z, fmaps=nf0 * s0 * s0, gain=np.sqrt(2) / 4, use_wscale=False), name='f_1')
            net = leaky_relu(tf.reshape(fc, [-1, nf0, s0, s0]))

        with tf.variable_scope('block1'):
            net = leaky_relu(bn(conv2d(net, fmaps=256, kernel=3, use_wscale=False), name='b_1')) #8*8
            net = upscale2d(net, factor=2)

        with tf.variable_scope('block2'):
            net = leaky_relu(bn(conv2d(net, fmaps=256, kernel=3, use_wscale=False), name='b_2')) #16*16
            net = upscale2d(net, factor=2)

        with tf.variable_scope('block3'):
            net = leaky_relu(bn(conv2d(net, fmaps=128, kernel=3, use_wscale=False), name='b_3')) #32*32
            net = upscale2d(net, factor=2)

        with tf.variable_scope('block4'):
            net = leaky_relu(bn(conv2d(net, fmaps=64, kernel=3, use_wscale=False), name='b_4')) #64*64
            net = upscale2d(net, factor=2)

        with tf.variable_scope('output'):
            net = apply_bias(conv2d(net, fmaps=3, kernel=3, use_wscale=False))  # 64*64
            net = tf.nn.sigmoid(net)

        return net


def generator_z(input_x, reuse=False):

    with tf.variable_scope('generator_z', reuse=reuse):

        with tf.variable_scope('block1'):
            net = leaky_relu(bn(conv2d(input_x, fmaps=64, kernel=5, use_wscale=False), name='b_1')) #32*32
            net = downscale2d(net, factor=2)

        with tf.variable_scope('block2'):
            net = leaky_relu(bn(conv2d(net, fmaps=128, kernel=3, use_wscale=False), name='b_2')) #16*16
            net = downscale2d(net, factor=2)

        with tf.variable_scope('block3'):
            net = leaky_relu(bn(conv2d(net, fmaps=256, kernel=3, use_wscale=False), name='b_3')) #8*8
            net = downscale2d(net, factor=2)

        with tf.variable_scope('block4'):
            net = leaky_relu(bn(conv2d(net, fmaps=256, kernel=3, use_wscale=False), name='b_4')) #4*4
            net = downscale2d(net, factor=2)

        with tf.variable_scope('fc1'):
            fc = leaky_relu(bn(dense(net, fmaps=512, gain=1, use_wscale=False), name='f_1'))

        with tf.variable_scope('mu'):
            mu = apply_bias(dense(fc, fmaps=256, gain=1, use_wscale=False))

        with tf.variable_scope('sigma'):
            sigma = apply_bias(dense(fc, fmaps=256, gain=1, use_wscale=False))

        eps = tf.random_normal(shape=tf.shape(mu))

        return mu + tf.exp(sigma / 2) * eps

def discriminator(input_x, input_z, reuse=False):

    with tf.variable_scope('discriminator', reuse=reuse):

        with tf.variable_scope('input_x'):

            with tf.variable_scope('block1'):
                net = leaky_relu(conv2d(input_x, fmaps=64, kernel=7, use_wscale=False))
                net = tf.nn.dropout(net, keep_prob=0.8)

            with tf.variable_scope('block2'):
                net = conv2d(net, fmaps=128, kernel=5, use_wscale=False)
                net = downscale2d(net, factor=2)
                net = tf.nn.dropout(leaky_relu(bn(net, name='b_1')), keep_prob=0.8)

            with tf.variable_scope('block3'):
                net = conv2d(net, fmaps=256, kernel=3, use_wscale=False)
                net = downscale2d(net, factor=2)
                net = tf.nn.dropout(leaky_relu(bn(net, name='b_2')), keep_prob=0.8)

            with tf.variable_scope('block4'):
                net = conv2d(net, fmaps=256, kernel=3, use_wscale=False)
                net = downscale2d(net, factor=2)
                net = tf.nn.dropout(leaky_relu(bn(net, name='b_3')), keep_prob=0.8)

            with tf.variable_scope('block5'):
                net = conv2d(net, fmaps=512, kernel=3, use_wscale=False)
                net = downscale2d(net, factor=2)
                net = tf.nn.dropout(leaky_relu(bn(net, name='b_4')), keep_prob=0.8)

            with tf.variable_scope('fc_1'):
                fc_1 = leaky_relu(bn(dense(net, fmaps=512, use_wscale=False), name='f_1'))
                fc_1 = tf.nn.dropout(fc_1, keep_prob=0.8)

        with tf.variable_scope('input_z'):

            with tf.variable_scope('z_1'):
                fc_z = leaky_relu(dense(input_z, fmaps=1024, use_wscale=False))
                fc_z = tf.nn.dropout(fc_z, keep_prob=0.8)

            with tf.variable_scope('z_2'):
                fc_z = leaky_relu(dense(fc_z, fmaps=1024, use_wscale=False))
                fc_z = tf.nn.dropout(fc_z, keep_prob=0.8)


        with tf.variable_scope('input_xz'):

            xz = tf.concat([fc_1, fc_z], axis=1)
            print('xz shape:', xz.shape)
            with tf.variable_scope('xz_1'):
                fc_xz = leaky_relu(apply_bias(dense(xz, fmaps=2048, use_wscale=False)))
                fc_xz = tf.nn.dropout(fc_xz, keep_prob=0.8)

            with tf.variable_scope('xz_2'):
                fc_xz = leaky_relu(apply_bias(dense(fc_xz, fmaps=2048, use_wscale=False)))
                fc_xz = tf.nn.dropout(fc_xz, keep_prob=0.8)

            with tf.variable_scope('output'):
                fc_xz = apply_bias(dense(fc_xz, fmaps=1, use_wscale=False))

            logits = fc_xz

            output = tf.nn.sigmoid(fc_xz)

            return output, logits



