import tensorflow as tf
import numpy as np
import os
import argparse
from model import generator_x, generator_z, discriminator
from utils import imwrite, immerge

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', '-e', default='cifar10')
parser.add_argument('--data_dir', '-d', default='./-r06.tfrecords', type=str)
parser.add_argument('--log_dir', '-l', default='log')
parser.add_argument('--checkpoint_dir', '-c', default='checkpoints/model-ckpt')
parser.add_argument('--load_model', default=None, type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--beta1', default=0.5, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--restore_path', default='', type=str)
parser.add_argument('--learning_rate', default=0.0001, type=float)
args = parser.parse_args()

def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                    np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def tensorflow_session():
    # Init session and params
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    #config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess

def get_train_data(sess, data_dir, batch_size):
    dset = tf.data.TFRecordDataset(data_dir)
    dset = dset.map(parse_tfrecord_tf)
    dset = dset.repeat().batch(batch_size)
    train_iterator = tf.data.Iterator.from_structure(dset.output_types, dset.output_shapes)
    training_init_op = train_iterator.make_initializer(dset)
    image_batch = train_iterator.get_next()
    sess.run(training_init_op)
    return image_batch

def main():

    image_size = 64
    z_dim = 256
    with tf.name_scope('input'):
        real = tf.placeholder('float32', [args.batch_size, 3, image_size, image_size], name='real_image')
        z = tf.placeholder('float32', [args.batch_size, z_dim], name='Gaussian')
        lr_g = tf.placeholder(tf.float32, None, name='learning_rate_g')
        lr_d = tf.placeholder(tf.float32, None, name='learning_rate_d')

    G_x = generator_x(input_z=z, reuse=False)
    G_z = generator_z(input_x=real, reuse=False)

    dis_fake, dis_fake_logit = discriminator(input_x=G_x, input_z=z, reuse=False)
    dis_real, dis_real_logit = discriminator(input_x=real, input_z=G_z, reuse=True)

    Reconstruction = generator_x(generator_z(real, reuse=True), reuse=True)

    with tf.variable_scope('generator_loss'):
        G_loss_img = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_logit, labels=tf.ones_like(dis_fake_logit)))
        G_loss_z = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real_logit, labels=tf.zeros_like(dis_real_logit)))
        G_loss = G_loss_img + G_loss_z

    with tf.variable_scope('discriminator_loss'):
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real_logit, labels=tf.ones_like(dis_real_logit)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_logit, labels=tf.zeros_like(dis_fake_logit)))
        D_loss = D_loss_real + D_loss_fake

    Genx_vars = [v for v in tf.global_variables() if v.name.startswith("generator_x")]
    Genz_vars = [v for v in tf.global_variables() if v.name.startswith("generator_z")]
    Dis_vars = [v for v in tf.global_variables() if v.name.startswith("discriminator")]

    G_solver = tf.train.AdamOptimizer(lr_g, args.beta1, args.beta2).minimize(G_loss, var_list=Genx_vars+Genz_vars)
    D_solver = tf.train.AdamOptimizer(lr_d, args.beta1, args.beta2).minimize(D_loss, var_list=Dis_vars)

    saver = tf.train.Saver(max_to_keep=10)
    sess = tensorflow_session()

    if args.restore_path != '':
        print('resotre weights from {}'.format(args.restore_path))
        saver.restore(sess, tf.train.latest_checkpoint(args.restore_path))
        print('Load weights finished!!!')
    else:
        sess.run(tf.global_variables_initializer())

    print('Getting training HQ data...')
    image_batch = get_train_data(sess, data_dir=args.data_dir, batch_size=args.batch_size)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    for it in range(500000):

        latent_z = np.random.randn(args.batch_size, 256).astype(np.float32)
        batch_images = sess.run(image_batch)
        batch_images = adjust_dynamic_range(batch_images.astype(np.float32), [0, 255], [0., 1.])

        feed_dict_1 = {real: batch_images, z: latent_z, lr_g: args.learning_rate, lr_d: args.learning_rate*0.1}
        _, g_loss_img, g_loss_z = sess.run([G_solver, G_loss_img, G_loss_z], feed_dict=feed_dict_1)
        _, d_loss_real, d_loss_fake = sess.run([D_solver, D_loss_real, D_loss_fake], feed_dict=feed_dict_1)

        if it % 20 == 0:
            print('Iter: {}  g_loss_img: {} g_loss_z: {} d_r_loss: {} d_f_loss_: {}'.format(
                it, g_loss_img, g_loss_z, d_loss_real, d_loss_fake))

            if it % 500 == 0:
                samples1 = sess.run(G_x, feed_dict={z: latent_z})
                samples1 = adjust_dynamic_range(samples1.transpose(0, 2, 3, 1), drange_in=[0, 1], drange_out=[-1,1])
                imwrite(immerge(samples1, 8, 8), '%s/epoch_%d_sampling.jpg' % (args.log_dir, it))

                recon = sess.run(Reconstruction, feed_dict={real: batch_images})
                recon = adjust_dynamic_range(recon.transpose(0, 2, 3, 1), drange_in=[0, 1], drange_out=[-1, 1])
                imwrite(immerge(recon, 8, 8), '%s/epoch_%d_recon.jpg' % (args.log_dir, it))

                batch_images = adjust_dynamic_range(batch_images.transpose(0, 2, 3, 1), drange_in=[0, 1], drange_out=[-1, 1])
                imwrite(immerge(batch_images, 8, 8), '%s/epoch_%d_orin.jpg' % (args.log_dir, it))


        if np.mod(it, 10000) == 0 and it != 0:
            saver.save(sess, args.checkpoint_dir, global_step=it)


if __name__ == '__main__':
    main()
