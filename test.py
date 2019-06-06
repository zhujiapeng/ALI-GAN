import tensorflow as tf
import numpy as np
import os
import argparse
from model_128 import generator_x, generator_z, discriminator
from utils import imwrite, immerge
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', '-e', default='cifar10')
parser.add_argument('--data_dir', '-d', default='./-r07.tfrecords', type=str)
parser.add_argument('--test', '-l', default='log_128')
parser.add_argument('--checkpoints', '-c', default='checkpoints/model-ckpt')
parser.add_argument('--load_model', default=None, type=str)
parser.add_argument('--batch_size', default=25, type=int)
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
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess

def get_train_data(sess, data_dir, batch_size):
    dset = tf.data.TFRecordDataset(data_dir)
    dset = dset.map(parse_tfrecord_tf)
    dset = dset.batch(batch_size)
    train_iterator = tf.data.Iterator.from_structure(dset.output_types, dset.output_shapes)
    training_init_op = train_iterator.make_initializer(dset)
    image_batch = train_iterator.get_next()
    sess.run(training_init_op)
    return image_batch

def main():

    image_size = 128
    z_dim = 256
    with tf.name_scope('input'):
        real = tf.placeholder('float32', [args.batch_size, 3, image_size, image_size], name='real_image')
        z = tf.placeholder('float32', [args.batch_size, z_dim], name='Gaussian')

    G_x = generator_x(input_z=z, reuse=False)
    G_z = generator_z(input_x=real, reuse=False)

    Reconstruction = generator_x(generator_z(real, reuse=True), reuse=True)

    saver = tf.train.Saver()
    sess = tensorflow_session()

    if args.restore_path != '':
        print('resotre weights from {}'.format(args.restore_path))
        saver.restore(sess, tf.train.latest_checkpoint(args.restore_path))
        print('Load weights finished!!!')

    print('Getting training Test data...')
    image_batch = get_train_data(sess, data_dir=args.data_dir, batch_size=args.batch_size)

    if not os.path.exists(args.test_dir):
        os.mkdir(args.log_dir)


    ## Reconstruction
    for it in tqdm(range(200)):

        batch_images = sess.run(image_batch)
        batch_images = adjust_dynamic_range(batch_images.astype(np.float32), [0, 255], [0., 1.])

        recon = sess.run(Reconstruction, feed_dict={real: batch_images})
        recon = adjust_dynamic_range(recon.transpose(0, 2, 3, 1), drange_in=[0, 1], drange_out=[-1, 1])
        imwrite(immerge(recon[:25, :, :, :], 5, 5), '%s/epoch_%04d_recon.png' % (args.log_dir, it))

        batch_images = adjust_dynamic_range(batch_images.transpose(0, 2, 3, 1), drange_in=[0, 1], drange_out=[-1, 1])
        imwrite(immerge(batch_images[:25, :, :, :], 5, 5), '%s/epoch_%04d_orin.png' % (args.log_dir, it))

    # Sampling
    for it in tqdm(range(200)):
        latent_z = np.random.randn(args.batch_size, z_dim).astype(np.float32)
        samples1 = sess.run(G_x, feed_dict={z: latent_z})
        samples1 = adjust_dynamic_range(samples1.transpose(0, 2, 3, 1), drange_in=[0, 1], drange_out=[-1,1])
        imwrite(immerge(samples1[:25, :, :, :], 5, 5), '%s/epoch_%04d_sampling.png' % (args.log_dir, it))



if __name__ == '__main__':
    main()
