import argparse

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from utils import load_discrete_lfw_attributes, reduce_img_size, load_images
from vgg19_lfw import Vgg19


def parse_arg():
    """Parse commandline arguments
    :return: argument dict
    """
    print('Parsing arguments')
    parser = argparse.ArgumentParser('Deep Feature Interpolation')
    parser.add_argument('--data-dir', '-d', default='data', type=str,
                        help='Path to data directory containing the images')

    return parser.parse_args()


def build_rgb_means(X):
    return np.mean(X, axis=(0, 1, 2))


def run(FLAGS):
    atts = load_discrete_lfw_attributes(FLAGS.data_dir)
    # atts = atts.head(n=20)
    X = reduce_img_size(load_images(*atts.path))
    del atts['path']
    del atts['person']
    y = [row.as_matrix() for idx, row in atts.iterrows()]
    y = (np.array(y) + 1) / 2.0

    # Randomize
    np.random.seed(42)
    np.random.shuffle(X)
    np.random.seed(42)
    np.random.shuffle(y)


    batch_size = 50
    g = tf.Graph()
    with g.as_default():
        with tf.Session(graph=g) as sess:
            vgg = Vgg19(model=None, input_placeholder=True, data_dir=FLAGS.data_dir)
            # the optimizer slots added by AdamOptimizer().
            init_op = tf.initialize_all_variables()
            correct_prediction = tf.equal(vgg.preds > 0.5, vgg.train_labels > 0.5)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

            # Actually intialize the variables
            train_writer = tf.train.SummaryWriter('run')
            loss_sum = tf.scalar_summary('loss', vgg.loss)
            acc_sum = tf.scalar_summary('accuracy', accuracy)

            sess.run(init_op)
            epochs = 10
            for _ in tqdm(range(epochs)):
                for counter in tqdm(np.arange(start=0, stop=len(X)-batch_size, step=batch_size)):
                    X_batch = X[counter:counter + batch_size]
                    y_batch = y[counter:counter + batch_size]

                    vgg.optimizer.run(feed_dict={
                        vgg.inputRGB: X_batch,
                        vgg.train_labels: y_batch
                    })

                    # Summary
                    if counter % 100 == 0:
                        train_accuracy, sumloss, sumacc = sess.run([accuracy, loss_sum, acc_sum], feed_dict={
                            vgg.inputRGB: X_batch,
                            vgg.train_labels: y_batch
                        })
                        print("step %d, training accuracy %g" % (counter, train_accuracy))
                        train_writer.add_summary(sumloss, global_step=counter)
                        train_writer.add_summary(sumacc, global_step=counter)
                        # loss = vgg.loss.eval({
                        #         vgg.inputRGB: X_batch,
                        #         vgg.train_labels: y_batch
                        #     })
                        # print(loss)

if __name__ == '__main__':
    run(parse_arg())
    exit(0)
