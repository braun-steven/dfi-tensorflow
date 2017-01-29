import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def test_loss(image):
    with tf.Session() as sess:
        random_imag = np.random.rand(224,224,3)*255
        z = tf.Variable(initial_value=random_imag)
        obj = tf.constant(image)
        loss = tf.reduce_sum(tf.abs(z-obj))
        train_opt = tf.train.AdamOptimizer(learning_rate=5,epsilon=10E-8).minimize(loss)
        sess.run(tf.initialize_all_variables())
        for i in range(100):
            train_opt.run()
            if i%20 == 0:
                ret = sess.run(z)
                print('test')
                plt.imsave(fname='{}.png'.format(i),arr=ret/255)
            print('{}: {}'.format(i,sess.run(loss)))

        ret=sess.run(z)

    z_result = np.abs(ret / 255.0)

    # imgplot = plt.imshow(z)
    print('Dumping result')
    plt.imsave(fname='z.png',
               arr=z_result)
    diff_img = np.abs((ret - image) / 255.0)
    print('Max diff pixel: {}'.format(diff_img.max()))
    plt.imsave(fname='diff.png',
               arr=diff_img)

    diff2 = (ret - random_imag)
    print('Max diff random and return: {}'.format(diff2.max()))
    plt.imsave(fname='diff2.png',arr=diff2)
    return

if __name__ == '__main__':
    atts = load_discrete_lfw_attributes('/home/manfred/dfi-tensorflow/data')
    imgs_path = atts['path'].values
    start_img = reduce_img_size(load_images(*[imgs_path[0]]))[0]
    test_loss(start_img)