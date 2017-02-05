from time import strftime, gmtime

import matplotlib as mpl
import tqdm as tqdm

mpl.use('Agg')

import matplotlib.pyplot as plt

plt.style.use('ggplot')
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from utils import *
from vgg19 import Vgg19
import os.path


class DFI:
    """Deep Feature Interpolation procedure

    Implementation of DFI as described here:
    https://arxiv.org/pdf/1611.05507v1.pdf
    """

    def __init__(self, FLAGS):
        """
        Initialize the DFI procedure
        :param k: Number of nearest neighbours
        :param alpha: Scalar factor for w
        :param lamb: Scalar factor for total variation in loss
        :param beta: Scalar in exponent of total variation
        :param model_path: Path to the vgg19 pretrained model
        :param num_layers: Number of layers of which the deep
        features shall be extracted
        :param gpu: Compute or gpu
        :param data_dir: Directory for images
        """
        self._model = load_model(FLAGS.model_path)
        self._conv_layer_tensors = []
        self._summaries = []
        self.FLAGS = FLAGS
        self._loss_log = []

        self._conv_layer_tensor_names = ['conv3_1/Relu:0',
                                         'conv4_1/Relu:0',
                                         'conv5_1/Relu:0']
        self._sess = None

    def run(self):
        """Start the DFI process"""
        print('Starting DFI')
        # Name-scope for tensorboard
        # Setup        # Set device

        # Get phi(z) = phi(x) + alpha*w
        phi_z = self._get_phi_z_const()

        # Get reverse mapped z = phi^-1(phi(z))
        reverse_mapped_z = self._reverse_map_z(phi_z)

        self._save_output(reverse_mapped_z)

    def _reverse_map_z(self, phi_z):
        """Reverse map z from phi(z)"""
        device = '/gpu:0' if self.FLAGS.gpu else '/cpu:0'
        # Open second session with
        with tf.device(device):
            self._graph_var = tf.Graph()
            with self._graph_var.as_default():
                self._nn = Vgg19(model=self._model, input_placeholder=False,
                                 data_dir=self.FLAGS.data_dir,
                                 random_start=self.FLAGS.random_start,
                                 start_image_path=self.FLAGS.person_image)

                with tf.Session(graph=self._graph_var) as self._sess:
                    self._sess.run(tf.initialize_all_variables())

                    self._conv_layer_tensors = [
                        self._graph_var.get_tensor_by_name(
                            self._conv_layer_tensor_names[idx])
                        for idx in range(self.FLAGS.num_layers)]

                    # Set z_tensor reference
                    self._z_tensor = self._nn.inputRGB

                    return self._optimize_z_tf(phi_z)

    def _get_phi_z_const(self):
        """
        Calculates the constant phi(z) = phi(x) + alpha * w
        :return: phi(z) = phi(x) + alpha * w
        """

        device = '/gpu:0' if self.FLAGS.gpu else '/cpu:0'
        if self.FLAGS.rebuild_cache or not os.path.isfile('cache.ch.npy'):
            print('Using device: {}'.format(device))
            with tf.device(device):
                self._graph_ph = tf.Graph()
                with self._graph_ph.as_default():
                    self._nn = Vgg19(model=self._model, input_placeholder=True,
                                     start_image_path=self.FLAGS.person_image)

                    with tf.name_scope('DFI-Graph') as scope:
                        # Run the graph in the session.
                        with tf.Session(graph=self._graph_ph) as self._sess:
                            self._sess.run(tf.initialize_all_variables())

                            self._conv_layer_tensors = [
                                self._graph_ph.get_tensor_by_name(
                                    self._conv_layer_tensor_names[idx])
                                for idx in range(self.FLAGS.num_layers)]

                            atts = load_discrete_lfw_attributes(
                                self.FLAGS.data_dir)
                            imgs_path = atts['path'].values

                            if self.FLAGS.person_image:
                                start_img_path = self.FLAGS.person_image
                            else:
                                start_img_path = imgs_path[
                                    self.FLAGS.person_index]

                            person_index = get_person_idx_by_path(atts,
                                                                  start_img_path)

                            start_img = \
                                reduce_img_size(load_images(*[start_img_path]))[
                                    0]
                            plt.imsave(fname='start_img.png', arr=start_img)

                            # Get image paths
                            pos_paths, neg_paths = self._get_sets(atts,
                                                                  self.FLAGS.feature,
                                                                  person_index)

                            # Reduce image sizes
                            pos_imgs = reduce_img_size(load_images(*pos_paths))
                            neg_imgs = reduce_img_size(load_images(*neg_paths))

                            # Get pos/neg deep features
                            pos_deep_features = self._phi(pos_imgs)
                            neg_deep_features = self._phi(neg_imgs)

                            # Calc W
                            w = np.mean(pos_deep_features, axis=0) - np.mean(
                                neg_deep_features,
                                axis=0)
                            w /= np.linalg.norm(w)

                            inv = -1 if self.FLAGS.invert else 1

                            # Calc phi(z)
                            phi = self._phi(start_img)
                            phi_z = phi + self.FLAGS.alpha * w * inv
                            np.save('cache.ch', phi_z)
        else:
            print('Loading cached phi_z')
            phi_z = np.load('cache.ch.npy')

        return phi_z

    def _optimize_z_tf(self, phi_z):
        """

        :param phi_z: phi(start_img) + alpha*w
        :param start_img: start img
        :return:
        """

        # Init constant tensor
        phi_z_const_tensor = tf.constant(phi_z, dtype=tf.float32,
                                         name='phi_x_alpha_w')

        # Rescale image
        tensor = self._z_tensor
        min = tf.reduce_min(tensor)
        max = tf.reduce_max(tensor)
        rescaled_img_tensor = 255 * (tensor - min) / (max - min)

        # Define loss
        loss, diff_loss_tensor, tv_loss_tensor = self._minimize_z_tensor(
            phi_z_const_tensor, self._z_tensor)

        # Logging
        person_name = self.FLAGS.person_image.split('/')[len(self.FLAGS.person_image.split('/')) - 1]
        log_path = 'log/run_k-{}_alpha-{}_feat-{}_lamb-{}_lr-{}_rand-{}_opt-{}_pers-{}.{}'.format(
            self.FLAGS.k, self.FLAGS.alpha, self.FLAGS.feature, self.FLAGS.lamb,
            self.FLAGS.lr, self.FLAGS.random_start, self.FLAGS.optimizer,
            person_name,
            strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        )

        train_writer = tf.train.SummaryWriter(log_path)

        # Use placeholder as learning rate to enable decreasing lr
        lr = tf.placeholder(dtype=tf.float32, shape=None, name='learning_rate')

        # Add the optimizer
        if self.FLAGS.optimizer == 'adam':
            train_op = tf.train.AdamOptimizer(learning_rate=lr) \
                .minimize(loss, var_list=[self._z_tensor])

        # Add the ops to initialize variables.  These will include
        # the optimizer slots added by AdamOptimizer().
        init_op = tf.initialize_all_variables()

        # Actually intialize the variables
        self._sess.run(init_op)

        # If verbose, display loss per step, else show progress
        if self.FLAGS.verbose:
            it = range(self.FLAGS.steps + 1)
        else:
            # Show progressbar
            it = tqdm.tqdm(range(self.FLAGS.steps + 1))

        # Steps
        for i in it:
            # Decreasing learning rate
            if i < 2000:
                train_op.run(feed_dict={lr: self.FLAGS.lr})
            elif 2000 <= i < 4000:
                train_op.run(feed_dict={lr: self.FLAGS.lr / 3.0})
            elif 4000 <= i:
                train_op.run(feed_dict={lr: self.FLAGS.lr / 9.0})

            self._log_step(i, train_writer, rescaled_img_tensor, loss,
                           diff_loss_tensor, tv_loss_tensor)  # Store image

        # Save the output image
        return self._sess.run(rescaled_img_tensor)[0]

    def _save_output(self, dfi_z):
        """Save the output in a feature and a person directory"""

        # Plot image
        split = self.FLAGS.person_image.split('/')
        person_name = split[len(split) - 1][:-4]
        feat_name = self.FLAGS.feature.replace(' ', '_')
        person_prefix = self.FLAGS.output+ '/persons/' + person_name + '/'
        person_suffix = feat_name + '_alpha-' + str(self.FLAGS.alpha)
        feat_prefix = self.FLAGS.output+ '/features/' + feat_name + '/'
        feat_suffix = person_name + '_alpha-' + str(self.FLAGS.alpha)

        ensure_dir(person_prefix)
        ensure_dir(feat_prefix)

        time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        person_path = person_prefix + person_suffix + time
        feature_path = feat_prefix + feat_suffix + time

        plt.imsave(fname=person_path + '.png', arr=dfi_z/255)
        plt.imsave(fname=feature_path + '.png', arr=dfi_z/255)

        # Plot loss
        loss_log = np.array(self._loss_log)
        plt.plot(loss_log[:, 1], loss_log[:, 0])
        plt.ylim((0.02, 0.2))
        plt.xlim((-100, 2100))
        plt.title('')

        legend = 'alpha: {}\n' \
                 'k: {}\n' \
                 'lr: {}'.format(self.FLAGS.alpha,
                                 self.FLAGS.k,
                                 self.FLAGS.lr)

        plt.legend('test', loc='upper right', title=legend)
        plt.title('Feature: {}, Person: {}'.format(self.FLAGS.feature,
                                                   person_name),
                  fontsize=16)
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.savefig(person_path + '_loss.png')
        plt.savefig(feature_path + '_loss.png')

        f_pers = open(person_path + '_loss.csv', 'w')
        f_feat = open(feature_path + '_loss.csv', 'w')
        for loss, step in loss_log:
            f_pers.write('{},{}\n'.format(step, loss))
            f_feat.write('{},{}\n'.format(step, loss))
        f_pers.close()
        f_feat.close()

    def _log_step(self, i, train_writer, rescaled_img_tensor, loss,
                  diff_loss_tensor, tv_loss_tensor):
        """Logs the intermediate values of the optimization step"""
        # Output 100 summary values
        if i % math.ceil(self.FLAGS.steps / 100.0) == 0:
            self._loss_log.append((loss.eval(), i))

            for sum_op in self._summaries:
                summary = self._sess.run(sum_op)
                train_writer.add_summary(summary, i)

            if self.FLAGS.verbose:
                temp_loss, diff_loss, tv_loss = \
                    self._sess.run(
                        [loss, diff_loss_tensor, tv_loss_tensor])
                # train_writer.add_summary(summary, i)
                print('Step: {}'.format(i))
                print('{:>14.10f} - loss'.format(temp_loss))
                print('{:>14.10f} - tv_loss'.format(tv_loss))
                print('{:>14.10f} - diff_loss'.format(diff_loss))

        # Output 4 images
        if i % math.ceil(self.FLAGS.steps / 4.0) == 0:
            im_sum_op = tf.image_summary('img{}'.format(i),
                                         tensor=rescaled_img_tensor,
                                         name='img{}'.format(i))
            im_sum = self._sess.run(im_sum_op)
            train_writer.add_summary(im_sum, global_step=i)

    def _minimize_z_tensor(self, phi_z_const_tensor, z_tensor):
        """
        Objective function implemented with tensors
        :param phi_z_const_tensor: phi(x) + alpha*w
        :param z_tensor: Variable
        :return: loss
        """

        with tf.name_scope('summaries'):
            phi_z_prime = self._phi_tensor()
            subtract = phi_z_prime - phi_z_const_tensor
            square = tf.square(subtract)
            reduce_sum = tf.reduce_sum(square)

            # Get the tv-reg term
            regularization = self._total_variation_regularization(z_tensor,
                                                                  self.FLAGS.beta)
            with tf.name_scope('tv_loss'):
                tv_loss = self.FLAGS.lamb * tf.reduce_sum(regularization)

            with tf.name_scope('diff_loss'):
                diff_loss = 0.5 * reduce_sum

            # Create loss on values lower than zero
            shape = tf.constant([224, 224, 3], dtype=tf.float32)
            with tf.name_scope('loss_lower'):
                loss_lower = -1 * tf.reduce_sum(
                    (z_tensor - tf.abs(z_tensor)) / 2.0) / tf.reduce_prod(shape)

            # Create loss on values higher than 255
            with tf.name_scope('loss_upper'):
                sub = (z_tensor - 255)
                loss_upper = tf.reduce_sum(
                    (sub + tf.abs(sub)) / 2.0) / tf.reduce_prod(shape)

            with tf.name_scope('loss'):
                loss = diff_loss + tv_loss + loss_upper + loss_lower

            # Add summaries
            self._summaries.append(tf.scalar_summary('loss', loss))
            self._summaries.append(tf.scalar_summary('tv_loss', tv_loss))
            self._summaries.append(tf.scalar_summary('diff_loss', diff_loss))
            self._summaries.append(tf.scalar_summary('loss_lower', loss_lower))
            self._summaries.append(tf.scalar_summary('loss_upper', loss_upper))

            return loss, diff_loss, tv_loss

    def _total_variation_regularization(self, x, beta=2):
        """
        Idea from:
        https://github.com/antlerros/tensorflow-fast-neuralstyle/blob/master/net.py
        """
        wh = tf.constant([[[[1], [1], [1]]], [[[-1], [-1], [-1]]]], tf.float32)
        ww = tf.constant([[[[1], [1], [1]], [[-1], [-1], [-1]]]], tf.float32)
        tvh = lambda x: self._conv2d(x, wh, p='SAME')
        tvw = lambda x: self._conv2d(x, ww, p='SAME')
        dh = tvh(x)
        dw = tvw(x)
        tv = (tf.add(tf.reduce_sum(dh ** 2, [1, 2, 3]),
                     tf.reduce_sum(dw ** 2, [1, 2, 3]))) ** (beta / 2.)

        tv /= np.prod(x.get_shape().as_list(), dtype=np.float32) * 255

        return tv

    def _conv2d(self, x, W, strides=[1, 1, 1, 1], p='SAME', name=None):
        """2d Convolution"""
        return tf.nn.conv2d(x, W, strides=strides, padding=p, name=name)

    def _phi_tensor(self):
        """
        Implementation of the deep feature function in tensorflow
        :return: tensor of z in the deep feature space
        """
        # Start with first layer tensor
        res = tf.reshape(self._conv_layer_tensors[0], [-1], name='phi_z')

        # Concatenate the rest
        for i in np.arange(1, self.FLAGS.num_layers):
            tmp = tf.reshape(self._conv_layer_tensors[i], [-1])
            res = tf.concat(0, [res, tmp])

        square = tf.square(res)
        reduce_sum = tf.reduce_sum(square, name='phi_tensor_sum')
        sqrt = tf.sqrt(reduce_sum, name='phi_tensor_sqrt')

        return res / sqrt

    def _phi(self, imgs):
        """Transform list of images into deep feature space

        :param imgs: input images
        :return: deep feature transformed images
        """

        if not isinstance(imgs, list):
            input_images = [imgs]
        else:
            input_images = imgs

        fd = {self._nn.inputRGB: input_images}
        ret = self._sess.run(self._conv_layer_tensors,
                             feed_dict=fd)
        res = []

        # Create result list
        for img_idx in range(len(input_images)):
            phi_img = np.array([])

            # Append all layer results to a (M,) vector
            for layer_idx in range(self.FLAGS.num_layers):
                phi_img = np.append(phi_img,
                                    ret[layer_idx][img_idx].reshape(-1))

            res.append(phi_img)

        # Handle correct return type and normalize (L2)
        if not isinstance(imgs, list):
            return res[0] / np.linalg.norm(res[0])  # Single image
        else:
            return [x / np.linalg.norm(x) for x in res]  # List of images

    def _get_sets(self, atts, feat, person_index):
        """
        Generates the subsets of the knn given a person index and a feature
        :param atts: Attribute dataframe
        :param feat: Feature
        :param person_index: Person index
        :return: Positive and negative subset of the k nearest neighbours of the given person
        """
        person = atts.loc[person_index]
        del person['person']
        del person['path']

        # Remove person from df
        atts = atts.drop(person_index)

        # Split by feature
        pos_set = atts.loc[atts[feat] == 1]
        neg_set = atts.loc[atts[feat] == -1]
        pos_paths = self._get_k_neighbors(pos_set, person)
        neg_paths = self._get_k_neighbors(neg_set, person)

        return pos_paths.as_matrix(), neg_paths.as_matrix()

    def _get_k_neighbors(self, subset, person):
        """
        Gets the KNN, given the subset and the person
        :param subset: Subset in which the knn should look at
        :param person: Starting point
        :return: Image paths of the k nearest neighbours of the person in the subset
        """
        del subset['person']
        paths = subset['path']
        del subset['path']

        knn = KNeighborsClassifier(n_jobs=-1)
        dummy_target = [0 for x in range(subset.shape[0])]
        knn.fit(X=subset.as_matrix(), y=dummy_target)
        knn_indices = \
            knn.kneighbors(person.as_matrix(), n_neighbors=self.FLAGS.k,
                           return_distance=False)[0]

        neighbor_paths = paths.iloc[knn_indices]

        return neighbor_paths

    def features(self):
        """
        Generate a list of possible attributes to choose from
        :return: List of attributes
        """

        atts = load_lfw_attributes(self.FLAGS.data_dir)
        del atts['path']
        del atts['person']
        return atts.columns.values
