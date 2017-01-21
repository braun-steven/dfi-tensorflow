import numpy as np
import tensorflow as tf

_VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    """
    A VGG-19 Network implementation using TensorFlow library.
    The network takes an image of size 224x224 with RGB channels and returns
    category scores of size 1000.

    The network configuration:
    - RGB: 224x224x3
    - BGR: 224x224x3
    - conv1: 224x224x64
    - conv2: 112x112x128
    - conv3: 56x56x256
    - conv4: 28x28x512
    - conv5: 14x14x512
    - fc6: 25088(=7x7x512)x4096
    - fc7: 4096x4096
    - fc8: 4096x1000
    """

    WIDTH = 224
    "The fixed width of the input image."
    HEIGHT = 224
    "The fixed height of the input image."
    CHANNELS = 3
    "The fixed channels number of the input image."

    model = {}
    "The model storing the kernels, weights and biases."
    model_save_path = None
    "The model save path, especially for the training process."
    model_save_freq = 0
    """
    The frequency to save the model in the training process. e.g. Save the
    model every 1000 iteration.
    """

    learning_rate = 0.05
    "Learning rate for the gradient descent."

    _inputRGB = None
    _inputBGR = None
    _inputNormalizedBGR = None

    _conv1_1 = None
    _conv1_2 = None
    _pool1 = None

    _conv2_1 = None
    _conv2_2 = None
    _pool2 = None

    _conv3_1 = None
    _conv3_2 = None
    _conv3_3 = None
    _conv3_4 = None
    _pool3 = None

    _conv4_1 = None
    _conv4_2 = None
    _conv4_3 = None
    _conv4_4 = None
    _pool4 = None

    _conv5_1 = None
    _conv5_2 = None
    _conv5_3 = None
    _conv5_4 = None
    _pool5 = None

    _fc6 = None
    _relu6 = None

    _fc7 = None
    _relu7 = None

    _fc8 = None

    _preds = None
    "The predictions tensor, shape of [?, 1000]"

    _loss = None
    _optimizer = None
    _train_labels = None
    "The train labels tensor, a placeholder."

    def __init__(self,
                 model=None,
                 model_save_path=None,
                 model_save_freq=0):
        """
        :param model: The model either for back-propagation or
        :param model_save_path: The model path for training process.
        :param model_save_freq: Save the model (in training process) every N
        iterations.
        forward-propagation.
        """
        self.model = self._init_empty_model() if not model else model
        self.model_save_path = model_save_path
        self.model_save_freq = model_save_freq

        # Define the train labels.
        self._train_labels = tf.placeholder(tf.float32,
                                            [None, 1000])

        # Define the input placeholder with RGB channels.
        # Size: 224x224x3
        self._inputRGB = tf.placeholder(tf.float32,
                                        [None,
                                         Vgg19.WIDTH,
                                         Vgg19.HEIGHT,
                                         Vgg19.CHANNELS])

        # Convert RGB to BGR order
        # Size: 224x224x3
        red, green, blue = tf.split(3, 3, self._inputRGB)
        self._inputBGR = tf.concat(3, [
            blue,
            green,
            red,
        ])

        # normalize the input so that the elements all have nearly equal
        # variances.
        # Size: 224x224x3
        self._inputNormalizedBGR = tf.concat(3, [
            blue - _VGG_MEAN[0],
            green - _VGG_MEAN[1],
            red - _VGG_MEAN[2],
        ])

        # Setup the VGG-Net graph.
        # Size: 224x224x64
        self._conv1_1 = self._conv_layer(self._inputNormalizedBGR, "conv1_1")
        self._conv1_2 = self._conv_layer(self._conv1_1, "conv1_2")
        # Size: 112x112x64
        self._pool1 = self._max_pool(self._conv1_2, 'pool1')

        # Size: 112x112x128
        self._conv2_1 = self._conv_layer(self._pool1, "conv2_1")
        self._conv2_2 = self._conv_layer(self._conv2_1, "conv2_2")
        # Size: 56x56x128
        self._pool2 = self._max_pool(self._conv2_2, 'pool2')

        # Size: 56x56x256
        self._conv3_1 = self._conv_layer(self._pool2, "conv3_1")
        self._conv3_2 = self._conv_layer(self._conv3_1, "conv3_2")
        self._conv3_3 = self._conv_layer(self._conv3_2, "conv3_3")
        self._conv3_4 = self._conv_layer(self._conv3_3, "conv3_4")
        # Size: 28x28x256
        self._pool3 = self._max_pool(self._conv3_4, 'pool3')

        # Size: 28x28x512
        self._conv4_1 = self._conv_layer(self._pool3, "conv4_1")
        self._conv4_2 = self._conv_layer(self._conv4_1, "conv4_2")
        self._conv4_3 = self._conv_layer(self._conv4_2, "conv4_3")
        self._conv4_4 = self._conv_layer(self._conv4_3, "conv4_4")
        # Size: 14x14x512
        self._pool4 = self._max_pool(self._conv4_4, 'pool4')

        # Size: 14x14x512
        self._conv5_1 = self._conv_layer(self._pool4, "conv5_1")
        self._conv5_2 = self._conv_layer(self._conv5_1, "conv5_2")
        self._conv5_3 = self._conv_layer(self._conv5_2, "conv5_3")
        self._conv5_4 = self._conv_layer(self._conv5_3, "conv5_4")
        # Size: 7x7x512
        self._pool5 = self._max_pool(self._conv5_4, 'pool5')

        # Size: 25088(=7x7x512)x4096
        self._fc6 = self._fc_layer(self._pool5, "fc6")
        self._relu6 = tf.nn.relu(self._fc6)

        # Size: 4096x4096
        self._fc7 = self._fc_layer(self._relu6, "fc7")
        self._relu7 = tf.nn.relu(self._fc7)

        # Size: 4096x1000
        self._fc8 = self._fc_layer(self._relu7, "fc8")

        # For predicting.
        self._preds = tf.nn.softmax(self._fc8, name="prediction")

        # For training.
        self._loss = tf.nn.softmax_cross_entropy_with_logits(self._fc8,
                                                             self._train_labels)
        self._optimizer = tf.train \
            .GradientDescentOptimizer(self.learning_rate) \
            .minimize(self._loss)

    @property
    def inputRGB(self):
        """
        The input RGB images tensor of channels in RGB order.
        Shape must be of [?, 224, 224, 3]
        """
        return self._inputRGB

    @property
    def inputBGR(self):
        """
        The input RGB images tensor of channels in BGR order.
        Shape must be of [?, 224, 224, 3]
        """
        return self._inputBGR

    @property
    def preds(self):
        """
        The prediction(s) tensor, shape of [?, 1000].
        """
        return self._preds

    @property
    def train_labels(self):
        """
        The train labels tensor, shape of [?, 1000].
        """
        return self._train_labels

    @property
    def loss(self):
        """
        The loss tensor.
        """
        return self._loss

    @property
    def optimizer(self):
        """
        The optimizer tensor, used for the training.
        """
        return self._optimizer

    def _avg_pool(self, value, name):
        return tf.nn.avg_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def _max_pool(self, value, name):
        return tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def _conv_layer(self, value, name):
        with tf.variable_scope(name):
            filt = self._get_conv_filter(name)

            conv = tf.nn.conv2d(value, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self._get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def _fc_layer(self, value, name):
        with tf.variable_scope(name):
            shape = value.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(value, [-1, dim])

            weights = self._get_fc_weight(name)
            biases = self._get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def _get_conv_filter(self, name):
        return tf.Variable(self.model[name][0], name="filter")

    def _get_bias(self, name):
        return tf.Variable(self.model[name][1], name="biases")

    def _get_fc_weight(self, name):
        return tf.Variable(self.model[name][0], name="weights")

    def _init_empty_model(self):
        self.model = {
            # All the following things follows [0] = weights, [1] = biases.
            # Conv-layer 1.
            "conv1_1": [np.ndarray([3, 3, 3, 64]),
                        np.ndarray([64])],
            "conv1_2": [np.ndarray([3, 3, 64, 64]),
                        np.ndarray([64])],
            # Conv-layer 2.
            "conv2_1": [np.ndarray([3, 3, 64, 128]),
                        np.ndarray([128])],
            "conv2_2": [np.ndarray([3, 3, 128, 128]),
                        np.ndarray([128])],
            # Conv-layer 3.
            "conv3_1": [np.ndarray([3, 3, 128, 256]),
                        np.ndarray([256])],
            "conv3_2": [np.ndarray([3, 3, 256, 256]),
                        np.ndarray([256])],
            "conv3_3": [np.ndarray([3, 3, 256, 256]),
                        np.ndarray([256])],
            "conv3_4": [np.ndarray([3, 3, 256, 256]),
                        np.ndarray([256])],
            # Conv-layer 4.
            "conv4_1": [np.ndarray([3, 3, 256, 512]),
                        np.ndarray([512])],
            "conv4_2": [np.ndarray([3, 3, 512, 512]),
                        np.ndarray([512])],
            "conv4_3": [np.ndarray([3, 3, 512, 512]),
                        np.ndarray([512])],
            "conv4_4": [np.ndarray([3, 3, 512, 512]),
                        np.ndarray([512])],
            # Conv-layer 5.
            "conv5_1": [np.ndarray([3, 3, 512, 512]),
                        np.ndarray([512])],
            "conv5_2": [np.ndarray([3, 3, 512, 512]),
                        np.ndarray([512])],
            "conv5_3": [np.ndarray([3, 3, 512, 512]),
                        np.ndarray([512])],
            "conv5_4": [np.ndarray([3, 3, 512, 512]),
                        np.ndarray([512])],
            # FC layer.
            "fc6": [np.ndarray([25088, 4096]),
                    np.ndarray([4096])],
            "fc7": [np.ndarray([4096, 4096]),
                    np.ndarray([4096])],
            "fc8": [np.ndarray([4096, 1000]),
                    np.ndarray([1000])]}

        return self.model
