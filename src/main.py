import argparse
import os.path
from pprint import pprint

import numpy as np

from DFI import DFI


def parse_arg():
    """Parse commandline arguments
    :return: argument dict
    """
    print('Parsing arguments')
    parser = argparse.ArgumentParser('Deep Feature Interpolation')
    parser.add_argument('--data-dir', '-d', default='data', type=str,
                        help='Path to data directory containing the images')
    parser.add_argument('--model-path', '-m', default='model/vgg19.npy',
                        type=str,
                        help='Path to the model file (*.npy)')
    parser.add_argument('--gpu', '-g', default=False, action='store_true',
                        help='Enable gpu computing')
    parser.add_argument('--num-layers', '-n', default=3, type=int,
                        help='Number of layers. One of {1,2,3}')
    parser.add_argument('--feature', '-f', default='No Beard', type=str,
                        help='Name of the Feature.')
    parser.add_argument('--person-index', '-p', default=0, type=int,
                        help="Index of the start image.")
    parser.add_argument('--person-image', default='lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg', type=str,
                        help="Start image path.")
    parser.add_argument('--list-features', '-l', default=False,
                        action='store_true', help='List all available '
                                                  'features.')
    parser.add_argument('--optimizer', '-o', type=str, help='Optimizer type')

    parser.add_argument('--lr', type=float, default=1,
                        help='Learning rate interval in log10')
    parser.add_argument('--steps', '-s', type=int, default=2000,
                        help='Number of steps')
    parser.add_argument('--eps', '-e', type=str, help='Epsilon interval in log10')
    parser.add_argument('--tk', help='Use TkInter', default=False, action='store_true')
    parser.add_argument('--k', '-k', help='Number of nearest neighbours', type=int, default=10)
    parser.add_argument('--alpha', '-a', help='Alpha param', type=float, default=0.4)
    parser.add_argument('--beta', '-b', help='Beta param', type=float, default=2)
    parser.add_argument('--lamb', help='Lambda param', type=float, default=0.001)
    parser.add_argument('--rebuild-cache', '-rc', help='Rebuild the cache', default=False, action='store_true')
    parser.add_argument('--random-start', '-rs', help='Use random start_img', default=False, action='store_true')
    parser.add_argument('--verbose', '-v', help='Set verbose', default=False, action='store_true')
    parser.add_argument('--invert', '-i', help='Invert deep feature difference (No Beard -> Beard)', default=False, action='store_true')
    args = parser.parse_args()

    # Check argument constraints
    if args.num_layers not in np.arange(1, 4):
        raise argparse.ArgumentTypeError(
            "%s is an invalid int value. (1 <= n <= 3)" % args.num_layers)

    if not os.path.exists(args.data_dir):
        raise argparse.ArgumentTypeError(
            "Directory %s does not exist." % args.data_dir)

    if not os.path.exists(args.model_path):
        raise argparse.ArgumentTypeError(
            "%File s does not exist." % args.model_path)

    return args


def main():
    """
    Main method
    :return: None
    """
    # Get args
    FLAGS = parse_arg()

    # Init DFI
    print('Creating DFI Object')
    dfi = DFI(FLAGS)

    # List features
    if FLAGS.list_features:
        print('Available features to select:')
        print(' - ' + '\n - '.join(sorted(dfi.features())))
        exit(0)

    # Run
    dfi.run(feat=FLAGS.feature, person_index=FLAGS.person_index)


if __name__ == '__main__':
    main()
