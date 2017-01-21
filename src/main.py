import argparse
import os.path

import numpy as np

from DFI import DFI


def parse_arg():
    """Parse commandline arguments
    :return: argument dict
    """
    print('Parsing arguments')
    parser = argparse.ArgumentParser('Deep Feature Interpolation')
    parser.add_argument('--data_dir', '-d', default='data', type=str,
                        help='Path to data directory containing the images')
    parser.add_argument('--model_path', '-m', default='model/vgg19.npy',
                        type=str,
                        help='Path to the model file (*.npy)')
    parser.add_argument('--gpu', '-g', default=False, action='store_true',
                        help='Enable gpu computing')
    parser.add_argument('--num_layers', '-n', default=3, type=int,
                        help='Number of layers. One of {1,2,3}')
    parser.add_argument('--feature', '-f', default='No Beard', type=str,
                        help='Name of the Feature.')
    parser.add_argument('--person_index', '-p', default=0, type=int,
                        help="Index of the start image.")
    parser.add_argument('--list_features', '-l', default=False,
                        action='store_true', help='List all available '
                                                  'features.')
    parser.add_argument('--tf', default=True, action='store_true',
                        help="Use Tensorflow for the optimization step")
    args = vars(parser.parse_args())

    # Check argument constraints
    if args['num_layers'] not in np.arange(1, 4):
        raise argparse.ArgumentTypeError(
            "%s is an invalid int value. (1 <= n <= 3)" % args['num_layers'])

    if not os.path.exists(args['data_dir']):
        raise argparse.ArgumentTypeError(
            "Directory %s does not exist." % args['data_dir'])

    if not os.path.exists(args['model_path']):
        raise argparse.ArgumentTypeError(
            "%File s does not exist." % args['model_path'])

    return args


def main():
    """
    Main method
    :return: None
    """
    # Get args
    args = parse_arg()

    # Init DFI
    print('Creating DFI Object')
    dfi = DFI(**args)

    # List features
    if args['list_features']:
        print('Available features to select:')
        print(' - ' + '\n - '.join(sorted(dfi.features())))
        exit(0)

    # Run
    dfi.run(feat=args['feature'], person_index=args['person_index'])


if __name__ == '__main__':
    main()
