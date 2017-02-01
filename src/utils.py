import numpy as np
import pandas
from scipy.misc import imresize
from scipy import ndimage


def load_images(*paths):
    """
    Load multiple images.
    :param paths: The image paths.
    """
    imgs = []
    for path in paths:
        img = ndimage.imread(path, mode="RGB").astype(float)
        imgs.append(img)
    return imgs


def load_lfw_attributes(data_dir):
    """Loads the lfw attribute file

    :return: Pandas dataframe containing the lfw attributes for each image
    """
    path = '{}/lfw_attributes.txt'.format(data_dir)
    df = pandas.read_csv(path, sep='\t')

    paths = []

    if data_dir.endswith('/'):
        data_dir = data_dir[:-1]

    for idx, row in df.iterrows():
        name = row[0]
        img_idx = str(row[1])
        name = name.replace(' ', '_')

        while len(img_idx) < 4:
            img_idx = '0' + img_idx

        path = '{0}/lfw-deepfunneled/{1}/{1}_{2}.jpg'.format(data_dir, name,
                                                             img_idx)
        paths.append(path)
    df['path'] = paths
    del df['imagenum']
    return df

def get_person_idx_by_path(df, path):
    """Gets the index of a given person image"""
    try:
        index = df[df['path'] == path].index.tolist()[0]
    except Exception:
        raise Exception('Image with the path {} could not be found'.format(path))
    return index

def load_discrete_lfw_attributes(data_dir):
    """Loads the discretized lfw attributes

    :return: Discretized lfw attributes
    """
    df = load_lfw_attributes(data_dir)

    for column in df:
        if column == 'person' or column == 'path':
            continue
        df[column] = df[column].apply(np.sign)

    return df


def reduce_img_size(imgs):
    cropped = [img[50:-50,50:-50] for img in imgs]
    resized = [imresize(img, (224,224)) for img in cropped]
    # for idx, img in enumerate(imgs):
    #     imgs[idx] = img[13:-13, 13:-13]
    return resized


def load_model(model_path):
    print('Loading model at {}'.format(model_path))
    return np.load(model_path, encoding='latin1').item()
