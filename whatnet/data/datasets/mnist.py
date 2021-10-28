from .base import *

import numpy as np
import gzip
import logging


SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"
TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
TRAIN_LABELS = "train-labels-idx1-ubyte.gz"
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
TEST_LABELS = "t10k-labels-idx1-ubyte.gz"

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
    logging.info("Extractint %s", f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        return data.reshape(num_images, rows, cols, 1)

def extract_labels(f):
    logging.info('Extracting %s', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
        num_labels = _read32(bytestream)
        buf = bytestream.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

def _read_data_sets(data_directory):
    train_images = download_if_not_exist(SOURCE_URL + TRAIN_IMAGES, 
                             os.path.join(data_directory, 'mnist'), 
                             TRAIN_IMAGES)
    with open(train_images, 'rb') as f:
        train_images = extract_images(f)
        
    train_labels = download_if_not_exist(SOURCE_URL + TRAIN_LABELS, 
                             os.path.join(data_directory, 'mnist'), 
                             TRAIN_LABELS)
    with open(train_labels, 'rb') as f:
        train_labels = extract_labels(f)
    assert train_images.shape[0] == train_labels.shape[0], (
          'images.shape: {} labels.shape {}'.format(train_images.shape, train_labels.shape))
        
    test_images = download_if_not_exist(SOURCE_URL + TEST_IMAGES, 
                             os.path.join(data_directory, 'mnist'), 
                             TEST_IMAGES)
    with open(test_images, 'rb') as f:
        test_images = extract_images(f)
        
    test_labels = download_if_not_exist(SOURCE_URL + TEST_LABELS, 
                             os.path.join(data_directory, 'mnist'), 
                             TEST_LABELS)
    with open(test_labels, 'rb') as f:
        test_labels = extract_labels(f)
    assert test_images.shape[0] == test_labels.shape[0], (
          'images.shape: {} labels.shape {}'.format(test_images.shape, test_labels.shape))
        
    return DataSets(train=DataSet(data=train_images, target=train_labels),
               validation=DataSet(data=None, target=None),
               test=DataSet(data=test_images, target=test_labels))

def read_data_sets(data_directory):
    try:
        return _read_data_sets(data_directory)
    except Exception as e:
        logging.error('', exc_info=True)
        raise e