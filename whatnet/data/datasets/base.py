import urllib.request
import os
import shutil
import logging
import collections

DataSet = collections.namedtuple('Dataset', ['data', 'target'])
DataSets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def download_if_not_exist(url, working_directory, filename):
    if not os.path.isdir(working_directory):
        os.makedirs(working_directory)
    filepath = os.path.join(working_directory, filename)
    if not os.path.isfile(filepath):
        with urllib.request.urlopen(url) as response, open(filepath, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        logging.info("Successfully download %s", filepath)
    return filepath
            