from matplotlib.pylab import csv2rec
import numpy as np
import yaml

import tsh; logger = tsh.create_logger(__name__)

def read_listfile(filename):
    """
    Reads in data sample list stored as a CSV file with optional meta-data.

    Parameters:
    -----------
    filename: string
        Input filename.

    Returns:
    --------
    meta: dict
        Meta-data from the header.
    data: recarray
        Data from the CSV table.
    """
    with open(filename, 'r') as f:
        meta = [ line[1:] for line in f.readlines() if line.startswith('#') ]
    meta = yaml.load(''.join(meta))
    data = csv2rec(filename, delimiter='\t', comments='#')
    assert 'id' in data.dtype.names
    assert len(data) == len(np.unique(data['id']))
    return meta, data


def read_truthfile(filename):
    meta, data = read_listfile(filename)
    return meta, data['id'].tolist(), data[meta['truth']]


def read_argsfile(filename):
    """
    Reads in arguments from file.

    Parameters:
    -----------
    filename: string
        Input filename.
    """
    return tsh.deserialize(filename)


def read_featurefile(filename):
    """
    """
    meta, features = read_listfile(filename)
    feature_names = np.array(features.dtype.names)
    feature_names = feature_names[feature_names != 'id'].tolist()
    meta['feature_names'] = feature_names
    return meta, features['id'].tolist(), features[feature_names].view(np.float64).reshape(len(features), -1)


def _dump_meta(filespec, meta):
    def dump_to_file_obj(f, meta):
        meta = yaml.dump(meta)
        meta = meta.split('\n')
        for line in meta:
            f.write('#%s\n' % line)

    if isinstance(filespec, str):
        with open(filespec, 'w') as f:
            dump_to_file_obj(f, meta)
    else:
        dump_to_file_obj(filespec, meta)


def write_featurefile(filename, sample_ids, features, feature_names=None, **kwargs):
    """
    Writes features into a CSV file with optional additional meta-data.

    Parameters:
    -----------
    filename: string
        Output filename
    sample_ids: list
        Unique data IDs.
    features: float ndarray
        Features - different features in columns, different data samples.
    feature_names: list of strings
        Names of different features in the same order as columns of the features matrix.
    kwargs
        Any additional meta-data.
    """
    assert feature_names != None

    with open(filename, 'w') as f:
        _dump_meta(f, kwargs)
        f.write('ID\t' + '\t'.join(feature_names) + '\n')
        for sample_id, feature in zip(sample_ids, features):
            f.write(str(sample_id) + '\t' + '\t'.join([str(v) for v in feature]) + '\n')

def write_predfile(filename, sample_ids, pred, column_names=None, **kwargs):
    assert column_names != None
    with open(filename, 'w') as f:
        _dump_meta(f, kwargs)
        f.write('ID\t' + '\t'.join(column_names) + '\n')
        for sample_id, feature in zip(sample_ids, pred):
            f.write(str(sample_id) + '\t' + '\t'.join([str(v) for v in feature]) + '\n')

def write_classifierfile(filename, classifier):
    tsh.serialize(filename, classifier)

def read_classifierfile(filename):
    return tsh.deserialize(filename)

def clean_args(args):
    for name in args['unserialized']:
        del args[name]
    #del args['unserialized']

