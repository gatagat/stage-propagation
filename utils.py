from matplotlib.pylab import csv2rec
import numpy as np
import yaml

import tsh; logger = tsh.create_logger(__name__)


def _dump_meta(filespec, meta):
    """
    Dumps meta data into a file.

    Parameters:
    -----------
    filespec: file object, or str
        File object opened for writing in text mode, or a filename.
    meta: any YAML-serializable Python data
        Meta data to be dumped.
    """
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


def read_argsfile(filename):
    """
    Reads in arguments from file.

    Parameters:
    -----------
    filename: string
        Input filename.
    """
    return tsh.deserialize(filename)


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


def write_listfile(filename, data, **kwargs):
    """
    Writes a general list file.

    List file is a CSV file with optional meta data stored in YAML inside
    comments on the first lines of the output file.

    Delimiter is a tab character (\\t).

    Parameters:
    -----------
    filename: str
        Output filename.
    data: recarray
        Tabular data to be stored as a CSV. One of the columns has to be called 'id'.
    kwargs: dict
        Meta data.
    """
    with open(filename, 'w') as f:
        _dump_meta(f, kwargs)
        column_names = [n for n in data.dtype.names if n != 'id']
        f.write('\t'.join(['id'] + column_names) + '\n')
        for sample in data:
            f.write(str(sample['id']) + '\t' + '\t'.join([str(sample[n]) for n in column_names]) + '\n')


def read_truthfile(filename):
    """
    Reads in a truth file, which is a list file, see read_listfile.

    Parameters:
    -----------
    filename: str
        Input filename.

    Returns:
    --------
    meta, ids, truth: dict, list, 1D ndarray
        Meta data, sample IDs, truth column from the input.
    """
    meta, data = read_listfile(filename)
    return meta, data['id'].tolist(), data[meta['truth']]


def read_featurefile(filename):
    """
    Reads in a feature file, which is a listfile, see read_listfile.

    Parameters:
    -----------
    filename: str
        Input filename.

    Returns:
    --------
    meta, ids, features: dict, list, 2D ndarray
        Meta data, sample IDs, feature values (samples - rows, features - columns).
    """
    meta, features = read_listfile(filename)
    sample_ids = features['id']
    cols = [n for n in features.dtype.names if n != 'id']
    meta['feature_names'] = cols
    features = features[cols].view(np.float64).reshape(len(features), -1)
    return meta, sample_ids.tolist(), features


def read_weightsfile(filename):
    """
    Reads in a weitghts file.

    Parameters:
    -----------
    filename: str
        Input filename.

    Returns:
    --------
    meta, ids, weights: dict, list, 2D ndarray
        Meta data, sample IDs, weights matrix.
    """
    meta, weights = read_listfile(filename)
    sample_ids = weights['id']
    cols = [str(i) for i in sample_ids]
    weights = weights[cols].view(np.float64).reshape(len(weights), -1)
    return meta, sample_ids.tolist(), weights


def read_classifierfile(filename):
    return tsh.deserialize(filename)


def write_classifierfile(filename, classifier):
    tsh.serialize(filename, classifier)


def read_propagatorfile(filename):
    return tsh.deserialize(filename)


def write_propagatorfile(filename, propagator):
    tsh.serialize(filename, propagator)


def clean_args(args):
    if 'unserialized' not in args:
        return
    for name in args['unserialized']:
        if name in args:
            del args[name]

