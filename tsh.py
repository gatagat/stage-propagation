import os
import errno
import bz2
import cPickle as pickle
import numpy as np
import itertools
import matplotlib.pylab as plt

from ._chaincode import chaincode, chaincode_to_string, string_to_chaincode, normalize_chaincode
from ._distances import pdist2
from ._bw import bwareaopen, bwboundaries, bwperim


PICKLE_FORMAT = 0x01
YAML_FORMAT = 0x02
JOBLIB_FORMAT = 0x04
BZIP2_FORMAT = 0x80
DEFAULT_FORMAT = PICKLE_FORMAT | BZIP2_FORMAT
try:
    import yaml
    has_yaml = True
except ImportError:
    has_yaml = False
try:
    from joblib import dump as joblib_dump
    from joblib import load as joblib_load
    has_joblib = True
    DEFAULT_FORMAT = JOBLIB_FORMAT
except ImportError:
    has_joblib = False


def _setup_log():
    """Configure tsh logger.

    """
    import logging
    import sys

    logging.basicConfig()
    log = logging.getLogger()
    log.handlers = []
    try:
        handler = logging.StreamHandler(stream=sys.stderr)
    except TypeError:
        handler = logging.StreamHandler(strm=sys.stderr)
    formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y.%m.%d %H:%M:%S'
        )
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.WARNING)


_setup_log()


def create_logger(name, level=None):
    import logging
    if name is None:
        name = 'tsh'
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def makedirs(path):
    try:
        os.makedirs(path)
        return True
    except OSError as e:
        if e.errno == errno.EEXIST:
            return False
        else:
            raise


def serialize(filename, obj, format=DEFAULT_FORMAT):
    if format & JOBLIB_FORMAT:
        if not has_joblib:
            raise RuntimeError(
                    'Missing library. Format (JOBLIB_FORMAT) not available.')
        joblib_dump(obj, filename, compress=3 if format & BZIP2_FORMAT else 0)
        return
    if format & BZIP2_FORMAT:
        open_fn = bz2.BZ2File
    else:
        open_fn = open
    with open_fn(filename, 'wb') as f:
        if format & PICKLE_FORMAT:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        elif format & YAML_FORMAT:
            if not has_yaml:
                raise RuntimeError(
                        'Missing library. Format (YAML_FORMAT) not available.')
            yaml.dump(obj, stream=f)
        else:
            raise ValueError('Unknown format value.')


def deserialize(filename, format=DEFAULT_FORMAT):
    if not os.path.exists(filename):
        raise RuntimeError('File %s does not exist' % filename)
    if format & JOBLIB_FORMAT:
        if not has_joblib:
            raise RuntimeError(
                    'Missing library. Format (JOBLIB_FORMAT) not available.')
        return joblib_load(filename)
    if format & BZIP2_FORMAT:
        open_fn = bz2.BZ2File
    else:
        open_fn = open
    with open_fn(filename, 'rb') as f:
        if format & PICKLE_FORMAT:
            return pickle.load(f)
        elif format & YAML_FORMAT:
            if not has_yaml:
                raise RuntimeError(
                        'Missing library. Format (YAML_FORMAT) not available.')
            return yaml.load(f)
        else:
            raise ValueError('Unknown format value.')


def dict_values(d, keys):
    return [d[k] for k in keys if k in d.keys()]


def standardize_image(image):
    image -= image.mean()
    image /= max(image.std(), 1e-20)


def read_config(opts=None, file=None, throw=False):
    from ConfigParser import ConfigParser

    config = ConfigParser()
    if hasattr(opts, '__dict__') and 'config' in vars(opts).keys() \
            and opts.config is not None:
        filename = opts.config
    elif file is not None:
        filename = os.path.join(
                os.path.dirname(os.path.abspath(file)), '.config')
    else:
        raise 'No config available'
    if throw and not os.path.exists(filename):
        raise IOError('File does not exist: %s' % filename)
    config.read(filename)
    return config


def read_gray_image(path, check=True):
    import cv2 as cv

    image = cv.imread(path, cv.CV_LOAD_IMAGE_GRAYSCALE)
    if check and image is None:
        print('Error loading image %s', path)
        raise IOError
    return image


def supervised_confusion_matrix(true, pred, labels=None):
    if labels is None:
        labels = np.unique(sorted(set(true) | set(pred)))
    labels_idx = dict([(v, k) for k, v in enumerate(labels)])
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int)
    for t, p in zip(true, pred):
        confusion[labels_idx[t], labels_idx[p]] += 1
    return confusion


def supervised_recall(true, pred, labels=None):
    if labels is None:
        labels = np.unique(sorted(set(true) | set(pred)))
    return [((pred == s) * (true == s)).sum() / ((true == s).sum() + 1e-10)
            for s in labels]


def supervised_precision(true, pred, labels=None):
    if labels is None:
        labels = np.unique(sorted(set(true) | set(pred)))
    return [((pred == s) * (true == s)).sum() / ((pred == s).sum() + 1e-10)
            for s in labels]


def supervised_accuracy(true, pred):
    return (pred == true).sum() / float(len(pred) + 1e-10)


def unsupervised_jaccard_coefficient(true, pred):
    import sklearn
    import sklearn.metrics
    contingency = sklearn.metrics.cluster.contingency_matrix(true, pred)

    def comb2(n):
        from scipy.misc import comb
        return comb(n, 2, exact=True)
    a = sum([comb2(cij) for cij in contingency.flatten()])
    a_plus_b_plus_c = sum([comb2(ci) for ci in contingency.sum(axis=1)]) \
        + sum([comb2(cj) for cj in contingency.sum(axis=0)]) - a
    jaccard = a / (a_plus_b_plus_c + 1e-10)
    return jaccard


def stratified_indices(X, min_count=None):
    from collections import Counter
    counts = Counter(X)
    #logger.debug(str(counts))
    k = min(counts.values())
    if min_count == None:
        min_count = k
    #logger.debug('k=%d, min_count=%d', k, min_count)
    sampled = []
    if isinstance(X, np.ndarray):
        X_a = X
    else:
        X_a = np.array(X)
    for l in counts.keys():
        if min_count > counts[l]:
            indices = np.sort(np.r_[np.arange(counts[l]), np.random.random_integers(0, high=counts[l]-1, size=min_count - counts[l])])
        else:
            indices = np.sort(np.random.permutation(np.arange(counts[l]))[:min_count])
        indices = np.nonzero(X_a == l)[0][indices]
        sampled += indices.tolist()
    #logger.debug('len=%d', len(sampled))
    return sampled


def plot_all_scatterplots(data, names, c=None, bins=10, scatter_alpha=None):
    numdata, numvars = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    target = None
    if c is not None and len(c) == len(data):
        unique_c = np.unique(c)
        if len(unique_c) < 6: # draw a stacked histogram only with small number of colors
            target_classes = range(len(unique_c))
            c2target = dict(zip([str(u) for u in unique_c], target_classes))
            target = np.array([c2target[str(u)] for u in c])

    for x, y in zip(*np.triu_indices_from(axes, k=1)):
        axes[x, y].scatter(data[:, x], data[:, y], c=c, alpha=scatter_alpha)
        axes[y, x].scatter(data[:, y], data[:, x], c=c, alpha=scatter_alpha)
    for x in range(numvars):
        if target is None:
            axes[x, x].hist(data[:, x], bins=bins)
        else:
            axes[x, x].hist([data[target == t, x] for t in target_classes], histtype='barstacked', color=unique_c)
	axes[x, x].annotate(names[x], (.95, .95), xycoords='axes fraction', ha='right', va='top')
    for x, y in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[y, x].xaxis.set_visible(True)
        axes[x, y].yaxis.set_visible(True)
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.92, top=0.92, hspace=0.05, wspace=0.05)
    return fig


def scatterplot_matrix(data, names, c=None, bins=10):
    from warnings import warn
    warn('scatterplot_matrix got renamed to plot_all_scatterplots', DeprecationWarning, stacklevel=2)
    plot_all_scatterplots(data, names, c=c, bins=bins)


def plot_confusion_matrix(conf, labels=None, color=None, xlabel=None, ylabel=None):
    if xlabel == None:
        xlabel = 'Predicted'
    if ylabel == None:
        ylabel = 'True'
    import matplotlib.pyplot as plt
    if color == None or color == 'recall':
        norm_conf = [[float(j)/(float(sum(i, 0))+1e-10) for j in i] for i in conf]
    elif color == 'precision':
        norm_conf = np.transpose([[float(j)/(float(sum(i, 0))+1e-10) for j in i] for i in np.transpose(conf)])
    else:
        raise ValueError
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, vmin=0., vmax=1., interpolation='nearest')
    N = len(conf)
    for y in xrange(N):
        assert len(conf[y]) == N
        for x in xrange(N):
            ax.annotate(str(conf[y][x]), xy=(x, y),
                        horizontalalignment='center',
                        verticalalignment='center', size='small')
    fig.colorbar(res)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels == None:
        labels = [ str(n) for n in range(N) ]
    if np.any([ len(l) > 8 for l in labels ]):
        plt.xticks(range(N), labels, rotation=90, size='small')
        plt.yticks(range(N), labels, size='small')
        plt.subplots_adjust(left=0.05, bottom=0.30, right=0.75, top=0.90, hspace=0.20, wspace=0.20)
    else:
        plt.xticks(range(N), labels)
        plt.yticks(range(N), labels)
        plt.subplots_adjust(left=0.05, bottom=0.10, right=0.95, top=0.90, hspace=0.20, wspace=0.20)
    return fig
