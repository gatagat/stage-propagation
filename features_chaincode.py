from collections import Counter
import numpy as np
import os
from PIL import Image
import Pycluster as pcl

import tsh; logger = tsh.create_logger(__name__)

def get_chaincode_from_image(filename, scale):
    #print 'Computing chaincode from %s at scale %d' % (filename, scale)
    assert scale >= 1
    mask = tsh.read_gray_image(filename)
    normal_size = (800/scale, 400/scale)
    mask = tsh.pil_to_array(Image.fromarray(mask, 'L').resize(normal_size, Image.ANTIALIAS))[::-1, :]
    contour = tsh.bwboundaries(mask > 127)
    contour = np.fliplr(contour) # (X, Y) -> (Y, X)
    try:
        cc, _ = tsh.chaincode(contour)
    except ValueError:
        logger.error(filename)
        raise
    return cc

def project_single(chaincode, dictionary, substring_length, normalize_chaincode, do_cache=False, cache=None):
    if do_cache and cache == None:
        cache = {}
    hist = np.zeros(len(dictionary))
    chaincode = np.r_[chaincode, chaincode[:substring_length]]
    for start in range(len(chaincode)-substring_length):
        word = chaincode[start:start+substring_length]
        if normalize_chaincode:
            word = tsh.normalize_chaincode(word)
        n = None
        if do_cache:
            cc = tsh.chaincode_to_string(word)
            if cc in cache:
                n = cache[cc]
                #logger.debug('Cache hit for ' + str(word))
        if n == None:
            dists = np.array([ (word != w).sum() for w in dictionary ])
            n = dists == dists.min()
            if do_cache:
                cache[cc] = n
        hist[n] += 1. / n.sum()
    return hist, cache

def get_chaincode_features(sample, dictionary=None, cache=None, bow_options=None, **kwargs):
    assert bow_options != None
    assert dictionary != None

    mask_prefix = os.path.expanduser(kwargs['mask_prefix'])

    # Calculate chaincode
    chaincode = {}
    for scale in bow_options['scales']:
        chaincode[scale] = get_chaincode_from_image(os.path.join(mask_prefix, sample['mask']), scale)

    # Project
    dictionary_size = np.sum([ len(d) for d in dictionary.values() ])
    hist = np.zeros(dictionary_size, dtype=np.float64)
    pos = 0
    for scale in bow_options['scales']:
        if scale not in cache:
            cache[scale] = None
        h, cache[scale] = project_single(
                chaincode[scale],
                dictionary[scale],
                bow_options['substring_length'],
                bow_options['normalize_chaincode'],
                bow_options['project_do_cache'],
                cache[scale])
        hist[pos:pos+len(dictionary[scale])] = h / h.sum()
        pos += len(dictionary[scale])

    return hist


def cluster(D, k):
    labels, _, _ = pcl.kmedoids(D, nclusters=k, npass=10, initialid=None)
    errors = np.array([ D[labels[i], i] for i in range(len(labels)) ])
    centroidids = np.unique(labels)
    cmap = np.zeros(labels.max()+1)
    for c in centroidids:
        cmap[c] = np.nonzero(centroidids == c)[0][0]
    labels = cmap[labels]
    logger.debug('k-medoids (k=%i): %.2f.' % (k, errors.sum()))
    return labels, { 'method': 'kmedoids',
                     'init': 'random',
                     'k': k,
                     'centroidids': centroidids,
                     'errors': errors,
                     'error': errors.sum(),
                     'error-label': 'sum of distances' }


def select_dictionary_entries(D, **kwargs):
    if kwargs['dictionary_method'] == 'cluster':
        k = kwargs['cluster_dictionary_size']
        clustering = cluster(D, k)
        return clustering[1]['centroidids'], [ (clustering[0] == i).sum() for i in range(k) ]
    elif kwargs['dictionary_method'] == 'vq':
        A = (D < kwargs['vq_distance_threshold']).astype(np.float64)
        if 'lpname' in kwargs.keys():
            logger.info('Saving LP problem')
            tsh.serialize(kwargs['lpname'], { 'D': D, 'A': A })
        indices = tsh.vq(A, verbose=True, solver='clp')
        counts = np.zeros(len(indices))
        for i in range(D.shape[0]):
            representatives = D[i, indices] == D[i, indices].min()
            counts[representatives] += 1. / representatives.sum()
        return indices, counts
    elif kwargs['dictionary_method'] == 'random':
        indices = np.random.permutation(D.shape[0])[:kwargs['random_dictionary_size']]
        counts = np.zeros(len(indices))
        for i in range(D.shape[0]):
            representatives = D[i, indices] == D[i, indices].min()
            counts[representatives] += 1. / representatives.sum()
        return indices, counts

    raise NotImplementedError


def substring_histogram(data, n=8, grouping=None, circular=True, normalize_chaincode=False):
    count = Counter()
    if grouping != None:
        assert len(grouping) == len(data)
        group_count = dict([ (g, Counter()) for g in np.unique(grouping) ])
    i = 0
    for i in range(len(data)):
        d = data[i]
        string = np.r_[d, d[:n]]
        cc = []
        for start in range(len(string)-n+1):
            s = string[start:start+n]
            if normalize_chaincode:
                s = tsh.normalize_chaincode(s)
            cc += [ tsh.chaincode_to_string(s) ]

        count.update(cc)
        if grouping != None:
            group_count[grouping[i]].update(cc)
        i += 1

    histogram_dtype = [ ('word', 'O'), ('count', int) ]
    if grouping != None:
        histogram_dtype += [ ('group%d' % g, int) for g in np.unique(grouping) ] + [ ('idf', float) ]
        histogram = np.zeros(len(count), dtype=histogram_dtype)
        i = 0
        for cc, n in count.items():
            h = histogram[i]
            h['word'] = tsh.string_to_chaincode(cc)
            h['count'] = n
            df = 0
            for g in group_count.keys():
                if cc in group_count[g]:
                    h['group%d' % g] = group_count[g][cc]
                    df += 1
            h['idf'] = np.log(len(group_count) / float(df))
            i += 1
    else:
        histogram = np.array([ (tsh.string_to_chaincode(cc), n) for cc, n in count.items()], dtype=histogram_dtype)

    return histogram


def train_dictionary(chaincodes, labels=None, **kwargs):
    substring_length = kwargs['substring_length']
    substring_count = kwargs['substring_count']
    scales = kwargs['scales']

    dictionary = {}
    counts = {}
    for scale in scales:
        if kwargs['substring_selection'] == 'tf':
            hist = substring_histogram(chaincodes['cc%02d' % scale], n=substring_length, normalize_chaincode=kwargs['normalize_chaincode'])
            logger.info('Obtained %d different substrings', len(hist))
            sort_indices = np.argsort(hist['count'])[::-1]
        elif kwargs['substring_selection'] == 'tfidf-embryos':
            cc = chaincodes['cc%02d' % scale]
            hist = substring_histogram(cc, grouping=range(len(cc)), n=substring_length, normalize_chaincode=kwargs['normalize_chaincode'])
            logger.info('Obtained %d different substrings', len(hist))
            sort_indices = np.argsort(hist['count'] * hist['idf'])[::-1]
        elif kwargs['substring_selection'] == 'tfidf-classes':
            assert labels != None
            cc = chaincodes['cc%02d' % scale]
            hist = substring_histogram(cc, grouping=labels, n=substring_length, normalize_chaincode=kwargs['normalize_chaincode'])
            logger.info('Obtained %d different substrings', len(hist))
            sort_indices = np.argsort(hist['count'] * hist['idf'])[::-1]
        else:
            raise NotImplementedError
        hist = hist[sort_indices]

        #visualize_substring_hist('cc%02d-at%02d-most-frequent' % (substring_length, scale), hist[:substring_count])

        logger.info('Computing distances at scale %d...', scale)
        if substring_count == np.inf:
            words = hist['word']
        else:
            words = hist['word'][:substring_count]
        D = tsh.pdist2(words, words, distance=tsh.hamming)

        logger.info('Clustering at scale %d...', scale)
        sel = select_dictionary_entries(D, **kwargs)
        dictionary[scale] = words[sel[0]]
        counts[scale] = sel[1]

    logger.info('Dictionary done (%s).', ', '.join([str(len(dictionary[scale])) for scale in kwargs['scales']]))
    return dictionary, counts


def prepare_chaincode_features(data, dictionary_name=None, bow_options_name=None, input_name=None, output_dir=None, **kwargs):
    assert bow_options_name != None
    assert dictionary_name != None
    assert input_name != None
    assert output_dir != None

    bow_options = tsh.deserialize(bow_options_name)

    dictionary_name = dictionary_name.format(OUT=output_dir)
    ret = { 'bow_options': bow_options, 'dictionary_name': dictionary_name }
    if os.path.exists(dictionary_name):
        ret.update(tsh.deserialize(dictionary_name))
        ret['feature_names'] = [ 's%02di%03d' % (scale, i) for scale in bow_options['scales'] for i in range(len(ret['dictionary'][scale])) ]
        return ret

    # Compute dictionary - for this we need the chaincodes
    chaincode_dtype = [ ('id', data.dtype['id']) ] + zip(
                [ 'cc%02d' % scale for scale in bow_options['scales'] ],
                ['O'] * len(bow_options['scales']))
    chaincodes = np.zeros(len(data), dtype=chaincode_dtype)
    chaincodes['id'] = data['id']
    mask_prefix = os.path.expanduser(kwargs['mask_prefix'])
    for scale in bow_options['scales']:
        scale_name = 'cc%02d' % scale
        for i in range(len(data)):
            chaincodes[i][scale_name] = get_chaincode_from_image(os.path.join(mask_prefix, data[i]['mask']), scale)
    tsh.serialize(os.path.join(output_dir, input_name + '-chaincodes.dat'), chaincodes)
    lpname = os.path.join(output_dir, input_name + '-lp.dat')
    dictionary = dict(zip(
        ['dictionary', 'counts'],
        train_dictionary(chaincodes, lpname=lpname, labels=data[kwargs['truth']], **bow_options)))
    tsh.serialize(dictionary_name, dictionary)
    ret.update(dictionary)
    ret['feature_names'] = [ 's%02di%03d' % (scale, i) for scale in bow_options['scales'] for i in range(len(ret['dictionary'][scale])) ]
    return ret

