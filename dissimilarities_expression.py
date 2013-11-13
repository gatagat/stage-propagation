import cv2 as cv
from joblib import Parallel, delayed
import numpy as np
import os
from sklearn.metrics.pairwise import pairwise_distances

import tsh.obsolete as tsh; logger = tsh.create_logger(__name__)

all_measures = [ 'SSD', 'NCCone', 'NSSD', 'Jaccard', 'cosine' ]
normalized_dists = [ 'NSSD', 'NCCone', 'NCClog', 'NCCplus', 'NCCplusok' ]

def image_distance(expri, exprj, distance, rotation_invariance=False):
    if distance.endswith('hproj'):
        distance = distance[:-5]
        exprih = expri.mean(axis=0)
        exprjh = exprj.mean(axis=0)
        if distance in normalized_dists:
            tsh.standardize_image(exprih)
            tsh.standardize_image(exprjh)
        d = _image_distance(exprih, exprjh, distance)
        flip = 'I'
        if rotation_invariance:
            dlr = _image_distance(exprih, exprjh[::-1], distance)
            d, flip = min(zip([d, dlr], ['I', 'H']))
    elif distance.endswith('proj'):
        distance = distance[:-4]
        exprih = expri.mean(axis=0)
        expriv = expri.mean(axis=1)
        exprjh = exprj.mean(axis=0)
        exprjv = exprj.mean(axis=1)
        if distance in normalized_dists:
            tsh.standardize_image(exprih)
            tsh.standardize_image(expriv)
            tsh.standardize_image(exprjh)
            tsh.standardize_image(exprjv)
        expri = np.r_[exprih, expriv]
        d = _image_distance(expri, np.r_[exprjh, exprjv], distance)
        flip = 'I'
        if rotation_invariance:
            dud = _image_distance(expri, np.r_[exprjh, exprjv[::-1]], distance)
            dlr = _image_distance(expri, np.r_[exprjh[::-1], exprjv], distance)
            dlrud = _image_distance(expri, np.r_[exprjh[::-1], exprjv[::-1]], distance)
            d, flip = min(zip([d, dud, dlr, dlrud], ['I', 'V', 'H', 'X']))
    else:
        if distance in normalized_dists:
            tsh.standardize_image(expri)
            tsh.standardize_image(exprj)
        d = _image_distance(expri, exprj, distance)
        flip = 'I'
        if rotation_invariance:
            dud = _image_distance(expri, np.flipud(exprj), distance)
            dlr = _image_distance(expri, np.fliplr(exprj), distance)
            dlrud = _image_distance(expri, np.flipud(np.fliplr(exprj)), distance)
            d, flip = min(zip([d, dud, dlr, dlrud], ['I', 'V', 'H', 'X']))
    d = max(d, 0)
    return d, flip

def _image_distance(expri, exprj, distance):
    if distance.startswith('NCC'):
        if expri.std() == 0 and exprj.std() == 0:
            ncc = 1.
        else:
            ncc = (expri * exprj).sum() / expri.size
        if distance == 'NCClog':
            d = -np.log((ncc + 1.) / 2. + 1e-10)
        elif distance == 'NCCone':
            d = (1. - ncc) / 2.
        elif distance == 'NCCplus':
            d = float(ncc > 0.) * (1. - ncc)
        elif distance == 'NCCplusok':
            d = (1. - ncc) if ncc > 0. else 1.
    elif distance == 'SSD' or distance == 'NSSD':
        d = ((expri - exprj) ** 2).sum()
    elif distance == 'Jaccard' or distance == 'NJaccard':
        M = np.maximum(expri, exprj).sum()
        m = np.minimum(expri, exprj).sum()
        if M == 0:
            d = 0.
        else:
            d = 1. - m/M
    else:
        d = pairwise_distances(
                expri.flatten().reshape((1, expri.size)),
                exprj.flatten().reshape((1, exprj.size)),
                distance)
    return d


def extract_expression(image_file, mask_file, inside_file, expression_file,
        data_file, normalized_width, normalized_height):
    '''
    '''
    if os.path.exists(data_file):
        return tsh.deserialize(data_file)
    image = tsh.read_gray_image(image_file)
    mask = tsh.read_gray_image(mask_file)
    inside, expression = tsh.extract_embryo_gray(image, mask > 0, normalized_width, normalized_height)
    if inside_file is not None:
        cv.imwrite(inside_file, inside)
    if expression_file is not None:
        cv.imwrite(expression_file, 255*expression)
    tsh.serialize(data_file, expression)
    return expression


def get_dissimilarities(data, output_dir=None, input_name=None, image_prefix=None, mask_prefix=None, n_jobs=None, **kwargs):
    assert image_prefix != None
    assert mask_prefix != None
    assert output_dir != None
    assert input_name != None
    measure = kwargs['measure']
    distance_name = kwargs['distance_name']
    rotation_invariance = kwargs['rotation_invariance']
    normalized_width = kwargs['normalized_width']
    normalized_height = kwargs['normalized_height']
    if n_jobs == None:
        n_jobs = 1

    image_prefix = os.path.expanduser(image_prefix)
    mask_prefix = os.path.expanduser(image_prefix)
    distance_name = distance_name.format(OUT=output_dir, INPUTNAME=input_name, **kwargs)
    tsh.makedirs(os.path.dirname(distance_name))
    kwargs['distance_name'] = distance_name

    expr_dir = os.path.join(output_dir, 'expr/%04dx%04d' % (normalized_width, normalized_height))
    tsh.makedirs(expr_dir)

    # Make it easier for evaluate.py to create nice html reports.
    if 'create_links' in kwargs and kwargs['create_links'] == True:
        if os.path.exists('expr'):
            os.unlink('expr')
        try:
            os.symlink(expr_dir, 'expr')
        except:
            pass
        if os.path.exists('distance'):
            os.unlink('distance')
        try:
            os.symlink(os.path.dirname(distance_name), 'distance')
        except:
            pass

    save_expr_images = kwargs['save_expr_images'] if 'save_expr_images' in kwargs else False
    if os.path.exists(distance_name):
        D = tsh.deserialize(distance_name)['D']
    else:
        imagenames = [ os.path.join(image_prefix, sample['image']) for sample in data ]
        masknames = [ os.path.join(mask_prefix, sample['mask']) for sample in data ]
        n = len(data)
        logger.info('Extracting %d expressions...', n)
        Parallel(n_jobs=n_jobs, verbose=True,
            pre_dispatch='2*n_jobs')(
            delayed(_extract_expression)(
                imagenames[j],
                masknames[j],
                os.path.join(expr_dir, 'inside%02d.png' % data[j]['id']) if save_expr_images else None,
                os.path.join(expr_dir, 'expr%02d.png' % data[j]['id']) if save_expr_images else None,
                os.path.join(expr_dir, 'expr%02d.dat' % data[j]['id']),
                normalized_width,
                normalized_height
            ) for j in range(n))

        logger.info('Computing %d dissimilarities...', (n*(n-1))/2)
        results = Parallel(n_jobs=n_jobs, verbose=True,
            pre_dispatch='2*n_jobs')(
            delayed(_get_dissimilarity)(
                i, j,
                measure, rotation_invariance,
                os.path.join(expr_dir, 'expr%02d.dat' % data[i]['id']),
                os.path.join(expr_dir, 'expr%02d.dat' % data[j]['id'])
            ) for j in range(n) for i in range(j+1, n))

        logger.info('Transforming results...')
        D = np.zeros((n, n), dtype=np.float)
        tfxs = np.array([['I'] * n] * n)
        for i, j, d, t in results:
            D[j, i] = d
            D[i, j] = d
            tfxs[j, i] = t
            tfxs[i, j] = t

        logger.info('Saving results...')
        tsh.serialize(distance_name, {
                'D': D,
                'min': None, 'max': None,
                'tfxs': tfxs,
                'measure': measure,
                'rotation_invariance': rotation_invariance })

    return kwargs, D

def _extract_expression(image_file, mask_file, inside_file, expression_file,
        data_file, normalized_width, normalized_height):
    extract_expression(image_file, mask_file, inside_file, expression_file,
        data_file, normalized_width, normalized_height)
    return None

def _get_dissimilarity(i, j, measure, rotation_invariance, data_filei, data_filej):
    expri = tsh.deserialize(data_filei).astype(float)
    exprj = tsh.deserialize(data_filej).astype(float)
    d, t = image_distance(expri, exprj, measure, rotation_invariance)
    return i, j, d, t

