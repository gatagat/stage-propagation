import cv2 as cv
import numpy as np
import os
from sklearn.metrics.pairwise import pairwise_distances

import tsh; logger = tsh.create_logger(__name__)

all_measures = [ 'SSD', 'NCCone', 'NSSD', 'Jaccard', 'cosine' ]
normalized_dists = [ 'NSSD', 'NCCone', 'NCClog' ]

def image_distance(expri, exprj, distance, rotation_invariance=False):
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
        d = pairwise_distances(expri.flatten().reshape((1, expri.size)), exprj.flatten().reshape((1, exprj.size)), distance)
    return d


def extract_expression(image_file, mask_file, inside_file, expression_file, data_file, normalized_width, normalized_height):
    if os.path.exists(data_file):
        return tsh.deserialize(data_file)
    image = tsh.read_gray_image(image_file)
    mask = tsh.read_gray_image(mask_file)
    inside, expression = tsh.extract_embryo_gray(image, mask > 0, normalized_width, normalized_height)
    cv.imwrite(inside_file, inside)
    cv.imwrite(expression_file, 255*expression)
    tsh.serialize(data_file, expression)
    return expression


def get_distances(data, output_dir=None, image_prefix=None, mask_prefix=None, **kwargs):
    assert image_prefix != None
    assert mask_prefix != None
    assert output_dir != None
    measure = kwargs['measure']
    distance_name = kwargs['distance_name']
    rotation_invariance = kwargs['rotation_invariance']
    normalized_width = kwargs['normalized_width']
    normalized_height = kwargs['normalized_height']

    image_prefix = os.path.expanduser(image_prefix)
    mask_prefix = os.path.expanduser(image_prefix)
    distance_name = distance_name.format(OUT=output_dir)
    kwargs['distance_name'] = distance_name
    tsh.makedirs(os.path.join(output_dir, 'expr'))
    if os.path.exists(distance_name):
        D = tsh.deserialize(distance_name)['D']
    else:
        imagenames = [ os.path.join(image_prefix, sample['image']) for sample in data ]
        masknames = [ os.path.join(mask_prefix, sample['mask']) for sample in data ]
        n = len(data)
        D = np.zeros((n, n), dtype=np.float)
        tfxs = np.array([['I'] * n] * n)
        lo_limit = []
        hi_limit = []
        for j in range(n):
            logger.info('Expression distances for %s', str(data[j]['id']))
            exprj = extract_expression(
                    imagenames[j],
                    masknames[j],
                    os.path.join(output_dir, 'expr/inside%02d.png' % data[j]['id']),
                    os.path.join(output_dir, 'expr/expr%02d.png' % data[j]['id']),
                    os.path.join(output_dir, 'expr/expr%02d.dat' % data[j]['id']),
                    normalized_width,
                    normalized_height).astype(float)
            if len(lo_limit) == 0:
                lo_limit = np.tile(float('inf'), exprj.shape)
                hi_limit = np.tile(-float('inf'), exprj.shape)
            lo_limit = np.minimum(lo_limit, exprj)
            hi_limit = np.maximum(hi_limit, exprj)
            for i in range(j+1, n):
                expri = extract_expression(
                        imagenames[i],
                        masknames[i],
                        os.path.join(output_dir, 'expr/inside%02d.png' % data[i]['id']),
                        os.path.join(output_dir, 'expr/expr%02d.png' % data[i]['id']),
                        os.path.join(output_dir, 'expr/expr%02d.dat' % data[i]['id']),
                        normalized_width,
                        normalized_height).astype(float)

                dist, tfx = image_distance(expri, exprj, measure, rotation_invariance)
                D[j, i] = dist
                D[i, j] = dist
                tfxs[j, i] = tfx
                tfxs[i, j] = tfx

        tsh.serialize(distance_name, {
                'D': D,
                'min': lo_limit, 'max': hi_limit,
                'tfxs': tfxs,
                'measure': measure,
                'rotation_invariance': rotation_invariance })

    return kwargs, D
