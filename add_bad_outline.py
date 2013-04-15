#!/usr/bin/env python

from matplotlib.pylab import csv2rec, rec2csv
import os

import tsh

def is_blurred(fine_segmentation, dirname, ids):
    data = tsh.deserialize(os.path.join(fine_segmentation, dirname, 'blur-evaluation.dat'))
    return [ data['data'][data['data']['obj'] == int(metasys_id)][0][data['blur_measure']] < 1.46128831851
            for metasys_id in ids ]

def is_touching(coarse_segmentation, dirname, ids):
    d = os.path.join(coarse_segmentation, dirname)
    objects = csv2rec(os.path.join(d, 'objects.csv'), delimiter=';')
    components = csv2rec(os.path.join(d, 'components.csv'), delimiter=';')
    return [ components[components['bbox'] == objects[objects['obj'] == int(metasys_id)][0]['bbox']]['cut'].any()
            for metasys_id in ids ]

fine_segmentation_dir = '/groups/stark/projects/vt/Segmentation/Fine/MetaSys/'
coarse_segmentation_dir = '/groups/stark/projects/vt/Segmentation/Coarse/MetaSys/' 

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    data = csv2rec(filename, delimiter=',')
    dirname = data['dirname'][0]
    ids = data['metasys_id']
    blurred = is_blurred(fine_segmentation_dir, dirname, ids)
    touching = is_touching(coarse_segmentation_dir, dirname, ids)

    import pandas
    df = pandas.DataFrame(data)
    df['blurred'] = blurred
    df['touching'] = touching
    df['bad_outline'] = df['blurred'] + df['touching']

    outname = os.path.splitext(filename)
    outname = outname[0] + '-wb.csv'
    rec2csv(df.to_records(), outname, delimiter=',')

