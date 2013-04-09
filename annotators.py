#!/usr/bin/env python

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

import tsh; logger = tsh.create_logger(__name__)
from utils import read_listfile

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare annotators.')
    parser.add_argument('-a', dest='list1', nargs='*', required=True, action='store', default=None, help='List file(s) of annotator A.')
    parser.add_argument('-b', dest='list2', nargs='*', required=True, action='store', default=None, help='List file(s) of annotator B.')
    parser.add_argument('-m', dest='name1', required=False, action='store', default=None, help='Name of annotator A.')
    parser.add_argument('-n', dest='name2', required=False, action='store', default=None, help='Name of annotator B.')
    opts = parser.parse_args()

    name1 = opts.name1 if opts.name1 != None else 'Annotator A'
    name2 = opts.name2 if opts.name2 != None else 'Annotator B'

    data1 = None
    data2 = None
    for listname1, listname2 in zip(opts.list1, opts.list2):
        meta, d1 = read_listfile(listname1)
        _, d2 = read_listfile(listname2)
        if data1 == None:
            data1 = d1
            data2 = d2
        else:
            data1 = np.r_[data1, d1]
            data2 = np.r_[data2, d2]

    labels = meta['stage_labels']
    sorted_class_nums = sorted(labels.keys())
    sorted_class_labels = tsh.dict_values(labels, sorted_class_nums)

    mask1 = data1['stage'] != 0
    mask2 = data2['stage'] != 0
    
    ids = set(data2[mask2]['id']).intersection(data1[mask1]['id'])
    n_both = len(ids)
    labels1 = [data1[data1['id'] == i][0]['stage'] for i in ids]
    labels2 = [data2[data2['id'] == i][0]['stage'] for i in ids]

    cm = sklearn.metrics.confusion_matrix(labels1, labels2, labels=sorted_class_nums)
    acc = cm.diagonal().sum() / float(n_both)
    label_acc = np.diagonal(cm).astype(np.float64) / np.sum(cm, axis=1).astype(np.float64)
    label_avg_acc = np.nansum(label_acc) / np.sum(np.isfinite(label_acc))

    tsh.plot_confusion_matrix(cm, labels=sorted_class_labels, xlabel=name1, ylabel=name2)
    plt.title('Sample accuracy: %.2f, label accuracy: %.2f' % (acc, label_avg_acc))
    cmname = 'annotators-cm.svg'
    plt.savefig(cmname)
    plt.close()

    print 'Annotated in both: %d' % n_both
    print 'Sample accuracy: %.2f, label accuracy: %.2f' % (acc, label_avg_acc)
    for i in range(len(sorted_class_nums)):
            print '%s accuracy: %3f' % (labels[sorted_class_nums[i]], label_acc[i])

