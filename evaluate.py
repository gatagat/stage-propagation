#!/usr/bin/env python

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics
import tempfile

import tsh; logger = tsh.create_logger(__name__)
from utils import read_listfile, read_truthfile, select

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluates predictions against known truth.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-p', '--predictions', dest='predictions', required=True, action='store', default=None, help='Predictions file.')
    parser.add_argument('-t', '--truth', dest='truth', required=True, action='store', default=None, help='Truth file.')
    parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    opts = parser.parse_args()
    if opts.output == None:
        outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
        logger.info('Output directory %s', outdir)
    else:
        outdir = opts.output
        if not os.path.exists(outdir):
            tsh.makedirs(outdir)
    predname = os.path.splitext(os.path.basename(opts.predictions))[0]
    config = tsh.read_config(opts, __file__)
    truth_meta, truth_ids, truth = read_truthfile(opts.truth)
    pred_meta, pred = read_listfile(opts.predictions)
    pred = select(pred, 'id', truth_ids)
    logger.info('Using %d predicted samples with ground truth to evaluate', len(pred))
    assert (np.array(truth_ids) == pred['id']).all()

    truth_name = truth_meta['truth']
    labels = truth_meta[truth_name + '_labels']
    for class_num, class_label in labels.items():
        true = truth == class_num
        #prob = pred['pred'] == class_num
        #prob = pred['pred_argmax'] == class_num
        prob = pred['prob%d' % class_num]
        fpr, tpr, _ = sklearn.metrics.roc_curve(true, prob)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(true, prob)
        prc_auc = sklearn.metrics.auc(recall, precision)
        # XXX: get rid of the precision = 1 for recall = 0
        if len(precision) > 1:
            precision[-1] = precision[-2]

        plt.clf()
        plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC - ' + truth_name.capitalize() + ' ' + class_label)
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(outdir, predname + '-roc-' + truth_name + '-%d' % class_num + '.svg'))
        plt.close()

        plt.clf()
        plt.plot(recall, precision, label='AUC = %0.2f' % prc_auc)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([-0.01, 1.01])
        plt.xlim([-0.01, 1.01])
        plt.title('Precision-Recall curve - ' + truth_name.capitalize() + ' ' + class_label)
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(outdir, predname + '-prc-' + truth_name + '-%d' % class_num + '.svg'))
        plt.close()

    sorted_class_nums = sorted(labels.keys())
    sorted_class_labels = tsh.dict_values(labels, sorted_class_nums)
    cm = sklearn.metrics.confusion_matrix(truth, pred['pred'], labels=sorted_class_nums)
    acc = (np.diag(cm).sum() / float(np.sum(cm)))
    tsh.plot_confusion_matrix(cm, labels=sorted_class_labels)
    print 'Accuracy: %.2f' % acc
    plt.title('Accuracy: %.2f' % acc)
    plt.savefig(os.path.join(outdir, predname + '-cm.svg'))
    plt.close()
