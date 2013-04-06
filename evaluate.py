#!/usr/bin/env python

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics
import tempfile
import operator
import scipy.interpolate as spi
import sys
from jinja2 import Environment, FileSystemLoader

import tsh; logger = tsh.create_logger(__name__)
from utils import read_listfile, read_truthfile, select

def create_roc_curve(true, prob):
    if true.any():
        fpr, tpr, _ = sklearn.metrics.roc_curve(true, prob)
        return fpr, tpr
    else:
        return None, None

def plot_roc_curve(fpr, tpr, title=None, filename=None):
    if fpr != None and tpr != None:
        roc_auc = sklearn.metrics.auc(fpr, tpr)
    else:
        fpr = [0]
        tpr = [0]
        roc_auc = np.nan
    plt.clf()
    plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    if title != None:
        plt.title(title)
    plt.legend(loc='lower right')
    if filename != None:
        plt.savefig(filename)
        plt.close()

def create_prc_curve(true, prob, title=None, filename=None):
    if true.any():
        precision, recall, _ = sklearn.metrics.precision_recall_curve(true, prob)
        # XXX: get rid of the precision = 1 for recall = 0
        if len(precision) > 1:
            precision[-1] = precision[-2]
        return precision, recall
    else:
        return None, None

def plot_prc_curve(precision, recall, title=None, filename=None):
    if precision != None and recall != None:
        prc_auc = sklearn.metrics.auc(recall, precision)
    else:
        recall = [0]
        precision = [0]
        prc_auc = np.nan
    plt.clf()
    plt.plot(recall, precision, label='AUC = %0.2f' % prc_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([-0.01, 1.01])
    plt.xlim([-0.01, 1.01])
    if title != None:
        plt.title(title)
    plt.legend(loc='lower right')
    if filename != None:
        plt.savefig(filename)
        plt.close()


def constant_extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x <= xs[0]: #XXX: <= instead of < because some TPR start with [0, 0, ...] - causes division by zero when dividing by slope
            return ys[0]
        elif x >= xs[-1]:
            return ys[-1]
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(map(pointwise, np.array(xs)))

    return ufunclike


def average_curves(x, y, n=1000):
   f = [constant_extrap1d(spi.interp1d(xi, yi, kind='linear')) for xi, yi in zip(x, y)]
   average_x = np.linspace(0, 1, num=n)
   average_y = np.mean([fi(average_x) for fi in f], axis=0)
   return average_x, average_y


def process(pred_filename, truth_filename, fprs=None, tprs=None, precisions=None, recalls=None, cms=None, accs=None):
    predname = os.path.splitext(os.path.basename(pred_filename))[0]
    truth_meta, truth_ids, truth = read_truthfile(truth_filename)
    pred_meta, all_pred = read_listfile(pred_filename)
    pred = select(all_pred, 'id', truth_ids)
    logger.info('Using %d predicted samples with ground truth to evaluate', len(pred))
    assert (np.array(truth_ids) == pred['id']).all()

    truth_name = truth_meta['truth']
    labels = truth_meta[truth_name + '_labels']
    roccurves = []
    prcurves = []
    for class_num, class_label in labels.items():
        true = truth == class_num
        prob = pred['prob%d' % class_num]
        #prob = pred['pred'] == class_num
        #prob = pred['pred_argmax'] == class_num
        fpr, tpr = create_roc_curve(true, prob)
        rocname = os.path.join(outdir, predname + '-roc-' + truth_name + '-%d' % class_num + '.svg')
        plot_roc_curve(fpr, tpr,
            title='ROC - ' + truth_name.capitalize() + ' ' + class_label,
            filename=rocname)
        roccurves += [rocname]
        if fpr != None:
            fprs[class_num] += [fpr]
        if tpr != None:
            tprs[class_num] += [tpr]
        precision, recall = create_prc_curve(true, prob)
        #print class_label, precision, recall
        prcname = os.path.join(outdir, predname + '-prc-' + truth_name + '-%d' % class_num + '.svg')
        plot_prc_curve(precision, recall,
            title='Precision-Recall curve - ' + truth_name.capitalize() + ' ' + class_label,
            filename=prcname)
        prcurves += [prcname]
        if precision != None:
            precisions[class_num] += [precision]
        if recall != None:
            recalls[class_num] += [recall]

    sorted_class_nums = sorted(labels.keys())
    sorted_class_labels = tsh.dict_values(labels, sorted_class_nums)
    cm = sklearn.metrics.confusion_matrix(truth, pred['pred'], labels=sorted_class_nums)
    acc = (np.diag(cm).sum() / float(np.sum(cm)))
    tsh.plot_confusion_matrix(cm, labels=sorted_class_labels)
    plt.title('Accuracy: %.2f' % acc)
    print 'Accuracy: %.2f' % acc
    with open(os.path.join(outdir, predname + '.txt'), 'w') as f:
        f.write('Accuracy: %.3f\n' % acc)
    cmname = os.path.join(outdir, predname + '-cm.svg')
    plt.savefig(cmname)
    plt.close()

    samples = []
    truth_meta, truth = read_listfile(truth_filename)
    for t in truth:
        samples += [{
            'id': t['id'],
            'image': os.path.join('image', os.path.relpath(os.path.join(truth_meta['image_prefix'], t['image']), '/home/imp/kazmar/vt_project/Segmentation/Fine/MetaSys/')),
            'mask': os.path.join('image', os.path.relpath(os.path.join(truth_meta['mask_prefix'], t['mask']), '/home/imp/kazmar/vt_project/Segmentation/Fine/MetaSys/')),
            'expr': os.path.join('expr', 'expr%d.png' % t['id']),
            'truth': labels[t[truth_name]],
            'prediction': labels[pred['pred'][truth['id'] == t['id']][0]] }]
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    env = Environment(loader=FileSystemLoader(template_dir))
    open(os.path.join(outdir, predname + '.html'), 'w').write(env.get_template('evaluation.html').render(
            title=predname + ' ' + truth_name, cm=cmname, roccurves=roccurves, prcurves=prcurves, samples=samples, predictions=all_pred))

    cms += [cm]
    accs += [acc]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluates predictions against known truth.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-p', '--predictions', dest='predictions', nargs='*', required=True, action='store', default=None, help='Predictions file(s).')
    parser.add_argument('-t', '--truth', dest='truth', nargs='*', required=True, action='store', default=None, help='Truth file(s).')
    parser.add_argument('--all-prefix', dest='all_prefix', required=False, action='store', default=None, help='Prefix for summary curves for multiple inputs.')
    parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    opts = parser.parse_args()
    config = tsh.read_config(opts, __file__)
    if opts.output == None:
        outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
        logger.info('Output directory %s', outdir)
    else:
        outdir = opts.output
        if not os.path.exists(outdir):
            tsh.makedirs(outdir)

    truth_meta, _, _= read_truthfile(opts.truth[0])
    truth_name = truth_meta['truth']
    labels = truth_meta[truth_name + '_labels']

    all_fprs = dict(zip(labels.keys(), [[] for _ in labels.keys()]))
    all_tprs = dict(zip(labels.keys(), [[] for _ in labels.keys()]))
    all_precisions = dict(zip(labels.keys(), [[] for _ in labels.keys()]))
    all_recalls = dict(zip(labels.keys(), [[] for _ in labels.keys()]))
    all_cms = []
    all_accs = []
    datasets = []
    for pred_filename, truth_filename in zip(opts.predictions, opts.truth):
        logger.info('Prediction: %s, truth: %s', pred_filename, truth_filename)
        accs = []
        process(pred_filename, truth_filename, fprs=all_fprs, tprs=all_tprs, precisions=all_precisions, recalls=all_recalls, cms=all_cms, accs=accs)
        all_accs += accs
        predname = os.path.splitext(os.path.basename(pred_filename))[0]
        datasets += [{'url': predname + '.html', 'label': predname, 'accuracy': np.mean(accs)}]

    if opts.all_prefix == None:
        if len(opts.predictions) == 1:
            sys.exit(0)
        all_prefix = 'all'
    else:
        all_prefix = opts.all_prefix

    roccurves = []
    prcurves = []
    for class_num, class_label in labels.items():
        if len(all_fprs[class_num]) == 0:
            tpr = None
            fpr = None
        else:
            fpr, tpr = average_curves(all_fprs[class_num], all_tprs[class_num])
        if len(all_recalls[class_num]) == 0:
            recall = None
            precision = None
        else:
            recall, precision = average_curves(map(lambda l: l[::-1], all_recalls[class_num]), map(lambda l: l[::-1], all_precisions[class_num]))
            recall = recall[::-1]
            precision = precision[::-1]
        rocname = os.path.join(outdir, all_prefix + '-roc-' + truth_name + '-%d' % class_num + '.svg')
        plot_roc_curve(fpr, tpr,
            title='ROC - ' + truth_name.capitalize() + ' ' + class_label,
            filename=rocname)
        roccurves += [rocname]
        prname = os.path.join(outdir, all_prefix + '-prc-' + truth_name + '-%d' % class_num + '.svg')
        plot_prc_curve(precision, recall,
            title='Precision-Recall curve - ' + truth_name.capitalize() + ' ' + class_label,
            filename=prname)
        prcurves += [prname]

    all_tprs = reduce(operator.concat, [all_tprs[class_num] for class_num in labels.keys() if len(all_tprs[class_num]) != 0])
    all_fprs = reduce(operator.concat, [all_fprs[class_num] for class_num in labels.keys() if len(all_fprs[class_num]) != 0])
    if len(all_fprs) == 0:
        tpr = None
        fpr = None
    else:
        fpr, tpr = average_curves(all_fprs, all_tprs)
    rocname = os.path.join(outdir, all_prefix + '-roc-' + truth_name + '.svg')
    plot_roc_curve(fpr, tpr,
        title='ROC - ' + truth_name.capitalize(),
        filename=rocname)
    roccurves += [rocname]
    all_recalls = reduce(operator.concat, [all_recalls[class_num] for class_num in labels.keys() if len(all_recalls[class_num]) != 0])
    all_precisions = reduce(operator.concat, [all_precisions[class_num] for class_num in labels.keys() if len(all_precisions[class_num]) != 0])
    if len(all_recalls) == 0:
        recall = None
        precision = None
    else:
        recall, precision = average_curves(map(lambda l: l[::-1], all_recalls), map(lambda l: l[::-1], all_precisions))
        recall = recall[::-1]
        precision = precision[::-1]
    prname = os.path.join(outdir, all_prefix + '-prc-' + truth_name + '.svg')
    plot_prc_curve(precision, recall,
        title='Precision-Recall curve - ' + truth_name.capitalize(),
        filename=prname)
    prcurves += [prname]

    cm = np.sum(all_cms, axis=0)
    avg_acc = np.mean(all_accs)
    acc = (np.diag(cm).sum() / float(np.sum(cm)))
    sorted_class_nums = sorted(labels.keys())
    sorted_class_labels = tsh.dict_values(labels, sorted_class_nums)
    tsh.plot_confusion_matrix(cm, labels=sorted_class_labels)
    plt.title('Accuracy: %.3f, datasets average accuracy: %.3f' % (acc, avg_acc))
    print all_prefix + ' accuracy: %.3f, average: %.3f' % (acc, avg_acc)
    with open(os.path.join(outdir, all_prefix + '.txt'), 'w') as f:
        f.write('Accuracy: %.3f, datasets average accuracy: %.3f\n' % (acc, avg_acc))
    cmname = os.path.join(outdir, all_prefix + '-cm.svg')
    plt.savefig(cmname)
    plt.close()

    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    env = Environment(loader=FileSystemLoader(template_dir))
    open(os.path.join(outdir, all_prefix + '-evaluation.html'), 'w').write(env.get_template('evaluation.html').render(
            title=all_prefix, cm=cmname, roccurves=roccurves, prcurves=prcurves, datasets=datasets, samples=None, predictions=None))

