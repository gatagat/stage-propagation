import numpy as np
import os
import scipy as sp
import tsh; logger = tsh.create_logger(__name__)
from matplotlib.pylab import csv2rec

# Read similarity and known truth
data = tsh.deserialize('013456-NSSD-rot.dat')
similarity = data['similarity']
imagelist = '013456'
ids, names = tsh.read_imagelist(imagelist, basedir='/Users/kazmar/tmp/Coarse_Images/013456')
vtline = '013456'
vt_truth_dir = '/Users/kazmar/tmp/Ground_Truth/' + vtline
full_truth = csv2rec(os.path.join(vt_truth_dir, 't.csv'), delimiter=',', converterd={ 'vt': lambda s: tsh.normalize_vt(s) })
full_truth = full_truth[full_truth['vt'] == vtline]
truth = ...

n = len(ids)
l_indices = np.in1d(ids, truth['embryo_id'])
#l_indices &= np.in1d(ids, truth[truth['stage'] != 'stg13-14']['embryo_id'])
pos_stage = 'stg15-16'
l_pos_indices = np.in1d(ids, truth[truth['stage'] == pos_stage]['embryo_id'])
l_neg_indices = l_indices & ~l_pos_indices
#l_neg_indices = np.random.permutation(np.nonzero(l_neg_indices)[0])[:l_pos_indices.sum()]
Y = np.zeros((n,), dtype=float)
Y[l_neg_indices] = -1.
Y[l_pos_indices] = 1.

# XXX: when going multi-class, do one-vs-rest, or one-vs-one, or something completely different?

# Solve label propagation
mu = 1e-2
eps = mu * 1e-1
W = np.exp(-similarity['D']*100.)
W[np.diag_indices_from(W)] = 0.
D = np.diag(W.sum(axis=1))
L = D - W
S = np.zeros((n, n), dtype=float)
S[l_indices, l_indices] = 1.
M = S + mu*L + eps*np.eye(n, dtype=float)
#M = mu*L + eps*np.eye(n, dtype=float)
cho = sp.linalg.cho_factor(M)
Yp = sp.linalg.cho_solve(cho, np.dot(S, Y))
new_l_pos_indices = Yp > 0.
new_l_count = (new_l_pos_indices & ~l_pos_indices).sum()

# Propose images based on propagated labels and their confidence

#??? how to get the confidence / how to setup the parameters of weights and costs

yp_sort_indices = np.argsort(-Yp)
print 'Known:'
for i in yp_sort_indices:
    if i not in np.nonzero(l_pos_indices)[0]:
        continue
    truth_i = full_truth['embryo_id'] == ids[i]
    print ids[i], full_truth[truth_i][['stage', 'orientation']], Yp[i]
print 'New ones:'
for i in yp_sort_indices:
    if i not in np.nonzero(new_l_pos_indices & ~l_pos_indices)[0]:
        continue
    truth_i = full_truth['embryo_id'] == ids[i]
    print ids[i], full_truth[truth_i][['stage', 'orientation']], Yp[i]

tpr = (full_truth[np.in1d(full_truth['embryo_id'], ids[new_l_pos_indices & ~l_pos_indices])]['stage'] == pos_stage).sum() / float(new_l_count)
print 'TPR: %.2f' % tpr

# Compare with 10 most similar ones
D = data['similarity']['D'].copy()
D[np.diag_indices_from(D)] = np.inf
dists = {}
for l in np.nonzero(l_pos_indices)[0]:
    for i in set(range(n)) - set(np.nonzero(l_indices)[0]):
        if i not in dists.keys():
            dists[i] = D[l, i]
        if D[l, i] < dists[i]:
            dists[i] = D[l, i]
sort_indices = np.argsort(dists.values())[:max(10, new_l_count)]
print 'Best similar:'
for i in sort_indices:
    truth_i = full_truth['embryo_id'] == ids[dists.keys()[i]]
    print ids[dists.keys()[i]], full_truth[truth_i][['stage', 'orientation']], dists.values()[i]

best_ids = ids[np.array(dists.keys())[sort_indices]]
tpr = (full_truth[np.in1d(full_truth['embryo_id'], best_ids)]['stage'] == pos_stage).sum() / float(len(best_ids))
print 'TPR: %.2f' % tpr
