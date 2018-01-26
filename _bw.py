import numpy as np
import cv2 as cv
from scipy.ndimage.measurements import label

def bwareaopen(mask, p):
    '''
    Removes all foreground connected components smaller than p pixels.

    mask: foreground mask
    p:    minimum number of pixels to keep the component

    Returns the new mask.
    '''
    lbl, max_label = label(mask)
    h, _ = np.histogram(lbl, np.arange(max_label + 2) - 0.5)
    opened = mask.copy()
    opened.flat[np.in1d(lbl, np.nonzero(h[1:] < p)[0] + 1)] = 0
    return opened


def bwboundaries(label):
    '''
    XXX: returns just the longest contour!
    returns contour as a list [[x0, y0], [x1, y1], ..., [xn, yn]]
    '''
    label_ext = np.zeros((label.shape[0]+2, label.shape[1]+2), dtype=label.dtype)
    label_ext[1:-1, 1:-1] = label
    if label.dtype == 'bool':
        c = cv.findContours(label_ext * np.uint8(255), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if c[1] == None:
            return np.array([])
        return (c[0][np.argmax([len(ci) for ci in c[0]])]).reshape(-1, 2) - 1

    boundaries = {}
    for l in np.unique(label[:]):
        if l == 0:
            continue
        c = cv.findContours(np.array(label_ext == l, dtype=np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if c[1] == None:
            continue
        boundaries[l] = (c[0][np.argmax([len(ci) for ci in c[0]])]).reshape(-1, 2) - 1
    return boundaries


def bwperim(bw, n=4):
    if bw.dtype != bool:
        bw = bw > 0
    ret = np.zeros(bw.shape, bool)
    diff = bw[:, :-1] != bw[:, 1:]
    ret[:, 1:] |= diff
    ret[:, :-1] |= diff
    diff = bw[:-1, :] != bw[1:, :]
    ret[1:, :] |= diff
    ret[:-1, :] |= diff
    return ret & bw
