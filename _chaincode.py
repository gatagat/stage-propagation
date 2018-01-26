'''
Freeman Chain Code

Original author Alessandro Mannini <mannini@esod.it>, 2010.
Translated to Python and extended by Tomas Kazmar, 24.8.2012.
'''

import numpy as np
import os
from ..io import svg

def chaincode(b, unwrap=False):
    '''
Returns Freeman chain code 8-connected representation of a boundary.

Parameters
----------
b : Nx2 int array
   Boundary given as array of (y,x) pixel coordinates.
   unwrap  - (optional, default=false) unwrap code
             if enable phase inversions are eliminated

Returns
-------
cc : list
    8-connected Freeman chain code of length N (close boundary), or N-1 (open
    boundary).

(x0, y0) : tuple of ints
    Starting point.

ucode : list
    Unwrapped 8-connected Freeman chain code (if requested)

Returns (cc, (x0, y0)) or (cc, (x0, y0), ucode) if unwrap is True.


Notes
-----
Direction-to-code convention:
    --------------------------
    | deltax | deltay | code |
    |------------------------|    y
    |    0   |   +1   |   2  |    ^     3  2  1
    |    0   |   -1   |   6  |    |      \ | /
    |   -1   |   +1   |   3  |    |   4 -- P -- 0
    |   -1   |   -1   |   5  |    |      / | \\
    |   +1   |   +1   |   1  |    |     5  6  7
    |   +1   |   -1   |   7  |    |
    |   -1   |    0   |   4  |    +-------------> x
    |   +1   |    0   |   0  |
    --------------------------
'''

    # compute dx,dy by a circular shift on coords arrays by 1 element
    delta = np.zeros(b.shape, dtype=int)
    delta[:-1, :] = b[1:, :] - b[:-1, :]
    delta[-1, :] = b[0, :] - b[-1, :]

    # check if boundary is 8-connected
    if ((np.abs(delta[:, 0]) > 1) + (np.abs(delta[:, 1]) > 1)).any():
        raise ValueError('Curve is not 8-connected.')

    # check if boundary is close, if so cut last element
    if (np.abs(delta[-1, :]) == 0).all():
        delta = delta[:-1, :]

    if ((np.abs(delta[:, 0]) == 0) * (np.abs(delta[:, 1]) == 0)).any():
        raise ValueError('Curve is degenerate.')

    # Take dy, dx to be a two-digit base-3 number (after adding one to both dy, dx),
    # and use this as an index into the following map:
    #   --------------------------------------
    #   | deltax | deltay | code | (base-3)+1 |
    #   |-------------------------------------|
    #   |    0   |   +1   |   2  |      8     |
    #   |    0   |   -1   |   6  |      2     |
    #   |   -1   |   +1   |   3  |      7     |
    #   |   -1   |   -1   |   5  |      1     |
    #   |   +1   |   +1   |   1  |      9     |
    #   |   +1   |   -1   |   7  |      3     |
    #   |   -1   |    0   |   4  |      4     |
    #   |   +1   |    0   |   0  |      6     |
    #   ---------------------------------------
    #
    idx = 3 * delta[:, 0] + delta[:, 1] + 4
    cm = np.array([5, 6, 7, 4, -1, 0, 3, 2, 1])
    cc = cm[idx]

    if unwrap:
        #
        # unwrapped_0 = cc_0
        # unwrapped_k = argmin_{u \in Z} |u - unwrapped_{k-1}|
        #       subject to:
        #               unwrapped_k - cc_k = 0 (mod 8)
        #
        ucc = cc.copy()
        for i in range(1, ucc.shape[0]):
            ucc[i] += 8*np.round((ucc[i-1] - cc[i])/8.)
        return cc, b[0, :], ucc
    else:
        return cc, b[0, :]

def chaincode_to_string(cc):
    '''
    Transforms a chaincode into a string of 0-7.

    Parameters:
    -----------
    cc : array
        A chaincode.

    Returns:
    --------
    s : string
        Corresponding string containing characters '0'..'7'.
    '''
    return ''.join([ chr(c + 48) for c in cc ])

def string_to_chaincode(s):
    '''
    Transforms a string of 0-7 into a chaincode.

    Parameters:
    -----------
    s : string
        A string containing characters '0'..'7'.

    Returns:
    --------
    cc : array
        Corresponding chaincode.
    '''
    return np.array([ ord(c) - 48 for c in s ])

def chaincode_to_coords(cc, start=None):
    '''
    '''
    deltax = np.array([ 1, 1, 0, -1, -1, -1, 0, 1 ])
    deltay = np.array([ 0, 1, 1, 1, 0, -1, -1, -1 ])
    x = np.zeros((len(cc)+1,), dtype=float)
    x[1:] = np.cumsum(deltax[cc])
    y = np.zeros((len(cc)+1,), dtype=float)
    y[1:] = np.cumsum(deltay[cc])
    if start != None:
        x += start[0]
        y += start[0]
    return x, y

def normalize_chaincode(cc):
    '''
    Normalizes the orientation of the chaincode.

    Flips the chaincode vertically/horizontally so that x_start <= x_end, and y_start <= y_end.

    Parameters:
    -----------
    cc : array
        Chaincode.

    Returns:
    --------
    normalized : array
        Normalized chaincode.
    '''
    x, y = chaincode_to_coords(cc)
    flip_map = np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8 ])
    if x[0] > x[-1]:
        flip_map[[0, 1, 3, 4, 5, 7]] = flip_map[[4, 3, 1, 0, 7, 5]]
    if y[0] > y[-1]:
        flip_map[[1, 2, 3, 5, 6, 7]] = flip_map[[7, 6, 5, 3, 2, 1]]
    return flip_map[cc]

def save_chaincode_svg(filename, cc, scale=20., margin=10.):
    '''
    Writes chaincode into an SVG file.

    Parameters:
    -----------
    filename : string
        Output filename.
    cc : array
        Chaincode.
    '''
    x, y = chaincode_to_coords(cc)
    x *= scale
    y *= scale
    x += margin - x.min()
    y += margin - y.min()
    assert (x >= margin).all()
    assert (y >= margin).all()

    shadow_color = (127, 127, 127)
    fg_color = (0, 0, 0)
    hilight_color = (255, 255, 255)

    scene = svg.Scene(os.path.splitext(os.path.basename(filename))[0])
    scene.add(svg.Polyline(zip(x, y), None, shadow_color, 8))
    for cx, cy in zip(x[1:], y[1:]):
        scene.add(svg.Circle([cx, cy], scale*0.1, fg_color, shadow_color, 1))
    scene.add(svg.Polyline(zip(x, y), None, fg_color, 6))
    scene.add(svg.Circle([x[0], y[0]], scale*0.1, hilight_color, shadow_color, 1))
    #scene.add(Text((50,50),"Chaincode: ",24,(0,0,0)))
    scene.width = max(x) + margin
    scene.height = max(y) + margin
    scene.write_svg(filename)
