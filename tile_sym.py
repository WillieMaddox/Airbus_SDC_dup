import numpy as np
import hashlib
# [00, 01, 02]
# [10, 11, 12]
# [20, 21, 22]


def eq(g, h):
    assert(np.all(g == h))


overlap_patterns = (
    'single_corner', 'double_vertical', 'double_horizontal', 'double_corner',
    'thin_vertical', 'thin_horizontal', 'thick_vertical', 'thick_horizontal')

q = np.array([[10, 11, 12, 13, 14, 15, 16],
              [17, 18, 19, 20, 21, 22, 23],
              [24, 25,  1,  2,  3, 26, 27],
              [28, 29,  4,  5,  6, 30, 31],
              [32, 33,  7,  8,  9, 34, 35],
              [36, 37, 38, 39, 40, 41, 42],
              [43, 44, 45, 46, 47, 48, 49]])

ky = {}  # A dict that maps tags to numpy array ops.
ky['hv1'] = (2, 4)  # drop these to make 1 row (or col) overlap
ky['hv2'] = (1, 3, 5)  # drop these to make 2 adjacent rows (or cols) overlap
ky['hv3'] = (0, 1, 5, 6)  # drop these to make 3 rows (or cols) overlap
ky['ul1'] = 0  # keeps only the top row or the left most column
ky['ul2'] = slice(None, 2)  # keeps the top 2 rows or the left 2 columns
ky['ul3'] = slice(None, 3)  # keeps the top 3 rows or the left 3 columns
ky['ccc'] = slice(None, None)  # keeps all rows or all columns
ky['lr3'] = slice(-3, None)  # keeps the bottom 3 rows or the right 3 columns
ky['lr2'] = slice(1, None)  # keeps the bottom 2 rows or the right 2 columns
ky['lr1'] = 2  # keeps only the bottom row or the right most column

f_drop = {}  # which (rows, columns) to drop from q above for constructing each pattern.
f_drop['single_corner'] = ('hv1', 'hv1')
f_drop['double_vertical'] = ('hv2', 'hv1')
f_drop['double_horizontal'] = ('hv1', 'hv2')
f_drop['double_corner'] = ('hv2', 'hv2')
f_drop['thin_vertical'] = ('hv3', 'hv1')
f_drop['thin_horizontal'] = ('hv1', 'hv3')
f_drop['thick_vertical'] = ('hv3', 'hv2')
f_drop['thick_horizontal'] = ('hv2', 'hv3')

combos = {}
combos['single_corner'] = [
    {
        'bloc': {'g': ('ul3', 'ul3'), 'h': ('lr3', 'lr3')},
        'test': {'g': ('lr1', 'lr1'), 'h': ('ul1', 'ul1')},
        'maps': np.array([[[2, 2], [0, 0]]])
    }, {
        'bloc': {'g': ('lr3', 'lr3'), 'h': ('ul3', 'ul3')},
        'test': {'g': ('ul1', 'ul1'), 'h': ('lr1', 'lr1')},
        'maps': np.array([[[0, 0], [2, 2]]])
    }, {
        'bloc': {'g': ('ul3', 'lr3'), 'h': ('lr3', 'ul3')},
        'test': {'g': ('lr1', 'ul1'), 'h': ('ul1', 'lr1')},
        'maps': np.array([[[2, 0], [0, 2]]])
    }, {
        'bloc': {'g': ('lr3', 'ul3'), 'h': ('ul3', 'lr3')},
        'test': {'g': ('ul1', 'lr1'), 'h': ('lr1', 'ul1')},
        'maps': np.array([[[0, 2], [2, 0]]])
    }]
combos['double_vertical'] = ('hv2', 'hv1')
combos['double_horizontal'] = ('hv1', 'hv2')
combos['double_corner'] = ('hv2', 'hv2')
combos['thin_vertical'] = ('hv3', 'hv1')
combos['thin_horizontal'] = ('hv1', 'hv3')
combos['thick_vertical'] = ('hv3', 'hv2')
combos['thick_horizontal'] = ('hv2', 'hv3')

# single corner
# overlap pattern: 1x1
# combinations: 4
# example:
# 000
# 000
# 00100
#   000
#   000

f = np.delete(q, ky[f_drop['single_corner'][0]], axis=0)
f = np.delete(f, ky[f_drop['single_corner'][1]], axis=1)
for combo in combos['single_corner']:
    g = f[ky[combo['bloc']['g'][0]], ky[combo['bloc']['g'][1]]]
    h = f[ky[combo['bloc']['h'][0]], ky[combo['bloc']['h'][1]]]
    eq(g[ky[combo['test']['g'][0]], ky[combo['test']['g'][1]]],
       h[ky[combo['test']['h'][0]], ky[combo['test']['h'][1]]])
    overlap_map = combo['maps']
    overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
    for idx1, idx2 in overlap_map:
        eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

f = np.delete(q, (2, 4), axis=0)
f = np.delete(f, (2, 4), axis=1)

g, h = f[ul3, ul3], f[lr3, lr3]
eq(g[2, 2], h[0, 0])
overlap_map = np.array([[[2, 2], [0, 0]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

g, h = f[lr3, lr3], f[ul3, ul3]
eq(g[0, 0], h[2, 2])
overlap_map = np.array([[[0, 0], [2, 2]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

g, h = f[ul3, lr3], f[lr3, ul3]
eq(g[2, 0], h[0, 2])
overlap_map = np.array([[[2, 0], [0, 2]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

g, h = f[lr3, ul3], f[ul3, lr3]
eq(g[0, 2], h[2, 0])
overlap_map = np.array([[[0, 2], [2, 0]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])


# double vertical
# overlap pattern: 2x1
# combinations: 4
# example:
# 000
# 00100
# 00100
#   000

f = np.delete(q, (1, 3, 5), axis=0)
f = np.delete(f, (2, 4), axis=1)

g, h = f[ul3, ul3], f[lr3, lr3]
eq(g[1:, 2], h[:2, 0])
overlap_map = np.array([[[1, 2], [0, 0]],
                        [[2, 2], [1, 0]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

g, h = f[lr3, lr3], f[ul3, ul3]
eq(g[:2, 0], h[1:, 2])
overlap_map = np.array([[[0, 0], [1, 2]],
                        [[1, 0], [2, 2]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

g, h = f[ul3, lr3], f[lr3, ul3]
eq(g[1:, 0], h[:2, 2])
overlap_map = np.array([[[1, 0], [0, 2]],
                        [[2, 0], [1, 2]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

g, h = f[lr3, ul3], f[ul3, lr3]
eq(g[:2, 2], h[1:, 0])
overlap_map = np.array([[[0, 2], [1, 0]],
                        [[1, 2], [2, 0]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

# double horizontal
# overlap pattern: 1x2
# combinations: 4
# example:
# 000
# 000
# 0110
#  000
#  000

f = np.delete(q, (2, 4), axis=0)
f = np.delete(f, (1, 3, 5), axis=1)

g, h = f[ul3, ul3], f[lr3, lr3]
eq(g[2, 1:], h[0, :2])
overlap_map = np.array([[[2, 1], [0, 0]],
                        [[2, 2], [0, 1]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

g, h = f[lr3, lr3], f[ul3, ul3]
eq(g[0, :2], h[2, 1:])
overlap_map = np.array([[[0, 0], [2, 1]],
                        [[0, 1], [2, 2]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

g, h = f[ul3, lr3], f[lr3, ul3]
eq(g[2, :2], h[0, 1:])
overlap_map = np.array([[[2, 0], [0, 1]],
                        [[2, 1], [0, 2]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

g, h = f[lr3, ul3], f[ul3, lr3]
eq(g[0, 1:], h[2, :2])
overlap_map = np.array([[[0, 1], [2, 0]],
                        [[0, 2], [2, 1]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])


# thin vertical
# overlap pattern: 3x1
# combinations: 2
# example:
# 00100
# 00100
# 00100

f = np.delete(q, (0, 1, 5, 6), axis=0)
f = np.delete(f, (2, 4), axis=1)

g, h = f[ul3, ul3], f[lr3, lr3]
eq(g[:, 2], h[:, 0])
overlap_map = np.array([[[0, 2], [0, 0]],
                        [[1, 2], [1, 0]],
                        [[2, 2], [2, 0]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

g, h = f[lr3, lr3], f[ul3, ul3]
eq(g[:, 0], h[:, 2])
overlap_map = np.array([[[0, 0], [0, 2]],
                        [[1, 0], [1, 2]],
                        [[2, 0], [2, 2]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])


# thin horizontal
# overlap pattern: 1x3
# combinations: 2
# example:
# 000
# 000
# 111
# 000
# 000

f = np.delete(q, (2, 4), axis=0)
f = np.delete(f, (0, 1, 5, 6), axis=1)

g, h = f[ul3, lr3], f[lr3, ul3]
eq(g[2, :], h[0, :])
overlap_map = np.array([[[2, 0], [0, 0]],
                        [[2, 1], [0, 1]],
                        [[2, 2], [0, 2]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

g, h = f[lr3, ul3], f[ul3, lr3]
eq(g[0, :], h[2, :])
overlap_map = np.array([[[0, 0], [2, 0]],
                        [[0, 1], [2, 1]],
                        [[0, 2], [2, 2]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

# double corner
# overlap pattern: 2x2
# combinations: 4
# example:
# 000
# 0110
# 0110
#  000

f = np.delete(q, (1, 3, 5), axis=0)
f = np.delete(f, (1, 3, 5), axis=1)

g, h = f[ul3, ul3], f[lr3, lr3]
eq(g[1:, 1:], h[:2, :2])
overlap_map = np.array([[[1, 1], [0, 0]],
                        [[1, 2], [0, 1]],
                        [[2, 1], [1, 0]],
                        [[2, 2], [1, 1]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

g, h = f[lr3, lr3], f[ul3, ul3]
eq(g[:2, :2], h[1:, 1:])
overlap_map = np.array([[[0, 0], [1, 1]],
                        [[0, 1], [1, 2]],
                        [[1, 0], [2, 1]],
                        [[1, 1], [2, 2]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

g, h = f[ul3, lr3], f[lr3, ul3]
eq(g[1:, :2], h[:2, 1:])
overlap_map = np.array([[[1, 0], [0, 1]],
                        [[1, 1], [0, 2]],
                        [[2, 0], [1, 1]],
                        [[2, 1], [1, 2]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

g, h = f[lr3, ul3], f[ul3, lr3]
eq(g[:2, 1:], h[1:, :2])
overlap_map = np.array([[[0, 1], [1, 0]],
                        [[0, 2], [1, 1]],
                        [[1, 1], [2, 0]],
                        [[1, 2], [2, 1]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])


# thick vertical
# overlap pattern: 3x2
# combinations: 2
# example:
# 0110
# 0110
# 0110

f = np.delete(q, (0, 1, 5, 6), axis=0)
f = np.delete(f, (1, 3, 5), axis=1)

g, h = f[ul3, ul3], f[lr3, lr3]
eq(g[:, 1:], h[:, :2])
overlap_map = np.array([[[0, 1], [0, 0]],
                        [[0, 2], [0, 1]],
                        [[1, 1], [1, 0]],
                        [[1, 2], [1, 1]],
                        [[2, 1], [2, 0]],
                        [[2, 2], [2, 1]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

g, h = f[lr3, lr3], f[ul3, ul3]
eq(g[:, :2], h[:, 1:])
overlap_map = np.array([[[0, 0], [0, 1]],
                        [[0, 1], [0, 2]],
                        [[1, 0], [1, 1]],
                        [[1, 1], [1, 2]],
                        [[2, 0], [2, 1]],
                        [[2, 1], [2, 2]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

# thick horizontal
# overlap pattern: 2x3
# combinations: 2
# example:
# 000
# 111
# 111
# 000

f = np.delete(q, (1, 3, 5), axis=0)
f = np.delete(f, (0, 1, 5, 6), axis=1)

g, h = f[ul3, lr3], f[lr3, ul3]
eq(g[1:, :], h[:2, :])
overlap_map = np.array([[[1, 0], [0, 0]],
                        [[1, 1], [0, 1]],
                        [[1, 2], [0, 2]],
                        [[2, 0], [1, 0]],
                        [[2, 1], [1, 1]],
                        [[2, 2], [1, 2]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])

g, h = f[lr3, ul3], f[ul3, lr3]
eq(g[:2, :], h[1:, :])
overlap_map = np.array([[[0, 0], [1, 0]],
                        [[0, 1], [1, 1]],
                        [[0, 2], [1, 2]],
                        [[1, 0], [2, 0]],
                        [[1, 1], [2, 1]],
                        [[1, 2], [2, 2]]])

overlap_hash = hashlib.md5(overlap_map.tobytes()).hexdigest()[:6]
for idx1, idx2 in overlap_map:
    eq(g[idx1[0], idx1[1]], h[idx2[0], idx2[1]])
