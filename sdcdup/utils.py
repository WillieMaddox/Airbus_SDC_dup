import os
from shutil import copyfile
from datetime import datetime
from collections import Counter
from functools import lru_cache
import numpy as np
import networkx as nx
import pandas as pd
import cv2
from cv2 import img_hash
from skimage.measure import shannon_entropy
from skimage.feature import greycomatrix, greycoprops

EPS = np.finfo(np.float32).eps

idx_chan_map = {0: 'H', 1: 'L', 2: 'S'}
chan_idx_map = {'H': 0, 'L': 1, 'S': 2}
chan_cv2_scale_map = {'H': 256, 'L': 256, 'S': 256}
chan_gimp_scale_map = {'H': 360, 'L': 200, 'S': 100}


@lru_cache(maxsize=None)
def generate_ij_pairs(dim=3):
    return tuple([(i, j) for i in range(dim) for j in range(dim)])

# There are 24 distinct ways a $3\times3$ grid can overlap with another $3\times3$ grid.


B = 256

ij_pairs_3x3 = ((0, 0), (0, 1), (0, 2),
                (1, 0), (1, 1), (1, 2),
                (2, 0), (2, 1), (2, 2))

idx2ijpair = {idx: ij for idx, ij in enumerate(ij_pairs_3x3)}
ijpair2idx = {ij: idx for idx, ij in enumerate(ij_pairs_3x3)}

overlap_tag_pairs = {
    '0000': '2222',
    '0001': '2122',
    '0002': '2022',
    '0102': '2021',
    '0202': '2020',
    '0010': '1222',
    '0011': '1122',
    '0012': '1022',
    '0112': '1021',
    '0212': '1020',
    '0020': '0222',
    '0021': '0122',
    '0022': '0022',
    '0122': '0021',
    '0222': '0020',
    '1020': '0212',
    '1021': '0112',
    '1022': '0012',
    '1122': '0011',
    '1222': '0010',
    '2020': '0202',
    '2021': '0102',
    '2022': '0002',
    '2122': '0001',
    '2222': '0000'}

boundingbox_corners = {
    '0000': np.array([[0, 0], [1, 1]]) * B,
    '0001': np.array([[0, 0], [2, 1]]) * B,
    '0002': np.array([[0, 0], [3, 1]]) * B,
    '0102': np.array([[1, 0], [3, 1]]) * B,
    '0202': np.array([[2, 0], [3, 1]]) * B,
    '0010': np.array([[0, 0], [1, 2]]) * B,
    '0011': np.array([[0, 0], [2, 2]]) * B,
    '0012': np.array([[0, 0], [3, 2]]) * B,
    '0112': np.array([[1, 0], [3, 2]]) * B,
    '0212': np.array([[2, 0], [3, 2]]) * B,
    '0020': np.array([[0, 0], [1, 3]]) * B,
    '0021': np.array([[0, 0], [2, 3]]) * B,
    '0022': np.array([[0, 0], [3, 3]]) * B,
    '0122': np.array([[1, 0], [3, 3]]) * B,
    '0222': np.array([[2, 0], [3, 3]]) * B,
    '1020': np.array([[0, 1], [1, 3]]) * B,
    '1021': np.array([[0, 1], [2, 3]]) * B,
    '1022': np.array([[0, 1], [3, 3]]) * B,
    '1122': np.array([[1, 1], [3, 3]]) * B,
    '1222': np.array([[2, 1], [3, 3]]) * B,
    '2020': np.array([[0, 2], [1, 3]]) * B,
    '2021': np.array([[0, 2], [2, 3]]) * B,
    '2022': np.array([[0, 2], [3, 3]]) * B,
    '2122': np.array([[1, 2], [3, 3]]) * B,
    '2222': np.array([[2, 2], [3, 3]]) * B}

overlap_tag_maps = {
    '0000': np.array([0]),
    '0001': np.array([0, 1]),
    '0002': np.array([0, 1, 2]),
    '0102': np.array([1, 2]),
    '0202': np.array([2]),
    '0010': np.array([0, 3]),
    '0011': np.array([0, 1, 3, 4]),
    '0012': np.array([0, 1, 2, 3, 4, 5]),
    '0112': np.array([1, 2, 4, 5]),
    '0212': np.array([2, 5]),
    '0020': np.array([0, 3, 6]),
    '0021': np.array([0, 1, 3, 4, 6, 7]),
    '0022': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    '0122': np.array([1, 2, 4, 5, 7, 8]),
    '0222': np.array([2, 5, 8]),
    '1020': np.array([3, 6]),
    '1021': np.array([3, 4, 6, 7]),
    '1022': np.array([3, 4, 5, 6, 7, 8]),
    '1122': np.array([4, 5, 7, 8]),
    '1222': np.array([5, 8]),
    '2020': np.array([6]),
    '2021': np.array([6, 7]),
    '2022': np.array([6, 7, 8]),
    '2122': np.array([7, 8]),
    '2222': np.array([8])}

far_away_corners = {
    '0000': ('2222',),
    '0001': ('2222',),
    '0002': ('2222', '2020'),
    '0102': ('2020',),
    '0202': ('2020',),
    '0010': ('2222',),
    '0011': ('2222',),
    '0012': ('2222', '2020'),
    '0112': ('2020',),
    '0212': ('2020',),
    '0020': ('2222', '0202'),
    '0021': ('2222', '0202'),
    '0022': ('2222', '2020', '0202', '0000'),
    '0122': ('2020', '0000'),
    '0222': ('2020', '0000'),
    '1020': ('0202',),
    '1021': ('0202',),
    '1022': ('0202', '0000'),
    '1122': ('0000',),
    '1222': ('0000',),
    '2020': ('0202',),
    '2021': ('0202',),
    '2022': ('0202', '0000'),
    '2122': ('0000',),
    '2222': ('0000',)}


def generate_overlap_tag_slices():

    # sd -> short for slice dictionary if that even means anything.
    sd = {
        '00': slice(None, 1*B),  # top row (left column)
        '01': slice(None, 2*B),  # top 2 rows (left 2 columns)
        '02': slice(None, None),   # all rows (all columns)
        '12': slice(1*B, None),  # bottom 2 rows (right 2 columns)
        '22': slice(2*B, None),  # bottom row (right column)
    }

    return {'0000': (sd['00'], sd['00']),
            '0001': (sd['00'], sd['01']),
            '0002': (sd['00'], sd['02']),
            '0102': (sd['00'], sd['12']),
            '0202': (sd['00'], sd['22']),
            '0010': (sd['01'], sd['00']),
            '0011': (sd['01'], sd['01']),
            '0012': (sd['01'], sd['02']),
            '0112': (sd['01'], sd['12']),
            '0212': (sd['01'], sd['22']),
            '0020': (sd['02'], sd['00']),
            '0021': (sd['02'], sd['01']),
            '0022': (sd['02'], sd['02']),
            '0122': (sd['02'], sd['12']),
            '0222': (sd['02'], sd['22']),
            '1020': (sd['12'], sd['00']),
            '1021': (sd['12'], sd['01']),
            '1022': (sd['12'], sd['02']),
            '1122': (sd['12'], sd['12']),
            '1222': (sd['12'], sd['22']),
            '2020': (sd['22'], sd['00']),
            '2021': (sd['22'], sd['01']),
            '2022': (sd['22'], sd['02']),
            '2122': (sd['22'], sd['12']),
            '2222': (sd['22'], sd['22'])}


overlap_tag_slices = generate_overlap_tag_slices()


def generate_pair_tag_lookup():
    ptl = {}
    for tag1, tag2 in overlap_tag_pairs.items():
        for idx1, idx2 in zip(overlap_tag_maps[tag1], overlap_tag_maps[tag2]):
            ptl[(idx1, idx2)] = tag1
    return ptl


def generate_tag_pair_lookup():
    tpl = {}
    for tag1, tag2 in overlap_tag_pairs.items():
        tpl[tag1] = list(zip(overlap_tag_maps[tag1], overlap_tag_maps[tag2]))
    return tpl


def generate_overlap_tag_nines_mask():
    overlap_tag_nines_mask = {}
    for overlap_tag, overlap_map in overlap_tag_maps.items():
        arr9 = np.zeros(9, dtype=np.bool8)
        for idx in overlap_map:
            arr9[idx] = True
        overlap_tag_nines_mask[overlap_tag] = arr9
    return overlap_tag_nines_mask


def vec2ijpairs(vec, dim0=3):
    assert len(vec) % dim0 == 0
    dim1 = len(vec) // dim0
    ijpairs = np.zeros((len(vec), 2), dtype=int)
    for i, v in enumerate(vec):
        ijpairs[i, 0] = v // dim0
        ijpairs[i, 1] = v % dim1
    return ijpairs


def ijpairs2vec(ijpairs, dim0=3):
    assert len(ijpairs) % dim0 == 0
    vec = np.zeros(ijpairs, dtype=int)
    for i, ijpair in enumerate(ijpairs):
        vec[i] = ijpair[0] * dim0 + ijpair[1]
    return vec


def convert_nine2tups(dups9):
    """
    [0, 1, 0, 0, 0, 1, 1, 0, 0]
    [(0, 1, 0), (0, 0, 1), (1, 0, 0)]
    [(0, 1), (1, 2), (2, 0)]

    :param dups9:
    :return:
    """
    if type(dups9[0]) == str:
        b9 = np.array([int(i) for i in dups9])
    else:
        b9 = np.array(dups9)
    b33 = b9.reshape((3, 3))
    x2 = np.argwhere(b33)
    return [tuple(x) for x in x2]


def convert_tups2nine(tups):
    """
    [(0, 1), (1, 2), (2, 0)]
    [(0, 1, 0), (0, 0, 1), (1, 0, 0)]
    [0, 1, 0, 0, 0, 1, 1, 0, 0]

    :param tups:
    :return:
    """
    b33 = np.zeros((3, 3), dtype=int)
    for i, j in tups:
        b33[i, j] = 1
    b9 = b33.reshape(-1)
    return b9


def get_datetime_now(t=None, fmt='%Y_%m%d_%H%M'):
    """Return timestamp as a string; default: current time, format: YYYY_DDMM_hhmm_ss."""
    if t is None:
        t = datetime.now()
    return t.strftime(fmt)


def quick_stats(arr):
    print(arr.shape, arr.dtype, np.min(arr), np.max(arr), np.mean(arr), np.std(arr), np.sum(arr))


def bce(y_true, y_pred, **kwargs):
    y_pred = np.clip(y_pred, EPS, 1. - EPS)
    out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return np.mean(out, axis=-1)


def pad_string(x, n):
    padding = n - len(x)
    x_new = x if padding <= 0 else ''.join(['0' * padding, x])
    return x_new


def hex_to_int(hash_hex):
    return int(hash_hex, 16)


def int_to_hex(hash_int, hash_len):
    hash_hex = hex(hash_int)[2:]
    return pad_string(hash_hex, hash_len)


def get_hamming_distance_score(hash1, hash2, normalize=False, as_score=True):
    """
    The args should be the same datatype as the output type of opencv img_hash blockMeanHash.
    Order does not matter. i.e. hash1, hash2 will produce the same result as hash2, hash1.

    :param hash1: len 32 ndarray of uint8
    :param hash2: len 32 ndarray of uint8
    :param normalize: bool. If True, normalize the metric [0, 1]
    :param as_score: bool. flips the hamming metric. The larger the number, the more perfect the match.
    :return: float if normalize is True, uint8 otherwise
    """
    h1 = np.unpackbits(hash1)
    h2 = np.unpackbits(hash2)

    hamming_metric = np.sum(h1 ^ h2, dtype=np.int)
    if as_score:
        hamming_metric = 256 - hamming_metric
    if normalize:
        hamming_metric = hamming_metric / 256

    return hamming_metric


def get_best_model_name(run_dir):
    best_model = None
    min_loss = 999.9
    run_dir2 = os.path.join('out', run_dir)
    for filename in os.listdir(run_dir2):
        if not filename.endswith('.hdf5'):
            continue
        if '.last.' in filename:
            continue
        filebase = filename.rsplit('.', maxsplit=1)[0]
        loss = float(filebase.split('-')[1])
        if loss <= min_loss:
            best_model = filename
            min_loss = loss
    best_model_filename = os.path.join('out', run_dir, best_model)
    print(best_model_filename)
    return best_model_filename


def get_tile(img, idx, sz=256):
    i, j = idx2ijpair[idx]
    return img[i * sz:(i + 1) * sz, j * sz:(j + 1) * sz, :]


def relative_diff(val1, val2, func='max'):
    funcs = {
        'mean': lambda x, y: (x + y) / 2.0,
        'mean_abs': lambda x, y: (np.abs(x) + np.abs(y)) / 2.0,
        'min': lambda x, y: np.min([x, y], axis=0),
        'min_abs': lambda x, y: np.min([np.abs(x), np.abs(y)], axis=0),
        'max': lambda x, y: np.max([x, y], axis=0),
        'max_abs': lambda x, y: np.max([np.abs(x), np.abs(y)], axis=0),
    }
    f = funcs[func]
    num = val1 - val2
    den = f(val1, val2)
    return np.abs(num / (den + (den == 0)))


def percent_diff(val1, val2):
    return relative_diff(val1, val2) * 100


def fuzzy_join(tile1, tile2):
    maxab = np.max(np.stack([tile1, tile2]), axis=0)
    a = maxab - tile2
    b = maxab - tile1
    return a + b


def fuzzy_diff(tile1, tile2):
    ab = fuzzy_join(tile1, tile2)
    return np.sum(ab)


def fuzzy_norm(tile1, tile2):
    ab = fuzzy_join(tile1, tile2)
    n = 255 * np.sqrt(np.prod(ab.shape))
    return np.linalg.norm(255 - ab) / n


def fuzzy_compare(tile1, tile2):
    ab = fuzzy_join(tile1, tile2)
    n = 255 * np.prod(ab.shape)
    return np.sum(255 - ab) / n


def check_exact_match(img1, img2, img1_overlap_tag):
    img1_slice = img1[overlap_tag_slices[img1_overlap_tag]]
    img2_slice = img2[overlap_tag_slices[overlap_tag_pairs[img1_overlap_tag]]]
    return np.all(img1_slice == img2_slice)


def check_fuzzy_diff(img1, img2, img1_overlap_tag):
    img1_slice = img1[overlap_tag_slices[img1_overlap_tag]]
    img2_slice = img2[overlap_tag_slices[overlap_tag_pairs[img1_overlap_tag]]]
    return fuzzy_diff(img1_slice, img2_slice)


def check_fuzzy_score(img1, img2, img1_overlap_tag):
    img1_slice = img1[overlap_tag_slices[img1_overlap_tag]]
    img2_slice = img2[overlap_tag_slices[overlap_tag_pairs[img1_overlap_tag]]]
    return fuzzy_compare(img1_slice, img2_slice)


def get_issolid_flags(tile):
    issolid_flags = np.array([-1, -1, -1])
    for chan in range(3):
        pix = np.unique(tile[:, :, chan].flatten())
        if len(pix) == 1:
            issolid_flags[chan] = pix[0]
    return issolid_flags


def gen_entropy(tile):
    entropy_shannon = np.zeros(3)
    for chan in range(3):
        entropy_shannon[chan] = shannon_entropy(tile[:, :, chan])

    return entropy_shannon


def gen_greycop_hash(tile, n_metrics):
    # TODO: Add second order entropy using 3x3 kernel.
    #  https://www.harrisgeospatial.com/docs/backgroundtexturemetrics.html

    # def get_sub_tile(tile, i, j, sz=256):
    #     return tile[i * sz:(i + 1) * sz, j * sz:(j + 1) * sz, :]

    assert tile.shape == (256, 256, 3)
    # if tile_power is None:
    #     tile_power = np.log2(256).astype(int)

    distances = [1]
    angles = np.array([0, 2]) * np.pi / 4
    properties = ['contrast', 'homogeneity', 'energy', 'correlation']
    assert n_metrics == len(properties) + 1
    # kernel = diamond(1)
    glcm_feats = np.zeros((len(properties) + 1, 3), dtype=np.float)
    # glcm_entropy = np.zeros((3, ))
    for chan in range(3):
        # mode0_img = modal(tile[:, :, chan], kernel)
        # entr0_img = entropy(tile[:, :, chan], kernel)

        glcm = greycomatrix(tile[:, :, chan], distances=distances, angles=angles, symmetric=True, normed=True)
        # glcm = np.squeeze(glcm)
        feats = np.array([greycoprops(glcm, prop) for prop in properties])
        glcm_feats[:-1, ..., chan] = np.squeeze(np.mean(feats, axis=-1))
        # feats = np.hstack(feats)
        # feats.append(glcm_entropy)
        glcm_entropy = -np.sum(glcm * np.log(glcm + (glcm == 0)), axis=(0, 1))
        glcm_feats[-1, ..., chan] = np.squeeze(np.mean(glcm_entropy, axis=-1))

    # pyramid_pixels = np.zeros((tile_power, 3), dtype=np.float)
    # pyramid_weights1 = np.zeros((tile_power, 3), dtype=np.float)
    # pyramid_weights2 = np.zeros((tile_power, 3), dtype=np.float)
    # for ii in range(tile_power):
    #
    #     len_sub_tiles = 2 ** ii
    #     sub_tile_dim = 256 // len_sub_tiles
    #     n_sub_tiles = len_sub_tiles * len_sub_tiles
    #     sub_tile_size = sub_tile_dim * sub_tile_dim
    #     pixel1 = np.zeros((n_sub_tiles, 3), dtype=np.uint8)
    #     weight1 = np.zeros((n_sub_tiles, 3), dtype=np.int64)
    #     entropy_shannon = np.zeros((n_sub_tiles, 3), dtype=np.float)
    #     entropy_shannon1 = np.zeros((n_sub_tiles, 3), dtype=np.float)
    #     entropy_shannon2 = np.zeros((n_sub_tiles, 3), dtype=np.float)
    #     entropy_shannon3 = np.zeros((n_sub_tiles, 3), dtype=np.float)
    #
    #     ij_pairs = generate_ij_pairs(len_sub_tiles)
    #     for idx in range(n_sub_tiles):
    #
    #         sub_tile = get_sub_tile(tile, *ij_pairs[idx], sub_tile_dim)
    #         selem = square(3)
    #         for chan in range(3):
    #
    #             pix, cts = np.unique(sub_tile[:, :, chan].flatten(), return_counts=True)
    #             max_idx = np.argmax(cts)
    #             pixel1[idx, chan] = pix[max_idx]
    #             weight1[idx, chan] = cts[max_idx]
    #             probs = cts / np.sum(cts)
    #             entropy_shannon3[idx, chan] = -np.sum(probs * np.log2(probs))
    #             mode_img = modal(sub_tile[:, :, chan], selem)
    #             entropy_shannon[idx, chan] = shannon_entropy(mode_img)
    #             entr_img = entropy(sub_tile[:, :, chan], selem)
    #             entropy_shannon1[idx, chan] = shannon_entropy(entr_img)
    #             entropy_shannon2[idx, chan] = shannon_entropy(sub_tile[:, :, chan])
    #
    #     pyramid_pixels[ii] = np.mean(pixel1, axis=0) / 255.
    #     pyramid_weights[ii] = np.mean(weight1, axis=0) / sub_tile_size
    #     pyramid_weights1[ii] = np.mean(entropy_shannon1, axis=0)
    #     pyramid_weights2[ii] = np.mean(entropy_shannon2, axis=0)
    #
    # pixel_mean = np.mean(pyramid_pixels, axis=0)
    # pixel_stdev = np.std(pyramid_pixels, axis=0)
    # weight_mean = np.mean(pyramid_weights, axis=1)
    # pyramid_hash = np.hstack([pixel_mean, pixel_stdev, weight_mean])

    return np.mean(glcm_feats, axis=1)


def to_hls(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS_FULL)


def to_bgr(hls):
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR_FULL)


def channel_shift(img, chan, val):
    """
    img must already be in hls (hue, lightness, saturation) format.
    img values must be uint8. [0, 255] so that hue will wrap around correctly.
    """

    gimp_scale = chan_gimp_scale_map[chan]
    idx = chan_idx_map[chan]

    # TODO: Add a plot showing how we arrived at each of the three scaling options.
    if idx == 0:  # hue

        scaled_val = 255. * val / gimp_scale
        scaled_val = np.around(scaled_val).astype(np.uint8)
        scaled_img = np.copy(img)
        scaled_img[:, :, idx] += scaled_val  # this line won't work correctly if img is not in bytes.

    elif idx == 1:  # lightness

        l = img[:, :, idx] * (1. / 255.)
        v2 = val / gimp_scale
        one_m_v2 = 1 - v2
        one_p_v2 = 1 + v2
        l_shifted = l * one_m_v2 + v2 if val > 0 else l * one_p_v2
        l_shifted = np.clip(l_shifted, 0, 1)
        scaled_img = np.copy(img)
        scaled_img[:, :, idx] = np.around(255 * l_shifted).astype(np.uint8)

    elif idx == 2:  # saturation

        scaled_val = (val / gimp_scale) + 1.
        s_shifted = img[:, :, idx] * scaled_val
        s_shifted = np.clip(s_shifted, 0, 255)
        scaled_img = np.copy(img)
        scaled_img[:, :, idx] = np.around(s_shifted).astype(np.uint8)

    else:
        raise ValueError

    return scaled_img


def read_duplicate_truth(filename):
    """
    Reads files with the following line format,
    "73fec0637.jpg 9a2f9d347.jpg 0022 0"

    :param filename:
    :return:
    """
    duplicate_truth = {}

    with open(filename, 'r') as ifs:
        for line in ifs.readlines():
            img1_id, img2_id, img1_overlap_tag, is_duplicate = line.strip().split(' ')
            if img1_id > img2_id:
                img1_id, img2_id = img2_id, img1_id
                img1_overlap_tag = overlap_tag_pairs[img1_overlap_tag]
            if (img1_id, img2_id, img1_overlap_tag) in duplicate_truth:
                continue
            duplicate_truth[(img1_id, img2_id, img1_overlap_tag)] = int(is_duplicate)

    return duplicate_truth


def load_duplicate_truth(filepath='data', filename='duplicate_truth.txt', from_chunks=True, chunk_type='all'):
    """
    Load in the main single file (duplicate_truth.txt) or load and concatenate all chunk_truth files.
    Should get the same result either way.

    :param filepath: str, path to filename or chunk files.
    :param filename: str, name of complete truth file.
    :param from_chunks: bool, If true, load truth from chunk files.
    :param chunk_type: str, if 'manual' load the verified truth only, if 'auto' load the automatic, otherwise load both.
    :return: dict
    """

    chunk_prefix = 'chunk_'
    if chunk_type == 'manual':
        chunk_prefix += 'truth_'
    elif chunk_type == 'auto':
        chunk_prefix += 'auto_'

    duplicate_truth = None

    if from_chunks:
        chunk = {}
        for fname in sorted(os.listdir(filepath)):
            if not fname.startswith(chunk_prefix):
                continue
            chunk.update(read_duplicate_truth(os.path.join(filepath, fname)))

        duplicate_truth = {}
        for k, v in sorted(chunk.items()):
            duplicate_truth[k] = v
    else:
        filename = os.path.join(filepath, filename)
        if os.path.exists(filename):
            duplicate_truth = read_duplicate_truth(filename)

    return duplicate_truth


def write_duplicate_truth(filename, duplicate_truth):

    with open(filename, 'w') as ofs:
        for (img1_id, img2_id, img1_overlap_tag), is_duplicate in sorted(duplicate_truth.items()):
            ofs.write(' '.join([img1_id, img2_id, img1_overlap_tag, str(is_duplicate)]) + '\n')


def update_duplicate_truth(pre_chunk, filepath='data', filename='duplicate_truth.txt', auto=True):

    duplicate_truth = load_duplicate_truth(filepath=filepath, filename=filename)

    chunk = {}
    for (img1_id, img2_id, img1_overlap_tag), is_duplicate in pre_chunk.items():
        if (img1_id, img2_id, img1_overlap_tag) in duplicate_truth:
            if duplicate_truth[(img1_id, img2_id, img1_overlap_tag)] != is_duplicate:
                raise ValueError(f"({img1_id}, {img2_id}, {img1_overlap_tag}) cannot both be {duplicate_truth[(img1_id, img2_id, img1_overlap_tag)]} and {is_duplicate}")
            continue
        if (img1_id, img2_id, img1_overlap_tag) in chunk:
            continue
        chunk[(img1_id, img2_id, img1_overlap_tag)] = int(is_duplicate)

    if len(chunk) > 0:

        # First save chunk to a new file.
        chunk_type = 'auto' if auto else 'truth'
        datetime_now = get_datetime_now()
        n_lines_in_chunk = len(chunk)
        n_lines_in_chunk_str = pad_string(str(n_lines_in_chunk), 6)
        chunk_filename = '_'.join(['chunk', chunk_type, datetime_now, n_lines_in_chunk_str]) + '.txt'
        write_duplicate_truth(os.path.join(filepath, chunk_filename), chunk)

        # Then update the duplicate_truth.txt file.
        chunk.update(duplicate_truth)
        duplicate_truth = {}
        for k, v in sorted(chunk.items()):
            duplicate_truth[k] = v
        write_duplicate_truth(os.path.join(filepath, filename), duplicate_truth)

        return duplicate_truth


def create_dataset_from_tiles_and_truth(sdcic):

    tpl = generate_tag_pair_lookup()
    dup_truth = load_duplicate_truth()

    dup_pairs = set()
    img_overlap_pairs = []

    # First collect all image pairs flagged as duplicates.
    for (img1_id, img2_id, img1_overlap_tag), is_dup in dup_truth.items():
        if is_dup:
            for idx1, idx2 in tpl[img1_overlap_tag]:
                img_overlap_pairs.append((img1_id, img2_id, idx1, idx2, is_dup))
            # Keep a record of all duplicate image pairs for later reference.
            dup_pairs.add((img1_id, img2_id))

    n_dup_tile_pairs = len(img_overlap_pairs)
    print(f"Number of non-dup/dup tiles: {0:>8}/{n_dup_tile_pairs}")

    # For the second pass, record the non-dups as non-dups unless the hashes of
    # overlapping tile are equal in which case just ignore that tile pair.
    # Also, if the two images have already been flagged duplicate (possibly for
    # a different overlap), then exclude all other overlaps we might have
    # accidentally picked up.
    done = False
    for (img1_id, img2_id, img1_overlap_tag), is_dup in dup_truth.items():
        if is_dup or (img1_id, img2_id) in dup_pairs:
            continue

        for idx1, idx2 in tpl[img1_overlap_tag]:
            # If 2 tiles are the same then skip them since they are actually dups.
            # Remember a dup corresponds to the "entire" overlay.  if the overlay
            # is flagged as non-dup then at least one of the tiles is different.
            if sdcic.tile_md5hash_grids[img1_id][idx1] == sdcic.tile_md5hash_grids[img2_id][idx2]:
                continue
            img_overlap_pairs.append((img1_id, img2_id, idx1, idx2, is_dup))
            if len(img_overlap_pairs) > 2 * n_dup_tile_pairs:
                done = True
                break
        if done:
            break

    print(f"Number of non-dup/dup tiles: {len(img_overlap_pairs) - n_dup_tile_pairs:>8}/{n_dup_tile_pairs}")

    if done:
        return img_overlap_pairs

    # Now go through the rest of the possible matches not yet verified in truth
    # and pick out the two tiles that, if the images were dups, would be the
    # tiles farthest away and set those as non-dups. These are good non dup
    # candidates because they are most likely very similar images but also most
    # likely not duplicates. Verify by comparing hashes.
    n_matching_tiles_list = [9, 6, 4, 3, 2, 1]
    for n_matching_tiles in n_matching_tiles_list:
        possible_matches_file = os.path.join("data", f"overlap_bmh_tile_scores_{n_matching_tiles}.pkl")
        df = pd.read_pickle(possible_matches_file)
        # load matches -> [(img1_id, img2_id, img1_overlap_tag), ...]

        possible_matches = {(i1, i2, o1): s for i1, i2, o1, *s in df.to_dict('split')['data']}
        for img1_id, img2_id, img1_overlap_tag in possible_matches:
            # We've already accounted for these earlier up above.
            if (img1_id, img2_id) in dup_pairs or (img1_id, img2_id, img1_overlap_tag) in dup_truth:
                continue
            # Find scores for far_away_corners.
            far_scores = []
            for img1_far_tag in far_away_corners[img1_overlap_tag]:
                bmh_scores = sdcic.get_bmh_scores(img1_id, img2_id, img1_far_tag)
                far_scores.append(bmh_scores[0])
            # The score has to be lower than sdcic.overlap_bmh_min_score.
            if min(far_scores) > sdcic.overlap_bmh_min_score:
                continue
            # Keep the one that has the lowest score.
            idx1, idx2 = tpl[far_away_corners[img1_overlap_tag][far_scores.index(min(far_scores))]][0]
            img_overlap_pairs.append((img1_id, img2_id, idx1, idx2, 0))
            if len(img_overlap_pairs) > 2 * n_dup_tile_pairs:
                done = True
                break
        if done:
            break

    print(f"Number of non-dup/dup tiles: {len(img_overlap_pairs) - n_dup_tile_pairs:>8}/{n_dup_tile_pairs}")

    if done:
        return img_overlap_pairs

    # Finally, if we still don't have a 50/50 split between dup/non-dup datapoints,
    # choose random images from the dataset and a random tile from each image and set
    # those as non-dups and continue until we have 50/50 split. Verify by comparing hashes.
    img_ids = os.listdir(sdcic.train_image_dir)
    corners = ['0000', '0202', '2020', '2222']

    while True:

        img1_id, img2_id = np.random.choice(img_ids, 2)
        # TODO: implement option to choose ANY random tiles not just the ones on opposite corners.
        img1_overlap_tag = np.random.choice(corners)

        # Probably won't ever happen but just in case...
        if (img1_id, img2_id) in dup_pairs or (img1_id, img2_id, img1_overlap_tag) in dup_truth:
            continue
        # The score has to be lower than sdcic.overlap_bmh_min_score.
        if sdcic.get_bmh_scores(img1_id, img2_id, img1_overlap_tag)[0] > sdcic.overlap_bmh_min_score:
            continue
        # Keep the one that has the lowest score.
        idx1, idx2 = tpl[img1_overlap_tag][0]
        img_overlap_pairs.append((img1_id, img2_id, idx1, idx2, 0))
        if len(img_overlap_pairs) > 2 * n_dup_tile_pairs:
            break

    print(f"Number of non-dup/dup tiles: {len(img_overlap_pairs) - n_dup_tile_pairs:>8}/{n_dup_tile_pairs}")

    return img_overlap_pairs


def even_split(n_samples, batch_size, split):
    # split the database into train/val sizes such that
    # batch_size divides them both evenly.
    # Hack until I can figure out how to ragged end of the database.
    n_data_crumbs = n_samples % batch_size
    n_samples = n_samples - n_data_crumbs
    train_percent = split / 100.
    n_train = round(n_samples * train_percent)
    n_valid = n_samples - n_train
    assert n_train % batch_size == 0
    assert n_valid % batch_size == 0
    return n_train, n_valid


def update_tile_cliques(G, tile1_hash, tile2_hash):
    G.add_edge(tile1_hash, tile2_hash)
    tile1_neighbors = set(nx.neighbors(G, tile1_hash))
    tile2_neighbors = set(nx.neighbors(G, tile2_hash))
#     assert len(tile1_neighbors & tile2_neighbors) == 0
    tile12_neighbors = tile1_neighbors | tile2_neighbors
    T = nx.complete_graph(tile12_neighbors)
    T.add_edges_from([(n, n) for n in tile12_neighbors])
    G.update(T)
