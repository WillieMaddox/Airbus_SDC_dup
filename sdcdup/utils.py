import os
import operator
from shutil import copyfile
from datetime import datetime
from collections import Counter
from collections import namedtuple
from functools import lru_cache
from tqdm import tqdm
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

tilebox_corners = (
    np.array([[0, 0], [1, 1]]) * B,
    np.array([[1, 0], [2, 1]]) * B,
    np.array([[2, 0], [3, 1]]) * B,
    np.array([[0, 1], [1, 2]]) * B,
    np.array([[1, 1], [2, 2]]) * B,
    np.array([[2, 1], [3, 2]]) * B,
    np.array([[0, 2], [1, 3]]) * B,
    np.array([[1, 2], [2, 3]]) * B,
    np.array([[2, 2], [3, 3]]) * B)

new_tag_mapping = {
    '0000': '00',
    '0001': '01',
    '0002': '02',
    '0102': '12',
    '0202': '22',
    '0010': '03',
    '0011': '04',
    '0012': '05',
    '0112': '15',
    '0212': '25',
    '0020': '06',
    '0021': '07',
    '0022': '08',
    '0122': '18',
    '0222': '28',
    '1020': '36',
    '1021': '37',
    '1022': '38',
    '1122': '48',
    '1222': '58',
    '2020': '66',
    '2021': '67',
    '2022': '68',
    '2122': '78',
    '2222': '88'}

overlap_tag_pairs = {
    '00': '88',
    '01': '78',
    '02': '68',
    '12': '67',
    '22': '66',
    '03': '58',
    '04': '48',
    '05': '38',
    '15': '37',
    '25': '36',
    '06': '28',
    '07': '18',
    '08': '08',
    '18': '07',
    '28': '06',
    '36': '25',
    '37': '15',
    '38': '05',
    '48': '04',
    '58': '03',
    '66': '22',
    '67': '12',
    '68': '02',
    '78': '01',
    '88': '00'}

boundingbox_corners = {
    '00': np.array([[0, 0], [1, 1]]) * B,
    '01': np.array([[0, 0], [2, 1]]) * B,
    '02': np.array([[0, 0], [3, 1]]) * B,
    '12': np.array([[1, 0], [3, 1]]) * B,
    '22': np.array([[2, 0], [3, 1]]) * B,
    '03': np.array([[0, 0], [1, 2]]) * B,
    '04': np.array([[0, 0], [2, 2]]) * B,
    '05': np.array([[0, 0], [3, 2]]) * B,
    '15': np.array([[1, 0], [3, 2]]) * B,
    '25': np.array([[2, 0], [3, 2]]) * B,
    '06': np.array([[0, 0], [1, 3]]) * B,
    '07': np.array([[0, 0], [2, 3]]) * B,
    '08': np.array([[0, 0], [3, 3]]) * B,
    '18': np.array([[1, 0], [3, 3]]) * B,
    '28': np.array([[2, 0], [3, 3]]) * B,
    '36': np.array([[0, 1], [1, 3]]) * B,
    '37': np.array([[0, 1], [2, 3]]) * B,
    '38': np.array([[0, 1], [3, 3]]) * B,
    '48': np.array([[1, 1], [3, 3]]) * B,
    '58': np.array([[2, 1], [3, 3]]) * B,
    '66': np.array([[0, 2], [1, 3]]) * B,
    '67': np.array([[0, 2], [2, 3]]) * B,
    '68': np.array([[0, 2], [3, 3]]) * B,
    '78': np.array([[1, 2], [3, 3]]) * B,
    '88': np.array([[2, 2], [3, 3]]) * B}

overlap_tag_maps = {
    '00': np.array([0]),
    '01': np.array([0, 1]),
    '02': np.array([0, 1, 2]),
    '12': np.array([1, 2]),
    '22': np.array([2]),
    '03': np.array([0, 3]),
    '04': np.array([0, 1, 3, 4]),
    '05': np.array([0, 1, 2, 3, 4, 5]),
    '15': np.array([1, 2, 4, 5]),
    '25': np.array([2, 5]),
    '06': np.array([0, 3, 6]),
    '07': np.array([0, 1, 3, 4, 6, 7]),
    '08': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    '18': np.array([1, 2, 4, 5, 7, 8]),
    '28': np.array([2, 5, 8]),
    '36': np.array([3, 6]),
    '37': np.array([3, 4, 6, 7]),
    '38': np.array([3, 4, 5, 6, 7, 8]),
    '48': np.array([4, 5, 7, 8]),
    '58': np.array([5, 8]),
    '66': np.array([6]),
    '67': np.array([6, 7]),
    '68': np.array([6, 7, 8]),
    '78': np.array([7, 8]),
    '88': np.array([8])}

far_away_corners = {
    '00': ('88',),
    '01': ('88',),
    '02': ('88', '66'),
    '12': ('66',),
    '22': ('66',),
    '03': ('88',),
    '04': ('88',),
    '05': ('88', '66'),
    '15': ('66',),
    '25': ('66',),
    '06': ('88', '22'),
    '07': ('88', '22'),
    '08': ('88', '66', '22', '00'),
    '18': ('66', '00'),
    '28': ('66', '00'),
    '36': ('22',),
    '37': ('22',),
    '38': ('22', '00'),
    '48': ('00',),
    '58': ('00',),
    '66': ('22',),
    '67': ('22',),
    '68': ('22', '00'),
    '78': ('00',),
    '88': ('00',)}


def generate_overlap_tag_slices():

    # sd -> short for slice dictionary if that even means anything.
    sd = {
        '00': slice(None, 1*B),  # top row (left column)
        '01': slice(None, 2*B),  # top 2 rows (left 2 columns)
        '02': slice(None, None),   # all rows (all columns)
        '12': slice(1*B, None),  # bottom 2 rows (right 2 columns)
        '22': slice(2*B, None),  # bottom row (right column)
    }

    return {'00': (sd['00'], sd['00']),
            '01': (sd['00'], sd['01']),
            '02': (sd['00'], sd['02']),
            '12': (sd['00'], sd['12']),
            '22': (sd['00'], sd['22']),
            '03': (sd['01'], sd['00']),
            '04': (sd['01'], sd['01']),
            '05': (sd['01'], sd['02']),
            '15': (sd['01'], sd['12']),
            '25': (sd['01'], sd['22']),
            '06': (sd['02'], sd['00']),
            '07': (sd['02'], sd['01']),
            '08': (sd['02'], sd['02']),
            '18': (sd['02'], sd['12']),
            '28': (sd['02'], sd['22']),
            '36': (sd['12'], sd['00']),
            '37': (sd['12'], sd['01']),
            '38': (sd['12'], sd['02']),
            '48': (sd['12'], sd['12']),
            '58': (sd['12'], sd['22']),
            '66': (sd['22'], sd['00']),
            '67': (sd['22'], sd['01']),
            '68': (sd['22'], sd['02']),
            '78': (sd['22'], sd['12']),
            '88': (sd['22'], sd['22'])}


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


def rle_decode(rle_string, shape=(768, 768)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def rle_to_full_mask(rle_list, shape=(768, 768)):
    """
    Convert a list of run-length encoded masks to a binary mask.
    This assumes that there are no overlapping segmentations.

    :param rle_list: List of rle strings corresponding to a single image.
    :param shape: shape of the single image.
    :return: return shape will be (1, shape[0], shape[1])
    """
    all_masks = np.zeros(shape=shape, dtype=np.uint8)
    for mask in rle_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask, shape=shape)
    in_mask = np.array(all_masks, dtype=bool)
    return np.expand_dims(in_mask, -1)


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


def get_hamming_distance(hash1, hash2, normalize=False, as_score=False):
    """
    The args should be the same datatype as the output type of opencv img_hash blockMeanHash.
    Order does not matter. i.e. hash1, hash2 will produce the same result as hash2, hash1.

    :param hash1: len 32 ndarray of uint8
    :param hash2: len 32 ndarray of uint8
    :param normalize: bool. If True, normalize the metric [0, 1]
    :param as_score: bool. flips the hamming metric. The larger the score, the more perfect the match.
    :return: float if normalize is True, uint8 otherwise
    """
    h1 = np.unpackbits(hash1)
    h2 = np.unpackbits(hash2)

    hamming_metric = np.sum(h1 ^ h2, dtype=np.int)
    hamming_metric = 256 - hamming_metric if as_score else hamming_metric
    hamming_metric = hamming_metric / 256 if normalize else hamming_metric

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
    "73fec0637.jpg 9a2f9d347.jpg 08 0"

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


def create_dataset_from_tiles(sdcic):
    """
    is_dup issolid  action
     i==j   i   j    skip?
    ----------------------
        1   1   1     1
        1   1   0     1 Does not exist?
        1   0   1     1 Does not exist?
        1   0   0     0
    ----------------------
        0   1   1     1
        0   1   0     0 Could present problems if other tile is "near" solid.
        0   0   1     0 Could present problems if other tile is "near" solid.
        0   0   0     0

    :param sdcic:
    :return:
    """
    img_overlap_pairs_dup_keys = []
    img_overlap_pairs_non_dup_all = []

    KeyScore = namedtuple('keyscore', 'key score')
    for img_id, tile_md5hash_grid in tqdm(sdcic.tile_md5hash_grids.items()):
        for idx1, tile1_md5hash in enumerate(tile_md5hash_grid):
            for idx2, tile2_md5hash in enumerate(tile_md5hash_grid):

                if idx1 > idx2:
                    continue

                tile1_issolid = np.all(sdcic.tile_issolid_grids[img_id][idx1] >= 0)
                tile2_issolid = np.all(sdcic.tile_issolid_grids[img_id][idx2] >= 0)

                if idx1 == idx2:
                    if tile1_issolid:
                        continue
                    if tile2_issolid:
                        continue
                    img_overlap_pairs_dup_keys.append((img_id, img_id, idx1, idx2, 1))
                    continue

                # if idx1 != idx2:
                if tile1_md5hash == tile2_md5hash:
                    continue
                if tile1_issolid and tile2_issolid:
                    continue

                bmh1 = sdcic.tile_bm0hash_grids[img_id][idx1]
                bmh2 = sdcic.tile_bm0hash_grids[img_id][idx2]
                score = get_hamming_distance(bmh1, bmh2, as_score=True)

                if score == 256:
                    tile1 = sdcic.get_tile(sdcic.get_img(img_id), idx1)
                    tile2 = sdcic.get_tile(sdcic.get_img(img_id), idx2)
                    tile3 = fuzzy_join(tile1, tile2)
                    pix3, cts3 = np.unique(tile3.flatten(), return_counts=True)
                    if np.max(cts3 / (256 * 256 * 3)) > 0.97:
                        # skip all the near solid (i.e. blue edge) tiles.
                        continue

                img_overlap_pairs_non_dup_all.append(KeyScore((img_id, img_id, idx1, idx2, 0), score/256))

    img_overlap_pairs_non_dup_keys_sorted = []
    for candidate in tqdm(sorted(img_overlap_pairs_non_dup_all, key=operator.attrgetter('score'), reverse=True)):
        img_overlap_pairs_non_dup_keys_sorted.append(candidate.key)

    img_overlap_pairs_non_dup_keys = img_overlap_pairs_non_dup_keys_sorted[:len(img_overlap_pairs_dup_keys)]
    img_overlap_pairs = img_overlap_pairs_non_dup_keys + img_overlap_pairs_dup_keys

    # non_dup_scores = []
    # img_overlap_pairs_non_dup_all_sorted = []
    # for candidate in tqdm(sorted(img_overlap_pairs_non_dup_all, key=operator.attrgetter('score'), reverse=True)):
    #     non_dup_scores.append(candidate.score)
    #     img_overlap_pairs_non_dup_all_sorted.append(candidate)
    # assert min(non_dup_scores) == non_dup_scores[0], (min(non_dup_scores), non_dup_scores[0])
    # assert max(non_dup_scores) == non_dup_scores[-1], (max(non_dup_scores), non_dup_scores[-1])
    # non_dup_scores = non_dup_scores[:len(img_overlap_pairs_dup_keys)]
    # assert max(non_dup_scores) == non_dup_scores[-1], (max(non_dup_scores), non_dup_scores[-1])
    # np.random.shuffle(non_dup_scores)
    # img_overlap_pairs_dup = []
    # for key, score in zip(img_overlap_pairs_dup_keys, non_dup_scores):
    #     img_overlap_pairs_dup.append(KeyScore(key, score))
    # img_overlap_pairs_non_dup_sorted = img_overlap_pairs_non_dup_all_sorted[:len(img_overlap_pairs_dup_keys)]
    # img_overlap_pairs = img_overlap_pairs_non_dup_sorted + img_overlap_pairs_dup

    return img_overlap_pairs


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
        # load matches -> [(img1_id, img2_id, img1_overlap_tag), ...]
        possible_matches_file = os.path.join("data", f"overlap_bmh_tile_scores_{n_matching_tiles}.pkl")
        df = pd.read_pickle(possible_matches_file)
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
    corners = ['00', '22', '66', '88']

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
    n_batches = n_samples // batch_size
    n_train_batches = round(n_batches * split)
    n_valid_batches = n_batches - n_train_batches
    n_train = n_train_batches * batch_size
    n_valid = n_valid_batches * batch_size
    assert n_train + n_valid <= n_samples, n_train
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


class CSVLogger:
    def __init__(self, filename, header):
        self.filename = filename
        self.header = header

    def on_epoch_end(self, stats):

        if not os.path.exists(self.filename):
            with open(self.filename, 'w') as ofs:
                ofs.write(','.join(self.header) + '\n')

        with open(self.filename, 'a') as ofs:
            ofs.write(','.join(map(str, stats)) + '\n')
