import os
import random
from datetime import datetime
from functools import lru_cache

import numpy as np
import networkx as nx
import cv2

import torch
from torch._six import int_classes as _int_classes
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pathlib import Path
from dotenv import load_dotenv, find_dotenv


def get_project_root() -> str:
    """Returns project root folder."""
    return os.fspath(Path(__file__).parent.parent)


load_dotenv(find_dotenv())

project_root = get_project_root()
models_dir = os.path.join(project_root, 'models')
train_image_dir = os.path.join(project_root, os.getenv('RAW_DATA_DIR'), 'train_768')
interim_data_dir = os.path.join(project_root, os.getenv('INTERIM_DATA_DIR'))
train_tile_dir = os.path.join(project_root, os.getenv('PROCESSED_DATA_DIR'), 'train_256')
processed_data_dir = os.path.join(project_root, os.getenv('PROCESSED_DATA_DIR'))

EPS = np.finfo(np.float32).eps

idx_chan_map = {0: 'H', 1: 'L', 2: 'S'}
chan_idx_map = {'H': 0, 'L': 1, 'S': 2}
chan_cv2_scale_map = {'H': 256, 'L': 256, 'S': 256}
chan_gimp_scale_map = {'H': 360, 'L': 200, 'S': 100}


# There are 24 distinct ways a $3\times3$ grid can overlap with another $3\times3$ grid.


B = 256

ij_pairs_3x3 = ((0, 0), (0, 1), (0, 2),
                (1, 0), (1, 1), (1, 2),
                (2, 0), (2, 1), (2, 2))

idx2ijpair = {idx: ij for idx, ij in enumerate(ij_pairs_3x3)}
ijpair2idx = {ij: idx for idx, ij in enumerate(ij_pairs_3x3)}

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

overlap_tags = list(overlap_tag_maps)

overlap_tag_pairs = dict(zip(overlap_tags, overlap_tags[::-1]))


def generate_boundingbox_corners():
    bbc = {}
    for overlap_tag in overlap_tags:
        idx1, idx2 = list(map(int, list(overlap_tag)))
        i, j = idx1 % 3, idx1 // 3
        k, l = idx2 % 3 + 1, idx2 // 3 + 1
        bbc[overlap_tag] = np.array([[i, j], [k, l]]) * B
    return bbc


def generate_third_party_overlaps():
    overlap_tag_matrix = np.array(overlap_tags).reshape((5, 5))
    third_party_overlaps = {}
    for overlap_tag in overlap_tags:
        third_party_overlaps[overlap_tag] = []
        center_row, center_col = np.argwhere(overlap_tag_matrix == overlap_tag)[0]
        for ii in range(max(0, center_row - 2), min(center_row + 3, 5)):
            for jj in range(max(0, center_col - 2), min(center_col + 3, 5)):
                third_party_overlaps[overlap_tag].append(overlap_tags[ii * 5 + jj])
    return third_party_overlaps


def generate_overlap_tag_slices():

    # sd -> short for slice dictionary if that even means anything.
    sd = {
        '00': slice(None, 1*B),  # top row (left column)
        '01': slice(None, 2*B),  # top 2 rows (left 2 columns)
        '02': slice(None, None),  # all rows (all columns)
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


@lru_cache(maxsize=None)
def generate_ij_pairs(dim=3):
    return tuple([(i, j) for i in range(dim) for j in range(dim)])


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


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


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


def pad_string(x, n):
    padding = n - len(x)
    x_new = x if padding <= 0 else ''.join(['0' * padding, x])
    return x_new


def get_datetime_now(t=None, fmt='%Y_%m%d_%H%M'):
    """Return timestamp as a string; default: current time, format: YYYY_DDMM_hhmm_ss."""
    if t is None:
        t = datetime.now()
    return t.strftime(fmt)


def get_hamming_distance(hash1, hash2, normalize=False, as_score=False):
    """
    The args should be the same datatype as the output type of opencv img_hash blockMeanHash.
    Order does not matter. i.e. hash1, hash2 will produce the same result as hash2, hash1.

    :param hash1: len 32 or 96 (3*32) ndarray of uint8
    :param hash2: len 32 or 96 (3*32) ndarray of uint8
    :param normalize: bool. If True, normalize the metric [0, 1]
    :param as_score: bool. flips the hamming metric. The larger the score, the more perfect the match.
    :return: float if normalize is True, uint8 otherwise
    """
    h1 = np.unpackbits(hash1)
    h2 = np.unpackbits(hash2)

    hamming_metric = np.sum(h1 ^ h2, dtype=np.int)
    hamming_metric = len(h1) - hamming_metric if as_score else hamming_metric
    hamming_metric = hamming_metric / len(h1) if normalize else hamming_metric

    return hamming_metric


def get_hamming_distance_array(hash1, hash2, normalize=False, as_score=False, axis=1):
    """
    The args should be the same datatype as the output type of opencv img_hash blockMeanHash.
    Order does not matter. i.e. hash1, hash2 will produce the same result as hash2, hash1.

    :param hash1: len 32 ndarray of uint8
    :param hash2: len 32 ndarray of uint8
    :param normalize: bool. If True, normalize the metric [0, 1]
    :param as_score: bool. flips the hamming metric. The larger the score, the more perfect the match.
    :return: float if normalize is True, uint8 otherwise
    """
    h1 = np.unpackbits(hash1, axis=axis)
    h2 = np.unpackbits(hash2, axis=axis)

    hamming_metric = np.sum(h1 ^ h2, dtype=np.int, axis=axis)
    hamming_metric = h1.shape[-1] - hamming_metric if as_score else hamming_metric
    hamming_metric = hamming_metric / h1.shape[-1] if normalize else hamming_metric

    return hamming_metric


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


def read_duplicate_truth(filename):
    """
    Reads files with the following format:
    img1_id, img2_id, img1_overlap_map, is_dup
    e.g. "73fec0637.jpg 9a2f9d347.jpg 08 0"

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


def load_duplicate_truth(filepath=processed_data_dir, filename=None, chunk_type='truth'):
    """
    Load in the main single file (duplicate_truth.txt) or load and concatenate all chunk_truth files.
    Should get the same result either way.

    :param filepath: str, path to filename or chunk files.
    :param filename: str, file to read truth from. If None, load truth from all files in filepath that match chunk_type.
    :param chunk_type: str, if 'manual' load the verified truth only, if 'auto' load the automatic, otherwise load both.
    :return: dict
    """

    chunk_prefix = 'chunk_'
    if chunk_type == 'truth':
        chunk_prefix += 'truth_'
    elif chunk_type == 'auto':
        chunk_prefix += 'auto_'

    duplicate_truth = None

    if filename:
        filename = os.path.join(filepath, filename)
        if os.path.exists(filename):
            duplicate_truth = read_duplicate_truth(filename)
    else:
        chunk = {}
        for fname in sorted(os.listdir(filepath)):
            if not fname.startswith(chunk_prefix):
                continue
            chunk.update(read_duplicate_truth(os.path.join(filepath, fname)))

        duplicate_truth = {}
        for k, v in sorted(chunk.items()):
            duplicate_truth[k] = v

    return duplicate_truth


def write_duplicate_truth(filename, duplicate_truth):

    with open(filename, 'w') as ofs:
        for (img1_id, img2_id, img1_overlap_tag), is_duplicate in sorted(duplicate_truth.items()):
            ofs.write(' '.join([img1_id, img2_id, img1_overlap_tag, str(is_duplicate)]) + '\n')


def update_duplicate_truth(pre_chunk, filepath=processed_data_dir, verified=False):

    chunk_type = 'truth' if verified else 'auto'

    duplicate_truth = load_duplicate_truth(filepath=filepath, chunk_type=chunk_type)

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

        datetime_now = get_datetime_now()
        n_lines_in_chunk = pad_string(str(len(chunk)), 6)
        chunk_filename = '_'.join(['chunk', chunk_type, datetime_now, n_lines_in_chunk]) + '.txt'
        write_duplicate_truth(os.path.join(filepath, chunk_filename), chunk)

        for k, v in sorted(chunk.items()):
            duplicate_truth[k] = v

        if chunk_type == 'truth':
            valid_auto_files = []
            valid_auto_chunks = {}
            for fname in sorted(os.listdir(filepath)):
                if not fname.startswith('chunk_auto_'):
                    continue
                auto_chunks = read_duplicate_truth(os.path.join(filepath, fname))
                to_del_arr = []
                for k, v in chunk.items():
                    if k in auto_chunks:
                        to_del_arr.append(k)
                if len(to_del_arr) > 0:
                    for to_del in to_del_arr:
                        auto_is_dup = auto_chunks.pop(to_del)
                        if chunk[to_del] != auto_is_dup:
                            print(fname, to_del)
                    valid_auto_chunks.update(auto_chunks)
                    valid_auto_files.append(fname)

            if len(valid_auto_chunks) > 0:
                datetime_now = get_datetime_now()
                n_lines_in_chunk = pad_string(str(len(valid_auto_chunks)), 6)
                chunk_filename = '_'.join(['chunk_auto', datetime_now, n_lines_in_chunk]) + '.txt'
                write_duplicate_truth(os.path.join(filepath, chunk_filename), valid_auto_chunks)

            if len(valid_auto_files) > 0:
                invalid_auto_files = ['_'.join(['invalid', auto_file]) for auto_file in valid_auto_files]
                for src, dst in zip(valid_auto_files, invalid_auto_files):
                    os.rename(os.path.join(filepath, src), os.path.join(filepath, dst))

    return duplicate_truth


def bce_loss(ytrue, yprob):
    return -1 * (np.log(np.max([EPS, yprob])) if ytrue else np.log(np.max([EPS, 1 - yprob])))


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


def get_img(img_id):
    return cv2.imread(os.path.join(train_image_dir, img_id))


def get_tile(img, idx, sz=256):
    i, j = idx2ijpair[idx]
    return img[i * sz:(i + 1) * sz, j * sz:(j + 1) * sz, :]


def to_hls(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS_FULL)


def to_bgr(hls):
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR_FULL)


def hls_shift(img, chan, val):
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


class ImgMod:
    """
    Reads a single image to be modified by hls.
    """

    def __init__(self, filename):
        self.filename = filename
        self.img_id = filename.split('/')[-1]

        self._hls_chan = None
        self._hls_gain = None

        self._parent_bgr = None
        self._parent_hls = None
        self._parent_rgb = None
        self._cv2_hls = None
        self._cv2_bgr = None
        self._cv2_rgb = None

    def brightness_shift(self, chan, gain):
        self._hls_chan = chan
        self._hls_gain = gain
        self._cv2_hls = None
        return self.cv2_rgb

    @property
    def shape(self):
        return self.parent_bgr.shape

    @property
    def parent_bgr(self):
        if self._parent_bgr is None:
            self._parent_bgr = cv2.imread(self.filename)
        return self._parent_bgr

    @property
    def parent_hls(self):
        if self._parent_hls is None:
            self._parent_hls = self.to_hls(self.parent_bgr)
        return self._parent_hls

    @property
    def parent_rgb(self):
        if self._parent_rgb is None:
            self._parent_rgb = self.to_rgb(self.parent_bgr)
        return self._parent_rgb

    @property
    def cv2_hls(self):
        if self._cv2_hls is None:
            if self._hls_gain is None:
                self._cv2_hls = self.parent_hls
            else:
                self._cv2_hls = hls_shift(self.parent_hls, self._hls_chan, self._hls_gain)
        return self._cv2_hls

    @property
    def cv2_bgr(self):
        if self._cv2_bgr is None:
            self._cv2_bgr = self.to_bgr(self.cv2_hls)
        return self._cv2_bgr

    @property
    def cv2_rgb(self):
        if self._cv2_rgb is None:
            self._cv2_rgb = self.to_rgb(self.cv2_bgr)
        return self._cv2_rgb

    def to_hls(self, bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS_FULL)

    def to_bgr(self, hls):
        return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR_FULL)

    def to_rgb(self, bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class RandomHorizontalFlip:
    """Horizontally flip the given numpy array randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Image): Image to be flipped.

        Returns:
            Image: Randomly flipped image.
        """
        if np.random.random() < self.p:
            return cv2.flip(img, 1)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomTransformC4:
    """Rotate a n-D tensor by 90 degrees in the H x W plane.

    Args:
        with_identity (bool): whether or not to include 0 degrees as a probable rotation
    """

    def __init__(self, with_identity=True):
        self.with_identity = with_identity
        self.n90s = (0, 1, 2, 3) if self.with_identity else (1, 2, 3)

    def __call__(self, img):
        """
        Args:
            img (Image): Image to be rotated.

        Returns:
            Image: Randomly rotated image but in 90 degree increments.
        """
        k = random.choice(self.n90s)
        return torch.rot90(img, k, (1, 2))

    def __repr__(self):
        return self.__class__.__name__ + '(with_identity={})'.format(self.with_identity)


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


class ReduceLROnPlateau2(ReduceLROnPlateau):

    def __init__(self, *args, **kwargs):
        super(ReduceLROnPlateau2, self).__init__(*args, **kwargs)

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


class SubsetSampler(data.Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ImportanceSampler(data.Sampler):
    r"""Samples elements from [0,..,len(weights)-1] with given probabilities (weights).

    Arguments:
        num_records (int): Total number of samples in the dataset.
        num_samples (int): Number of samples to draw from the dataset.
        batch_size (int): Size of mini-batch.

    """

    def __init__(self, num_records, num_samples, batch_size):

        if not isinstance(num_records, _int_classes) or isinstance(num_records, bool) or num_records <= 0:
            raise ValueError('num_records should be a positive integral value, but got num_records={}'.format(num_records))
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError('num_samples should be a positive integral value, but got num_samples={}'.format(num_samples))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integral value, but got batch_size={}'.format(batch_size))
        if num_records < num_samples < batch_size:
            raise ValueError('num_samples must be less than num_records and greater than batch_size')
        if num_samples % batch_size != 0:
            raise ValueError(f'batch_size ({batch_size}) must divide num_samples ({num_samples}) evenly.')

        self.num_steps = 0
        self.num_epochs = 0
        self.num_records = num_records
        self.num_samples = num_samples
        self.num_batches = num_samples // batch_size
        self.batch_size = batch_size
        self.drop_last = True

        self.ages = np.zeros(num_records, dtype=int)
        self.visits = np.zeros(num_records, dtype=int)
        # self.losses = np.zeros(num_records) - np.log(0.5)  # dup or non-dup
        self.losses = np.ones(num_records)

        self.epoch_losses = np.ones(num_samples) * -1.0
        self._epoch_ages = None

        self.indices = np.random.choice(self.num_records, self.num_samples, replace=False)
        self.sampler = SubsetSampler(self.indices)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    @property
    def epoch_ages(self):
        if self._epoch_ages is None:
            # plus 1 since we're always lagging behind by 1 gradient step.
            x = np.arange(self.num_batches)[::-1] + 1
            self._epoch_ages = np.repeat(x, self.batch_size)
            assert len(self._epoch_ages) == self.num_samples
        return self._epoch_ages

    def update(self, batch_losses):
        idx = self.num_steps * self.batch_size
        self.epoch_losses[idx:idx + self.batch_size] = batch_losses[:, 0]
        self.num_steps += 1

    def on_epoch_end(self):
        """Use losses, visits and ages to update weights for samples"""

        assert np.min(self.epoch_losses) >= 0, np.min(self.epoch_losses)
        # age all records by the number of batches seen this epoch.
        self.ages += self.num_batches
        # only update the sampled records since their ages got reset.
        self.ages[self.indices] = self.epoch_ages
        # increment visits for samples by one.
        self.visits[self.indices] += 1
        # update losses
        self.losses[self.indices] = self.epoch_losses
        self.num_epochs += 1

        # normalize
        norm_ages = self.ages / np.sum(self.ages)
        # log_ages = np.log(self.ages)
        # norm_log_ages = log_ages / np.sum(log_ages)

        non_visits = self.num_epochs - self.visits
        norm_non_visits = non_visits / np.sum(non_visits)

        norm_losses = self.losses / np.sum(self.losses)
        weights = norm_ages + norm_non_visits + norm_losses
        # weights = log_ages * (np.sum(self.losses) / np.sum(log_ages)) + self.losses
        # norm_weights = weights / np.sum(weights)

        # ucb = self.losses + 2 * np.sqrt(np.log(self.ages + 1) / (self.visits + 1))
        # self.indices = np.argsort(ucb)[-self.num_samples:]
        # self.indices = np.random.choice(self.num_records, self.num_samples, replace=False, p=self.norm_weights)
        self.indices = np.argsort(weights)[::-1][:self.num_samples]
        np.random.shuffle(self.indices)

        self.sampler = SubsetSampler(self.indices)
        self.num_steps = 0
        self.epoch_losses *= -1.0

        # print(self.num_epochs)
        # print(self.ages)
        # print(non_visits)
        # print(self.losses)
        # mask = np.zeros((self.num_records,), dtype=int)
        # mask[self.indices] = 11
        # print(mask)
