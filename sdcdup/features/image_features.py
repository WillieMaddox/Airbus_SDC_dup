import os
import time
import hashlib
import operator
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from collections import namedtuple

from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from cv2 import img_hash
import torch
from torch.utils import data

from sdcdup.utils import idx2ijpair
from sdcdup.utils import get_project_root
from sdcdup.utils import rle_to_full_mask
from sdcdup.utils import get_hamming_distance
from sdcdup.utils import get_hamming_distance_array
from sdcdup.utils import generate_tag_pair_lookup
from sdcdup.utils import generate_pair_tag_lookup
from sdcdup.utils import overlap_tag_pairs
from sdcdup.utils import overlap_tag_maps
from sdcdup.utils import get_overlap_matches
from sdcdup.utils import relative_diff
from sdcdup.utils import fuzzy_diff
from sdcdup.data import EvalDataset as Dataset
from sdcdup.data import WrappedDataLoader
from sdcdup.models import load_checkpoint

project_root = get_project_root()
models_dir = os.path.join(project_root, 'models')
raw_data_dir = os.path.join(project_root, os.getenv('RAW_DATA_DIR'))
interim_data_dir = os.path.join(project_root, os.getenv('INTERIM_DATA_DIR'))
processed_data_dir = os.path.join(project_root, os.getenv('PROCESSED_DATA_DIR'))

tag_pair_lookup = generate_tag_pair_lookup()
pair_tag_lookup = generate_pair_tag_lookup()


def filter_duplicates(img_ids):
    df = pd.read_csv(os.path.join(processed_data_dir, 'dup_blacklist_6.csv'))
    blacklist = []
    for idx, row in df.iterrows():
        blacklist.append(row['ImageId1'])
    new_img_ids = [img_id for img_id in img_ids if img_id not in blacklist]
    # print(len(img_ids), len(blacklist), len(new_img_ids))
    return new_img_ids


def get_rles():
    df = pd.read_csv(os.path.join(raw_data_dir, 'train_ship_segmentations_v2.csv'))
    df = pd.merge(df, df.groupby('ImageId').size().reset_index(name='cts'))
    df['cts'] = df.apply(lambda c_row: c_row['cts'] if isinstance(c_row['EncodedPixels'], str) else 0, 1)
    df = df[df['cts'] >= 1]
    return {k: list(v) for k, v in df.groupby('ImageId')['EncodedPixels']}


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


def get_subhist3d(k, hist_rgb, r, g, b):
    rmin = max(0, r - k)
    gmin = max(0, g - k)
    bmin = max(0, b - k)
    rmax = min(256, r + k + 1)
    gmax = min(256, g + k + 1)
    bmax = min(256, b + k + 1)
    return hist_rgb[rmin:rmax, gmin:gmax, bmin:bmax]


class SDCImageContainer:

    def __init__(self, **kwargs):

        # This class assumes images are square and that height and width are divisible by tile_size.
        super().__init__(**kwargs)

        self.train_image_dir = os.path.join(raw_data_dir, 'train_768')
        self.sz = 256  # tile_size
        self.n_rows = 3
        self.n_cols = 3
        self.n_tiles = self.n_rows * self.n_cols
        self.tile_score_max = self.sz * self.sz * 3 * 255  # 3 color channels, uint8
        self.max_num_tile_pixels = self.sz * self.sz

        self.img_metrics_config = {
            'md5': {
                'len': 8,
                'dtype': f'<U8',
                'file': os.path.join(interim_data_dir, 'image_md5.pkl'),
                'shape': (self.n_tiles, ),
                'func': self.get_md5hash},
            'bmh32': {
                'len': 32,
                'dtype': np.uint8,
                'file': os.path.join(interim_data_dir, 'image_bmh32.pkl'),
                'shape': (self.n_tiles, 32),
                'func': self.get_bm0hash32},
            'bmh96': {
                'len': 96,
                'dtype': np.uint8,
                'file': os.path.join(interim_data_dir, 'image_bmh96.pkl'),
                'shape': (self.n_tiles, 96),
                'func': self.get_bm0hash96},
            'cmh': {
                'len': 42,
                'dtype': np.float,
                'file': os.path.join(interim_data_dir, 'image_cmh.pkl'),
                'shape': (self.n_tiles, 42),
                'func': self.get_cm0hash},
            'enp': {
                'len': 3,
                'dtype': np.float,
                'file': os.path.join(interim_data_dir, 'image_enp.pkl'),
                'shape': (self.n_tiles, 3),
                'func': self.get_entropy},
            'hst': {
                'len': 256,
                'dtype': np.uint16,
                'file': os.path.join(interim_data_dir, 'image_hst.pkl'),
                'shape': (self.n_tiles, 256),
                'func': self.get_histstats},
            'avg': {
                'len': 768,
                'dtype': np.uint8,
                'file': os.path.join(interim_data_dir, 'image_avg.pkl'),
                'shape': (self.n_tiles, 768),
                'func': self.get_avgpool},
            'sol': {
                'len': 3,
                'dtype': np.int,
                'file': os.path.join(interim_data_dir, 'image_sol.pkl'),
                'shape': (self.n_tiles, 3),
                'func': self.get_issolid},
            'shp': {
                'len': 1,
                'dtype': np.int,
                'file': os.path.join(interim_data_dir, 'image_shp.pkl'),
                'shape': (self.n_tiles, ),
                'func': self.get_shipcnt}
        }
        self.img_metrics = {}
        self.new_metric_ids = []

        # self.tile_greycop_len = 5
        # self.tile_greycop_dtype = np.float
        # self.tile_greycop_file = os.path.join(interim_data_dir, filename_greycop)

        self._rles = None
        self.matches = None

        self.return_args_with_overlap_scores = False
        self.overlap_scores_config = {
            'md5': {
                'dtype': np.bool_,
                'func': self.get_md5_scores,
                'image_metric': 'md5',
                'filename': 'overlap_md5.pkl'},
            'bmh32': {
                'dtype': np.float,
                'func': self.get_bmh32_scores,
                'image_metric': 'bmh32',
                'filename': 'overlap_bmh32.pkl'},
            'bmh96': {
                'dtype': np.float,
                'func': self.get_bmh96_scores,
                'image_metric': 'bmh96',
                'filename': 'overlap_bmh96.pkl'},
            'cmh': {
                'dtype': np.float,
                'func': self.get_cmh_scores,
                'image_metric': 'cmh',
                'filename': 'overlap_cmh.pkl'},
            # 'gcm': {
            #     'dtype': np.float,
            #     'func': self.get_gcm_scores,
            #     'image_metric': 'gcm'},
            'enp': {
                'dtype': np.float,
                'func': self.get_enp_scores,
                'image_metric': 'enp',
                'filename': 'overlap_enp.pkl'},
            'hst': {
                'dtype': np.uint16,
                'func': self.gen_hst_scores,
                'image_metric': 'hst',
                'filename': 'overlap_hst.pkl'},
            'avg': {
                'dtype': np.uint8,
                'func': self.gen_avg_scores,
                'image_metric': 'avg',
                'filename': 'overlap_avg.pkl'},
            'pix': {
                'dtype': np.int,
                'func': self.gen_pix_scores,
                'image_metric': None,
                'filename': 'overlap_pix.pkl'},
            'px0': {
                'dtype': np.int,
                'func': self.gen_px0_scores,
                'image_metric': None,
                'filename': 'overlap_px0.pkl'},
            'shp': {
                'dtype': np.int,
                'func': self.gen_shp_scores,
                'image_metric': 'shp',
                'filename': 'overlap_shp.pkl'},
            'dnn': {
                'dtype': np.float,
                'func': self.gen_dnn_scores,
                'image_metric': None,
                'model_basename': 'dup_model2',
                'model_id': '2019_1129_1702',
                'filename': 'overlap_dnn_2019_1129_1702.pkl'},
        }

    def get_md5hash(self, tile):
        return hashlib.md5(tile.tobytes()).hexdigest()[:8]

    def get_bm0hash32(self, tile):
        return img_hash.blockMeanHash(tile, mode=0)[0]

    def get_bm0hash96(self, tile):
        hash0 = img_hash.blockMeanHash(tile[..., 0], mode=0)
        hash1 = img_hash.blockMeanHash(tile[..., 1], mode=0)
        hash2 = img_hash.blockMeanHash(tile[..., 2], mode=0)
        return np.hstack([hash0, hash1, hash2])[0]

    def get_cm0hash(self, tile):
        return img_hash.colorMomentHash(tile)[0]

    def get_histstats(self, tile):
        hist_rgb = cv2.calcHist([tile], [0, 1, 2], None, [256] * 3, [0, 256, 0, 256, 0, 256]).astype(np.int)
        r, g, b = np.unravel_index(np.argmax(hist_rgb), hist_rgb.shape)
        old_neighbor_total = 0
        neighbor_counts = np.zeros((256,), dtype=np.uint16)
        for k in range(256):
            new_neighbor_total = np.sum(get_subhist3d(k, hist_rgb, r, g, b))
            neighbor_counts[k] = new_neighbor_total - old_neighbor_total
            if new_neighbor_total == self.max_num_tile_pixels:
                break
            old_neighbor_total = new_neighbor_total
        return neighbor_counts

    def get_avgpool(self, tile):
        M, N, C = tile.shape
        MK = M // 16
        NL = N // 16
        res = np.median(tile[:MK * 16, :NL * 16, :].reshape(MK, 16, NL, 16, C), axis=(1, 3))
        return res.reshape(768)

    def get_issolid(self, tile):
        issolid_flags = np.array([-1, -1, -1])
        for chan in range(3):
            pix = np.unique(tile[:, :, chan].flatten())
            if len(pix) == 1:
                issolid_flags[chan] = pix[0]
        return issolid_flags

    def get_entropy(self, tile):
        entropy_shannon = np.zeros(3)
        for chan in range(3):
            entropy_shannon[chan] = shannon_entropy(tile[:, :, chan])
        return entropy_shannon

    def get_shipcnt(self, tile):
        return np.sum(tile)

    def generate_image_metric(self, img, metric_config):
        metric_array = np.zeros(metric_config['shape'], dtype=metric_config['dtype'])
        for idx in range(self.n_tiles):
            tile = self.get_tile(img, idx)
            metric_array[idx] = metric_config['func'](tile)
        return metric_array

    def process_image(self, img_id):
        img_metric = {}
        img = self.get_img(img_id)
        for metric_id in self.new_metric_ids:
            metric_config = self.img_metrics_config[metric_id]
            img_metric[metric_id] = self.generate_image_metric(img, metric_config)
        return img_metric

    def load_metrics(self, filename):
        df = pd.read_pickle(filename)
        img_metrics = {key: value for key, value in df.to_dict('split')['data']}
        return img_metrics

    def dump_metrics(self, metrics, filename):
        records = [{'ImageId': key, 'TileData': value} for key, value in sorted(metrics.items())]
        df = pd.DataFrame().append(records)
        df.to_pickle(filename)

    def load_image_metrics(self, metric_ids=None):
        metric_ids = metric_ids or ['md5', 'sol']
        metric_ids = set(metric_ids) | {'md5', 'sol'}

        for m_id in metric_ids:
            if not os.path.exists(self.img_metrics_config[m_id]['file']):
                if m_id == 'shp':
                    self.create_label_metrics(('shp',))
                else:
                    self.new_metric_ids.append(m_id)

        if len(self.new_metric_ids) > 0:
            self.create_image_metrics()

        for m_id in metric_ids:
            if m_id not in self.img_metrics:
                self.img_metrics[m_id] = self.load_metrics(self.img_metrics_config[m_id]['file'])

    def create_image_metrics(self):

        img_ids = os.listdir(self.train_image_dir)
        img_metrics = {m_id: {} for m_id in self.new_metric_ids}
        with ThreadPoolExecutor() as executor:
            n_records = len(img_ids)
            for img_id, img_metric in tqdm(zip(img_ids, executor.map(self.process_image, img_ids)), total=n_records):
                for metric_id, metric_array in img_metric.items():
                    img_metrics[metric_id][img_id] = metric_array

        for m_id in self.new_metric_ids:
            self.dump_metrics(img_metrics[m_id], self.img_metrics_config[m_id]['file'])

    def create_label_metrics(self, metric_ids=('shp',)):
        """
        Slightly different approach here since we are reading everything from a single rle csv file,
        rather than multiple image files.
        Args:
            metric_ids:

        Returns:

        """
        img_ids = os.listdir(self.train_image_dir)
        lbl_metrics = {m_id: {} for m_id in metric_ids}
        for img_id in tqdm(sorted(img_ids)):
            metric_array = np.zeros(self.img_metrics_config['shp']['shape'], dtype=self.img_metrics_config['shp']['dtype'])
            if img_id in self.rles:
                img = rle_to_full_mask(self.rles[img_id])
                for idx in range(self.n_tiles):
                    tile = self.get_tile(img, idx)
                    metric_array[idx] = self.img_metrics_config['shp']['func'](tile)
            lbl_metrics['shp'][img_id] = metric_array

        for m_id in metric_ids:
            self.dump_metrics(lbl_metrics[m_id], self.img_metrics_config[m_id]['file'])

    def get_img(self, filename, path=None):
        path = self.train_image_dir if path is None else path
        return cv2.imread(os.path.join(path, filename))

    def _get_tile(self, img, i, j):
        return img[i * self.sz:(i + 1) * self.sz, j * self.sz:(j + 1) * self.sz, :]

    def get_tile(self, img, idx):
        i, j = idx2ijpair[idx]
        return self._get_tile(img, i, j)

    @property
    def rles(self):
        if self._rles is None:
            self._rles = get_rles()
        return self._rles

    def get_md5_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = np.zeros((9,), dtype=np.bool_)
        for ii, (idx1, idx2) in enumerate(zip(img1_overlap_map, img2_overlap_map)):
            m1 = self.img_metrics['md5'][img1_id][idx1]
            m2 = self.img_metrics['md5'][img2_id][idx2]
            scores[ii] = m1 == m2
        if self.return_args_with_overlap_scores:
            return (img1_id, img2_id, img1_overlap_tag, *scores)
        return scores

    def get_bmh32_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        m1 = self.img_metrics['bmh32'][img1_id][img1_overlap_map]
        m2 = self.img_metrics['bmh32'][img2_id][img2_overlap_map]
        scores = get_hamming_distance_array(m1, m2, normalize=True, as_score=True)

        if self.return_args_with_overlap_scores:
            return (img1_id, img2_id, img1_overlap_tag, *scores)
        return scores

    def get_bmh96_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        m1 = self.img_metrics['bmh96'][img1_id][img1_overlap_map]
        m2 = self.img_metrics['bmh96'][img2_id][img2_overlap_map]
        scores = get_hamming_distance_array(m1, m2, normalize=True, as_score=True)

        if self.return_args_with_overlap_scores:
            return (img1_id, img2_id, img1_overlap_tag, *scores)
        return scores

    def get_cmh_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = []
        for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
            m1 = self.img_metrics['cmh'][img1_id][idx1]
            m2 = self.img_metrics['cmh'][img2_id][idx2]
            score = np.exp(-np.linalg.norm(m1 - m2))
            scores.append(score)
        if self.return_args_with_overlap_scores:
            return (img1_id, img2_id, img1_overlap_tag, *scores)
        return scores

    # def get_gcm_scores(self, img1_id, img2_id, img1_overlap_tag):
    #     img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
    #     img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
    #     scores = []
    #     for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
    #         gcm1 = self.tile_greycop_grids[img1_id][idx1]
    #         gcm2 = self.tile_greycop_grids[img2_id][idx2]
    #         score = relative_diff(gcm1, gcm2)
    #         scores.append(score)
    #     return np.array(scores)

    def get_enp_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = []
        for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
            m1 = self.img_metrics['enp'][img1_id][idx1]
            m2 = self.img_metrics['enp'][img2_id][idx2]
            score = np.exp(-np.linalg.norm(m1 - m2))
            scores.append(score)
        if self.return_args_with_overlap_scores:
            return (img1_id, img2_id, img1_overlap_tag, *np.array(scores))
        return np.array(scores)

    def gen_hst_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = np.zeros((9,), dtype=int)
        for ii, (idx1, idx2) in enumerate(zip(img1_overlap_map, img2_overlap_map)):
            h1 = self.img_metrics['hst'][img1_id][idx1]
            h2 = self.img_metrics['hst'][img2_id][idx2]
            scores[ii] = fuzzy_diff(h1, h2)
        if self.return_args_with_overlap_scores:
            return (img1_id, img2_id, img1_overlap_tag, *scores)
        return scores

    def gen_avg_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = np.zeros((9,), dtype=int)
        for ii, (idx1, idx2) in enumerate(zip(img1_overlap_map, img2_overlap_map)):
            a1 = self.img_metrics['avg'][img1_id][idx1]
            a2 = self.img_metrics['avg'][img2_id][idx2]
            scores[ii] = fuzzy_diff(a1, a2)
        if self.return_args_with_overlap_scores:
            return (img1_id, img2_id, img1_overlap_tag, *scores)
        return scores

    def gen_pix_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        img1 = self.get_img(img1_id)
        img2 = self.get_img(img2_id)
        scores = np.zeros((9,), dtype=int)
        for ii, (idx1, idx2) in enumerate(zip(img1_overlap_map, img2_overlap_map)):
            tile1 = self.get_tile(img1, idx1)
            tile2 = self.get_tile(img2, idx2)
            scores[ii] = fuzzy_diff(tile1, tile2)
        if self.return_args_with_overlap_scores:
            return (img1_id, img2_id, img1_overlap_tag, *scores)
        return scores

    def gen_px0_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        img1 = self.get_img(img1_id)
        img2 = self.get_img(img2_id)
        scores = np.zeros((9,), dtype=int)
        for ii, (idx1, idx2) in enumerate(zip(img1_overlap_map, img2_overlap_map)):
            tile1 = self.get_tile(img1, idx1)
            tile2 = self.get_tile(img2, idx2)
            scores[ii] = np.sum(tile1 != tile2)
        if self.return_args_with_overlap_scores:
            return (img1_id, img2_id, img1_overlap_tag, *scores)
        return scores

    def gen_shp_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = np.zeros((9,), dtype=int)
        for ii, (idx1, idx2) in enumerate(zip(img1_overlap_map, img2_overlap_map)):
            m1 = self.img_metrics['shp'][img1_id][idx1]
            m2 = self.img_metrics['shp'][img2_id][idx2]
            scores[ii] = m1 + m2
        if self.return_args_with_overlap_scores:
            return (img1_id, img2_id, img1_overlap_tag, *scores)
        return scores

    def create_image_overlap_properties(self, score_type, overlap_matches):

        # bmh: blockMeanHash scores: Hamming distance between 2 blockMeanHash scores
        # cmh: colorMomentHash scores: L2 norm between 2 colorMomentHash scores
        # FIXME gcm: skimage greycomatrix scores: Relative difference between 2 greycop scores.
        # FIXME enp: Entropy scores: Exponential of the negative L2 norm between 2 entropy scores.
        # pix: Pixel scores: Fuzzy difference between 2 images pixelwise. Requires reading images so can be slow.
        # px0: Pixel scores: Hamming distance between 2 images pixelwise. Requires reading images so can be slow.
        # shp: Number of pixels that belong to ships:

        self.return_args_with_overlap_scores = True

        overlap_scores_list = []
        func = self.overlap_scores_config[score_type]['func']

        if score_type in ('pix', 'px0'):
            with ThreadPoolExecutor() as executor:
                n_matches = len(overlap_matches)
                for overlap_match in tqdm(executor.map(func, *zip(*overlap_matches)), total=n_matches):
                    overlap_scores_list.append(overlap_match)
        elif score_type in ('dnn',):
            overlap_scores_list = func(overlap_matches)
        else:
            for overlap_match in tqdm(sorted(overlap_matches)):
                overlap_scores_list.append(func(*overlap_match))

        self.return_args_with_overlap_scores = False

        return overlap_scores_list

    def load_image_overlap_properties(self, matches_files, score_types='default'):

        if score_types is None:
            score_types = []

        if score_types == 'default':
            score_types = ['bmh96', 'cmh', 'enp', 'pix', 'px0']

        image_metrics = []
        for score_type in score_types:
            image_metric = self.overlap_scores_config[score_type].get('image_metric')
            if image_metric:
                image_metrics.append(image_metric)
        self.load_image_metrics(image_metrics)

        if self.matches is None:
            self.matches = get_overlap_matches(matches_files)

        overlap_scores = {}
        for score_type in score_types:

            print('score_type =', score_type)

            overlap_scores_filename = self.overlap_scores_config[score_type].get('filename')
            overlap_scores_file = os.path.join(interim_data_dir, overlap_scores_filename)

            if os.path.exists(overlap_scores_file):
                df = pd.read_pickle(overlap_scores_file)
                overlap_scores_list = [tuple(record) for record in df.to_dict('split')['data']]

                overlap_tile_keys = set()
                for img1_id, img2_id, img1_overlap_tag, *_ in tqdm(overlap_scores_list):
                    overlap_tile_keys.add((img1_id, img2_id, img1_overlap_tag))

                new_matches = []
                for img1_id, img2_id, img1_overlap_tag in tqdm(self.matches):
                    if (img1_id, img2_id, img1_overlap_tag) not in overlap_tile_keys:
                        new_matches.append((img1_id, img2_id, img1_overlap_tag))

                if len(new_matches) > 0:
                    overlap_scores_list += self.create_image_overlap_properties(score_type, new_matches)
                    df = pd.DataFrame(overlap_scores_list)
                    df.to_pickle(overlap_scores_file)
            else:
                overlap_scores_list = self.create_image_overlap_properties(score_type, self.matches)
                df = pd.DataFrame(overlap_scores_list)
                df.to_pickle(overlap_scores_file)

            overlap_tile_scores = {}
            for img1_id, img2_id, img1_overlap_tag, *scores9 in tqdm(overlap_scores_list):
                scores = np.array(scores9[:len(overlap_tag_maps[img1_overlap_tag])])
                overlap_tile_scores[(img1_id, img2_id, img1_overlap_tag)] = scores

            overlap_scores[score_type] = overlap_tile_scores

        if len(score_types) == 0:
            return {}

        print('overlap scores')

        Overlap_Scores = namedtuple('overlap_scores', score_types)
        overlap_image_maps = defaultdict(defaultdict)
        for img1_id, img2_id, img1_overlap_tag in tqdm(sorted(self.matches)):
            scores_list = [overlap_scores[s][(img1_id, img2_id, img1_overlap_tag)] for s in score_types]
            overlap_image_maps[(img1_id, img2_id)][img1_overlap_tag] = Overlap_Scores(*scores_list)

        return overlap_image_maps

    def gen_dnn_scores(self, overlap_matches):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def preprocess(x):
            return x.view(-1, 6, 256, 256).to(device)

        TilePairs = namedtuple('TilePairs', 'img1_id img2_id img1_overlap_tag overlap_idx idx1 idx2')

        tile_pairs = []
        for img1_id, img2_id, img1_overlap_tag in tqdm(overlap_matches):
            for overlap_idx, (idx1, idx2) in enumerate(tag_pair_lookup[img1_overlap_tag]):
                tile_pairs.append(TilePairs(img1_id, img2_id, img1_overlap_tag, overlap_idx, idx1, idx2))

        test_ds = Dataset(tile_pairs)
        test_dl = data.DataLoader(test_ds, batch_size=256, num_workers=22)
        test_dl = WrappedDataLoader(test_dl, preprocess)

        dnn_model_id = self.overlap_scores_config['dnn'].get('model_id')
        dnn_model_basename = self.overlap_scores_config['dnn'].get('model_basename')
        dnn_model_file = os.path.join(models_dir, f"{dnn_model_basename}.{dnn_model_id}.best.pth")

        model = load_checkpoint(dnn_model_file, dnn_model_basename)
        model.to(device)

        model.eval()
        with torch.no_grad():
            yprobs0 = [model(xb) for xb in tqdm(test_dl)]
            yprobs = np.vstack([l.cpu() for l in yprobs0]).reshape(-1)

        overlap_dnn_tile_scores = {}
        for tp, yprob in tqdm(zip(tile_pairs, yprobs)):
            if (tp.img1_id, tp.img2_id, tp.img1_overlap_tag) not in overlap_dnn_tile_scores:
                n_overlapping_tiles = len(tag_pair_lookup[tp.img1_overlap_tag])
                overlap_dnn_tile_scores[(tp.img1_id, tp.img2_id, tp.img1_overlap_tag)] = np.zeros(n_overlapping_tiles)
            overlap_dnn_tile_scores[(tp.img1_id, tp.img2_id, tp.img1_overlap_tag)][tp.overlap_idx] = yprob

        overlap_scores_list = []
        for (img1_id, img2_id, img1_overlap_tag), scores in tqdm(overlap_dnn_tile_scores.items()):
            overlap_scores_list.append((img1_id, img2_id, img1_overlap_tag, *scores))

        return overlap_scores_list


if __name__ == '__main__':

    t0 = time.time()
    score_types = ('md5', 'bmh96', 'cmh', 'enp', 'avg', 'hst', 'pix', 'px0', 'shp', 'dnn')
    sdcic = SDCImageContainer()
    matches_files = ['matches_bmh96_0.9.csv', 'matches_bmh32_0.9.csv']
    overlap_image_maps = sdcic.load_image_overlap_properties(matches_files, score_types=score_types)
    print(f'Done in {time.time() - t0}')
