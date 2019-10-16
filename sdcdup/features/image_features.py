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

from sdcdup.utils import idx2ijpair
from sdcdup.utils import get_project_root
from sdcdup.utils import rle_to_full_mask
from sdcdup.utils import get_hamming_distance
from sdcdup.utils import generate_pair_tag_lookup
from sdcdup.utils import overlap_tag_pairs
from sdcdup.utils import overlap_tag_maps
from sdcdup.utils import relative_diff
from sdcdup.utils import fuzzy_diff

project_root = get_project_root()
raw_data_dir = os.path.join(project_root, os.getenv('RAW_DATA_DIR'))
interim_data_dir = os.path.join(project_root, os.getenv('INTERIM_DATA_DIR'))
processed_data_dir = os.path.join(project_root, os.getenv('PROCESSED_DATA_DIR'))

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


class SDCImageContainer:

    def __init__(self,
                 cache_size=5000,
                 matches_params=('bmh', 0.9),
                 **kwargs):

        # This class assumes images are square and that height and width are divisible by tile_size.
        super().__init__(**kwargs)

        self.train_image_dir = os.path.join(raw_data_dir, 'train_768')
        self.sz = 256  # tile_size
        self.n_rows = 3
        self.n_cols = 3
        self.n_tiles = self.n_rows * self.n_cols
        self.tile_score_max = self.sz * self.sz * 3 * 255  # 3 color channels, uint8

        self.img_metrics_config = {
            'md5': {
                'len': 8,
                'dtype': f'<U8',
                'file': os.path.join(interim_data_dir, 'image_md5.pkl'),
                'shape': (self.n_tiles, ),
                'func': self.get_md5hash},
            'bmh': {
                'len': 32,
                'dtype': np.uint8,
                'file': os.path.join(interim_data_dir, 'image_bmh.pkl'),
                'shape': (self.n_tiles, 32),
                'func': self.get_bm0hash},
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
        self._sorted_hash_dict = None

        self.matches_metric = matches_params[0]  # 'bmh'
        self.matches_threshold = matches_params[1]  # 0.9 ~= 1 - ((5 + 20) / 256)
        matches_file = f'matches_{self.matches_metric}_{self.matches_threshold}.csv'
        self.matches_file = os.path.join(interim_data_dir, matches_file)
        self.matches_white = {
            'bmh': tuple(np.ones(32, dtype='uint8') * 255)
        }
        self.return_args_with_overlap_scores = False
        self.overlap_scores_config = {
            'bmh': {
                'dtype': np.float,
                'func': self.get_bmh_scores,
                'image_metric': 'bmh'},
            'cmh': {
                'dtype': np.float,
                'func': self.get_cmh_scores,
                'image_metric': 'cmh'},
            # 'gcm': {
            #     'dtype': np.float,
            #     'func': self.get_gcm_scores,
            #     'image_metric': 'gcm'},
            'enp': {
                'dtype': np.float,
                'func': self.get_enp_scores,
                'image_metric': 'enp'},
            'pix': {
                'dtype': np.int,
                'func': self.gen_pix_scores,
                'image_metric': None},
            'px0': {
                'dtype': np.int,
                'func': self.gen_px0_scores,
                'image_metric': None},
            'shp': {
                'dtype': np.int,
                'func': self.gen_shp_scores,
                'image_metric': 'shp'},
        }

    def get_md5hash(self, tile):
        return hashlib.md5(tile.tobytes()).hexdigest()[:8]

    def get_bm0hash(self, tile):
        return img_hash.blockMeanHash(tile, mode=0)[0]

    def get_cm0hash(self, tile):
        return img_hash.colorMomentHash(tile)[0]

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
        with ThreadPoolExecutor(max_workers=8) as executor:
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

    def get_bmh_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = []
        for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
            bmh1 = self.img_metrics['bmh'][img1_id][idx1]
            bmh2 = self.img_metrics['bmh'][img2_id][idx2]
            score = get_hamming_distance(bmh1, bmh2, normalize=True, as_score=True)
            scores.append(score)
        if self.return_args_with_overlap_scores:
            return (img1_id, img2_id, img1_overlap_tag, *scores)
        return scores

    def get_cmh_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = []
        for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
            cmh1 = self.img_metrics['cmh'][img1_id][idx1]
            cmh2 = self.img_metrics['cmh'][img2_id][idx2]
            score = np.exp(-np.linalg.norm(cmh1 - cmh2))
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
            enp1 = self.img_metrics['enp'][img1_id][idx1]
            enp2 = self.img_metrics['enp'][img2_id][idx2]
            score = np.exp(-np.linalg.norm(enp1 - enp2))
            # score = np.mean((enp1 + enp2) / 2.0)
            scores.append(score)
        if self.return_args_with_overlap_scores:
            return (img1_id, img2_id, img1_overlap_tag, *np.array(scores))
        return np.array(scores)

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
        scores = []
        for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
            shp1 = self.img_metrics['shp'][img1_id][idx1]
            shp2 = self.img_metrics['shp'][img2_id][idx2]
            score = shp1 + shp2
            scores.append(score)
        if self.return_args_with_overlap_scores:
            return (img1_id, img2_id, img1_overlap_tag, *np.array(scores))
        return np.array(scores)

    def find_valid_pairings_by_hash(self, hash_id, img_list):
        """
        This currently only works with the hashes stored in self.img_metrics['bmh']

        :param hash_id:
        :param img_list:
        :return:
        """

        hamming_lookup = {}
        for img_id in img_list:
            hamming_lookup[img_id] = [get_hamming_distance(bmh0, hash_id, normalize=True, as_score=True) for bmh0 in self.img_metrics['bmh'][img_id]]

        matches = []
        for img1_id in img_list:
            tiles1 = [idx for idx, bmhd in enumerate(hamming_lookup[img1_id]) if bmhd >= self.matches_threshold]
            for img2_id in img_list:
                if img2_id <= img1_id:
                    continue
                tiles2 = [idx for idx, bmhd in enumerate(hamming_lookup[img2_id]) if bmhd >= self.matches_threshold]

                # create a set of valid overlap_tags based on matching tiles between images.
                overlap_tags = set()
                for t1 in tiles1:
                    for t2 in tiles2:
                        overlap_tags.add(pair_tag_lookup.get((t1, t2)))

                for img1_overlap_tag in overlap_tags:
                    if (img1_id, img2_id, img1_overlap_tag) in matches:
                        continue
                    bmh_scores = self.get_bmh_scores(img1_id, img2_id, img1_overlap_tag)
                    if min(bmh_scores) < self.matches_threshold:
                        continue
                    matches.append((img1_id, img2_id, img1_overlap_tag))

        return matches

    @property
    def sorted_hash_dict(self):
        if self._sorted_hash_dict is None:
            img_ids = os.listdir(self.train_image_dir)
            # TODO: Use filter for all overlaps here?
            # img_ids = filter_duplicates(img_ids)

            hash_dict = defaultdict(set)
            for img_id in tqdm(img_ids):
                for h in self.img_metrics['bmh'][img_id]:
                    hash_dict[tuple(h)].add(img_id)

            hash_dict.pop(self.matches_white['bmh'])

            self._sorted_hash_dict = {}
            for key, dups in sorted(hash_dict.items(), key=lambda x: len(x[1]), reverse=True):
                if len(dups) > 1:
                    self._sorted_hash_dict[key] = dups

        return self._sorted_hash_dict

    def get_overlap_matches(self):

        if os.path.exists(self.matches_file):
            df = pd.read_csv(self.matches_file, dtype=str)
            overlap_matches = df.to_dict('split')['data']

        else:
            executor_args = [(key, sorted(val)) for key, val in self.sorted_hash_dict.items()]
            overlap_matches = []
            for hash_id, img_list in tqdm(executor_args):
                matches = self.find_valid_pairings_by_hash(hash_id, img_list)
                overlap_matches += matches
            overlap_matches = sorted(list(set(overlap_matches)))

            df = pd.DataFrame(overlap_matches)
            df.to_csv(self.matches_file, index=False)

        return overlap_matches

    def create_image_overlap_properties(self, score_type, overlap_matches):

        # bmh: blockMeanHash scores: Hamming distance between 2 blockMeanHash scores
        # cmh: colorMomentHash scores: L2 norm between 2 colorMomentHash scores
        # FIXME gcm: skimage greycomatrix scores: Relative difference between 2 greycop scores.
        # FIXME enp: Entropy scores: Exponential of the negative L2 norm between 2 entropy scores.
        # pix: Pixel scores: Fuzzy difference between 2 images pixelwise. Requires reading images so can be slow.
        # px0: Pixel scores: Hamming distance between 2 images pixelwise. Requires reading images so can be slow.
        # shp: Number of pixels that belong to ships:

        self.return_args_with_overlap_scores = True

        overlap_scores_file = os.path.join(interim_data_dir, f'overlap_{score_type}.pkl')

        print('score_type =', score_type)
        overlap_scores_list = []
        func = self.overlap_scores_config[score_type]['func']

        if score_type in ('pix', 'px0'):
            with ThreadPoolExecutor(max_workers=24) as executor:
                n_matches = len(overlap_matches)
                for overlap_match in tqdm(executor.map(func, *zip(*overlap_matches)), total=n_matches):
                    overlap_scores_list.append(overlap_match)
        else:
            for overlap_match in tqdm(sorted(overlap_matches)):
                overlap_scores_list.append(func(*overlap_match))

        df = pd.DataFrame(overlap_scores_list)
        df.to_pickle(overlap_scores_file)

        self.return_args_with_overlap_scores = False

    def load_image_overlap_properties(self, score_types=None):

        if score_types is None:
            score_types = ['bmh', 'cmh', 'enp', 'pix', 'px0']

        image_metrics = []
        for score_type in score_types:
            image_metric = self.overlap_scores_config[score_type].get('image_metric')
            if image_metric:
                image_metrics.append(image_metric)

        self.load_image_metrics(image_metrics)

        Overlap_Scores = namedtuple('overlap_scores', score_types)
        overlap_image_maps = defaultdict(dict)
        overlap_matches = self.get_overlap_matches()

        overlap_scores = {}
        for score_type in score_types:
            overlap_scores_file = os.path.join(interim_data_dir, f'overlap_{score_type}.pkl')

            if not os.path.exists(overlap_scores_file):
                self.create_image_overlap_properties(score_type, overlap_matches)

            df = pd.read_pickle(overlap_scores_file)
            overlap_tile_scores = defaultdict(dict)
            for img1_id, img2_id, img1_overlap_tag, *scores9 in df.to_dict('split')['data']:
                scores = np.array(scores9[:len(overlap_tag_maps[img1_overlap_tag])])
                overlap_tile_scores[(img1_id, img2_id)][img1_overlap_tag] = scores
            overlap_scores[score_type] = overlap_tile_scores

        for img1_id, img2_id, img1_overlap_tag in tqdm(sorted(overlap_matches)):
            scores_list = [overlap_scores[s][(img1_id, img2_id)][img1_overlap_tag] for s in score_types]
            overlap_image_maps[(img1_id, img2_id)][img1_overlap_tag] = Overlap_Scores(*scores_list)

        return overlap_image_maps


if __name__ == '__main__':

    t0 = time.time()
    score_types = ('bmh', 'cmh', 'enp', 'pix', 'px0', 'shp')
    sdcic = SDCImageContainer()
    overlap_image_maps = sdcic.load_image_overlap_properties(score_types=score_types)
    print(f'Done in {time.time() - t0}')
