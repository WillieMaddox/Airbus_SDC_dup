import os
import hashlib
import operator
from collections import defaultdict
from collections import namedtuple

from cachetools import LRUCache
from cachetools import cachedmethod
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


def get_rles(ship_file):
    df = pd.read_csv(ship_file)
    df = pd.merge(df, df.groupby('ImageId').size().reset_index(name='cts'))
    df['cts'] = df.apply(lambda c_row: c_row['cts'] if isinstance(c_row['EncodedPixels'], str) else 0, 1)
    df = df[df['cts'] >= 1]
    return {k: list(v) for k, v in df.groupby('ImageId')['EncodedPixels']}


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


class SDCImageContainer:

    def __init__(self,
                 cache_size=10000,
                 filename_md5hash='image_md5hash_grids.pkl',
                 filename_bm0hash='image_bm0hash_grids.pkl',
                 filename_cm0hash='image_cm0hash_grids.pkl',
                 filename_greycop='image_greycop_grids.pkl',
                 filename_entropy='image_entropy_grids.pkl',
                 filename_issolid='image_issolid_grids.pkl',
                 filename_shipcnt='image_shipcnt_grids.pkl',
                 **kwargs):

        # This class assumes images are square and height and width are divisible by tile_size.
        super().__init__(**kwargs)

        self.train_image_dir = os.path.join(raw_data_dir, 'train_768')
        self.rle_label_file = os.path.join(raw_data_dir, 'train_ship_segmentations_v2.csv')
        self.sz = 256  # tile_size
        self.n_rows = 3
        self.n_cols = 3
        self.n_tiles = self.n_rows * self.n_cols
        self.tile_score_max = self.sz * self.sz * 3 * 255  # 3 color channels, uint8
        self.tile_slice = slice(8, -8)
        self.tile_md5hash_len = 8
        self.tile_md5hash_dtype = f'<U{self.tile_md5hash_len}'
        self.tile_md5hash_grids = {}
        self.tile_md5hash_file = os.path.join(interim_data_dir, filename_md5hash)
        self.tile_bm0hash_len = 32
        self.tile_bm0hash_dtype = np.uint8
        self.tile_bm0hash_grids = {}
        self.tile_bm0hash_file = os.path.join(interim_data_dir, filename_bm0hash)
        self.tile_cm0hash_len = 42
        self.tile_cm0hash_dtype = np.float
        self.tile_cm0hash_grids = {}
        self.tile_cm0hash_file = os.path.join(interim_data_dir, filename_cm0hash)
        self.tile_greycop_len = 5
        self.tile_greycop_dtype = np.float
        self.tile_greycop_grids = {}
        self.tile_greycop_file = os.path.join(interim_data_dir, filename_greycop)
        self.tile_entropy_grids = {}
        self.tile_entropy_file = os.path.join(interim_data_dir, filename_entropy)
        self.tile_issolid_grids = {}
        self.tile_issolid_file = os.path.join(interim_data_dir, filename_issolid)
        self.tile_shipcnt_grids = {}
        self.tile_shipcnt_file = os.path.join(interim_data_dir, filename_shipcnt)
        self.matches = []
        self.matches_metric = 'bmh'
        self.matches_threshold = 0.90234375  # 1 - ((5 + 20) / 256)
        self.cache = LRUCache(maxsize=cache_size)
        self.score_funcs = {
            'bmh': self.get_bmh_scores,
            'cmh': self.get_cmh_scores,
            'gcm': self.get_gcm_scores,
            'pyr': self.get_pyr_scores,
            'enp': self.get_enp_scores,
            'pix': self.gen_pix_scores,
            'px0': self.gen_px0_scores,
            'shp': self.gen_shp_scores,
        }

    def preprocess_image_properties(self):

        img_md5hash_grids = {}
        if os.path.exists(self.tile_md5hash_file):
            df = pd.read_pickle(self.tile_md5hash_file)
            img_md5hash_grids = {key: val for key, val in df.to_dict('split')['data']}

        img_bm0hash_grids = {}
        if os.path.exists(self.tile_bm0hash_file):
            df = pd.read_pickle(self.tile_bm0hash_file)
            img_bm0hash_grids = {key: val for key, val in df.to_dict('split')['data']}

        img_cm0hash_grids = {}
        if os.path.exists(self.tile_cm0hash_file):
            df = pd.read_pickle(self.tile_cm0hash_file)
            img_cm0hash_grids = {key: val for key, val in df.to_dict('split')['data']}

        img_greycop_grids = {}
        if os.path.exists(self.tile_greycop_file):
            df = pd.read_pickle(self.tile_greycop_file)
            img_greycop_grids = {key: val for key, val in df.to_dict('split')['data']}

        img_entropy_grids = {}
        if os.path.exists(self.tile_entropy_file):
            df = pd.read_pickle(self.tile_entropy_file)
            img_entropy_grids = {key: val for key, val in df.to_dict('split')['data']}

        img_issolid_grids = {}
        if os.path.exists(self.tile_issolid_file):
            df = pd.read_pickle(self.tile_issolid_file)
            img_issolid_grids = {key: val for key, val in df.to_dict('split')['data']}

        mm = 0
        hh = 0
        cc = 0
        gg = 0
        ee = 0
        ss = 0

        md5hash_records = []
        bm0hash_records = []
        cm0hash_records = []
        greycop_records = []
        entropy_records = []
        issolid_records = []

        img_ids = os.listdir(self.train_image_dir)
        for img_id in tqdm(sorted(img_ids)):

            img = None

            tile_md5hash_grid = img_md5hash_grids.get(img_id)
            if tile_md5hash_grid is None:
                mm += 1
                img = self.get_img(img_id)
                tile_md5hash_grid = np.zeros(self.n_tiles, dtype=self.tile_md5hash_dtype)
                for idx in range(self.n_tiles):
                    tile = self.get_tile(img, idx)
                    tile_md5hash_grid[idx] = hashlib.md5(tile.tobytes()).hexdigest()[:self.tile_md5hash_len]

            md5hash_records.append({'ImageId': img_id, 'TileData': tile_md5hash_grid})  # str
            self.tile_md5hash_grids[img_id] = tile_md5hash_grid

            tile_bm0hash_grid = img_bm0hash_grids.get(img_id)
            if tile_bm0hash_grid is None:
                hh += 1
                img = self.get_img(img_id) if img is None else img
                tile_bm0hash_grid = np.zeros((self.n_tiles, self.tile_bm0hash_len), dtype=self.tile_bm0hash_dtype)
                for idx in range(self.n_tiles):
                    tile = self.get_tile(img, idx)
                    tile_bm0hash_grid[idx] = img_hash.blockMeanHash(tile, mode=0)[0]

            bm0hash_records.append({'ImageId': img_id, 'TileData': tile_bm0hash_grid})  # int
            self.tile_bm0hash_grids[img_id] = tile_bm0hash_grid

            tile_cm0hash_grid = img_cm0hash_grids.get(img_id)
            if tile_cm0hash_grid is None:
                cc += 1
                img = self.get_img(img_id) if img is None else img
                tile_cm0hash_grid = np.zeros((self.n_tiles, self.tile_cm0hash_len), dtype=self.tile_cm0hash_dtype)
                for idx in range(self.n_tiles):
                    tile = self.get_tile(img, idx)
                    tile_cm0hash_grid[idx] = img_hash.colorMomentHash(tile)[0]

            cm0hash_records.append({'ImageId': img_id, 'TileData': tile_cm0hash_grid})  # float
            self.tile_cm0hash_grids[img_id] = tile_cm0hash_grid

            tile_greycop_grid = img_greycop_grids.get(img_id)
            if tile_greycop_grid is None:
                gg += 1
                img = self.get_img(img_id) if img is None else img
                tile_greycop_grid = np.zeros((self.n_tiles, self.tile_greycop_len), dtype=self.tile_greycop_dtype)
                for idx in range(self.n_tiles):
                    tile = self.get_tile(img, idx)
                    tile_greycop_grid[idx] = gen_greycop_hash(tile, self.tile_greycop_len)

            greycop_records.append({'ImageId': img_id, 'TileData': tile_greycop_grid})  # float
            self.tile_greycop_grids[img_id] = tile_greycop_grid

            tile_entropy_grid = img_entropy_grids.get(img_id)
            if tile_entropy_grid is None:
                ee += 1
                img = self.get_img(img_id) if img is None else img
                tile_entropy_grid = np.zeros((self.n_tiles, 3), dtype=np.float)
                for idx in range(self.n_tiles):
                    tile = self.get_tile(img, idx)
                    tile_entropy_grid[idx] = gen_entropy(tile)

            entropy_records.append({'ImageId': img_id, 'TileData': tile_entropy_grid})  # float
            self.tile_entropy_grids[img_id] = tile_entropy_grid

            tile_issolid_grid = img_issolid_grids.get(img_id)
            if tile_issolid_grid is None:
                ss += 1
                img = self.get_img(img_id) if img is None else img
                tile_issolid_grid = np.zeros((self.n_tiles, 3), dtype=np.int)
                for idx in range(self.n_tiles):
                    tile = self.get_tile(img, idx)
                    tile_issolid_grid[idx] = get_issolid_flags(tile)

            issolid_records.append({'ImageId': img_id, 'TileData': tile_issolid_grid})  # int
            self.tile_issolid_grids[img_id] = tile_issolid_grid

            if mm >= 5000:
                df = pd.DataFrame().append(md5hash_records)
                df.to_pickle(self.tile_md5hash_file)
                mm = 0

            if hh >= 5000:
                df = pd.DataFrame().append(bm0hash_records)
                df.to_pickle(self.tile_bm0hash_file)
                hh = 0

            if cc >= 5000:
                df = pd.DataFrame().append(cm0hash_records)
                df.to_pickle(self.tile_cm0hash_file)
                cc = 0

            if gg >= 5000:
                df = pd.DataFrame().append(greycop_records)
                df.to_pickle(self.tile_greycop_file)
                gg = 0

            if ee >= 5000:
                df = pd.DataFrame().append(entropy_records)
                df.to_pickle(self.tile_entropy_file)
                ee = 0

            if ss >= 5000:
                df = pd.DataFrame().append(issolid_records)
                df.to_pickle(self.tile_issolid_file)
                ss = 0

        if mm > 0:
            df = pd.DataFrame().append(md5hash_records)
            df.to_pickle(self.tile_md5hash_file)

        if hh > 0:
            df = pd.DataFrame().append(bm0hash_records)
            df.to_pickle(self.tile_bm0hash_file)

        if cc > 0:
            df = pd.DataFrame().append(cm0hash_records)
            df.to_pickle(self.tile_cm0hash_file)

        if gg > 0:
            df = pd.DataFrame().append(greycop_records)
            df.to_pickle(self.tile_greycop_file)

        if ee > 0:
            df = pd.DataFrame().append(entropy_records)
            df.to_pickle(self.tile_entropy_file)

        if ss > 0:
            df = pd.DataFrame().append(issolid_records)
            df.to_pickle(self.tile_issolid_file)

    def preprocess_label_properties(self):

        img_shipcnt_grids = {}
        if os.path.exists(self.tile_shipcnt_file):
            df = pd.read_pickle(self.tile_shipcnt_file)
            img_shipcnt_grids = {key: val for key, val in df.to_dict('split')['data']}

        pp = 0

        rles = None

        shipcnt_records = []

        img_ids = os.listdir(self.train_image_dir)
        for img_id in tqdm(sorted(img_ids)):
            tile_shipcnt_grid = img_shipcnt_grids.get(img_id)
            if tile_shipcnt_grid is None:
                pp += 1
                rles = rles or get_rles(self.rle_label_file)
                tile_shipcnt_grid = np.zeros(self.n_tiles, dtype=np.int)
                if img_id in rles:
                    img = rle_to_full_mask(rles[img_id])
                    for idx in range(self.n_tiles):
                        tile = self.get_tile(img, idx)
                        tile_shipcnt_grid[idx] = np.sum(tile)

            shipcnt_records.append({'ImageId': img_id, 'TileData': tile_shipcnt_grid})  # int
            self.tile_shipcnt_grids[img_id] = tile_shipcnt_grid

            if pp >= 5000:
                df = pd.DataFrame().append(shipcnt_records)
                df.to_pickle(self.tile_shipcnt_file)
                pp = 0

        if pp > 0:
            df = pd.DataFrame().append(shipcnt_records)
            df.to_pickle(self.tile_shipcnt_file)

    @cachedmethod(operator.attrgetter('cache'))
    def get_img(self, filename, path=None):
        path = self.train_image_dir if path is None else path
        return cv2.imread(os.path.join(path, filename))

    def _get_tile(self, img, i, j):
        return img[i * self.sz:(i + 1) * self.sz, j * self.sz:(j + 1) * self.sz, :]

    def get_tile(self, img, idx):
        i, j = idx2ijpair[idx]
        return self._get_tile(img, i, j)

    def get_bmh_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = []
        for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
            bmh1 = self.tile_bm0hash_grids[img1_id][idx1]
            bmh2 = self.tile_bm0hash_grids[img2_id][idx2]
            score = get_hamming_distance(bmh1, bmh2, normalize=True, as_score=True)
            scores.append(score)
        return scores

    def get_cmh_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = []
        for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
            cmh1 = self.tile_cm0hash_grids[img1_id][idx1]
            cmh2 = self.tile_cm0hash_grids[img2_id][idx2]
            score = np.exp(-np.linalg.norm(cmh1 - cmh2))
            scores.append(score)
        return scores

    def get_gcm_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = []
        for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
            gcm1 = self.tile_greycop_grids[img1_id][idx1]
            gcm2 = self.tile_greycop_grids[img2_id][idx2]
            score = relative_diff(gcm1, gcm2)
            scores.append(score)
        return np.array(scores)

    def get_pyr_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = []
        for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
            pyr1 = self.tile_pyramid_grids[img1_id][idx1]
            pyr2 = self.tile_pyramid_grids[img2_id][idx2]
            score = np.exp(-np.linalg.norm(pyr1 - pyr2))
            # score = max(0.0, 1.0 - np.mean((pyr1 + pyr2) / 2.0))
            scores.append(score)
        return np.array(scores)

    def get_enp_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = []
        for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
            enp1 = self.tile_entropy_grids[img1_id][idx1]
            enp2 = self.tile_entropy_grids[img2_id][idx2]
            score = np.exp(-np.linalg.norm(enp1 - enp2))
            # score = np.mean((enp1 + enp2) / 2.0)
            scores.append(score)
        return np.array(scores)

    def gen_pix_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        img1 = self.get_img(img1_id)
        img2 = self.get_img(img2_id)
        scores = []
        for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
            tile1 = self.get_tile(img1, idx1)
            tile2 = self.get_tile(img2, idx2)
            score = fuzzy_diff(tile1, tile2)
            scores.append(score)
        return np.array(scores)

    def gen_px0_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        img1 = self.get_img(img1_id)
        img2 = self.get_img(img2_id)
        scores = []
        for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
            tile1 = self.get_tile(img1, idx1)
            tile2 = self.get_tile(img2, idx2)
            score = np.sum(tile1 != tile2)
            scores.append(score)
        return np.array(scores)

    def gen_shp_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = []
        for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
            shp1 = self.tile_shipcnt_grids[img1_id][idx1]
            shp2 = self.tile_shipcnt_grids[img2_id][idx2]
            score = shp1 + shp2
            scores.append(score)
        return np.array(scores)

    def find_valid_pairings_by_hash(self, hash_id, sorted_hash_dict, level_overlap_tags):

        img_list = list(sorted(sorted_hash_dict[hash_id]))

        hamming_distance_lookup = {}
        for img_id in img_list:
            hamming_list = [get_hamming_distance(bmh0, hash_id) for bmh0 in self.tile_bm0hash_grids[img_id]]
            hamming_distance_lookup[img_id] = np.array(hamming_list)

        for i, img1_id in enumerate(img_list):
            tiles1 = [idx for idx, bmhd in enumerate(hamming_distance_lookup[img1_id]) if bmhd <= 5]
            for j, img2_id in enumerate(img_list):
                if j <= i:
                    continue
                tiles2 = [idx for idx, bmhd in enumerate(hamming_distance_lookup[img2_id]) if bmhd <= 5]

                # create a set of valid overlap_tags based on matching tiles between images.
                overlap_tags = set()
                for t1 in tiles1:
                    for t2 in tiles2:
                        overlap_tags.add(pair_tag_lookup.get((t1, t2)))

                overlap_tags.intersection_update(level_overlap_tags)

                if len(overlap_tags) == 0:
                    continue

                assert img1_id < img2_id
                for img1_overlap_tag in overlap_tags:
                    if (img1_id, img2_id, img1_overlap_tag) in self.matches:
                        continue
                    bmh_scores = self.get_bmh_scores(img1_id, img2_id, img1_overlap_tag)
                    if min(bmh_scores) < self.matches_threshold:
                        continue
                    self.matches.append((img1_id, img2_id, img1_overlap_tag))

        return

    def create_overlap_matches_filename(self, n_matching_tiles):
        self.matches = []  # empty the matches container so we don't mix unequal tiles.
        file_tag = f'{n_matching_tiles}_{self.matches_metric}_{self.matches_threshold}'
        return os.path.join(interim_data_dir, f'overlap_matches_{file_tag}.csv')


class SDCImage:

    def __init__(self, img_id, train_image_dir,
                 tile_size=256,
                 tile_dups=None,
                 tile_slice=None,
                 tile_bm0hash_len=None,
                 tile_bm0hash_grid=None,
                 tile_bm0hash_dtype=None,
                 tile_entropy_grid=None):

        # This class assumes the image is square and can be divided perfectly by the tile_size.
        self.img_id = img_id
        self.filename = os.path.join(train_image_dir, img_id)
        self.sz = tile_size
        self.n_rows = 3
        self.n_cols = 3
        self.n_tiles = self.n_rows * self.n_cols
        self.tile_slice = tile_slice
        self.tile_bm0hash_len = tile_bm0hash_len
        self.tile_bm0hash_dtype = tile_bm0hash_dtype
        self._tile_bm0hash_grid = tile_bm0hash_grid
        self._tile_entropy_grid = tile_entropy_grid
        self._tile_dups = tile_dups
        self.overlap_image_maps = {tag: defaultdict(str) for tag in overlap_tag_maps}
        self.overlap_image_names = {tag: '' for tag in overlap_tag_maps}
        self.overlap_image_scores = {tag: 0.0 for tag in overlap_tag_maps}
        self.overlap_tile_scores = np.zeros((3, 3, 3, 3), dtype=np.float64)

    def get_img(self):
        return cv2.imread(self.filename)

    def get_tile(self, img, idx):
        i, j = idx2ijpair[idx]
        return img[i * self.sz:(i + 1) * self.sz, j * self.sz:(j + 1) * self.sz, :]

    @property
    def tile_bm0hash_grid(self):
        if self._tile_bm0hash_grid is None:
            self._tile_bm0hash_grid = np.zeros((self.n_tiles, self.tile_bm0hash_len), dtype=self.tile_bm0hash_dtype)
            img = self.get_img()
            for idx in range(self.n_tiles):
                tile = self.get_tile(img, idx)
                # tile = np.delete(tile, self.tile_slice, axis=0)
                # tile = np.delete(tile, self.tile_slice, axis=1)
                self._tile_bm0hash_grid[idx] = img_hash.blockMeanHash(tile, mode=0)
        return self._tile_bm0hash_grid

    @property
    def tile_entropy_grid(self):
        if self._tile_entropy_grid is None:
            self._tile_entropy_grid = np.zeros((self.n_tiles, 2), dtype=np.float)  # (idx, chan)
            img = self.get_img()
            for idx in range(self.n_tiles):
                tile = self.get_tile(img, idx)
                self._tile_entropy_grid[idx] = gen_entropy(tile)
        return self._tile_entropy_grid

    def update_overlap_map(self, other_sdc, tile_scores, tag):

        if self.overlap_image_names[tag] not in ('', other_sdc.img_id):  # sanity check
            print(tag, self.img_id, self.overlap_image_names[tag], other_sdc.img_id)

        self.overlap_image_names[tag] = other_sdc.img_id

        for (idx1, idx2, s) in zip(overlap_tag_maps[tag], overlap_tag_maps[overlap_tag_pairs[tag]], tile_scores):
            i, j = idx2ijpair[idx1]
            k, l = idx2ijpair[idx2]
            self.overlap_tile_scores[i, j, k, l] = s

        self.overlap_image_maps[tag][other_sdc.img_id] = {'tile_scores': tile_scores}


def load_image_overlap_properties(n_matching_tiles_list, score_types=None):

    sdcic = SDCImageContainer()
    sdcic.preprocess_image_properties()
    sdcic.preprocess_label_properties()

    if score_types is None:
        score_types = ['bmh', 'cmh', 'enp', 'pix', 'px0', 'shp']

    Overlap_Scores = namedtuple('overlap_scores', score_types)

    overlap_image_maps = {}
    for n_matching_tiles in n_matching_tiles_list:

        overlap_matches = get_overlap_matches(sdcic, n_matching_tiles)
        overlap_scores = {}
        for score_type in score_types:
            overlap_tile_scores_filename = f'overlap_{score_type}_tile_scores_{n_matching_tiles}.pkl'
            df = pd.read_pickle(os.path.join(interim_data_dir, overlap_tile_scores_filename))
            overlap_tile_scores = {}
            for img1_id, img2_id, img1_overlap_tag, *scores in df.to_dict('split')['data']:
                if (img1_id, img2_id) not in overlap_tile_scores:
                    overlap_tile_scores[(img1_id, img2_id)] = {}
                overlap_tile_scores[(img1_id, img2_id)][img1_overlap_tag] = np.array(scores)
            overlap_scores[score_type] = overlap_tile_scores

        for img1_id, img2_id, img1_overlap_tag in tqdm(sorted(overlap_matches)):

            scores_list = []
            for score_type in score_types:
                scores_list.append(overlap_scores[score_type][(img1_id, img2_id)][img1_overlap_tag])

            if (img1_id, img2_id) not in overlap_image_maps:
                overlap_image_maps[(img1_id, img2_id)] = {}
            overlap_image_maps[(img1_id, img2_id)][img1_overlap_tag] = Overlap_Scores(*scores_list)

    return overlap_image_maps


def get_overlap_matches(sdcic, n_matching_tiles):

    overlap_matches_file = sdcic.create_overlap_matches_filename(n_matching_tiles)

    if os.path.exists(overlap_matches_file):
        df = pd.read_csv(overlap_matches_file, dtype=str)
        overlap_matches = df.to_dict('split')['data']

    else:
        level_overlap_tags = {tag for tag, tiles in overlap_tag_maps.items() if len(tiles) in (n_matching_tiles,)}
        img_ids = os.listdir(sdcic.train_image_dir)
        # TODO: Use filter for all overlaps here? or just n_matching_tiles?
        # img_ids = filter_duplicates(img_ids)

        hash_dict = defaultdict(set)
        for img_id in tqdm(img_ids):
            for h in sdcic.tile_bm0hash_grids[img_id]:
                hash_dict[tuple(h)].add(img_id)

        sorted_hash_dict = {}
        for key, dups in sorted(hash_dict.items(), key=lambda x: len(x[1]), reverse=True):
            if len(dups) > 1:
                sorted_hash_dict[key] = dups

        hash_ids = list(sorted_hash_dict)
        for hash_id in tqdm(hash_ids):
            sdcic.find_valid_pairings_by_hash(hash_id, sorted_hash_dict, level_overlap_tags)
        overlap_matches = sdcic.matches

        df = pd.DataFrame(overlap_matches)
        df.to_csv(overlap_matches_file, index=False)

    return overlap_matches


def create_image_overlap_properties(n_matching_tiles_list, score_types):

    # bmh: blockMeanHash scores: Hamming distance between 2 blockMeanHash scores
    # cmh: colorMomentHash scores: L2 norm between 2 colorMomentHash scores
    # FIXME gcm: skimage greycomatrix scores: Relative difference between 2 greycop scores.
    # enp: Entropy scores: Exponential of the negative L2 norm between 2 entropy scores.
    # pix: Pixel scores: Fuzzy difference between 2 images pixelwise. Requires reading images so can be slow.
    # px0: Pixel scores: Hamming distance between 2 images pixelwise. Requires reading images so can be slow.
    # shp: Number of pixels that belong to ships:

    sdcic = SDCImageContainer()
    sdcic.preprocess_image_properties()
    sdcic.preprocess_label_properties()

    for n_matching_tiles in n_matching_tiles_list:
        overlap_matches = get_overlap_matches(sdcic, n_matching_tiles)

        for score_type in score_types:
            overlap_tile_scores_filename = f'overlap_{score_type}_tile_scores_{n_matching_tiles}.pkl'
            overlap_tile_scores_file = os.path.join(interim_data_dir, overlap_tile_scores_filename)
            if not os.path.exists(overlap_tile_scores_file):
                overlap_tile_scores_list = []

                for img1_id, img2_id, img1_overlap_tag in tqdm(sorted(overlap_matches)):
                    scores = sdcic.score_funcs[score_type](img1_id, img2_id, img1_overlap_tag)
                    overlap_tile_scores_list.append((img1_id, img2_id, img1_overlap_tag, *scores))
                df = pd.DataFrame(overlap_tile_scores_list)
                df.to_pickle(overlap_tile_scores_file)


if __name__ == '__main__':

    # n_matching_tiles = 9  # 376407 matches 2:40,    259 pixel_scores 00:03
    # n_matching_tiles = 6  # 376407 matches 3:12,  82823 pixel_scores 16:25
    # n_matching_tiles = 4  # 376407 matches 2:36,  72629 pixel_scores 13:51
    # n_matching_tiles = 3  # 376407 matches 2:43,  75936 pixel_scores 12:40
    # n_matching_tiles = 2  # 376407 matches 2:38, 149106 pixel_scores 20:26
    # n_matching_tiles = 1

    n_matching_tiles_list = [9, 6, 4, 3, 2, 1]
    score_types = ('bmh', 'cmh', 'enp', 'pix', 'px0', 'shp')
    create_image_overlap_properties(n_matching_tiles_list, score_types)
    # overlap_image_maps = load_image_overlap_properties(n_matching_tiles_list)

    print('done')
