import os
import hashlib
import operator

from cachetools import LRUCache
from cachetools import cachedmethod
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from cv2 import img_hash
from collections import defaultdict
from collections import namedtuple
from sdcdup.utils import idx2ijpair
from sdcdup.utils import rle_to_full_mask
from sdcdup.utils import get_hamming_distance
from sdcdup.utils import generate_pair_tag_lookup
from sdcdup.utils import overlap_tag_pairs
from sdcdup.utils import overlap_tag_maps
from sdcdup.utils import relative_diff
from sdcdup.utils import fuzzy_diff
from sdcdup.utils import gen_entropy
from sdcdup.utils import gen_greycop_hash
from sdcdup.utils import get_issolid_flags

EPS = np.finfo(np.float32).eps

pair_tag_lookup = generate_pair_tag_lookup()


def filter_duplicates(img_ids):
    df = pd.read_csv('data/processed/dup_blacklist_6.csv')
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


class SDCImageContainer:

    def __init__(self, cache_size=10000, **kwargs):
        # This class assumes images are square and height and width are divisible by tile_size.
        super().__init__(**kwargs)

        self.train_image_dir = 'data/raw/train_768/'
        self.rle_label_file = 'data/raw/train_ship_segmentations_v2.csv/'
        self.sz = 256  # tile_size
        self.n_rows = 3
        self.n_cols = 3
        self.n_tiles = self.n_rows * self.n_cols
        self.tile_score_max = self.sz * self.sz * 3 * 255  # 3 color channels, uint8
        self.tile_slice = slice(8, -8)
        self.tile_md5hash_len = 8
        self.tile_md5hash_dtype = f'<U{self.tile_md5hash_len}'
        self.tile_md5hash_grids = {}
        self.tile_bm0hash_len = 32
        self.tile_bm0hash_dtype = np.uint8
        self.tile_bm0hash_grids = {}
        self.tile_cm0hash_len = 42
        self.tile_cm0hash_dtype = np.float
        self.tile_cm0hash_grids = {}
        self.tile_greycop_len = 5
        self.tile_greycop_dtype = np.float
        self.tile_greycop_grids = {}
        self.tile_entropy_grids = {}
        self.tile_issolid_grids = {}
        self.tile_shipcnt_grids = {}
        self.matches = {}
        self.bmh_distance_max = 5
        self.overlap_bmh_min_score = 1 - ((self.bmh_distance_max + 20) / 256)
        self.overlap_cmh_min_score = 0.80  # cmh score has to be at least this good before assigning it to an image
        self.color_cts_solid = self.sz * self.sz
        self.cache = LRUCache(maxsize=cache_size)

    def preprocess_image_properties(
            self,
            filename_md5hash='data/interim/image_md5hash_grids.pkl',
            filename_bm0hash='data/interim/image_bm0hash_grids.pkl',
            filename_cm0hash='data/interim/image_cm0hash_grids.pkl',
            filename_greycop='data/interim/image_greycop_grids.pkl',
            filename_entropy='data/interim/image_entropy_grids.pkl',
            filename_issolid='data/interim/image_issolid_grids.pkl'):

        img_md5hash_grids = {}
        if os.path.exists(filename_md5hash):
            df = pd.read_pickle(filename_md5hash)
            img_md5hash_grids = {key: val for key, val in df.to_dict('split')['data']}

        img_bm0hash_grids = {}
        if os.path.exists(filename_bm0hash):
            df = pd.read_pickle(filename_bm0hash)
            img_bm0hash_grids = {key: val for key, val in df.to_dict('split')['data']}

        img_cm0hash_grids = {}
        if os.path.exists(filename_cm0hash):
            df = pd.read_pickle(filename_cm0hash)
            img_cm0hash_grids = {key: val for key, val in df.to_dict('split')['data']}

        img_greycop_grids = {}
        if os.path.exists(filename_greycop):
            df = pd.read_pickle(filename_greycop)
            img_greycop_grids = {key: val for key, val in df.to_dict('split')['data']}

        img_entropy_grids = {}
        if os.path.exists(filename_entropy):
            df = pd.read_pickle(filename_entropy)
            img_entropy_grids = {key: val for key, val in df.to_dict('split')['data']}

        img_issolid_grids = {}
        if os.path.exists(filename_issolid):
            df = pd.read_pickle(filename_issolid)
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
                df.to_pickle(filename_md5hash)
                mm = 0

            if hh >= 5000:
                df = pd.DataFrame().append(bm0hash_records)
                df.to_pickle(filename_bm0hash)
                hh = 0

            if cc >= 5000:
                df = pd.DataFrame().append(cm0hash_records)
                df.to_pickle(filename_cm0hash)
                cc = 0

            if gg >= 5000:
                df = pd.DataFrame().append(greycop_records)
                df.to_pickle(filename_greycop)
                gg = 0

            if ee >= 5000:
                df = pd.DataFrame().append(entropy_records)
                df.to_pickle(filename_entropy)
                ee = 0

            if ss >= 5000:
                df = pd.DataFrame().append(issolid_records)
                df.to_pickle(filename_issolid)
                ss = 0

        if mm > 0:
            df = pd.DataFrame().append(md5hash_records)
            df.to_pickle(filename_md5hash)

        if hh > 0:
            df = pd.DataFrame().append(bm0hash_records)
            df.to_pickle(filename_bm0hash)

        if cc > 0:
            df = pd.DataFrame().append(cm0hash_records)
            df.to_pickle(filename_cm0hash)

        if gg > 0:
            df = pd.DataFrame().append(greycop_records)
            df.to_pickle(filename_greycop)

        if ee > 0:
            df = pd.DataFrame().append(entropy_records)
            df.to_pickle(filename_entropy)

        if ss > 0:
            df = pd.DataFrame().append(issolid_records)
            df.to_pickle(filename_issolid)

    def preprocess_label_properties(
            self,
            filename_shipcnt='data/interim/image_shipcnt_grids.pkl'):

        img_shipcnt_grids = {}
        if os.path.exists(filename_shipcnt):
            df = pd.read_pickle(filename_shipcnt)
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
                df.to_pickle(filename_shipcnt)
                pp = 0

        if pp > 0:
            df = pd.DataFrame().append(shipcnt_records)
            df.to_pickle(filename_shipcnt)

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

    def get_greycop_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = []
        for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
            gcm1 = self.tile_greycop_grids[img1_id][idx1]
            gcm2 = self.tile_greycop_grids[img2_id][idx2]
            score = relative_diff(gcm1, gcm2)
            scores.append(score)
        return np.array(scores)

    def get_pyramid_scores(self, img1_id, img2_id, img1_overlap_tag):
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

    def get_entropy_scores(self, img1_id, img2_id, img1_overlap_tag):
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

    def gen_pixel_scores(self, img1_id, img2_id, img1_overlap_tag):
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

    def gen_shipcnt_scores(self, img1_id, img2_id, img1_overlap_tag):
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
                    if min(bmh_scores) < self.overlap_bmh_min_score:
                        continue
                    self.matches[(img1_id, img2_id, img1_overlap_tag)] = bmh_scores

        return


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


# TODO: add ability to specify which score types we want retrieved.
#  But first need to figure out what to do about gcm since it holds 5 different score types.
def load_image_overlap_properties(n_matching_tiles_list, score_types=None):

    overlap_image_maps = {}

    for n_matching_tiles in n_matching_tiles_list:

        overlap_bmh_tile_scores_file = f'data/interim/overlap_bmh_tile_scores_{n_matching_tiles}.pkl'
        overlap_cmh_tile_scores_file = f'data/interim/overlap_cmh_tile_scores_{n_matching_tiles}.pkl'
        overlap_gcm_tile_scores_file = f'data/interim/overlap_gcm_tile_scores_{n_matching_tiles}.pkl'
        overlap_enp_tile_scores_file = f'data/interim/overlap_enp_tile_scores_{n_matching_tiles}.pkl'
        overlap_pix_tile_scores_file = f'data/interim/overlap_pix_tile_scores_{n_matching_tiles}.pkl'
        overlap_px0_tile_scores_file = f'data/interim/overlap_px0_tile_scores_{n_matching_tiles}.pkl'
        overlap_shp_tile_scores_file = f'data/interim/overlap_shp_tile_scores_{n_matching_tiles}.pkl'

        df = pd.read_pickle(overlap_bmh_tile_scores_file)
        overlap_bmh_tile_scores = {}
        for img1_id, img2_id, img1_overlap_tag, *bmh_scores in df.to_dict('split')['data']:
            if (img1_id, img2_id) not in overlap_bmh_tile_scores:
                overlap_bmh_tile_scores[(img1_id, img2_id)] = {}
            overlap_bmh_tile_scores[(img1_id, img2_id)][img1_overlap_tag] = np.array(bmh_scores)

        df = pd.read_pickle(overlap_cmh_tile_scores_file)
        overlap_cmh_tile_scores = {}
        for img1_id, img2_id, img1_overlap_tag, *cmh_scores in df.to_dict('split')['data']:
            if (img1_id, img2_id) not in overlap_cmh_tile_scores:
                overlap_cmh_tile_scores[(img1_id, img2_id)] = {}
            overlap_cmh_tile_scores[(img1_id, img2_id)][img1_overlap_tag] = np.array(cmh_scores)

        df = pd.read_pickle(overlap_gcm_tile_scores_file)
        overlap_gcm_tile_scores = {}
        for img1_id, img2_id, img1_overlap_tag, *gcm_scores in df.to_dict('split')['data']:
            if (img1_id, img2_id) not in overlap_gcm_tile_scores:
                overlap_gcm_tile_scores[(img1_id, img2_id)] = {}
            overlap_gcm_tile_scores[(img1_id, img2_id)][img1_overlap_tag] = np.array(gcm_scores)

        df = pd.read_pickle(overlap_enp_tile_scores_file)
        overlap_enp_tile_scores = {}
        for img1_id, img2_id, img1_overlap_tag, *enp_scores in df.to_dict('split')['data']:
            if (img1_id, img2_id) not in overlap_enp_tile_scores:
                overlap_enp_tile_scores[(img1_id, img2_id)] = {}
            overlap_enp_tile_scores[(img1_id, img2_id)][img1_overlap_tag] = np.array(enp_scores)

        df = pd.read_pickle(overlap_pix_tile_scores_file)
        overlap_pix_tile_scores = {}
        for img1_id, img2_id, img1_overlap_tag, *pix_scores in df.to_dict('split')['data']:
            if (img1_id, img2_id) not in overlap_pix_tile_scores:
                overlap_pix_tile_scores[(img1_id, img2_id)] = {}
            overlap_pix_tile_scores[(img1_id, img2_id)][img1_overlap_tag] = np.array(pix_scores)

        df = pd.read_pickle(overlap_px0_tile_scores_file)
        overlap_px0_tile_scores = {}
        for img1_id, img2_id, img1_overlap_tag, *px0_scores in df.to_dict('split')['data']:
            if (img1_id, img2_id) not in overlap_px0_tile_scores:
                overlap_px0_tile_scores[(img1_id, img2_id)] = {}
            overlap_px0_tile_scores[(img1_id, img2_id)][img1_overlap_tag] = np.array(px0_scores)

        df = pd.read_pickle(overlap_shp_tile_scores_file)
        overlap_shp_tile_scores = {}
        for img1_id, img2_id, img1_overlap_tag, *shp_scores in df.to_dict('split')['data']:
            if (img1_id, img2_id) not in overlap_shp_tile_scores:
                overlap_shp_tile_scores[(img1_id, img2_id)] = {}
            overlap_shp_tile_scores[(img1_id, img2_id)][img1_overlap_tag] = np.array(shp_scores)

        Overlap_Scores = namedtuple(
            'overlap_scores',
            ['bmh', 'cmh', 'con', 'hom', 'eng', 'cor', 'epy', 'enp', 'pix', 'px0', 'shp']
        )
        for (img1_id, img2_id), img1_overlap_tags in tqdm(sorted(overlap_bmh_tile_scores.items())):
            for img1_overlap_tag in img1_overlap_tags:

                bmh_scores = overlap_bmh_tile_scores[(img1_id, img2_id)][img1_overlap_tag]
                cmh_scores = overlap_cmh_tile_scores[(img1_id, img2_id)][img1_overlap_tag]
                con_scores = overlap_gcm_tile_scores[(img1_id, img2_id)][img1_overlap_tag][:, 0]
                hom_scores = overlap_gcm_tile_scores[(img1_id, img2_id)][img1_overlap_tag][:, 1]
                eng_scores = overlap_gcm_tile_scores[(img1_id, img2_id)][img1_overlap_tag][:, 2]
                cor_scores = overlap_gcm_tile_scores[(img1_id, img2_id)][img1_overlap_tag][:, 3]
                epy_scores = overlap_gcm_tile_scores[(img1_id, img2_id)][img1_overlap_tag][:, 4]
                enp_scores = overlap_enp_tile_scores[(img1_id, img2_id)][img1_overlap_tag]
                pix_scores = overlap_pix_tile_scores[(img1_id, img2_id)][img1_overlap_tag]
                px0_scores = overlap_px0_tile_scores[(img1_id, img2_id)][img1_overlap_tag]
                shp_scores = overlap_shp_tile_scores[(img1_id, img2_id)][img1_overlap_tag]

                overlap_scores = Overlap_Scores(
                    bmh_scores, cmh_scores, con_scores, hom_scores, eng_scores, cor_scores,
                    epy_scores, enp_scores, pix_scores, px0_scores, shp_scores)

                if (img1_id, img2_id) not in overlap_image_maps:
                    overlap_image_maps[(img1_id, img2_id)] = {}
                overlap_image_maps[(img1_id, img2_id)][img1_overlap_tag] = overlap_scores

    return overlap_image_maps


def main():

    sdcic = SDCImageContainer()

    # image_md5hash_grids_file = 'data/interim/image_md5hash_grids.pkl'
    # image_bm0hash_grids_file = 'data/interim/image_bm0hash_grids.pkl'
    # image_cm0hash_grids_file = 'data/interim/image_cm0hash_grids.pkl'
    # image_greycop_grids_file = 'data/interim/image_greycop_grids.pkl'
    # image_entropy_grids_file = 'data/interim/image_entropy_grids.pkl'
    # image_issolid_grids_file = 'data/interim/image_issolid_grids.pkl'
    sdcic.preprocess_image_properties()

    # image_shipcnt_grids_file = 'data/interim/image_shipcnt_grids.pkl'
    sdcic.preprocess_label_properties()

    n_matching_tiles = 9  # 376407 matches 2:40,    259 pixel_scores 00:03
    # n_matching_tiles = 6  # 376407 matches 3:12,  82823 pixel_scores 16:25
    # n_matching_tiles = 4  # 376407 matches 2:36,  72629 pixel_scores 13:51
    # n_matching_tiles = 3  # 376407 matches 2:43,  75936 pixel_scores 12:40
    # n_matching_tiles = 2  # 376407 matches 2:38, 149106 pixel_scores 20:26
    # n_matching_tiles = 1
    overlap_bmh_tile_scores_file = f'data/interim/overlap_bmh_tile_scores_{n_matching_tiles}.pkl'
    overlap_cmh_tile_scores_file = f'data/interim/overlap_cmh_tile_scores_{n_matching_tiles}.pkl'
    overlap_gcm_tile_scores_file = f'data/interim/overlap_gcm_tile_scores_{n_matching_tiles}.pkl'
    overlap_enp_tile_scores_file = f'data/interim/overlap_enp_tile_scores_{n_matching_tiles}.pkl'
    overlap_pix_tile_scores_file = f'data/interim/overlap_pix_tile_scores_{n_matching_tiles}.pkl'
    overlap_px0_tile_scores_file = f'data/interim/overlap_px0_tile_scores_{n_matching_tiles}.pkl'
    overlap_shp_tile_scores_file = f'data/interim/overlap_shp_tile_scores_{n_matching_tiles}.pkl'

    # TODO: Use a "matches" file with only "(img1_id, img2_id), img1_overlap_tags"
    #  instead of overlap_bmh_tile_scores_file for a reference in image_features.py
    # blockMeanHash
    if not os.path.exists(overlap_bmh_tile_scores_file):

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

        overlap_bmh_tile_scores_list = []
        for (img1_id, img2_id, img1_overlap_tag), bmh_scores in sdcic.matches.items():
            overlap_bmh_tile_scores_list.append((img1_id, img2_id, img1_overlap_tag, *bmh_scores))

        df = pd.DataFrame(overlap_bmh_tile_scores_list)
        df.to_pickle(overlap_bmh_tile_scores_file)

    df = pd.read_pickle(overlap_bmh_tile_scores_file)
    overlap_bmh_tile_scores = {}
    for img1_id, img2_id, img1_overlap_tag, *bmh_scores in tqdm(df.to_dict('split')['data']):
        if (img1_id, img2_id) not in overlap_bmh_tile_scores:
            overlap_bmh_tile_scores[(img1_id, img2_id)] = {}
        overlap_bmh_tile_scores[(img1_id, img2_id)][img1_overlap_tag] = bmh_scores

    # colorMomentHash scores:
    # L2 norm between 2 colorMomentHash scores
    if not os.path.exists(overlap_cmh_tile_scores_file):
        overlap_cmh_tile_scores_list = []
        for (img1_id, img2_id), img1_overlap_tags in tqdm(sorted(overlap_bmh_tile_scores.items())):
            for img1_overlap_tag in img1_overlap_tags:
                cmh_scores = sdcic.get_cmh_scores(img1_id, img2_id, img1_overlap_tag)
                overlap_cmh_tile_scores_list.append((img1_id, img2_id, img1_overlap_tag, *cmh_scores))
        df = pd.DataFrame(overlap_cmh_tile_scores_list)
        df.to_pickle(overlap_cmh_tile_scores_file)

    # skimage greycomatrix property scores:
    # Exponential of the negative L2 norm between 2 greycop scores.
    if not os.path.exists(overlap_gcm_tile_scores_file):
        overlap_gcm_tile_scores_list = []
        for (img1_id, img2_id), img1_overlap_tags in tqdm(sorted(overlap_bmh_tile_scores.items())):
            for img1_overlap_tag in img1_overlap_tags:
                gcm_scores = sdcic.get_greycop_scores(img1_id, img2_id, img1_overlap_tag)
                overlap_gcm_tile_scores_list.append((img1_id, img2_id, img1_overlap_tag, *gcm_scores))
        df = pd.DataFrame(overlap_gcm_tile_scores_list)
        df.to_pickle(overlap_gcm_tile_scores_file)

    # Entropy scores:
    # Exponential of the negative L2 norm between 2 entropy scores.
    if not os.path.exists(overlap_enp_tile_scores_file):
        overlap_enp_tile_scores_list = []
        for (img1_id, img2_id), img1_overlap_tags in tqdm(sorted(overlap_bmh_tile_scores.items())):
            for img1_overlap_tag in img1_overlap_tags:
                enp_scores = sdcic.get_entropy_scores(img1_id, img2_id, img1_overlap_tag)
                overlap_enp_tile_scores_list.append((img1_id, img2_id, img1_overlap_tag, *enp_scores))
        df = pd.DataFrame(overlap_enp_tile_scores_list)
        df.to_pickle(overlap_enp_tile_scores_file)

    # Pixel scores:
    # Hamming distance between 2 images pixelwise. Requires reading images so can be slow.
    if not os.path.exists(overlap_pix_tile_scores_file):
        overlap_pix_tile_scores_list = []
        for (img1_id, img2_id), img1_overlap_tags in tqdm(sorted(overlap_bmh_tile_scores.items())):
            for img1_overlap_tag in img1_overlap_tags:
                pix_scores = sdcic.gen_pixel_scores(img1_id, img2_id, img1_overlap_tag)
                overlap_pix_tile_scores_list.append((img1_id, img2_id, img1_overlap_tag, *pix_scores))
        df = pd.DataFrame(overlap_pix_tile_scores_list)
        df.to_pickle(overlap_pix_tile_scores_file)

    # Pixel scores:
    # Hamming distance between 2 images pixelwise. Requires reading images so can be slow.
    if not os.path.exists(overlap_px0_tile_scores_file):
        overlap_px0_tile_scores_list = []
        for (img1_id, img2_id), img1_overlap_tags in tqdm(sorted(overlap_bmh_tile_scores.items())):
            for img1_overlap_tag in img1_overlap_tags:
                px0_scores = sdcic.gen_px0_scores(img1_id, img2_id, img1_overlap_tag)
                overlap_px0_tile_scores_list.append((img1_id, img2_id, img1_overlap_tag, *px0_scores))
        df = pd.DataFrame(overlap_px0_tile_scores_list)
        df.to_pickle(overlap_px0_tile_scores_file)

    # number of pixels that belong to ships:
    if not os.path.exists(overlap_shp_tile_scores_file):
        overlap_shp_tile_scores_list = []
        for (img1_id, img2_id), img1_overlap_tags in tqdm(sorted(overlap_bmh_tile_scores.items())):
            for img1_overlap_tag in img1_overlap_tags:
                shp_scores = sdcic.gen_shipcnt_scores(img1_id, img2_id, img1_overlap_tag)
                overlap_shp_tile_scores_list.append((img1_id, img2_id, img1_overlap_tag, *shp_scores))
        df = pd.DataFrame(overlap_shp_tile_scores_list)
        df.to_pickle(overlap_shp_tile_scores_file)


if __name__ == '__main__':
    main()
    print('done')
