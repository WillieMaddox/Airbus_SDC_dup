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
from collections import Counter
from utils import tile_idx2ij
from utils import get_hamming_distance_score
from utils import generate_pair_tag_lookup
from utils import overlap_tag_pairs
from utils import overlap_tag_maps
from utils import percent_diff
from utils import fuzzy_diff
from utils import gen_entropy
from utils import gen_greycop_hash
from utils import read_image_duplicate_tiles
from utils import write_image_duplicate_tiles
from utils import read_image_image_duplicate_tiles
from utils import update_image_image_duplicate_tiles

EPS = np.finfo(np.float32).eps

pair_tag_lookup = generate_pair_tag_lookup()


def filter_duplicates(img_ids):
    df = pd.read_csv('dup_blacklist_6.csv')
    blacklist = []
    for idx, row in df.iterrows():
        blacklist.append(row['ImageId1'])
    new_img_ids = [img_id for img_id in img_ids if img_id not in blacklist]
    # print(len(img_ids), len(blacklist), len(new_img_ids))
    return new_img_ids


class SDCImageContainer:

    def __init__(self, train_image_dir, cache_size=10000, **kwargs):
        # This class assumes images are square and height and width are divisible by tile_size.
        super().__init__(**kwargs)
        self.train_image_dir = train_image_dir
        self.h5_file = train_image_dir + '.h5'
        self.sz = 256  # tile_size
        self.n_rows = 3
        self.n_cols = 3
        self.n_tiles = self.n_rows * self.n_cols
        self.tile_score_max = self.sz * self.sz * 3 * 255  # 3 color channels, uint8
        self.tile_slice = slice(8, -8)
        self.tile_greycop_len = 5
        self.tile_greycop_dtype = np.float
        self.tile_greycop_grids = {}
        self.tile_md5hash_len = 8
        self.tile_md5hash_dtype = f'<U{self.tile_md5hash_len}'
        self.tile_md5hash_grids = {}
        self.tile_bm0hash_len = 32
        self.tile_bm0hash_dtype = np.uint8
        self.tile_bm0hash_grids = {}
        self.tile_cm0hash_len = 42
        self.tile_cm0hash_dtype = np.float
        self.tile_cm0hash_grids = {}
        self.tile_entropy_grids = {}
        self.image_duplicate_tiles = {}
        self.image_image_duplicate_tiles = {}
        self.matches = {}
        self.bmh_distance_max = 5
        self.overlap_bmh_min_score = 1 - ((self.bmh_distance_max + 20) / 256)
        self.overlap_cmh_min_score = 0.80  # cmh score has to be at least this good before assigning it to an image
        self.color_cts_solid = self.sz * self.sz
        self.cache = LRUCache(maxsize=cache_size)

    def preprocess_image_properties(self, filename_greycop, filename_md5hash, filename_bm0hash, filename_cm0hash, filename_entropy, filename_tile_dups):
        img_greycop_grids = {}
        if os.path.exists(filename_greycop):
            df = pd.read_pickle(filename_greycop)
            img_greycop_grids = {key: val for key, val in df.to_dict('split')['data']}

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

        img_entropy_grids = {}
        if os.path.exists(filename_entropy):
            df = pd.read_pickle(filename_entropy)
            img_entropy_grids = {key: val for key, val in df.to_dict('split')['data']}

        img_dups_vec = read_image_duplicate_tiles(filename_tile_dups)

        pp = 0
        mm = 0
        hh = 0
        cc = 0
        ee = 0
        dd = 0

        greycop_records = []
        md5hash_records = []
        bm0hash_records = []
        cm0hash_records = []
        entropy_records = []
        duplicate_records = {}

        img_ids = os.listdir(self.train_image_dir)
        for img_id in tqdm(sorted(img_ids)):

            img = None

            tile_greycop_grid = img_greycop_grids.get(img_id)
            if tile_greycop_grid is None:
                pp += 1
                img = self.get_img(img_id)
                tile_greycop_grid = np.zeros((self.n_tiles, self.tile_greycop_len), dtype=self.tile_greycop_dtype)
                for idx in range(self.n_tiles):
                    tile = self.get_tile(img, idx)
                    tile_greycop_grid[idx] = gen_greycop_hash(tile, self.tile_greycop_len)

            greycop_records.append({'ImageId': img_id, 'greycop_grid': tile_greycop_grid})  # float
            self.tile_greycop_grids[img_id] = tile_greycop_grid

            tile_md5hash_grid = img_md5hash_grids.get(img_id)
            if tile_md5hash_grid is None:
                mm += 1
                img = self.get_img(img_id) if img is None else img
                tile_md5hash_grid = np.zeros(self.n_tiles, dtype=self.tile_md5hash_dtype)
                for idx in range(self.n_tiles):
                    tile = self.get_tile(img, idx)
                    tile_md5hash_grid[idx] = hashlib.md5(tile.tobytes()).hexdigest()[:self.tile_md5hash_len]

            md5hash_records.append({'ImageId': img_id, 'md5hash_grid': tile_md5hash_grid})  # str
            self.tile_md5hash_grids[img_id] = tile_md5hash_grid

            tile_bm0hash_grid = img_bm0hash_grids.get(img_id)
            if tile_bm0hash_grid is None:
                hh += 1
                img = self.get_img(img_id) if img is None else img
                tile_bm0hash_grid = np.zeros((self.n_tiles, self.tile_bm0hash_len), dtype=self.tile_bm0hash_dtype)
                for idx in range(self.n_tiles):
                    tile = self.get_tile(img, idx)
                    tile_bm0hash_grid[idx] = img_hash.blockMeanHash(tile, mode=0)[0]

            bm0hash_records.append({'ImageId': img_id, 'bm0hash_grid': tile_bm0hash_grid})  # int
            self.tile_bm0hash_grids[img_id] = tuple(tuple(bm0) for bm0 in tile_bm0hash_grid)

            tile_cm0hash_grid = img_cm0hash_grids.get(img_id)
            if tile_cm0hash_grid is None:
                cc += 1
                img = self.get_img(img_id) if img is None else img
                tile_cm0hash_grid = np.zeros((self.n_tiles, self.tile_cm0hash_len), dtype=self.tile_cm0hash_dtype)
                for idx in range(self.n_tiles):
                    tile = self.get_tile(img, idx)
                    tile_cm0hash_grid[idx] = img_hash.colorMomentHash(tile)[0]

            cm0hash_records.append({'ImageId': img_id, 'cm0hash_grid': tile_cm0hash_grid})  # float
            self.tile_cm0hash_grids[img_id] = tile_cm0hash_grid

            tile_entropy_grid = img_entropy_grids.get(img_id)
            if tile_entropy_grid is None:
                ee += 1
                img = self.get_img(img_id) if img is None else img
                tile_entropy_grid = np.zeros((self.n_tiles, 3), dtype=np.float)
                for idx in range(self.n_tiles):
                    tile = self.get_tile(img, idx)
                    tile_entropy_grid[idx] = gen_entropy(tile)

            entropy_records.append({'ImageId': img_id, 'entropy_grid': tile_entropy_grid})  # float
            self.tile_entropy_grids[img_id] = tile_entropy_grid

            # Quick lookup just to tell us if any tiles within a single image are exact duplicates.
            tile_dups_vec = img_dups_vec.get(img_id)
            if tile_dups_vec is None:
                dd += 1
                img = self.get_img(img_id) if img is None else img
                tile_dups_vec = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
                for idx1 in range(self.n_tiles):
                    for idx2 in range(self.n_tiles):
                        if idx2 <= idx1:
                            continue
                        if self.tile_md5hash_grids[img_id][idx1] != self.tile_md5hash_grids[img_id][idx2]:
                            continue
                        if np.any(self.get_tile(img, idx1) != self.get_tile(img, idx2)):
                            # This check isn't really necessary.
                            # We do it anyway just to double check that the hashes aren't corrupted.
                            continue
                        tile_dups_vec[idx2] = min(tile_dups_vec[idx1], tile_dups_vec[idx2])

            duplicate_records[img_id] = tile_dups_vec
            self.image_duplicate_tiles[img_id] = tile_dups_vec

            if pp >= 500:
                df = pd.DataFrame().append(greycop_records)
                df.to_pickle(filename_greycop)
                pp = 0

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

            if ee >= 5000:
                df = pd.DataFrame().append(entropy_records)
                df.to_pickle(filename_entropy)
                ee = 0

            if dd >= 5000:
                write_image_duplicate_tiles(filename_tile_dups, duplicate_records)
                dd = 0

        if pp > 0:
            df = pd.DataFrame().append(greycop_records)
            df.to_pickle(filename_greycop)

        if mm > 0:
            df = pd.DataFrame().append(md5hash_records)
            df.to_pickle(filename_md5hash)

        if hh > 0:
            df = pd.DataFrame().append(bm0hash_records)
            df.to_pickle(filename_bm0hash)

        if cc > 0:
            df = pd.DataFrame().append(cm0hash_records)
            df.to_pickle(filename_cm0hash)

        if ee > 0:
            df = pd.DataFrame().append(entropy_records)
            df.to_pickle(filename_entropy)

        if dd > 0:
            write_image_duplicate_tiles(filename_tile_dups, duplicate_records)

    @cachedmethod(operator.attrgetter('cache'))
    def get_img(self, filename, path=None):
        path = self.train_image_dir if path is None else path
        return cv2.imread(os.path.join(path, filename))

    def get_tile(self, img, idx):
        i, j = tile_idx2ij[idx]
        return img[i * self.sz:(i + 1) * self.sz, j * self.sz:(j + 1) * self.sz, :]

    def get_bmh_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = []
        for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
            bmh1 = self.tile_bm0hash_grids[img1_id][idx1]
            bmh2 = self.tile_bm0hash_grids[img2_id][idx2]
            score = get_hamming_distance_score(bmh1, bmh2, normalize=True)
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

    def get_greycop_scores(self, img1_id, img2_id, img1_overlap_tag):
        img1_overlap_map = overlap_tag_maps[img1_overlap_tag]
        img2_overlap_map = overlap_tag_maps[overlap_tag_pairs[img1_overlap_tag]]
        scores = []
        for idx1, idx2 in zip(img1_overlap_map, img2_overlap_map):
            gcm1 = self.tile_greycop_grids[img1_id][idx1]
            gcm2 = self.tile_greycop_grids[img2_id][idx2]
            score = percent_diff(gcm1, gcm2)
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

    def update_image_image_duplicate_tiles(self, img1_id, img2_id):
        """
        record tiles from other images that match tiles from this image exactly.

        ImageId0      ImageId1      012345678 012345678
        ---------------------------------------------------------------
        012345678.jpg abcdef123.jpg 999999999 999999999  No equal pairs
        012345678.jpg abcdef123.jpg 099099099 999000000  (0, 3, 6) == (3, 4, 5, 6, 7, 8)
        012345678.jpg abcdef123.jpg 919399918 938918998  (1, 7) == 4, 3 == 1, 8 == (2, 5, 8)
        012345678.jpg abcdef123.jpg 012345678 678012345  0 == 3, ..., 3 == 6, ..., 8 == 2

        :param img1_id:
        :param img2_id:
        :return:
        """
        dup_self1 = self.image_duplicate_tiles[img1_id]
        dup_self2 = self.image_duplicate_tiles[img2_id]
        dup_cts1 = Counter(dup_self1)
        dup_cts2 = Counter(dup_self2)

        dup_idx1 = {c: np.array([idx for idx, tidx in enumerate(dup_self1) if tidx == c]) for c in dup_cts1}
        dup_idx2 = {c: np.array([idx for idx, tidx in enumerate(dup_self2) if tidx == c]) for c in dup_cts2}
        dup_tiles1 = np.array([9 if dup_cts1[idx] == 1 else tidx for idx, tidx in enumerate(dup_self1)])
        dup_tiles2 = np.array([9 if dup_cts2[idx] == 1 else tidx for idx, tidx in enumerate(dup_self2)])

        is_updated = False
        for idx1 in dup_cts1:
            for idx2 in dup_cts2:
                if self.tile_md5hash_grids[img1_id][idx1] != self.tile_md5hash_grids[img2_id][idx2]:
                    continue

                dup_tiles1[dup_idx1[idx1]] = idx1
                dup_tiles2[dup_idx2[idx2]] = idx1

        if (img1_id, img2_id) not in self.image_image_duplicate_tiles:
            self.image_image_duplicate_tiles[(img1_id, img2_id)] = (dup_tiles1, dup_tiles2)
            is_updated = True

        return is_updated

    def find_valid_pairings_by_hash(self, hash_id, sorted_hash_dict, level_overlap_tags):

        img_list = list(sorted(sorted_hash_dict[hash_id]))

        hamming_distance_lookup = {}
        for img_id in img_list:
            hamming_list = [get_hamming_distance_score(bmh0, hash_id, as_score=False) for bmh0 in self.tile_bm0hash_grids[img_id]]
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

    def find_valid_pairings_by_search(self, img_list):
        img_tag_scores = {img_id: self.images[img_id].overlap_image_scores for img_id in img_list}
        for ii, img1_id in enumerate(img_list):
            sdc1 = self.images[img1_id]
            img1 = sdc1.get_img()
            for jj, img2_id in enumerate(img_list):
                if jj <= ii:
                    continue
                sdc2 = self.images[img2_id]
                img2 = sdc2.get_img()

                for img1_overlap_tag, img1_overlap_map in overlap_tag_maps.items():
                    bmh_scores = self.get_bmh_scores(img1, img2, img1_overlap_tag)
                    if min(bmh_scores) > self.best_score_threshold:
                        img_tag_scores[img1_id][img1_overlap_tag] = min(bmh_scores)
                        img_tag_scores[img2_id][overlap_tag_pairs[img1_overlap_tag]] = min(bmh_scores)
                        sdc1.update_overlap_map(sdc2, bmh_scores, img1_overlap_tag)
                        sdc2.update_overlap_map(sdc1, bmh_scores, overlap_tag_pairs[img1_overlap_tag])
                        break

            print(f'{ii}/{len(img_list)}')
        return img_tag_scores


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
        i, j = tile_idx2ij[idx]
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
            self._tile_entropy_grid = np.zeros((self.n_tiles, 2), dtype=np.float) # (idx, chan)
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
            i, j = tile_idx2ij[idx1]
            k, l = tile_idx2ij[idx2]
            self.overlap_tile_scores[i, j, k, l] = s

        self.overlap_image_maps[tag][other_sdc.img_id] = {'tile_scores': tile_scores}


def main():
    ship_dir = "data/input"
    train_image_dir = os.path.join(ship_dir, 'train_768')
    image_md5hash_grids_file = os.path.join("data", "image_md5hash_grids.pkl")
    image_bm0hash_grids_file = os.path.join("data", "image_bm0hash_grids.pkl")
    image_cm0hash_grids_file = os.path.join("data", "image_cm0hash_grids.pkl")
    image_entropy_grids_file = os.path.join("data", "image_entropy_grids.pkl")
    image_greycop_grids_file = os.path.join("data", "image_greycop_grids.pkl")
    image_duplicate_tiles_file = os.path.join("data", "image_duplicate_tiles.txt")
    image_image_duplicate_tiles_file = os.path.join("data", "image_image_duplicate_tiles.txt")

    sdcic = SDCImageContainer(train_image_dir)
    sdcic.preprocess_image_properties(
        image_greycop_grids_file,
        image_md5hash_grids_file,
        image_bm0hash_grids_file,
        image_cm0hash_grids_file,
        image_entropy_grids_file,
        image_duplicate_tiles_file)

    n_matching_tiles = 9  # 376407 matches 2:40,    259 pixel_scores 00:03
    # n_matching_tiles = 6  # 376407 matches 3:12,  82823 pixel_scores 16:25
    # n_matching_tiles = 4  # 376407 matches 2:36,  72629 pixel_scores 13:51
    # n_matching_tiles = 3  # 376407 matches 2:43,  75936 pixel_scores 12:40
    # n_matching_tiles = 2  # 376407 matches 2:38, 149106 pixel_scores 20:26
    overlap_bmh_tile_scores_file = os.path.join("data", f"overlap_bmh_tile_scores_{n_matching_tiles}.pkl")
    overlap_cmh_tile_scores_file = os.path.join("data", f"overlap_cmh_tile_scores_{n_matching_tiles}.pkl")
    overlap_gcm_tile_scores_file = os.path.join("data", f"overlap_gcm_tile_scores_{n_matching_tiles}.pkl")
    overlap_enp_tile_scores_file = os.path.join("data", f"overlap_enp_tile_scores_{n_matching_tiles}.pkl")
    overlap_pix_tile_scores_file = os.path.join("data", f"overlap_pix_tile_scores_{n_matching_tiles}.pkl")

    # blockMeanHash
    if not os.path.exists(overlap_bmh_tile_scores_file):

        level_overlap_tags = {tag for tag, tiles in overlap_tag_maps.items() if len(tiles) in (n_matching_tiles,)}
        img_ids = os.listdir(train_image_dir)
        # TODO: Use filter for all overlaps here? or just n_matching_tiles?
        # img_ids = filter_duplicates(img_ids)

        hash_dict = defaultdict(set)
        for img_id in tqdm(img_ids):
            for h in sdcic.tile_bm0hash_grids[img_id]:
                hash_dict[h].add(img_id)

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

    sdcic.image_image_duplicate_tiles = read_image_image_duplicate_tiles(image_image_duplicate_tiles_file)

    updated_image_image_duplicate_tiles = {}
    for (img1_id, img2_id), img1_overlap_tags in tqdm(sorted(overlap_bmh_tile_scores.items())):
        if (img1_id, img2_id) in sdcic.image_image_duplicate_tiles:
            continue
        is_updated = sdcic.update_image_image_duplicate_tiles(img1_id, img2_id)
        if not is_updated:
            continue
        updated_image_image_duplicate_tiles[(img1_id, img2_id)] = sdcic.image_image_duplicate_tiles[(img1_id, img2_id)]

    if len(updated_image_image_duplicate_tiles) > 0:
        update_image_image_duplicate_tiles(image_image_duplicate_tiles_file, updated_image_image_duplicate_tiles)


if __name__ == '__main__':
    main()
    print('done')
