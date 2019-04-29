import os
import time
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from cv2 import img_hash
from collections import defaultdict
from collections import Counter
from utils import tile_idx2ij, tile_ij2idx
from utils import normalized_hamming_distance
from utils import generate_overlay_tag_slices
from utils import generate_pair_tag_lookup
from utils import overlay_tag_pairs
from utils import overlay_tag_maps
from utils import read_image_duplicate_tiles
from utils import write_image_duplicate_tiles
from utils import read_image_image_duplicate_tiles
from utils import update_image_image_duplicate_tiles

overlay_tag_slices = generate_overlay_tag_slices()

pair_tag_lookup = generate_pair_tag_lookup()

hash_algos = {
    # 'md5SumHash_16': lambda: None,
    'pHash': img_hash.pHash,
    # 'md5SumHash': lambda tile: hashlib.md5(tile.tobytes()).hexdigest(),
    'averageHash': img_hash.averageHash,
    # 'blockMeanHash0': lambda tile: img_hash.blockMeanHash(tile, mode=0),
    # 'blockMeanHash1': lambda tile: img_hash.blockMeanHash(tile, mode=1),
    # 'colorMomentHash': img_hash.colorMomentHash,
    # 'marrHildrethHash': img_hash.marrHildrethHash,
    # 'radialVarianceHash': img_hash.radialVarianceHash
}


def filter_duplicates(img_ids):
    df = pd.read_csv('dup_blacklist_6.csv')
    blacklist = []
    for idx, row in df.iterrows():
        blacklist.append(row['ImageId1'])
    new_img_ids = [img_id for img_id in img_ids if img_id not in blacklist]
    # print(len(img_ids), len(blacklist), len(new_img_ids))
    return new_img_ids


def get_channel_entropy(ctr, img_size):
    ctr_norm = {k: v / img_size for k, v in ctr.items()}
    ctr_entropy = {k: -v * np.log(v) for k, v in ctr_norm.items()}
    entropy = np.sum([k * v for k, v in ctr_entropy.items()])
    return entropy


def get_entropy(img):
    img_grad = np.gradient(img.astype(np.int), axis=(0, 1))
    entropy_list = []
    for channel_grad in img_grad:
        ctr = Counter(np.abs(channel_grad).flatten())
        entropy_list.append(get_channel_entropy(ctr, img.size))
    return np.array(entropy_list)


class SDCImageContainer:

    def __init__(self, train_image_dir, **kwargs):
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
        self.tile_counter_grids = {}
        self.tile_md5hash_len = 8
        self.tile_md5hash_dtype = f'<U{self.tile_md5hash_len}'
        self.tile_md5hash_grids = {}
        self.tile_bm0hash_len = 32
        self.tile_bm0hash_dtype = np.uint8
        self.tile_bm0hash_grids = {}
        self.tile_entropy_grids = {}
        self.image_duplicate_tiles = {}
        self.image_image_duplicate_tiles = {}
        self.image_image_duplicate_tiles_orig = {}
        self.overlay_image_maps = {}
        self.overlay_pixel_scores = {}
        self.minimum_score_threshold = 0.95  # overlay score has to be at least this good before assigning it to an image
        self.best_score_threshold = 0.99  # after this, don't have to check if better score exists.
        self.matches = defaultdict(list)

    def load_3x3_grids(self, filename_counter, filename_md5hash, filename_bm0hash, filename_entropy, filename_tile_dups):
        img_counter_grids = {}
        if os.path.exists(filename_counter):
            df = pd.read_pickle(filename_counter)
            img_counter_grids = {key: val for key, val in df.to_dict('split')['data']}

        img_md5hash_grids = {}
        if os.path.exists(filename_md5hash):
            df = pd.read_pickle(filename_md5hash)
            img_md5hash_grids = {key: val for key, val in df.to_dict('split')['data']}

        img_bm0hash_grids = {}
        if os.path.exists(filename_bm0hash):
            df = pd.read_pickle(filename_bm0hash)
            img_bm0hash_grids = {key: val for key, val in df.to_dict('split')['data']}

        img_entropy_grids = {}
        if os.path.exists(filename_entropy):
            df = pd.read_pickle(filename_entropy)
            img_entropy_grids = {key: val for key, val in df.to_dict('split')['data']}

        img_dups_vec = read_image_duplicate_tiles(filename_tile_dups)

        cc = 0
        mm = 0
        hh = 0
        ee = 0
        dd = 0

        counter_records = []
        md5hash_records = []
        bm0hash_records = []
        entropy_records = []
        duplicate_records = {}

        img_ids = os.listdir(self.train_image_dir)
        for img_id in tqdm(sorted(img_ids)):

            img = None

            prev_counter_grid = img_counter_grids.get(img_id)
            if prev_counter_grid is None:
                cc += 1
                img = self.get_img(img_id)
                prev_counter_grid = np.zeros((self.n_tiles, 3, 2), dtype=np.int64)
                for idx in range(self.n_tiles):
                    tile = self.get_tile(img, idx)
                    for chan in range(3):
                        unique, counts = np.unique(tile[:, :, chan].flatten(), return_counts=True)
                        max_idx = np.argmax(counts)
                        prev_counter_grid[idx, chan] = np.array([unique[max_idx], counts[max_idx]])

            tile_counter_grid = [tuple(tuple(pcg) for pcg in prev_counter_grid[idx]) for idx in range(self.n_tiles)]

            counter_records.append({'ImageId': img_id, 'counter_grid': prev_counter_grid})  # int
            self.tile_counter_grids[img_id] = tile_counter_grid

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

            tile_entropy_grid = img_entropy_grids.get(img_id)
            if tile_entropy_grid is None:
                ee += 1
                img = self.get_img(img_id) if img is None else img
                tile_entropy_grid = np.zeros((self.n_tiles, 2), dtype=np.float)
                for idx in range(self.n_tiles):
                    tile = self.get_tile(img, idx)
                    tile_entropy_grid[idx] = get_entropy(tile)

            entropy_records.append({'ImageId': img_id, 'entropy_grid': tile_entropy_grid})  # int
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

            if cc >= 5000:
                df = pd.DataFrame().append(counter_records)
                df.to_pickle(filename_counter)
                cc = 0

            if mm >= 5000:
                df = pd.DataFrame().append(md5hash_records)
                df.to_pickle(filename_md5hash)
                mm = 0

            if hh >= 5000:
                df = pd.DataFrame().append(bm0hash_records)
                df.to_pickle(filename_bm0hash)
                hh = 0

            if ee >= 5000:
                df = pd.DataFrame().append(entropy_records)
                df.to_pickle(filename_entropy)
                ee = 0

            if dd >= 5000:
                write_image_duplicate_tiles(filename_tile_dups, duplicate_records)
                dd = 0

        if cc > 0:
            df = pd.DataFrame().append(counter_records)
            df.to_pickle(filename_counter)

        if mm > 0:
            df = pd.DataFrame().append(md5hash_records)
            df.to_pickle(filename_md5hash)

        if hh > 0:
            df = pd.DataFrame().append(bm0hash_records)
            df.to_pickle(filename_bm0hash)

        if ee > 0:
            df = pd.DataFrame().append(entropy_records)
            df.to_pickle(filename_entropy)

        if dd > 0:
            write_image_duplicate_tiles(filename_tile_dups, duplicate_records)

    def get_img(self, filename, path=None):
        path = self.train_image_dir if path is None else path
        return cv2.imread(os.path.join(path, filename))

    def get_tile0(self, img, i, j):
        return img[i * self.sz:(i + 1) * self.sz, j * self.sz:(j + 1) * self.sz, :]

    def get_tile(self, img, idx):
        i, j = tile_idx2ij[idx]
        return img[i * self.sz:(i + 1) * self.sz, j * self.sz:(j + 1) * self.sz, :]

    def fuzzy_join(self, tile1, tile2):
        maxab = np.max(np.stack([tile1, tile2]), axis=0)
        a = maxab - tile2
        b = maxab - tile1
        return a + b

    def fuzzy_diff(self, tile1, tile2):
        ab = self.fuzzy_join(tile1, tile2)
        return np.sum(ab)

    def fuzzy_compare(self, tile1, tile2):
        ab = self.fuzzy_join(tile1, tile2)
        n = np.prod(ab.shape)
        return np.sum(255 - ab) / (255 * n)

    def check_exact_match(self, img1, img2, img1_overlay_tag):
        img1_slice = overlay_tag_slices[img1_overlay_tag]
        img2_slice = overlay_tag_slices[overlay_tag_pairs[img1_overlay_tag]]
        img1_overlay = img1[img1_slice]
        img2_overlay = img2[img2_slice]
        is_perfect_score = np.all(img1_overlay == img2_overlay)
        return is_perfect_score

    def get_bmh_score(self, img1, img2, img1_overlay_tag):
        img1_slice = overlay_tag_slices[img1_overlay_tag]
        img2_slice = overlay_tag_slices[overlay_tag_pairs[img1_overlay_tag]]
        img1_overlay = img1[img1_slice]
        img2_overlay = img2[img2_slice]
        bmh1 = img_hash.blockMeanHash(img1_overlay, mode=0)
        bmh2 = img_hash.blockMeanHash(img2_overlay, mode=0)
        score = self.fuzzy_compare(bmh1, bmh2)
        return score

    def get_overlay_score(self, img1_id, img2_id, img1_overlay_tag):
        img1_overlay_map = overlay_tag_maps[img1_overlay_tag]
        img2_overlay_map = overlay_tag_maps[overlay_tag_pairs[img1_overlay_tag]]
        bmh1_list = []
        bmh2_list = []
        for idx1, idx2 in zip(img1_overlay_map, img2_overlay_map):
            bmh1 = self.tile_bm0hash_grids[img1_id][idx1]
            bmh2 = self.tile_bm0hash_grids[img2_id][idx2]
            bmh1_list.append(bmh1)
            bmh2_list.append(bmh2)
        bmh1_arr = np.vstack(bmh1_list)
        bmh2_arr = np.vstack(bmh2_list)
        score = self.fuzzy_compare(bmh1_arr, bmh2_arr)
        return score

    def get_tile_scores(self, img1_id, img2_id, img1_overlay_tag):
        img1_overlay_map = overlay_tag_maps[img1_overlay_tag]
        img2_overlay_map = overlay_tag_maps[overlay_tag_pairs[img1_overlay_tag]]
        scores = []
        for idx1, idx2 in zip(img1_overlay_map, img2_overlay_map):
            bmh1 = self.tile_bm0hash_grids[img1_id][idx1]
            bmh2 = self.tile_bm0hash_grids[img2_id][idx2]
            score = self.fuzzy_compare(bmh1, bmh2)
            scores.append(score)
        return scores

    def get_pixel_scores(self, img1, img2, img1_overlay_tag):
        img1_overlay_map = overlay_tag_maps[img1_overlay_tag]
        img2_overlay_map = overlay_tag_maps[overlay_tag_pairs[img1_overlay_tag]]
        scores = []
        for idx1, idx2 in zip(img1_overlay_map, img2_overlay_map):
            tile1 = self.get_tile(img1, idx1)
            tile2 = self.get_tile(img2, idx2)
            score = self.fuzzy_diff(tile1, tile2)
            scores.append(score)
        return scores

    def find_valid_pairings_by_hash(self, hash_id, img_list, overlap_level):

        overlay_64321_tags = {tag for tag, tiles in overlay_tag_maps.items() if len(tiles) in (overlap_level,)}  # should pass as arg

        for i, img1_id in enumerate(img_list):
            tiles1 = [idx for idx, bm0hash in enumerate(self.tile_bm0hash_grids[img1_id]) if bm0hash == hash_id]
            for j, img2_id in enumerate(img_list):
                if j >= i:
                    continue
                tiles2 = [idx for idx, bm0hash in enumerate(self.tile_bm0hash_grids[img2_id]) if bm0hash == hash_id]

                # create a set of valid overlay_tags based on matching tiles between images.
                overlay_tags = set()
                for t1 in tiles1:
                    for t2 in tiles2:
                        overlay_tags.add(pair_tag_lookup.get((t1, t2)))

                overlay_tags.intersection_update(overlay_64321_tags)

                if len(overlay_tags) == 0:
                    continue

                # only consider unfilled tags
                # unfilled_tags = set()
                # for overlay_tag in overlay_tags:
                #     ois1 = sdc1.overlay_image_scores[overlay_tag]
                #     ois2 = sdc2.overlay_image_scores[overlay_tag_pairs[overlay_tag]]
                #     if ois1 > self.best_score_threshold and ois2 > self.best_score_threshold:
                #         # oin1 = sdc1.overlay_image_names[overlay_tag]
                #         # oin2 = sdc2.overlay_image_names[overlay_tag_pairs[overlay_tag]]
                #         # if oin1 != img2_id or oin2 != img1_id:
                #         #     print('\n', img1_id, img2_id, ois1, ois2)
                #         # raise ValueError
                #         continue
                #     unfilled_tags.add(overlay_tag)

                # only consider unfilled tags
                # unfilled_tags = set()
                # for overlay_tag in overlay_tags:
                #     oin1 = sdc1.overlay_image_names[overlay_tag]
                #     oin2 = sdc2.overlay_image_names[overlay_tag_pairs[overlay_tag]]
                #     if oin1 == img2_id and oin2 == img1_id:
                #         # We've already considered these 2 images for this particular overlay.
                #         continue
                #     unfilled_tags.add(overlay_tag)

                # overlay_tags.intersection_update(unfilled_tags)

                for overlay_tag in overlay_tags:

                    if img1_id < img2_id:  # lexicographic sort
                        self.update_overlay_matches(img1_id, img2_id, overlay_tag)
                    else:
                        self.update_overlay_matches(img2_id, img1_id, overlay_tag_pairs[overlay_tag])

                    # if overlay_score > self.minimum_score_threshold:
                    #     stats = self.compute_stats(sdc1, sdc2, img1, img2, overlay_tag)
                    #     if len(stats) == 0:
                    #         continue
                    #     fuzzy_matches.append(stats)

        return

    def update_overlay_matches(self, img1_id, img2_id, img1_overlay_tag):

        overlay_score = self.get_overlay_score(img1_id, img2_id, img1_overlay_tag)
        if overlay_score < self.minimum_score_threshold:
            return

        tile_scores = self.get_tile_scores(img1_id, img2_id, img1_overlay_tag)
        if min(tile_scores) < self.minimum_score_threshold:
            return

        self.matches[(img1_id, img2_id)].append((img1_overlay_tag, overlay_score, tile_scores))

    def update_overlay_maps(self, img1_id, img2_id, img1_overlay_tag, overlay_score=None, tile_scores=None, pixel_scores=None, entropy_score=None):

        overlay_score = overlay_score or self.get_overlay_score(img1_id, img2_id, img1_overlay_tag)
        if overlay_score < self.minimum_score_threshold:
            return

        tile_scores = tile_scores or self.get_tile_scores(img1_id, img2_id, img1_overlay_tag)
        assert len(tile_scores) == len(overlay_tag_maps[img1_overlay_tag])
        if min(tile_scores) < self.minimum_score_threshold:
            return

        if (img1_id, img2_id) not in self.overlay_image_maps:
            self.overlay_image_maps[(img1_id, img2_id)] = {}
        self.overlay_image_maps[(img1_id, img2_id)][img1_overlay_tag] = (overlay_score, tile_scores, pixel_scores, entropy_score)

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

        img1 = None
        img2 = None

        dup_idx1 = {c: np.array([idx for idx, tidx in enumerate(dup_self1) if tidx == c]) for c in dup_cts1}
        dup_idx2 = {c: np.array([idx for idx, tidx in enumerate(dup_self2) if tidx == c]) for c in dup_cts2}
        dup_tiles1 = np.array([9 if dup_cts1[idx] == 1 else tidx for idx, tidx in enumerate(dup_self1)])
        dup_tiles2 = np.array([9 if dup_cts2[idx] == 1 else tidx for idx, tidx in enumerate(dup_self2)])

        is_updated = False
        for t1 in dup_cts1:
            for t2 in dup_cts2:
                if self.tile_bm0hash_grids[img1_id][t1] != self.tile_bm0hash_grids[img2_id][t2]:
                    continue
                if img1 is None:
                    img1 = self.get_img(img1_id)
                if img2 is None:
                    img2 = self.get_img(img2_id)
                if np.any(self.get_tile(img1, t1) != self.get_tile(img2, t2)):
                    continue

                dup_tiles1[dup_idx1[t1]] = t1
                dup_tiles2[dup_idx2[t2]] = t1

        if (img1_id, img2_id) not in self.image_image_duplicate_tiles:
            self.image_image_duplicate_tiles[(img1_id, img2_id)] = (dup_tiles1, dup_tiles2)
            is_updated = True

        return is_updated

    def compute_stats(self, sdc1, sdc2, img1, img2, img1_overlay_tag):

        stats = {}
        stats['overlay_tag'] = (img1_overlay_tag, overlay_tag_pairs[img1_overlay_tag])
        stats['img_id'] = (sdc1.img_id, sdc2.img_id)
        # stats['overlay_score'] = overlay_score
        # stats['blockMeanHash0'] = tile_scores

        for algo in hash_algos:
            stats[algo] = []

        for idx1, idx2 in overlay_tag_maps[img1_overlay_tag]:
            i, j = tile_idx2ij[idx1]
            k, l = tile_idx2ij[idx2]
            tile1 = sdc1.get_tile(img1, i, j)
            tile2 = sdc2.get_tile(img2, k, l)
            for name, func in hash_algos.items():
                # if name == 'blockMeanHash':
                #     t1 = sdc1.tile_bm0hash_grid[(i, j)]
                #     t2 = sdc2.tile_bm0hash_grid[(k, l)]
                # else:
                t1 = func(tile1)
                t2 = func(tile2)

                if name == 'colorMomentHash':
                    b12 = t1 - t2
                    i12 = np.linalg.norm(b12)
                else:
                    b12 = normalized_hamming_distance(t1, t2)
                    i12 = 1.0 - b12

                stats[name].append(i12)
                # stats[name] += i12

        return stats

    def find_valid_pairings_by_search(self, img_list):
        img_tag_scores = {img_id: self.images[img_id].overlay_image_scores for img_id in img_list}
        for ii, img1_id in enumerate(img_list):
            sdc1 = self.images[img1_id]
            img1 = sdc1.get_img()
            for jj, img2_id in enumerate(img_list):
                if jj <= ii:
                    continue
                sdc2 = self.images[img2_id]
                img2 = sdc2.get_img()

                for img1_overlay_tag, img1_overlay_map in overlay_tag_maps.items():
                    overlay_scores = self.get_overlay_scores(img1, img2, img1_overlay_tag)
                    if min(overlay_scores) > self.best_score_threshold:
                        img_tag_scores[img1_id][img1_overlay_tag] = min(overlay_scores)
                        img_tag_scores[img2_id][overlay_tag_pairs[img1_overlay_tag]] = min(overlay_scores)
                        sdc1.update_overlay_map(sdc2, overlay_scores, img1_overlay_tag)
                        sdc2.update_overlay_map(sdc1, overlay_scores, overlay_tag_pairs[img1_overlay_tag])
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
        self.overlay_image_maps = {tag: defaultdict(str) for tag in overlay_tag_maps}
        self.overlay_image_names = {tag: '' for tag in overlay_tag_maps}
        self.overlay_image_scores = {tag: 0.0 for tag in overlay_tag_maps}
        self.overlay_tile_scores = np.zeros((3, 3, 3, 3), dtype=np.float64)

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
                self._tile_entropy_grid[idx] = get_entropy(tile)
        return self._tile_entropy_grid

    def update_overlay_map(self, other_sdc, tile_scores, overlay_score, tag):

        if self.overlay_image_names[tag] not in ('', other_sdc.img_id) or self.overlay_image_scores[tag] == overlay_score:  # sanity check
            print(tag, self.img_id, self.overlay_image_names[tag], self.overlay_image_scores[tag], other_sdc.img_id, overlay_score)

        self.overlay_image_names[tag] = other_sdc.img_id
        self.overlay_image_scores[tag] = overlay_score

        for (idx1, idx2, s) in zip(overlay_tag_maps[tag], overlay_tag_maps[overlay_tag_pairs[tag]], tile_scores):
            i, j = tile_idx2ij[idx1]
            k, l = tile_idx2ij[idx2]
            self.overlay_tile_scores[i, j, k, l] = s

        self.overlay_image_maps[tag][other_sdc.img_id] = {'overlay_score': overlay_score, 'tile_scores': tile_scores}


def main():
    ship_dir = "data/input"
    train_image_dir = os.path.join(ship_dir, 'train_768')
    image_counter_grids_file = os.path.join("data", "image_counter_grids.pkl")
    image_md5hash_grids_file = os.path.join("data", "image_md5hash_grids.pkl")
    image_bm0hash_grids_file = os.path.join("data", "image_bm0hash_grids.pkl")
    image_entropy_grids_file = os.path.join("data", "image_entropy_grids.pkl")
    image_duplicate_tiles_file = os.path.join("data", "image_duplicate_tiles.txt")
    image_image_duplicate_tiles_file = os.path.join("data", "image_image_duplicate_tiles.txt")

    sdcic = SDCImageContainer(train_image_dir)
    sdcic.load_3x3_grids(
        image_counter_grids_file,
        image_md5hash_grids_file,
        image_bm0hash_grids_file,
        image_entropy_grids_file,
        image_duplicate_tiles_file)

    # n_matching_tiles = 9
    # n_matching_tiles = 6  # 5:14 minutes
    # n_matching_tiles = 4  # 12:52 minutes
    n_matching_tiles = 3  # 15:43 minutes
    # n_matching_tiles = 2  # 26:27 minutes
    overlay_matches_file = os.path.join("data", f"overlay_matches_{n_matching_tiles}.pkl")
    overlay_pixel_scores_file = os.path.join("data", f"overlay_pixel_scores_{n_matching_tiles}.pkl")

    if not os.path.exists(overlay_matches_file):

        img_ids = os.listdir(train_image_dir)
        # TODO: Use filter for all overlays here? or just n_matching_tiles?
        img_ids = filter_duplicates(img_ids)

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
            img_list = list(sorted(sorted_hash_dict[hash_id]))
            sdcic.find_valid_pairings_by_hash(hash_id, img_list, n_matching_tiles)

        matches_list = []
        for key, values in sdcic.matches.items():
            for row in values:
                matches_list.append((key[0], key[1], row[0], row[1], *row[2]))

        df = pd.DataFrame(matches_list)
        df.to_pickle(overlay_matches_file)

    df = pd.read_pickle(overlay_matches_file)
    for row in tqdm(df.to_dict('split')['data']):
        sdcic.matches[(row[0], row[1])].append((row[2], row[3], row[4:]))

    if not os.path.exists(overlay_pixel_scores_file):

        img1_id_old = None
        image_cache = {}
        overlay_pixel_scores_list = []
        for (img1_id, img2_id), values in tqdm(sorted(sdcic.matches.items())):

            img1_overlay_tags = set([v[0] for v in values])
            img1 = image_cache.setdefault(img1_id, sdcic.get_img(img1_id))
            img2 = image_cache.setdefault(img2_id, sdcic.get_img(img2_id))

            for img1_overlay_tag in img1_overlay_tags:
                pixel_scores = sdcic.get_pixel_scores(img1, img2, img1_overlay_tag)
                overlay_pixel_scores_list.append((img1_id, img2_id, img1_overlay_tag, *pixel_scores))

            if img1_id_old is None:
                img1_id_old = img1_id
            if img1_id > img1_id_old:
                del image_cache[img1_id_old]
                img1_id_old = img1_id
            if len(image_cache) > 20000:
                image_cache = {}
                img1_id_old = None

        df = pd.DataFrame(overlay_pixel_scores_list)
        df.to_pickle(overlay_pixel_scores_file)

    df = pd.read_pickle(overlay_pixel_scores_file)
    for row in tqdm(df.to_dict('split')['data']):
        if (row[0], row[1]) not in sdcic.overlay_pixel_scores:
            sdcic.overlay_pixel_scores[(row[0], row[1])] = {}
        assert row[2] not in sdcic.overlay_pixel_scores[(row[0], row[1])]
        sdcic.overlay_pixel_scores[(row[0], row[1])][row[2]] = row[3:]

    sdcic.image_image_duplicate_tiles = read_image_image_duplicate_tiles(image_image_duplicate_tiles_file)

    updated_image_image_duplicate_tiles = {}
    for img_id12, values in tqdm(sorted(sdcic.matches.items())):
        img1_overlay_tags = set([v[0] for v in values])  # all have the same overlay_tag
        if len(values) == 0 or len(img1_overlay_tags) != 1:
            continue
        img1_overlay_tag, overlay_score, tile_scores = values[0]
        sdcic.update_overlay_maps(img_id12[0], img_id12[1], img1_overlay_tag, overlay_score, tile_scores)
        if img_id12 in sdcic.image_image_duplicate_tiles:
            continue
        is_updated = sdcic.update_image_image_duplicate_tiles(img_id12[0], img_id12[1])
        if not is_updated:
            continue
        updated_image_image_duplicate_tiles[img_id12] = sdcic.image_image_duplicate_tiles[img_id12]

    if len(updated_image_image_duplicate_tiles) > 0:
        update_image_image_duplicate_tiles(image_image_duplicate_tiles_file, updated_image_image_duplicate_tiles)


if __name__ == '__main__':
    main()
    print('done')
