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
from utils import normalized_hamming_distance
from utils import generate_overlay_tag_slices
from utils import generate_pair_tag_lookup

overlay_tag_slices = generate_overlay_tag_slices()

pair_tag_lookup = generate_pair_tag_lookup()

overlay_map_counter = Counter()

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
    ctr_norm = {k: v/img_size for k, v in ctr.items()}
    ctr_entropy = {k: -v*np.log(v) for k, v in ctr_norm.items()}
    entropy = np.sum([k * v for k, v in ctr_entropy.items()])
    return entropy


def get_entropy(img):
    img_grad = np.gradient(img.astype(np.int), axis=(0, 1))
    entropy_vec = []
    for channel_grad in img_grad:
        ctr = Counter(np.abs(channel_grad).flatten())
        entropy_vec.append(get_channel_entropy(ctr, img.size))
    return np.array(entropy_vec)


class SDCImageContainer(dict):

    def __init__(self, train_image_dir, **kwargs):
        # This class assumes all images are square and can be divided perfectly by the tile_size.
        super().__init__(**kwargs)
        self.train_image_dir = train_image_dir
        self.h5_file = train_image_dir + '.h5'
        self.sz = 256
        self.tile_score_max = self.sz * self.sz * 3 * 255
        self.tile_slice = slice(8, -8)
        self.tile_hash_len = 32
        # self.tile_hash_dtype = f'<U{self.tile_hash_len}'
        self.tile_hash_dtype = np.uint8
        self.minimum_score_threshold = 0.95  # overlay score has to be at least this good before assigning it to an image
        self.best_score_threshold = 0.99  # after this, don't have to check if better score exists.
        self.matches = defaultdict(list)

    def load_3x3_grids(self, filename_hash, filename_entropy):
        img_hash_grids = {}
        if os.path.exists(filename_hash):
            df = pd.read_pickle(filename_hash)
            img_hash_grids = {key: val for key, val in df.to_dict('split')['data']}

        img_entropy_grids = {}
        if os.path.exists(filename_entropy):
            df = pd.read_pickle(filename_entropy)
            img_entropy_grids = {key: val for key, val in df.to_dict('split')['data']}

        hh = 0
        ee = 0
        img_ids = os.listdir(self.train_image_dir)
        img_ids = filter_duplicates(img_ids)

        hash_records = []
        entropy_records = []
        for img_id in tqdm(img_ids):

            img = None

            prev_hash_grid = img_hash_grids.get(img_id)
            if prev_hash_grid is None:
                hh += 1
                img = self.get_img(os.path.join(self.train_image_dir, img_id))
                prev_hash_grid = np.zeros((3, 3, self.tile_hash_len), dtype=self.tile_hash_dtype)
                for i in range(3):
                    for j in range(3):
                        tile = self.get_tile(img, i, j)
                        prev_hash_grid[i, j] = img_hash.blockMeanHash(tile, mode=0)[0]
            tile_hash_grid = {}
            for i in range(3):
                for j in range(3):
                    tile_hash_grid[(i, j)] = tuple(prev_hash_grid[i, j])
            hash_records.append({'ImageId': img_id, 'hash_grid': prev_hash_grid})  # int

            prev_entropy_grid = img_entropy_grids.get(img_id)
            if prev_entropy_grid is None:
                ee += 1
                if img is None:
                    img = self.get_img(os.path.join(self.train_image_dir, img_id))
                prev_entropy_grid = np.zeros((3, 3, 2), dtype=np.float)
                for i in range(3):
                    for j in range(3):
                        tile = self.get_tile(img, i, j)
                        prev_entropy_grid[i, j] = get_entropy(tile)
            tile_entropy_grid = {}
            for i in range(3):
                for j in range(3):
                    tile_entropy_grid[(i, j)] = prev_entropy_grid[i, j]
            entropy_records.append({'ImageId': img_id, 'entropy_grid': prev_entropy_grid})  # int

            sdc_image = SDCImage(img_id, self.train_image_dir,
                                 tile_slice=self.tile_slice,
                                 tile_hash_len=self.tile_hash_len,
                                 tile_hash_grid=tile_hash_grid,
                                 tile_hash_dtype=self.tile_hash_dtype,
                                 tile_entropy_grid=tile_entropy_grid)

            self.__setitem__(img_id, sdc_image)

            if hh >= 5000:
                df = pd.DataFrame().append(hash_records)
                df.to_pickle(filename_hash)
                hh = 0

            if ee >= 1000:
                df = pd.DataFrame().append(entropy_records)
                df.to_pickle(filename_entropy)
                ee = 0

        if hh > 0:
            df = pd.DataFrame().append(hash_records)
            df.to_pickle(filename_hash)

        if ee > 0:
            df = pd.DataFrame().append(entropy_records)
            df.to_pickle(filename_entropy)

    def get_img(self, filename):
        return cv2.imread(filename)

    def get_tile(self, img, i, j):
        return img[i * self.sz:(i + 1) * self.sz, j * self.sz:(j + 1) * self.sz, :]

    def fuzzy_compare(self, tile1, tile2):
        maxab = np.max(np.stack([tile1, tile2]), axis=0)
        n = np.prod(maxab.shape)
        a = maxab - tile2
        b = maxab - tile1
        ab = a + b
        return np.sum(255 - ab) / (255 * n)

    def check_exact_match(self, img1, img2, overlay_tag):
        slice1 = overlay_tag_slices[overlay_tag]
        slice2 = overlay_tag_slices[overlay_tag_pairs[overlay_tag]]
        overlay1 = img1[slice1[0], slice1[1]]
        overlay2 = img2[slice2[0], slice2[1]]
        is_perfect_score = np.all(overlay1 == overlay2)
        return is_perfect_score

    def get_bmh_score(self, img1, img2, overlay_tag):
        slice1 = overlay_tag_slices[overlay_tag]
        slice2 = overlay_tag_slices[overlay_tag_pairs[overlay_tag]]
        overlay1 = img1[slice1[0], slice1[1]]
        overlay2 = img2[slice2[0], slice2[1]]
        bmh1 = img_hash.blockMeanHash(overlay1, mode=0)
        bmh2 = img_hash.blockMeanHash(overlay2, mode=0)
        score = self.fuzzy_compare(bmh1, bmh2)
        return score

    def get_overlay_score(self, sdc1, sdc2, overlay_tag):
        overlay_map1 = overlay_tag_maps[overlay_tag]
        overlay_map2 = overlay_tag_maps[overlay_tag_pairs[overlay_tag]]
        bmh1_list = []
        bmh2_list = []
        for ((i, j), (k, l)) in zip(overlay_map1, overlay_map2):
            bmh1 = sdc1.tile_hash_grid[(i, j)]
            bmh2 = sdc2.tile_hash_grid[(k, l)]
            bmh1_list.append(bmh1)
            bmh2_list.append(bmh2)
        bmh1_arr = np.vstack(bmh1_list)
        bmh2_arr = np.vstack(bmh2_list)
        score = self.fuzzy_compare(bmh1_arr, bmh2_arr)
        return score

    def get_tile_scores(self, sdc1, sdc2, overlay_tag):
        overlay_map1 = overlay_tag_maps[overlay_tag]
        overlay_map2 = overlay_tag_maps[overlay_tag_pairs[overlay_tag]]
        scores = []
        for ((i, j), (k, l)) in zip(overlay_map1, overlay_map2):
            bmh1 = sdc1.tile_hash_grid[(i, j)]
            bmh2 = sdc2.tile_hash_grid[(k, l)]
            score = self.fuzzy_compare(bmh1, bmh2)
            scores.append(score)
        return scores

    def find_valid_pairings_by_hash(self, hash_id, img_list, overlap_level):

        overlay_64321_tags = {tag for tag, tiles in overlay_tag_maps.items() if len(tiles) in (overlap_level,)}  # should pass as arg

        for i, img_id1 in enumerate(img_list):
            sdc1 = self.__getitem__(img_id1)
            tiles1 = [key for key, value in sdc1.tile_hash_grid.items() if value == hash_id]
            for j, img_id2 in enumerate(img_list):
                if j >= i:
                    continue
                sdc2 = self.__getitem__(img_id2)
                tiles2 = [key for key, value in sdc2.tile_hash_grid.items() if value == hash_id]

                # create a set of valid overlay_tags based on matching tiles between images.
                overlay_tags = set()
                for tile1 in tiles1:
                    for tile2 in tiles2:
                        pair_key = tuple([tuple(tile1), tuple(tile2)])
                        overlay_tags.add(pair_tag_lookup.get(pair_key))

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
                #         # if oin1 != sdc2.img_id or oin2 != sdc1.img_id:
                #         #     print('\n', sdc1.img_id, sdc2.img_id, ois1, ois2)
                #         # raise ValueError
                #         continue
                #     unfilled_tags.add(overlay_tag)

                # only consider unfilled tags
                # unfilled_tags = set()
                # for overlay_tag in overlay_tags:
                #     oin1 = sdc1.overlay_image_names[overlay_tag]
                #     oin2 = sdc2.overlay_image_names[overlay_tag_pairs[overlay_tag]]
                #     if oin1 == sdc2.img_id and oin2 == sdc1.img_id:
                #         # We've already considered these 2 images for this particular overlay.
                #         continue
                #     unfilled_tags.add(overlay_tag)

                # overlay_tags.intersection_update(unfilled_tags)

                for overlay_tag in overlay_tags:

                    if sdc1.img_id < sdc2.img_id:  # lexigraphic sort
                        self.update_overlay_matches(sdc1, sdc2, overlay_tag)
                    else:
                        self.update_overlay_matches(sdc2, sdc1, overlay_tag_pairs[overlay_tag])

                    # if overlay_score > self.minimum_score_threshold:
                    #     stats = self.compute_stats(sdc1, sdc2, img1, img2, overlay_tag)
                    #     if len(stats) == 0:
                    #         continue
                    #     fuzzy_matches.append(stats)

        return

    def update_overlay_matches(self, sdc1, sdc2, overlay_tag):

        overlay_score = self.get_overlay_score(sdc1, sdc2, overlay_tag)
        if overlay_score < self.minimum_score_threshold:
            return

        tile_scores = self.get_tile_scores(sdc1, sdc2, overlay_tag)
        if min(tile_scores) < self.minimum_score_threshold:
            return

        self.matches[(sdc1.img_id, sdc2.img_id)].append((overlay_tag, overlay_score, tile_scores))

    def update_overlay_maps(self, sdc1, sdc2, overlay_tag, overlay_score=None, tile_scores=None):

        overlay_score = overlay_score or self.get_overlay_score(sdc1, sdc2, overlay_tag)

        # if sdc1.overlay_image_scores[overlay_tag] != sdc2.overlay_image_scores[overlay_tag_pairs[overlay_tag]]:
        #     print(sdc1.overlay_image_scores[overlay_tag])
        #     print(sdc2.overlay_image_scores[overlay_tag_pairs[overlay_tag]])

        if overlay_score < self.minimum_score_threshold:
            return

        # if overlay_score < sdc1.overlay_image_scores[overlay_tag]:
        #     return

        if tile_scores is not None or len(tile_scores) != len(overlay_tag_maps[overlay_tag]):
            tile_scores = self.get_tile_scores(sdc1, sdc2, overlay_tag)

        if min(tile_scores) < self.minimum_score_threshold:
            return

        sdc1.update_overlay_map(sdc2, tile_scores, overlay_score, overlay_tag)
        sdc2.update_overlay_map(sdc1, tile_scores, overlay_score, overlay_tag_pairs[overlay_tag])

    def compute_stats(self, sdc1, sdc2, img1, img2, overlay_tag):

        stats = {}
        stats['overlay_tag'] = (overlay_tag, overlay_tag_pairs[overlay_tag])
        stats['img_id'] = (sdc1.img_id, sdc2.img_id)
        # stats['overlay_score'] = overlay_score
        # stats['blockMeanHash0'] = tile_scores

        for algo in hash_algos:
            stats[algo] = []

        for s, ((i, j), (k, l)) in enumerate(overlay_tag_maps[overlay_tag]):

            tile1 = sdc1.get_tile(img1, i, j)
            tile2 = sdc2.get_tile(img2, k, l)
            for name, func in hash_algos.items():
                # if name == 'blockMeanHash':
                #     t1 = sdc1.tile_hash_grid[(i, j)]
                #     t2 = sdc2.tile_hash_grid[(k, l)]
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
        img_tag_scores = {img_id: self.__getitem__(img_id).overlay_image_scores for img_id in img_list}
        for ii, img_id1 in enumerate(img_list):
            sdc1 = self.__getitem__(img_id1)
            img1 = sdc1.get_img()
            for jj, img_id2 in enumerate(img_list):
                if jj <= ii:
                    continue
                sdc2 = self.__getitem__(img_id2)
                img2 = sdc2.get_img()

                for overlay_tag, overlay_map in overlay_tag_maps.items():
                    overlay_scores = self.get_overlay_scores(img1, img2, overlay_map)
                    if min(overlay_scores) > self.best_score_threshold:
                        img_tag_scores[img_id1][overlay_tag] = min(overlay_scores)
                        img_tag_scores[img_id2][overlay_tag_pairs[overlay_tag]] = min(overlay_scores)
                        sdc1.update_overlay_map(sdc2, overlay_scores, overlay_tag)
                        sdc2.update_overlay_map(sdc1, overlay_scores, overlay_tag_pairs[overlay_tag])
                        break

            print(f'{ii}/{len(img_list)}')
        return img_tag_scores


class SDCImage:

    def __init__(self, img_id, train_image_dir,
                 tile_size=256,
                 tile_slice=None,
                 tile_hash_len=None,
                 tile_hash_grid=None,
                 tile_hash_dtype=None,
                 tile_entropy_grid=None):

        # This class assumes the image is square and can be divided perfectly by the tile_size.
        self.img_id = img_id
        self.filename = os.path.join(train_image_dir, img_id)
        self.sz = tile_size
        self.tile_slice = tile_slice
        self.tile_hash_len = tile_hash_len
        self.tile_hash_dtype = tile_hash_dtype
        self._tile_hash_grid = tile_hash_grid
        self._tile_entropy_grid = tile_entropy_grid

        self.overlay_image_names = {tag: '' for tag, tiles in overlay_tag_maps.items()}
        self.overlay_image_scores = {tag: 0.0 for tag, tiles in overlay_tag_maps.items()}
        self.overlay_tile_scores = np.zeros((3, 3, 3, 3), dtype=np.float64)

        self.overlay_image_maps = {tag: defaultdict(str) for tag, tiles in overlay_tag_maps.items()}

    def get_img(self):
        return cv2.imread(self.filename)

    def get_tile(self, img, i, j):
        return img[i * self.sz:(i + 1) * self.sz, j * self.sz:(j + 1) * self.sz, :]

    @property
    def n_mapped_tile_pairs(self):
        return np.round(np.sum(self.overlay_tile_scores))

    @property
    def tile_hash_grid(self):
        if self._tile_hash_grid is None:
            # self._tile_hash_grid = np.zeros((3, 3, self.tile_hash_len), dtype=self.tile_hash_dtype)
            self._tile_hash_grid = {}
            img = self.get_img()
            for i in range(3):
                for j in range(3):
                    tile = self.get_tile(img, i, j)
                    self._tile_hash_grid[(i, j)] = img_hash.blockMeanHash(tile, mode=0)
        return self._tile_hash_grid

    @property
    def tile_entropy_grid(self):
        if self._tile_entropy_grid is None:
            # self._tile_entropy_grid = np.zeros((3, 3, 3), dtype=np.float)
            self._tile_entropy_grid = {}
            img = self.get_img()
            for i in range(3):
                for j in range(3):
                    tile = self.get_tile(img, i, j)
                    self._tile_entropy_grid[(i, j)] = get_entropy(tile)
        return self._tile_entropy_grid

    def update_overlay_map(self, other_sdc, tile_scores, overlay_score, tag):

        if self.overlay_image_names[tag] not in ('', other_sdc.img_id) or self.overlay_image_scores[tag] == overlay_score:  # sanity check
            print(tag, self.img_id, self.overlay_image_names[tag], self.overlay_image_scores[tag], other_sdc.img_id, overlay_score)

        self.overlay_image_names[tag] = other_sdc.img_id
        self.overlay_image_scores[tag] = overlay_score

        for ((i, j), (k, l), s) in zip(overlay_tag_maps[tag], overlay_tag_maps[overlay_tag_pairs[tag]], tile_scores):
            self.overlay_tile_scores[i, j, k, l] = s

        self.overlay_image_maps[tag][other_sdc.img_id] = {'overlay_score': overlay_score, 'tile_scores': tile_scores}

        overlay_map_counter[tag] += 1


def main():
    ship_dir = ".data/input"
    train_image_dir = os.path.join(ship_dir, 'train_768')
    image_hash_grids_file = os.path.join(ship_dir, "image_hash_grids.pkl")
    image_entropy_grids_file = os.path.join(ship_dir, "image_entropy_grids.pkl")

    sdc_images = SDCImageContainer(train_image_dir)
    sdc_images.load_3x3_grids(image_hash_grids_file, image_entropy_grids_file)

    n_matching_tiles = 6  # 5:14 minutes
    overlay_matches_filename = f'overlay_matches_{n_matching_tiles}.pkl'

    if not os.path.exists(overlay_matches_filename):

        img_ids = os.listdir(train_image_dir)
        img_ids = filter_duplicates(img_ids)

        hash_dict = defaultdict(set)
        for img_id in tqdm(img_ids):
            for h in sdc_images[img_id].tile_hash_grid.values():
                hash_dict[h].add(img_id)

        sorted_hash_dict = {}
        for key, dups in sorted(hash_dict.items(), key=lambda x: len(x[1]), reverse=True):
            if len(dups) > 1:
                sorted_hash_dict[key] = dups

        hash_ids = list(sorted_hash_dict)
        for hash_id in tqdm(hash_ids):
            img_list = list(sorted(sorted_hash_dict[hash_id]))
            sdc_images.find_valid_pairings_by_hash(hash_id, img_list, n_matching_tiles)

        matches_list = []
        for key, values in sdc_images.matches.items():
            for row in values:
                matches_list.append((key[0], key[1], row[0], row[1], min(row[2])))
        df = pd.DataFrame(matches_list)
        df.to_pickle(overlay_matches_filename)

    else:

        df = pd.read_pickle(overlay_matches_filename)
        sdc_images.matches = defaultdict(list)
        for row in tqdm(df.to_dict('split')['data']):
            sdc_images.matches[(row[0], row[1])].append((row[2], row[3], [row[4]]))

    for key, values in sorted(sdc_images.matches.items(), key=lambda x: x[0]):
        n_hashes = len(values)
        p0 = len(set([v[0] for v in values]))  # all have the same overlay_tag
        # p1 = all([v[1] == 1 for v in values])  # all have perfect overlay_score
        # p2 = all([v[2] == 1 for v in values])  # all have perfect tile scores
        if n_hashes >= 1 and p0 == 1:  # and p1 and p2:
            sdc_images.update_overlay_maps(sdc_images[key[0]], sdc_images[key[1]], values[0][0], overlay_score=values[0][1], tile_scores=values[0][2])

    print(overlay_map_counter)


if __name__ == '__main__':
    main()
    print('done')
