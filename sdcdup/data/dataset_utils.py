import operator
import os
from collections import namedtuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm

from sdcdup.utils import generate_tag_pair_lookup
from sdcdup.utils import far_away_corners
from sdcdup.utils import get_hamming_distance
from sdcdup.utils import fuzzy_join
from sdcdup.utils import get_tile
from sdcdup.utils import get_img
from sdcdup.utils import to_hls
from sdcdup.utils import to_bgr
from sdcdup.utils import idx_chan_map
from sdcdup.utils import hls_shift
from sdcdup.utils import load_duplicate_truth
from sdcdup.utils import interim_data_dir
from sdcdup.utils import train_image_dir
from sdcdup.utils import train_tile_dir

from sdcdup.features import SDCImageContainer


def create_256_tiles(train_image_dir, train_tile_dir):
    n_tiles = 9
    os.makedirs(train_tile_dir, exist_ok=True)
    img_ids = os.listdir(train_image_dir)
    for img_id in tqdm(img_ids):
        img = None
        filebase, fileext = img_id.split('.')
        for idx in range(n_tiles):
            outfile = os.path.join(train_tile_dir, f'{filebase}_{idx}.{fileext}')
            if os.path.exists(outfile):
                continue
            if img is None:
                img = cv2.imread(os.path.join(train_image_dir, img_id))
            tile = get_tile(img, idx)
            cv2.imwrite(outfile, tile)


def create_dataset_from_tiles():
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
    sdcic = SDCImageContainer()
    sdcic.preprocess_image_properties()

    img_overlap_pairs_dup_keys = []
    img_overlap_pairs_non_dup_all = []

    KeyScore = namedtuple('keyscore', 'key score')
    for img_id, tile_md5hash_grid in tqdm(sdcic.img_metrics['md5'].items()):
        for idx1, tile1_md5hash in enumerate(tile_md5hash_grid):
            for idx2, tile2_md5hash in enumerate(tile_md5hash_grid):

                if idx1 > idx2:
                    continue

                tile1_issolid = np.all(sdcic.img_metrics['sol'][img_id][idx1] >= 0)
                tile2_issolid = np.all(sdcic.img_metrics['sol'][img_id][idx2] >= 0)

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

                bmh1 = sdcic.img_metrics['bmh'][img_id][idx1]
                bmh2 = sdcic.img_metrics['bmh'][img_id][idx2]
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


def create_dataset_from_tiles_and_truth():

    sdcic = SDCImageContainer()
    sdcic.preprocess_image_properties()

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
            if sdcic.img_metrics['md5'][img1_id][idx1] == sdcic.img_metrics['md5'][img2_id][idx2]:
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
        df = pd.read_pickle(os.path.join(interim_data_dir, f'overlap_bmh_tile_scores_{n_matching_tiles}.pkl'))
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
            # The score has to be lower than sdcic.matches_threshold.
            if min(far_scores) > sdcic.matches_threshold:
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
        # The score has to be lower than sdcic.matches_threshold.
        if sdcic.get_bmh_scores(img1_id, img2_id, img1_overlap_tag)[0] > sdcic.matches_threshold:
            continue
        # Keep the one that has the lowest score.
        idx1, idx2 = tpl[img1_overlap_tag][0]
        img_overlap_pairs.append((img1_id, img2_id, idx1, idx2, 0))
        if len(img_overlap_pairs) > 2 * n_dup_tile_pairs:
            break

    print(f"Number of non-dup/dup tiles: {len(img_overlap_pairs) - n_dup_tile_pairs:>8}/{n_dup_tile_pairs}")

    return img_overlap_pairs


class TrainDataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, img_overlaps, train_or_valid, image_transform,
                 in_shape=(6, 256, 256),
                 out_shape=(1,)):

        """Initialization"""
        self.img_overlaps = img_overlaps
        # TODO: handle case if train_or_valid == 'test'
        self.valid = train_or_valid == 'valid'
        self.image_transform = image_transform
        self.ij = ((0, 0), (0, 1), (0, 2),
                   (1, 0), (1, 1), (1, 2),
                   (2, 0), (2, 1), (2, 2))

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hls_limits = {'H': 10, 'L': 20, 'S': 20}
        if self.valid:
            self.img_augs = [self.get_random_augmentation() for _ in self.img_overlaps]

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.img_overlaps)

    def __getitem__(self, index):
        """Generates one sample of data"""
        if self.valid:
            img_aug = self.img_augs[index]
        else:
            img_aug = self.get_random_augmentation()
        return self.get_data_pair(self.img_overlaps[index], img_aug)  # X, y

    def get_random_augmentation(self):

        # So, we aren't always biasing the 'second' image with hls shifting...
        flip_img_order = np.random.random() > 0.5
        # The first tile will always come from either a slice of the image or from the saved slice.
        first_from_large = np.random.random() > 0.5
        second_from_large = np.random.random() > 0.5
        second_augment_hls = np.random.random() > 0.25
        flip_stacking_order = np.random.random() > 0.5

        hls_idx = np.random.choice(3)
        hls_chan = idx_chan_map[hls_idx]
        hls_gain = np.random.choice(self.hls_limits[hls_chan]) + 1
        hls_gain = hls_gain if np.random.random() > 0.5 else -1 * hls_gain

        return flip_img_order, first_from_large, second_from_large, second_augment_hls, hls_chan, hls_gain, flip_stacking_order

    def color_shift(self, img, chan, gain):
        hls = to_hls(img)
        hls_shifted = hls_shift(hls, chan, gain)
        return to_bgr(hls_shifted)

    def get_tile(self, img, idx, sz=256):
        i, j = self.ij[idx]
        return img[i * sz:(i + 1) * sz, j * sz:(j + 1) * sz, :]

    def read_from_large(self, img_id, idx):
        img = cv2.imread(os.path.join(train_image_dir, img_id))
        return self.get_tile(img, idx)

    def read_from_small(self, img_id, idx):
        filebase, fileext = img_id.split('.')
        tile_id = f'{filebase}_{idx}.{fileext}'
        return cv2.imread(os.path.join(train_tile_dir, tile_id))

    def get_data_pair(self, img_overlap, img_aug):

        flip_img_order, first_from_large, second_from_large, aug_hls, chan, gain, flip_stacking_order = img_aug
        if flip_img_order:
            img2_id, img1_id, idx2, idx1, is_dup = img_overlap
        else:
            img1_id, img2_id, idx1, idx2, is_dup = img_overlap

        read1 = self.read_from_large if first_from_large else self.read_from_small
        read2 = self.read_from_large if second_from_large else self.read_from_small
        same_image = img1_id == img2_id

        if same_image:  # img1_id == img2_id
            if is_dup:  # idx1 == idx2
                tile1 = read1(img1_id, idx1)
                if aug_hls:
                    tile2 = self.color_shift(tile1, chan, gain)
                else:
                    tile2 = read2(img2_id, idx2)
            else:  # idx1 != idx2
                if first_from_large and second_from_large:
                    img = cv2.imread(os.path.join(train_image_dir, img1_id))
                    tile1 = self.get_tile(img, idx1)
                    tile2 = self.get_tile(img, idx2)
                else:
                    tile1 = read1(img1_id, idx1)
                    tile2 = read2(img2_id, idx2)
        else:  # img1_id != img2_id
            tile1 = read1(img1_id, idx1)
            tile2 = read2(img2_id, idx2)

        # if is_dup == 0 and sdcic.img_metrics['md5'][img1_id][idx1] == sdcic.img_metrics['md5'][img2_id][idx2]:
        #     print(f'same_image, is_dup: {same_image*1}, {is_dup}')
        #     print(f'{img1_id} {idx1} -> ({self.ij[idx1][0]},{self.ij[idx1][1]})')
        #     print(f'{img2_id} {idx2} -> ({self.ij[idx2][0]},{self.ij[idx2][1]})')
        #     is_dup = 1

        tile1 = cv2.cvtColor(tile1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        tile2 = cv2.cvtColor(tile2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.

        X = np.dstack([tile2, tile1]) if flip_stacking_order else np.dstack([tile1, tile2])
        X = self.image_transform(X)
        y = np.array([is_dup], dtype=np.float32)
        return X, y


class EvalDataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, tile_pairs,
                 image_transform=None,
                 in_shape=(6, 256, 256),
                 out_shape=(1,)):
        """Initialization"""
        self.sz = 256
        self.tile_pairs = tile_pairs
        self.image_transform = image_transform
        self.ij = ((0, 0), (0, 1), (0, 2),
                   (1, 0), (1, 1), (1, 2),
                   (2, 0), (2, 1), (2, 2))

        self.in_shape = in_shape
        self.out_shape = out_shape

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.tile_pairs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        tp = self.tile_pairs[index]

        img1 = get_img(tp.img1_id)
        img2 = get_img(tp.img2_id)

        tile1 = cv2.cvtColor(self.get_tile(img1, *self.ij[tp.idx1]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        tile2 = cv2.cvtColor(self.get_tile(img2, *self.ij[tp.idx2]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.

        X = np.dstack([tile1, tile2])
        X = X.transpose((2, 0, 1))
        X = torch.from_numpy(X)
        return X

    def get_tile(self, img, i, j):
        return img[i * self.sz:(i + 1) * self.sz, j * self.sz:(j + 1) * self.sz, :]


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(b))
