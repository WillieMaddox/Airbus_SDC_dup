
import os
import time
from collections import Counter
from collections import defaultdict

import numpy as np
import networkx as nx
from tqdm import tqdm

from sdcdup.utils import overlap_tag_maps
from sdcdup.utils import overlap_tag_pairs
from sdcdup.utils import get_project_root
from sdcdup.utils import generate_tag_pair_lookup
from sdcdup.utils import generate_third_party_overlaps
from sdcdup.utils import load_duplicate_truth
from sdcdup.features import SDCImageContainer


project_root = get_project_root()
raw_data_dir = os.path.join(project_root, os.getenv('RAW_DATA_DIR'))
interim_data_dir = os.path.join(project_root, os.getenv('INTERIM_DATA_DIR'))

third_party_overlaps = generate_third_party_overlaps()
tag_pair_lookup = generate_tag_pair_lookup()


class SDCImage:

    def __init__(self, img_id):
        # This class assumes the image is square and can be divided perfectly by the tile_size.
        self._id = img_id
        self.overlaps = {overlap_tag: tuple() for overlap_tag in list(overlap_tag_maps)}

    def add_overlap(self, img2_id, img12_overlap_tag):
        self.overlaps[img12_overlap_tag] = tuple((img2_id,))

    @property
    def img_id(self):
        return self._id

    def __eq__(self, other):
        return isinstance(other, SDCImage) and self._id == other._id

    def __hash__(self):
        return self._id

    def __repr__(self):
        return self._id


def check_overlap(img1_id, img2_id, img12_overlap_tag, overlap_groups, overlap_image_maps, non_dups=None):
    non_dups = non_dups if non_dups else {}
    img32_overlap_tags = third_party_overlaps[img12_overlap_tag]
    img13_overlap_tags = img32_overlap_tags[::-1]
    missing_third_party_matches = set()
    for img13_overlap_tag, img32_overlap_tag in zip(img13_overlap_tags, img32_overlap_tags):
        for img3_id in overlap_groups[img1_id].overlaps[img13_overlap_tag]:
            if img3_id == img2_id:
                continue

            img23_overlap_tag = overlap_tag_pairs[img32_overlap_tag]

            if img3_id < img2_id:
                if len(overlap_groups[img3_id].overlaps[img32_overlap_tag]) != 0:
                    if overlap_groups[img3_id].overlaps[img32_overlap_tag][0] != img2_id:
                        return False, missing_third_party_matches

                if img3_id in non_dups:
                    if img32_overlap_tag in non_dups[img3_id]:
                        if img2_id in non_dups[img3_id][img32_overlap_tag]:
                            return False, missing_third_party_matches
                try:
                    scores = overlap_image_maps[(img3_id, img2_id, img32_overlap_tag)]
                except KeyError:
                    missing_third_party_matches.add((img3_id, img2_id, img32_overlap_tag))
                    continue

                if np.min(scores.dnn) < 0.5:
                    return False, missing_third_party_matches
            else:
                if len(overlap_groups[img2_id].overlaps[img23_overlap_tag]) != 0:
                    if overlap_groups[img2_id].overlaps[img23_overlap_tag][0] != img3_id:
                        return False, missing_third_party_matches

                if img2_id in non_dups:
                    if img23_overlap_tag in non_dups[img2_id]:
                        if img3_id in non_dups[img2_id][img23_overlap_tag]:
                            return False, missing_third_party_matches
                try:
                    scores = overlap_image_maps[(img2_id, img3_id, img23_overlap_tag)]
                except KeyError:
                    missing_third_party_matches.add((img2_id, img3_id, img23_overlap_tag))
                    continue

                if np.min(scores.dnn) < 0.5:
                    return False, missing_third_party_matches

    return True, missing_third_party_matches


def add_overlap(img1_id, img2_id, img12_overlap_tag, overlap_groups):
    img32_overlap_tags = third_party_overlaps[img12_overlap_tag]
    img13_overlap_tags = img32_overlap_tags[::-1]
    for img13_overlap_tag, img32_overlap_tag in zip(img13_overlap_tags, img32_overlap_tags):
        for img3_id in overlap_groups[img1_id].overlaps[img13_overlap_tag]:
            if img3_id == img2_id:
                continue
            overlap_groups[img3_id].add_overlap(img2_id, img32_overlap_tag)
    overlap_groups[img1_id].add_overlap(img2_id, img12_overlap_tag)


if __name__ == '__main__':

    t0 = time.time()

    matches_files = [
        'matches_bmh32_0.9_offset.csv',
        'matches_bmh96_0.9_offset.csv',
        'matches_bmh32_0.8.csv',
        'matches_bmh96_0.8.csv',
    ]
    sdcic = SDCImageContainer()
    score_types = ['avg', 'pix', 'dnn']
    overlap_image_maps = sdcic.load_image_overlap_properties(matches_files, score_types=['bmh96', 'dnn'])
    dup_truth = load_duplicate_truth()

    solid_hashes = {'b06a8fb9', 'b8e3e4c9', '715bd1bf', '232b4413', '571ea5a6'}  # the blues
    for img_id, tile_issolid_grid in sdcic.img_metrics['sol'].items():
        idxs = set(np.where(tile_issolid_grid >= 0)[0])
        for idx in idxs:
            if np.all(tile_issolid_grid[idx] >= 0):
                solid_hashes.add(sdcic.img_metrics['md5'][img_id][idx])

    non_dups = defaultdict(lambda: defaultdict(set))
    for (img1_id, img2_id, img12_overlap_tag), is_dup in tqdm(dup_truth.items()):
        img21_overlap_tag = overlap_tag_pairs[img12_overlap_tag]
        if not is_dup:
            non_dups[img1_id][img12_overlap_tag].add(img2_id)
            non_dups[img2_id][img21_overlap_tag].add(img1_id)
            continue

    sort_scores = {}
    for k, s in tqdm(overlap_image_maps.items()):
        sort_scores[k] = (np.mean(s.avg)/(16*16*3) + np.mean(s.pix)/(256*256*3), 9 - len(s.avg))

    overlap_groups = {}
    is_dup_truth = {}
    missing_third_party_matches = set()
    G = nx.Graph()

    for (img1_id, img2_id, img12_overlap_tag), sort_score in tqdm(sorted(sort_scores.items(), key=lambda x: x[1])):

        img21_overlap_tag = overlap_tag_pairs[img12_overlap_tag]

        if (img1_id, img2_id, img12_overlap_tag) in dup_truth:
            if dup_truth[(img1_id, img2_id, img12_overlap_tag)] == 0:
                continue

        img1_hashes = set(sdcic.img_metrics['md5'][img1_id][overlap_tag_maps[img12_overlap_tag]])
        if len(img1_hashes.difference(solid_hashes)) == 0:
            continue

        img2_hashes = set(sdcic.img_metrics['md5'][img2_id][overlap_tag_maps[img21_overlap_tag]])
        if len(img2_hashes.difference(solid_hashes)) == 0:
            continue

        if img1_id == img2_id:
            continue

        scores = overlap_image_maps[(img1_id, img2_id, img12_overlap_tag)]
        if np.min(scores.dnn) < 0.9:
            continue

        if img1_id in overlap_groups:
            if len(overlap_groups[img1_id].overlaps[img12_overlap_tag]) != 0:
                continue

        if img2_id in overlap_groups:
            if len(overlap_groups[img2_id].overlaps[img21_overlap_tag]) != 0:
                continue

        if img1_id not in overlap_groups:
            overlap_groups[img1_id] = SDCImage(img1_id)
        if img2_id not in overlap_groups:
            overlap_groups[img2_id] = SDCImage(img2_id)

        good_overlap, missing_matches = check_overlap(img1_id, img2_id, img12_overlap_tag, overlap_groups, overlap_image_maps, non_dups)
        if len(missing_matches) > 0:
            missing_third_party_matches |= missing_matches

        if not good_overlap:
            continue

        good_overlap, missing_matches = check_overlap(img2_id, img1_id, img21_overlap_tag, overlap_groups, overlap_image_maps, non_dups)
        if len(missing_matches) > 0:
            missing_third_party_matches |= missing_matches

        if not good_overlap:
            continue

        add_overlap(img1_id, img2_id, img12_overlap_tag, overlap_groups)
        add_overlap(img2_id, img1_id, img21_overlap_tag, overlap_groups)
        G.add_edge(img1_id, img2_id)

    neighbor_counts = Counter()
    for image_hashes in nx.connected_components(G):
        neighbor_counts[len(image_hashes)] += 1
    print(list(sorted(neighbor_counts.items())))

    for image_hashes in nx.connected_components(G):
        if len(image_hashes) < 50 or len(image_hashes) > 100:
            continue
        image_hashes0 = sorted(image_hashes)

    print(f'Done in {time.time() - t0}')
