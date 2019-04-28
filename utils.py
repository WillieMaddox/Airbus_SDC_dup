import os
from shutil import copyfile
from datetime import datetime
import numpy as np
import cv2

EPS = np.finfo(np.float32).eps

idx_chan_map = {0: 'H', 1: 'L', 2: 'S'}
chan_idx_map = {'H': 0, 'L': 1, 'S': 2}
chan_cv2_scale_map = {'H': 256, 'L': 256, 'S': 256}
chan_gimp_scale_map = {'H': 360, 'L': 200, 'S': 100}


# There are 24 distinct ways a $3\times3$ grid can overlap with another $3\times3$ grid.

ij_pairs_3x3 = [(0, 0), (0, 1), (0, 2),
                (1, 0), (1, 1), (1, 2),
                (2, 0), (2, 1), (2, 2)]

tile_idx2ij = {idx: ij for idx, ij in enumerate(ij_pairs_3x3)}
tile_ij2idx = {ij: idx for idx, ij in enumerate(ij_pairs_3x3)}

overlay_tag_pairs = {
    '0022': '0022',
    '0122': '0021',
    '0021': '0122',
    '1022': '0012',
    '0012': '1022',
    '1122': '0011',
    '0011': '1122',
    '1021': '0112',
    '0112': '1021',
    '0222': '0020',
    '0020': '0222',
    '2022': '0002',
    '0002': '2022',
    '1222': '0010',
    '0010': '1222',
    '1020': '0212',
    '0212': '1020',
    '2122': '0001',
    '0001': '2122',
    '2021': '0102',
    '0102': '2021',
    '2222': '0000',
    '0000': '2222',
    '2020': '0202',
    '0202': '2020',
}


overlay_tag_maps = {
    '0022': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    '0122': np.array([1, 2, 4, 5, 7, 8]),
    '0021': np.array([0, 1, 3, 4, 6, 7]),
    '1022': np.array([3, 4, 5, 6, 7, 8]),
    '0012': np.array([0, 1, 2, 3, 4, 5]),
    '1122': np.array([4, 5, 7, 8]),
    '0011': np.array([0, 1, 3, 4]),
    '1021': np.array([3, 4, 6, 7]),
    '0112': np.array([1, 2, 4, 5]),
    '0222': np.array([2, 5, 8]),
    '0020': np.array([0, 3, 6]),
    '2022': np.array([6, 7, 8]),
    '0002': np.array([0, 1, 2]),
    '1222': np.array([5, 8]),
    '0010': np.array([0, 3]),
    '1020': np.array([3, 6]),
    '0212': np.array([2, 5]),
    '2122': np.array([7, 8]),
    '0001': np.array([0, 1]),
    '2021': np.array([6, 7]),
    '0102': np.array([1, 2]),
    '2222': np.array([8]),
    '0000': np.array([0]),
    '2020': np.array([6]),
    '0202': np.array([2])}


def generate_overlay_tag_slices():

    # sd -> short for slice dictionary if that even means anything.
    sd = {
        '00': slice(None, 1*256),  # top row (left column)
        '01': slice(None, 2*256),  # top 2 rows (left 2 columns)
        '02': slice(None, None),   # all rows (all columns)
        '12': slice(1*256, None),  # bottom 2 rows (right 2 columns)
        '22': slice(2*256, None),  # bottom row (right column)
    }

    return {'0022': (sd['02'], sd['02']),
            '0122': (sd['02'], sd['12']),
            '0021': (sd['02'], sd['01']),
            '1022': (sd['12'], sd['02']),
            '0012': (sd['01'], sd['02']),
            '1122': (sd['12'], sd['12']),
            '0011': (sd['01'], sd['01']),
            '1021': (sd['12'], sd['01']),
            '0112': (sd['01'], sd['12']),
            '0222': (sd['02'], sd['22']),
            '0020': (sd['02'], sd['00']),
            '2022': (sd['22'], sd['02']),
            '0002': (sd['00'], sd['02']),
            '1222': (sd['12'], sd['22']),
            '0010': (sd['01'], sd['00']),
            '1020': (sd['12'], sd['00']),
            '0212': (sd['01'], sd['22']),
            '2122': (sd['22'], sd['12']),
            '0001': (sd['00'], sd['01']),
            '2021': (sd['22'], sd['01']),
            '0102': (sd['00'], sd['12']),
            '2222': (sd['22'], sd['22']),
            '0000': (sd['00'], sd['00']),
            '2020': (sd['22'], sd['00']),
            '0202': (sd['00'], sd['22'])}


def generate_pair_tag_lookup():
    ptl = {}
    for tag1, tag2 in overlay_tag_pairs.items():
        for idx1, idx2 in zip(overlay_tag_maps[tag1], overlay_tag_maps[tag2]):
            ptl[(idx1, idx2)] = tag1
    return ptl


def generate_overlay_tag_nines_mask():
    overlay_tag_nines_mask = {}
    for overlay_tag, overlay_map in overlay_tag_maps.items():
        arr9 = np.zeros(9, dtype=np.bool8)
        for idx in overlay_map:
            arr9[idx] = True
        overlay_tag_nines_mask[overlay_tag] = arr9
    return overlay_tag_nines_mask




def get_datetime_now(t=None, fmt='%Y_%m%d_%H%M'):
    """Return timestamp as a string; default: current time, format: YYYY_DDMM_hhmm_ss."""
    if t is None:
        t = datetime.now()
    return t.strftime(fmt)


def quick_stats(arr):
    print(arr.shape, arr.dtype, np.min(arr), np.max(arr), np.mean(arr), np.std(arr), np.sum(arr))


def bce(y_true, y_pred, **kwargs):
    y_pred = np.clip(y_pred, EPS, 1. - EPS)
    out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return np.mean(out, axis=-1)


def pad_string(x, n):
    padding = n - len(x)
    x_new = x if padding <= 0 else ''.join(['0' * padding, x])
    return x_new


def hex_to_int(hash_hex):
    return int(hash_hex, 16)


def int_to_hex(hash_int, hash_len):
    hash_hex = hex(hash_int)[2:]
    return pad_string(hash_hex, hash_len)


def normalized_hamming_distance(hash1, hash2):
    return np.mean((hash1 != hash2) * 1)


def get_best_model_name(run_dir):
    best_model = None
    min_loss = 999.9
    run_dir2 = os.path.join('out', run_dir)
    for filename in os.listdir(run_dir2):
        if not filename.endswith('.hdf5'):
            continue
        if '.last.' in filename:
            continue
        filebase = filename.rsplit('.', maxsplit=1)[0]
        loss = float(filebase.split('-')[1])
        if loss <= min_loss:
            best_model = filename
            min_loss = loss
    best_model_filename = os.path.join('out', run_dir, best_model)
    print(best_model_filename)
    return best_model_filename


def get_tile(img, idx, sz=256):
    i, j = tile_idx2ij[idx]
    return img[i * sz:(i + 1) * sz, j * sz:(j + 1) * sz, :]


def to_hls(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS_FULL)


def to_bgr(hls):
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR_FULL)


def channel_shift(img, chan, val):
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


def backup_file(filename, backup_str=None):

    if not os.path.exists(filename):
        return
    if backup_str is None:
        backup_str = get_datetime_now()
    else:
        assert type(backup_str) == str

    filebase, fileext = filename.rsplit('.', maxsplit=1)
    new_filename = ''.join([filebase, "_", backup_str, ".", fileext])
    copyfile(filename, new_filename)
    assert os.path.exists(new_filename)


def read_duplicate_truth(filename):

    duplicate_truth = {}
    if not os.path.exists(filename):
        return duplicate_truth

    with open(filename, 'r') as ifs:
        for line in ifs.readlines():
            img1_id, img2_id, img1_overlay_tag, is_duplicate = line.strip().split(' ')
            if img1_id > img2_id:
                img1_id, img2_id = img2_id, img1_id
                img1_overlay_tag = overlay_tag_pairs[img1_overlay_tag]
            if (img1_id, img2_id, img1_overlay_tag) in duplicate_truth:
                continue
            duplicate_truth[(img1_id, img2_id, img1_overlay_tag)] = int(is_duplicate)

    return duplicate_truth


def write_duplicate_truth(filename, duplicate_truth):

    with open(filename, 'w') as ofs:
        for (img1_id, img2_id, img1_overlay_tag), is_duplicate in sorted(duplicate_truth.items()):
            ofs.write(' '.join([img1_id, img2_id, img1_overlay_tag, str(is_duplicate)]))
            ofs.write('\n')


def update_duplicate_truth(filename, new_truth):

    has_updated = False
    duplicate_truth = read_duplicate_truth(filename)
    n_lines_in_original = len(duplicate_truth)

    for (img1_id, img2_id, img1_overlay_tag), is_duplicate in new_truth.items():
        if img1_id > img2_id:
            img1_id, img2_id = img2_id, img1_id
            img1_overlay_tag = overlay_tag_pairs[img1_overlay_tag]
        if (img1_id, img2_id, img1_overlay_tag) in duplicate_truth:
            if duplicate_truth[(img1_id, img2_id, img1_overlay_tag)] != is_duplicate:
                raise ValueError(f"{img1_id} and {img2_id} cannot both be {duplicate_truth[(img1_id, img2_id, img1_overlay_tag)]} and {is_duplicate}")
            continue
        duplicate_truth[(img1_id, img2_id, img1_overlay_tag)] = is_duplicate
        has_updated = True

    if has_updated:
        backup_str = pad_string(str(n_lines_in_original), 8)
        backup_file(filename, backup_str)
        write_duplicate_truth(filename, duplicate_truth)


def read_image_duplicate_tiles(filename):
    """
    filename format: img_id 011345666
    dup_tiles format: {img_id: np.array([0, 1, 1, 3, 4, 5, 6, 6, 6])}
    :param filename:
    :return:
    """
    dup_tiles = {}
    if not os.path.exists(filename):
        return dup_tiles

    with open(filename, 'r') as ifs:
        for line in ifs.readlines():
            img1_id, dup_tiles1 = line.strip().split(' ')
            dup_tiles[img1_id] = np.array(list(map(int, dup_tiles1)))

    return dup_tiles


def write_image_duplicate_tiles(filename, dup_tiles):
    """
    dup_tiles format: {img_id: np.array([0, 1, 1, 3, 4, 5, 6, 6, 6])}
    filename format: img_id 011345666

    :param dup_tiles:
    :param filename:
    :return:
    """
    with open(filename, 'w') as ofs:
        for img1_id, dup_tiles1 in sorted(dup_tiles.items()):
            ofs.write(img1_id + ' ')
            ofs.write(''.join(list(map(str, dup_tiles1))) + '\n')


def read_image_image_duplicate_tiles(filename):
    """
    file format: img1_id img2_id 919399918 938918998
    dup_tiles format: {(img1_id, img2_id): ((9, 1, 9, 3, 9, 9, 9, 1, 8), (9, 3, 8, 9, 1, 8, 9, 9, 8))}
    :param filename:
    :return:
    """
    dup_tiles = {}
    if not os.path.exists(filename):
        return dup_tiles

    with open(filename, 'r') as ifs:
        for line in ifs.readlines():
            img1_id, img2_id, dup_tiles1, dup_tiles2 = line.strip().split(' ')
            dup_tiles[(img1_id, img2_id)] = (np.array(list(map(int, dup_tiles1))), np.array(list(map(int, dup_tiles2))))

    return dup_tiles


def write_image_image_duplicate_tiles(filename, dup_tiles):
    """
    dup_tiles format: {(img1_id, img2_id): ((9, 1, 9, 3, 9, 9, 9, 1, 8), (9, 3, 8, 9, 1, 8, 9, 9, 8))}
    file format: img1_id img2_id 919399918 938918998

    :param dup_tiles:
    :param filename:
    :return:
    """

    with open(filename, 'w') as ofs:
        for (img1_id, img2_id), (dup_tiles1, dup_tiles2) in sorted(dup_tiles.items()):
            ofs.write(img1_id + ' ' + img2_id + ' ')
            ofs.write(''.join([str(d) for d in dup_tiles1]) + ' ')
            ofs.write(''.join([str(d) for d in dup_tiles2]) + '\n')


def update_image_image_duplicate_tiles(filename, new_tiles):
    has_updated = False
    duplicate_tiles = read_image_image_duplicate_tiles(filename)
    n_lines_in_original = len(duplicate_tiles)
    for (img1_id, img2_id), (dup_tiles1, dup_tiles2) in new_tiles.items():

        if (img1_id, img2_id) in duplicate_tiles:
            old_tiles1, old_tiles2 = duplicate_tiles[(img1_id, img2_id)]
            if old_tiles1 != dup_tiles1:
                raise ValueError(f"{img1_id}: old tiles vs. new tiles: {old_tiles1} != {dup_tiles1}")
            if old_tiles2 != dup_tiles2:
                raise ValueError(f"{img2_id}: old tiles vs. new tiles: {old_tiles2} != {dup_tiles2}")
        else:
            duplicate_tiles[(img1_id, img2_id)] = (dup_tiles1, dup_tiles2)
            has_updated = True

    if has_updated:
        backup_str = pad_string(str(n_lines_in_original), 8)
        backup_file(filename, backup_str)
        write_image_image_duplicate_tiles(filename, duplicate_tiles)


def even_split(n_samples, batch_size, split):
    # Hack until I can figure out how to ragged end of the database.
    train_percent = split / 100.
    train_pivot = int(n_samples * train_percent)
    n_train = train_pivot - train_pivot % batch_size

    valid_percent = 1. - train_percent
    valid_pivot = int(n_samples * valid_percent)
    n_valid = valid_pivot - valid_pivot % batch_size

    return n_train, n_valid


def create_dataset_from_tiles_and_truth(dup_tiles, dup_truth):
    overlay_tag_maps0 = {}
    for img1_overlay_tag, img1_overlay_map in overlay_tag_maps.items():
        img2_overlay_map = overlay_tag_maps[overlay_tag_pairs[img1_overlay_tag]]
        overlay_tag_maps0[img1_overlay_tag] = list(zip(img1_overlay_map, img2_overlay_map))

    used_ids = set()
    img_overlay_pairs = {}
    for (img1_id, img2_id, img1_overlay_tag), is_dup in dup_truth.items():
        if is_dup:
            img_overlay_pairs[(img1_id, img2_id, img1_overlay_tag)] = overlay_tag_maps0[img1_overlay_tag]
            used_ids.add(img1_id)
            used_ids.add(img2_id)

    for img_id, dup_vector in dup_tiles.items():
        if img_id in used_ids:
            continue
        if dup_vector.sum() == 36:  # if dup_vector == [0,1,2,3,4,5,6,7,8], all tiles are unique.
            img_overlay_pairs[(img_id, img_id, '0022')] = overlay_tag_maps0['0022']

    return img_overlay_pairs
