import os
from shutil import copyfile
from datetime import datetime
from tqdm import tqdm
import numpy as np
import h5py
import cv2

EPS = np.finfo(np.float32).eps

idx_chan_map = {0: 'H', 1: 'L', 2: 'S'}
chan_idx_map = {'H': 0, 'L': 1, 'S': 2}
chan_cv2_scale_map = {'H': 256, 'L': 256, 'S': 256}
chan_gimp_scale_map = {'H': 360, 'L': 200, 'S': 100}


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
    '0022': np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]),
    '0122': np.array([[0, 1], [0, 2], [1, 1], [1, 2], [2, 1], [2, 2]]),
    '0021': np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]),
    '1022': np.array([[1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]),
    '0012': np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]),
    '1122': np.array([[1, 1], [1, 2], [2, 1], [2, 2]]),
    '0011': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    '1021': np.array([[1, 0], [1, 1], [2, 0], [2, 1]]),
    '0112': np.array([[0, 1], [0, 2], [1, 1], [1, 2]]),
    '0222': np.array([[0, 2], [1, 2], [2, 2]]),
    '0020': np.array([[0, 0], [1, 0], [2, 0]]),
    '2022': np.array([[2, 0], [2, 1], [2, 2]]),
    '0002': np.array([[0, 0], [0, 1], [0, 2]]),
    '1222': np.array([[1, 2], [2, 2]]),
    '0010': np.array([[0, 0], [1, 0]]),
    '1020': np.array([[1, 0], [2, 0]]),
    '0212': np.array([[0, 2], [1, 2]]),
    '2122': np.array([[2, 1], [2, 2]]),
    '0001': np.array([[0, 0], [0, 1]]),
    '2021': np.array([[2, 0], [2, 1]]),
    '0102': np.array([[0, 1], [0, 2]]),
    '2222': np.array([[2, 2]]),
    '0000': np.array([[0, 0]]),
    '2020': np.array([[2, 0]]),
    '0202': np.array([[0, 2]])}


def generate_overlay_tag_slices():

    # sd -> short for slice dictionary
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
        for pair in zip(overlay_tag_maps[tag1], overlay_tag_maps[tag2]):
            pair_key = tuple([tuple(pair[0]), tuple(pair[1])])
            ptl[pair_key] = tag1
    return ptl


def get_datetime_now(t=None, fmt='%Y_%m%d_%H%M_%S'):
    """Return timestamp as a string; default: current time, format: YYYY_DDMM_hhmm_ss."""
    if t is None:
        t = datetime.now()
    return t.strftime(fmt)


def create_hdf5_dataset(data_dir=None, outfile=None):
    if data_dir in (None, '.'):
        data_dir = os.path.join(os.getcwd(), 'data', 'train_768')
    # if data_dir is None:
    #     data_dir = "/media/Borg_LS/DATA/geos/airbus/input/train_768"
    if outfile is None:
        outfile = data_dir + ".h5"

    img_ids = os.listdir(data_dir)

    with h5py.File(outfile, 'w') as h5:
        for img_id in tqdm(img_ids):
            img = cv2.imread(os.path.join(data_dir, img_id))
            h5.create_dataset(img_id, data=img)


def quick_stats(arr):
    print(arr.shape, arr.dtype, np.min(arr), np.max(arr), np.mean(arr), np.std(arr), np.sum(arr))


def bce(y_true, y_pred, **kwargs):
    y_pred = np.clip(y_pred, EPS, 1. - EPS)
    out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return np.mean(out, axis=-1)


def hex_to_int(hash_hex):
    return int(hash_hex, 16)


def int_to_hex(hash_int, hash_len):
    hash_hex = hex(hash_int)[2:]
    padding = hash_len - len(hash_hex)
    if padding > 0:
        hash_hex = ''.join(['0' * padding, hash_hex])
    return hash_hex


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


def get_tile(img, i, j, sz=256):
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


def backup_file(filename):
    if not os.path.exists(filename):
        return

    filebase, fileext = filename.rsplit('.')
    new_filename = ''.join([filebase, "_", get_datetime_now(), ".", fileext])
    copyfile(filename, new_filename)
    assert os.path.exists(new_filename)


def read_dup_truth(filename='dup_truth.txt'):

    dup_truth = {}
    if not os.path.exists(filename):
        return dup_truth

    with open(filename, 'r') as ifs:
        for line in ifs.readlines():
            img_id1, img_id2, overlay_tag, is_dup = line.strip().split(' ')
            if img_id1 > img_id2:
                img_id1, img_id2 = img_id2, img_id1
                overlay_tag = overlay_tag_pairs[overlay_tag]
            if (img_id1, img_id2, overlay_tag) in dup_truth:
                continue
            dup_truth[(img_id1, img_id2, overlay_tag)] = int(is_dup)

    return dup_truth


def write_dup_truth(dup_truth, filename='dup_truth.txt'):

    with open(filename, 'w') as ofs:
        for (img_id1, img_id2, overlay_tag), is_dup in sorted(dup_truth.items(), key=lambda x: x[0][0]):
            ofs.write(' '.join([img_id1, img_id2, overlay_tag, str(is_dup)]))
            ofs.write('\n')


def update_dup_truth(update_dict, dup_truth=None, filename='dup_truth.txt'):

    has_updated = False
    if dup_truth is None:
        dup_truth = read_dup_truth(filename=filename)

    for (img_id1, img_id2, overlay_tag), is_dup in update_dict.items():
        if img_id1 > img_id2:
            img_id1, img_id2 = img_id2, img_id1
            overlay_tag = overlay_tag_pairs[overlay_tag]
        if (img_id1, img_id2, overlay_tag) in dup_truth:
            if dup_truth[(img_id1, img_id2, overlay_tag)] != is_dup:
                raise ValueError(f"{img_id1} and {img_id2} cannot both be {dup_truth[(img_id1, img_id2, overlay_tag)]} and {is_dup}")
            continue
        dup_truth[(img_id1, img_id2, overlay_tag)] = is_dup
        has_updated = True

    if has_updated:
        backup_file(filename)
        write_dup_truth(dup_truth, filename=filename)

