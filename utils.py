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


def backup_dup_truth(filename='dup_truth.txt'):

    if not os.path.exists(filename):
        return

    filebase, fileext = filename.rsplit('.')
    new_filename = ''.join([filebase, "_", get_datetime_now(), ".", fileext])
    copyfile(filename, new_filename)
    assert os.path.exists(new_filename)


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
        backup_dup_truth(filename=filename)
        write_dup_truth(dup_truth, filename=filename)

