import os
from shutil import copyfile
import numpy as np
from datetime import datetime
from tqdm import tqdm
import h5py
import cv2

EPS = np.finfo(np.float32).eps

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


def perf_measure0(y_true, y_pred, axis=-1):
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)
    TP = np.sum(yt & yp, axis=axis)
    TN = np.sum(~yt & ~yp, axis=axis)
    FP = np.sum(~yt & yp, axis=axis)
    FN = np.sum(yt & ~yp, axis=axis)
    return TP, FP, TN, FN


def perf_measure(y_true, y_pred, axis=-1):
    TP = np.sum(y_true * y_pred, axis=axis)
    TN = np.sum((1 - y_true) * (1 - y_pred), axis=axis)
    FP = np.sum((1 - y_true) * y_pred, axis=axis)
    FN = np.sum(y_true * (1 - y_pred), axis=axis)
    return TP, FP, TN, FN


def confusion_matrix_variants(TP, FP, TN, FN):
    TPR = TP / (TP + FN)  # Sensitivity, hit rate, recall, or true positive rate
    TNR = TN / (TN + FP)  # Specificity or true negative rate
    PPV = TP / (TP + FP)  # Precision or positive predictive value
    NPV = TN / (TN + FN)  # Negative predictive value
    FPR = FP / (FP + TN)  # Fall out or false positive rate
    FNR = FN / (TP + FN)  # False negative rate
    FDR = FP / (TP + FP)  # False discovery rate
    ACC = (TP + TN) / (TP + FP + FN + TN)  # Overall accuracy

    #     print(f'{TP:>5}, {TN:>6}, {FP:>5}, {FN:>5}, {TPR:>.5f}, {TNR:>.5f}, {PPV:>.5f}, {NPV:>.5f}, {FPR:>.5f}, {FNR:>.5f}, {FDR:>.5f}, {ACC:>.5f}')
    return TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC


def recall(y_true, y_pred, axis=-1, smooth=1e-3):
    return (np.sum(y_true * y_pred, axis=axis) + smooth) / (np.sum(y_true, axis=axis) + smooth)


def fbeta(y_true, y_pred, beta=2, axis=-1, smooth=1e-3):
    tp, fp, tn, fn = perf_measure(y_true, y_pred, axis=axis)
    return ((beta ** 2 + 1) * tp + smooth) / ((beta ** 2 + 1) * tp + beta ** 2 * fn + fp + smooth)


def iou(y_true, y_pred, axis=-1):
    i = np.sum((y_true * y_pred) > 0.5, axis=axis) + EPS
    u = np.sum((y_true + y_pred) > 0.5, axis=axis) + EPS
    return i / u


IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def f2score(mask_true, mask_prob):

    f2_total = 0
    for th in IOU_THRESHOLDS:
        mask_pred = mask_prob > th
        tp, fp, _, fn = perf_measure0(mask_true, mask_pred)
        f2_total += (5 * tp + EPS) / (5 * tp + 4 * fn + fp + EPS)

    return f2_total / len(IOU_THRESHOLDS)


def bce(y_true, y_pred, **kwargs):
    y_pred = np.clip(y_pred, EPS, 1. - EPS)
    out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return np.mean(out, axis=-1)


def soft_dice_coef(y_true, y_pred, axis=-1, smooth=1e-3):
    AB = np.sum(y_true * y_pred, axis=axis)
    A = np.sum(y_true, axis=axis)
    B = np.sum(y_pred, axis=axis)
    return (2 * AB + smooth) / (A + B + smooth)


def soft_dice_loss(y_true, y_pred, axis=-1, smooth=1e-3):
    return 1 - soft_dice_coef(y_true, y_pred, axis=axis, smooth=smooth)


# https://www.jeremyjordan.me/semantic-segmentation/
def soft_dice_loss2(y_true, y_pred, epsilon=1e-6):
    """
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    """

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    return 1 - numerator / (denominator + epsilon)


def focal_loss_wrapper(gamma=2., alpha=.25, axis=-1):
    def focal_loss(y_true, y_pred):
        y_pred_c = np.clip(y_pred, EPS, 1. - EPS)
        pt_1 = np.where(np.equal(y_true, 1), y_pred_c, np.ones_like(y_pred))
        pt_0 = np.where(np.equal(y_true, 0), y_pred_c, np.zeros_like(y_pred))
        res1 = alpha * np.power(1. - pt_1, gamma) * np.log(pt_1)
        res0 = (1 - alpha) * np.power(pt_0, gamma) * np.log(1. - pt_0)
        return -np.mean(res1 + res0, axis=axis)
    return focal_loss


def bce_soft_dice_loss_n0(y_true, y_pred, bce_weight=0.5):
    return bce(y_true, y_pred) * bce_weight + soft_dice_loss(y_true, y_pred) * (1 - bce_weight)


def focal_soft_dice_loss_wrapper(gamma=2., alpha=.25, focal_coef=0.5, axis=-1, smooth=1e-3):
    focal_loss = focal_loss_wrapper(gamma=gamma, alpha=alpha, axis=axis)

    def focal_soft_dice_loss(y_true, y_pred):
        return focal_loss(y_true, y_pred) * focal_coef + soft_dice_loss(y_true, y_pred, axis=axis, smooth=smooth) * (1. - focal_coef)

    return focal_soft_dice_loss


def normalized_hamming_distance(hash1, hash2):
    return np.mean((hash1 != hash2) * 1)


def random_crop(img, crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y + dy), x:(x + dx), :]


def crop_generator(batch_generator, crop_length):
    """
    Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator
    """
    while True:
        batch_x, batch_y = next(batch_generator)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, batch_x.shape[-1]))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)


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
            img_id1, img_id2, is_dup = line.strip().split(' ')
            if img_id2 < img_id1:
                img_id1, img_id2 = img_id2, img_id1
            if (img_id1, img_id2) in dup_truth:
                continue
            dup_truth[(img_id1, img_id2)] = int(is_dup)

    return dup_truth


def write_dup_truth(dup_truth, filename='dup_truth.txt'):

    with open(filename, 'w') as ofs:
        for (img_id1, img_id2), is_dup in sorted(dup_truth.items(), key=lambda x: x[0][0]):
            ofs.write(' '.join([img_id1, img_id2, str(is_dup)]))
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

    for (img_id1, img_id2), is_dup in update_dict.items():
        if img_id2 < img_id1:
            img_id1, img_id2 = img_id2, img_id1
        if (img_id1, img_id2) in dup_truth:
            if dup_truth[(img_id1, img_id2)] != is_dup:
                raise ValueError(f"{img_id1} and {img_id2} cannot both be {dup_truth[(img_id1, img_id2)]} and {is_dup}")
            continue
        dup_truth[(img_id1, img_id2)] = is_dup
        has_updated = True

    if has_updated:
        backup_dup_truth(filename=filename)
        write_dup_truth(dup_truth, filename=filename)
