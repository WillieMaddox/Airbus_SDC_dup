import numpy as np
import matplotlib.pyplot as plt
import cv2


def fuzzy_diff(tile1, tile2, norm=False):
    maxab = np.max(np.stack([tile1, tile2]), axis=0)
    a = maxab - tile2
    b = maxab - tile1
    ab = a + b
    return ab / ab.max() if norm else ab


def slice_from_large(img, i, j, sz=256):
    return img[i * sz:(i + 1) * sz, j * sz:(j + 1) * sz, :]


minmax = lambda x: [x.dtype, np.min(x, axis=(0, 1)), np.max(x, axis=(0, 1))]


chan_idx_map = {'H': 0, 'L': 1, 'S': 2}
chan_cv2_scale_map = {'H': 256, 'L': 256, 'S': 256}
chan_gimp_scale_map = {'H': 360, 'L': 200, 'S': 200}


def to_hls(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS_FULL)


def to_bgr(hls):
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR_FULL)


def channel_shift(hls, chan, val):
    # NOTE: hls must be in bytes so that hue will wrap around correctly.

    gimp_scale = chan_gimp_scale_map[chan]
    idx = chan_idx_map[chan]

    # TODO: Add a plot showing how we arrived at each of the three scaling options.
    if idx == 0:  # hue
        scaled_val = 255. * val / gimp_scale
        scaled_val = np.around(scaled_val).astype(np.uint8)
        scaled_img = np.copy(hls)
        scaled_img[:, :, idx] += scaled_val

    elif idx == 1:  # lightness
        # TODO: Could this be simplified?
        l = hls[:, :, idx] * 1.
        min_0 = np.min(l)
        max_0 = np.max(l)
        if val < 0:
            scaled_val = -1 * val / 100.
            min_1 = min_0 / 2.
            max_1 = max_0 / 2.
        else:
            scaled_val = val / 100.
            min_1 = (min_0 + 255.) / 2.
            max_1 = (max_0 + 255.) / 2.
        min_v = (min_1 - min_0) * scaled_val + min_0
        max_v = (max_1 - max_0) * scaled_val + max_0
        # FIXME: Divide by zero error if image happens to be all black.
        l_shifted = (l - min_0) * (max_v - min_v) / (max_0 - min_0) + min_v
        l_shifted = np.clip(l_shifted, 0, 255)
        scaled_img = np.copy(hls)
        scaled_img[:, :, idx] = np.around(l_shifted).astype(np.uint8)

    elif idx == 2:  # saturation

        scaled_val = (val / 100.) + 1.
        s_shifted = hls[:, :, idx] * scaled_val
        s_shifted = np.clip(s_shifted, 0, 255)
        scaled_img = np.copy(hls)
        scaled_img[:, :, idx] = np.around(s_shifted).astype(np.uint8)

    else:
        raise ValueError

    return scaled_img


def plot_row(ax, row_num, rgb1, title1, rgb2, title2, dtick=256):
    n_ticks = rgb1.shape[1] // dtick + 1
    ticks = [i * dtick for i in range(n_ticks)]
    ax[row_num, 0].imshow(rgb1)
    ax[row_num, 0].set_title(title1)
    ax[row_num, 0].set_xticks(ticks)
    ax[row_num, 0].set_yticks(ticks)
    ax[row_num, 1].imshow(rgb2)
    ax[row_num, 1].set_title(title2)
    ax[row_num, 1].set_xticks(ticks)
    ax[row_num, 1].set_yticks(ticks)
    diff = fuzzy_diff(rgb1, rgb2)
#     max_diff = np.max(diff)
#     sum_diff = np.sum(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    ax[row_num, 2].imshow(fuzzy_diff(rgb1, rgb2, norm=True))
    ax[row_num, 2].set_title(f'mean: {mean_diff:>7.3f}, std: {std_diff:>7.3f}')
    ax[row_num, 2].set_xticks(ticks)
    ax[row_num, 2].set_yticks(ticks)


class ImgMod:
    """
    Reads hls modified images saved by gimp and compares to original image read by gimp.
    """

    def __init__(self, parent_hls, filename):
        self.filename = filename
        self.parent_prefix, self.img_id = filename.split('/')
        self.gmp_prefix, self.gmp_extension = self.img_id.split('.')
        self.hls_channel, pom, gain = self.gmp_prefix[0], self.gmp_prefix[1], int(self.gmp_prefix[2:])
        sign = 1 if pom == 'p' else -1
        self.hls_gain = sign * gain

        self.parent_hls = parent_hls

        self._gmp_bgr = None
        self._gmp_hls = None
        self._gmp_rgb = None
        self._cv2_hls = None
        self._cv2_bgr = None
        self._cv2_rgb = None

    @property
    def gmp_bgr(self):
        if self._gmp_bgr is None:
            self._gmp_bgr = cv2.imread(self.filename)
        return self._gmp_bgr

    @property
    def gmp_hls(self):
        if self._gmp_hls is None:
            self._gmp_hls = self.to_hls(self.gmp_bgr)
        return self._gmp_hls

    @property
    def gmp_rgb(self):
        if self._gmp_rgb is None:
            self._gmp_rgb = self.to_rgb(self.gmp_bgr)
        return self._gmp_rgb

    @property
    def cv2_hls(self):
        if self._cv2_hls is None:
            self._cv2_hls = channel_shift(self.parent_hls, self.hls_channel, self.hls_gain)
        return self._cv2_hls

    @property
    def cv2_bgr(self):
        if self._cv2_bgr is None:
            self._cv2_bgr = self.to_bgr(self.cv2_hls)
        return self._cv2_bgr

    @property
    def cv2_rgb(self):
        if self._cv2_rgb is None:
            self._cv2_rgb = self.to_rgb(self.cv2_bgr)
        return self._cv2_rgb

    def to_hls(self, bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS_FULL)

    def to_bgr(self, hls):
        return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR_FULL)

    def to_rgb(self, bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


src1 = "03a5fd8d2.jpg"
src2 = "676f4cfd0.jpg"

ibgr1 = cv2.imread("/media/Borg_LS/DATA/geos/airbus/input/train_768/" + src1)
ibgr2 = cv2.imread("/media/Borg_LS/DATA/geos/airbus/input/train_768/" + src2)

ihls1 = to_hls(ibgr1)
ihls2 = to_hls(ibgr2)

irgb1 = cv2.cvtColor(ibgr1, cv2.COLOR_BGR2RGB)
irgb2 = cv2.cvtColor(ibgr2, cv2.COLOR_BGR2RGB)

modkeysA = [
    'Sm100.bmp',
    # 'Sm099.bmp',
    # 'Sm098.bmp',
    # 'Sm097.bmp',
    # 'Sm096.bmp',
    # 'Sm095.bmp',
    # 'Sm090.bmp',
    'Sm075.bmp',
    'Sm050.bmp',
    'Sm025.bmp',
]
modkeysB = [
    # 'Sp005.bmp',
    # 'Sp010.bmp',
    # 'Sp015.bmp',
    # 'Sp020.bmp',
    'Sp025.bmp',
    'Sp050.bmp',
    'Sp075.bmp',
    'Sp100.bmp',
]

modsA = {k: ImgMod(ihls1, f"03a5fd8d2/{k}") for k in modkeysA}
modsB = {k: ImgMod(ihls1, f"03a5fd8d2/{k}") for k in modkeysB}

print(f"gain {'cv2':>9} {'gimp':>9}")
for k in modkeysA:
    print(f'{modsA[k].hls_gain:>4} {np.sum(fuzzy_diff(ibgr1, modsA[k].cv2_bgr)):>9} {np.sum(fuzzy_diff(ibgr1, modsA[k].gmp_bgr)):>9}')
print(f'{0:>4} {np.sum(fuzzy_diff(ibgr1, ibgr2)):>9}')
for k in modkeysB:
    print(f'{modsB[k].hls_gain:>4} {np.sum(fuzzy_diff(ibgr1, modsB[k].cv2_bgr)):>9} {np.sum(fuzzy_diff(ibgr1, modsB[k].gmp_bgr)):>9}')

n_rows = len(modkeysA) + 1 + len(modkeysB)
fig, ax = plt.subplots(n_rows, 3, figsize=(3*2, n_rows*2))

ii = 0
for k in modkeysA:
    # plot_row(ax, ii, irgb1, src1, modsA[k].cv2_rgb, f'OpenCV: {modsA[k].hls_channel} = {modsA[k].hls_gain}')
    # ii += 1
    # plot_row(ax, ii, irgb1, src1, modsA[k].gmp_rgb, modsA[k].img_id)
    # ii += 1
    plot_row(ax, ii, modsA[k].cv2_rgb, f'OpenCV: {modsA[k].hls_channel} = {modsA[k].hls_gain}', modsA[k].gmp_rgb, modsA[k].img_id)
    ii += 1
plot_row(ax, ii, irgb1, src1, irgb2, src2)
ii += 1
for k in modkeysB:
    # plot_row(ax, ii, irgb1, src1, modsB[k].cv2_rgb, f'OpenCV: {modsB[k].hls_channel} = {modsB[k].hls_gain}')
    # ii += 1
    # plot_row(ax, ii, irgb1, src1, modsB[k].gmp_rgb, modsB[k].img_id)
    # ii += 1
    plot_row(ax, ii, modsB[k].cv2_rgb, f'OpenCV: {modsB[k].hls_channel} = {modsB[k].hls_gain}', modsB[k].gmp_rgb, modsB[k].img_id)
    ii += 1

plt.tight_layout()
plt.show()
