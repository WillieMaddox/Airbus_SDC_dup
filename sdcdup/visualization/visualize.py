from collections import namedtuple
import numpy as np
import cv2
from scipy import stats

from sdcdup.utils import overlap_tag_pairs
from sdcdup.utils import boundingbox_corners
from sdcdup.utils import generate_overlap_tag_slices

overlap_tag_slices = generate_overlap_tag_slices()
ChannelShift = namedtuple('ChannelShift', 'method shared')


def holt_winters_second_order_ewma(x, span, beta):
    # Ref http://connor-johnson.com/2014/02/01/smoothing-with-exponentially-weighted-moving-averages/
    N = x.size
    alpha = 2.0 / (1 + span)
    s = np.zeros((N,))
    b = np.zeros((N,))
    s[0] = x[0]
    for i in range(1, N):
        s[i] = alpha * x[i] + (1 - alpha) * (s[i - 1] + b[i - 1])
        b[i] = beta * (s[i] - s[i - 1]) + (1 - beta) * b[i - 1]
    return s


def reversed_recombined_holt_winters(x, span=15, beta=0.3):
    # take EWMA in both directions with a smaller span term
    fwd = holt_winters_second_order_ewma(x, span, beta)
    bwd = holt_winters_second_order_ewma(x[::-1], span, beta)
    c = np.vstack((fwd, bwd[::-1]))  # lump fwd and bwd together
    c = np.mean(c, axis=0)  # average
    return c


def get_ticks(img_size=768, dtick=256, hist_size=256):
    n_ticks = img_size // dtick + 1
    return [i * dtick * hist_size // 256 for i in range(n_ticks)]


def subtract_channel_average(img1, img2, img1_overlap_tag, shift):
    slice1 = overlap_tag_slices[img1_overlap_tag]
    slice2 = overlap_tag_slices[overlap_tag_pairs[img1_overlap_tag]]
    if shift.shared:
        # Use the average channel value between both images.
        img12_stack = np.vstack([img1[slice1], img2[slice2]])
        if shift.method == 'median':
            m12 = np.median(img12_stack, axis=(0, 1), keepdims=True).astype(np.uint8)
        elif shift.method == 'mode':
            m12 = stats.mode(img12_stack.reshape(-1, 3), axis=0)[0]
        else:
            return
        img1[slice1] = img1[slice1] - m12
        img2[slice2] = img2[slice2] - m12
    else:
        # Each image uses its own average channel values
        if shift.method == 'median':
            m1 = np.median(img1[slice1], axis=(0, 1), keepdims=True).astype(np.uint8)
            m2 = np.median(img2[slice2], axis=(0, 1), keepdims=True).astype(np.uint8)
        elif shift.method == 'mode':
            m1 = stats.mode(img1[slice1].reshape(-1, 3), axis=0)[0]
            m2 = stats.mode(img2[slice2].reshape(-1, 3), axis=0)[0]
        else:
            return
        img1[slice1] = img1[slice1] - m1
        img2[slice2] = img2[slice2] - m2


def draw_tile_number(img, idx, value=None, img_size=768, font=cv2.FONT_HERSHEY_SIMPLEX, scale=5, color=None, thickness=8):

    value = value or idx
    if color is None:
        color = tuple(int(x) for x in np.random.randint(0, 256, 3))
    text_width, text_height = cv2.getTextSize(str(value), font, scale, thickness)[0]
    tile_col = (((idx * 256) % img_size) + 128 - text_width // 2)
    tile_row = (((idx * 256) // img_size) * 256 + 128 + text_height // 2)
    cv2.putText(img, str(value), (tile_col, tile_row), font, scale, color, thickness)


def draw_bbox(img, bbox, thickness, color, img_size=None):
    img_size = img_size or img.shape[0]
    offset = (thickness // 2) + 1
    bbox_pt1 = np.clip(bbox[0], offset, img_size - offset)
    bbox_pt2 = np.clip(bbox[1], offset, img_size - offset)
    cv2.rectangle(img, tuple(bbox_pt1), tuple(bbox_pt2), color, thickness)


def draw_tile_bbox(img, idx, thickness, color, img_size=None, tile_size=256):
    height, width, _ = img.shape
    if img_size is None:
        img_size = max(height, width)

    assert height % tile_size == 0
    assert width % tile_size == 0
    ncols = width // tile_size

    bbox_pt1 = np.array([idx % ncols, idx // ncols])
    bbox_pt2 = bbox_pt1 + 1
    bbox = np.stack([bbox_pt1, bbox_pt2]) * tile_size
    draw_bbox(img, bbox, thickness, color, img_size=img_size)


def draw_overlap_bbox(img, img_overlap_tag, thickness, color, img_size=None):
    draw_bbox(img, boundingbox_corners[img_overlap_tag], thickness, color, img_size=img_size)


def show_image(ax, img, title, ticks):
    ax.imshow(img)
    ax.set_title(title)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def show_image_pair(ax1, ax2, imgmod1, imgmod2, img1_overlap_tag, draw_bboxes, bbox_thickness, bbox_color, title1, title2, ticks, shift=ChannelShift('', True)):
    img1 = imgmod1.parent_rgb
    img2 = imgmod2.parent_rgb

    if shift.method in ('median', 'mode'):
        subtract_channel_average(img1, img2, img1_overlap_tag, shift)

    if draw_bboxes:
        draw_overlap_bbox(img1, img1_overlap_tag, bbox_thickness, bbox_color)
        draw_overlap_bbox(img2, overlap_tag_pairs[img1_overlap_tag], bbox_thickness, bbox_color)

    show_image(ax1, img1, title1, ticks)
    show_image(ax2, img2, title2, ticks)
