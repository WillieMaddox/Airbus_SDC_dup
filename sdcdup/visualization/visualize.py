
import numpy as np
import cv2

from sdcdup.utils import tilebox_corners
from sdcdup.utils import overlap_tag_pairs
from sdcdup.utils import boundingbox_corners
from sdcdup.utils import generate_overlap_tag_slices

overlap_tag_slices = generate_overlap_tag_slices()


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


def subtract_channel_median(img1, img2, img1_overlap_tag):
    slice1 = overlap_tag_slices[img1_overlap_tag]
    slice2 = overlap_tag_slices[overlap_tag_pairs[img1_overlap_tag]]
    m12 = np.median(np.vstack([img1[slice1], img2[slice2]]), axis=(0, 1), keepdims=True).astype(np.uint8)
    img1[slice1] = img1[slice1] - m12
    img2[slice2] = img2[slice2] - m12


def draw_tile_number(img, idx, font=cv2.FONT_HERSHEY_SIMPLEX, scale=5, color=None, thickness=8):
    if color is None:
        color = tuple(int(x) for x in np.random.randint(0, 256, 3))
    text_width, text_height = cv2.getTextSize(str(idx), font, scale, thickness)[0]
    tile_col = ((idx % 3) * 256 + 128 - text_width // 2)
    tile_row = ((idx // 3) * 256 + 128 + text_height // 2)
    cv2.putText(img, str(idx), (tile_col, tile_row), font, scale, color, thickness)


def draw_tile_bbox(img, idx, thickness, color):
    offset = (thickness // 2) + 1
    bbox_pt1, bbox_pt2 = tilebox_corners[idx]
    bbox_pt1 = np.clip(bbox_pt1, offset, 768 - offset)
    bbox_pt2 = np.clip(bbox_pt2, offset, 768 - offset)
    cv2.rectangle(img, tuple(bbox_pt1), tuple(bbox_pt2), color, thickness)


def draw_overlap_bbox(img, img_overlap_tag, thickness, color):
    offset = (thickness // 2) + 1
    img_bbox_pt1, img_bbox_pt2 = boundingbox_corners[img_overlap_tag]
    img_bbox_pt1 = np.clip(img_bbox_pt1, offset, 768 - offset)
    img_bbox_pt2 = np.clip(img_bbox_pt2, offset, 768 - offset)
    cv2.rectangle(img, tuple(img_bbox_pt1), tuple(img_bbox_pt2), color, thickness)


def show_image(ax, img, title, ticks):
    ax.imshow(img)
    ax.set_title(title)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def show_image_pair(ax1, ax2, imgmod1, imgmod2, img1_overlap_tag, draw_bboxes, bbox_thickness, bbox_color, title1, title2, ticks, median_color_shift):
    img1 = imgmod1.parent_rgb
    img2 = imgmod2.parent_rgb

    if median_color_shift:
        subtract_channel_median(img1, img2, img1_overlap_tag)

    if draw_bboxes:
        draw_overlap_bbox(img1, img1_overlap_tag, bbox_thickness, bbox_color)
        draw_overlap_bbox(img2, overlap_tag_pairs[img1_overlap_tag], bbox_thickness, bbox_color)

    show_image(ax1, img1, title1, ticks)
    show_image(ax2, img2, title2, ticks)
