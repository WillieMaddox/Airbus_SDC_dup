import pytest
from pytest import fixture
import numpy as np
from sdcdup.utils import overlap_tags
from sdcdup.utils import overlap_tag_pairs
from sdcdup.utils import generate_boundingbox_corners
from sdcdup.utils import generate_third_party_overlaps
from sdcdup.utils import generate_pair_tag_lookup
from sdcdup.utils import generate_tag_pair_lookup
from sdcdup.utils import rle_encode
from sdcdup.utils import rle_decode
from sdcdup.utils import pad_string


def test_overlap_tags():
    overlap_tags0 = [
        '00', '01', '02', '12', '22',
        '03', '04', '05', '15', '25',
        '06', '07', '08', '18', '28',
        '36', '37', '38', '48', '58',
        '66', '67', '68', '78', '88']

    for tag0, tag1 in zip(overlap_tags0, overlap_tags):
        assert tag0 == tag1


def test_overlap_tag_pairs():
    overlap_tag_pairs0 = {
        '00': '88',
        '01': '78',
        '02': '68',
        '12': '67',
        '22': '66',
        '03': '58',
        '04': '48',
        '05': '38',
        '15': '37',
        '25': '36',
        '06': '28',
        '07': '18',
        '08': '08',
        '18': '07',
        '28': '06',
        '36': '25',
        '37': '15',
        '38': '05',
        '48': '04',
        '58': '03',
        '66': '22',
        '67': '12',
        '68': '02',
        '78': '01',
        '88': '00'}

    for overlap_tag1, overlap_tag2 in overlap_tag_pairs.items():
        assert overlap_tag_pairs0[overlap_tag1] == overlap_tag2


def test_generate_boundingbox_corners():
    B = 256
    boundingbox_corners0 = {
        '00': np.array([[0, 0], [1, 1]]) * B,
        '01': np.array([[0, 0], [2, 1]]) * B,
        '02': np.array([[0, 0], [3, 1]]) * B,
        '12': np.array([[1, 0], [3, 1]]) * B,
        '22': np.array([[2, 0], [3, 1]]) * B,
        '03': np.array([[0, 0], [1, 2]]) * B,
        '04': np.array([[0, 0], [2, 2]]) * B,
        '05': np.array([[0, 0], [3, 2]]) * B,
        '15': np.array([[1, 0], [3, 2]]) * B,
        '25': np.array([[2, 0], [3, 2]]) * B,
        '06': np.array([[0, 0], [1, 3]]) * B,
        '07': np.array([[0, 0], [2, 3]]) * B,
        '08': np.array([[0, 0], [3, 3]]) * B,
        '18': np.array([[1, 0], [3, 3]]) * B,
        '28': np.array([[2, 0], [3, 3]]) * B,
        '36': np.array([[0, 1], [1, 3]]) * B,
        '37': np.array([[0, 1], [2, 3]]) * B,
        '38': np.array([[0, 1], [3, 3]]) * B,
        '48': np.array([[1, 1], [3, 3]]) * B,
        '58': np.array([[2, 1], [3, 3]]) * B,
        '66': np.array([[0, 2], [1, 3]]) * B,
        '67': np.array([[0, 2], [2, 3]]) * B,
        '68': np.array([[0, 2], [3, 3]]) * B,
        '78': np.array([[1, 2], [3, 3]]) * B,
        '88': np.array([[2, 2], [3, 3]]) * B}

    boundingbox_corners = generate_boundingbox_corners()
    for overlap_tag, bbox_corner in boundingbox_corners.items():
        assert np.all(bbox_corner == boundingbox_corners0[overlap_tag])


def test_generate_third_party_overlaps():
    third_party_overlaps0 = {
        '00': [overlap_tags[i] for i in [0, 1, 2, 5, 6, 7, 10, 11, 12]],
        '01': [overlap_tags[i] for i in [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]],
        '02': [overlap_tags[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],
        '12': [overlap_tags[i] for i in [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]],
        '22': [overlap_tags[i] for i in [2, 3, 4, 7, 8, 9, 12, 13, 14]],
        '03': [overlap_tags[i] for i in [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17]],
        '04': [overlap_tags[i] for i in [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18]],
        '05': [overlap_tags[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]],
        '15': [overlap_tags[i] for i in [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]],
        '25': [overlap_tags[i] for i in [2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19]],
        '06': [overlap_tags[i] for i in [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 20, 21, 22]],
        '07': [overlap_tags[i] for i in [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23]],
        '08': overlap_tags,
        '18': [overlap_tags[i] for i in [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]],
        '28': [overlap_tags[i] for i in [2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24]],
        '36': [overlap_tags[i] for i in [5, 6, 7, 10, 11, 12, 15, 16, 17, 20, 21, 22]],
        '37': [overlap_tags[i] for i in [5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23]],
        '38': [overlap_tags[i] for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]],
        '48': [overlap_tags[i] for i in [6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]],
        '58': [overlap_tags[i] for i in [7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24]],
        '66': [overlap_tags[i] for i in [10, 11, 12, 15, 16, 17, 20, 21, 22]],
        '67': [overlap_tags[i] for i in [10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23]],
        '68': [overlap_tags[i] for i in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]],
        '78': [overlap_tags[i] for i in [11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]],
        '88': [overlap_tags[i] for i in [12, 13, 14, 17, 18, 19, 22, 23, 24]]
    }

    third_party_overlaps = generate_third_party_overlaps()

    for overlap_tag in overlap_tags:
        for otp0, otp in zip(third_party_overlaps0[overlap_tag], third_party_overlaps[overlap_tag]):
            assert otp0 == otp, print(overlap_tag, otp0, otp)


def test_generate_pair_tag_lookup():
    ptl0 = {
        (0, 8): '00',
        (0, 7): '01', (1, 8): '01',
        (0, 6): '02', (1, 7): '02', (2, 8): '02',
        (1, 6): '12', (2, 7): '12',
        (2, 6): '22',
        (0, 5): '03', (3, 8): '03',
        (0, 4): '04', (1, 5): '04', (3, 7): '04', (4, 8): '04',
        (0, 3): '05', (1, 4): '05', (2, 5): '05', (3, 6): '05', (4, 7): '05', (5, 8): '05',
        (1, 3): '15', (2, 4): '15', (4, 6): '15', (5, 7): '15',
        (2, 3): '25', (5, 6): '25',
        (0, 2): '06', (3, 5): '06', (6, 8): '06',
        (0, 1): '07', (1, 2): '07', (3, 4): '07', (4, 5): '07', (6, 7): '07', (7, 8): '07',
        (0, 0): '08', (1, 1): '08', (2, 2): '08',
        (3, 3): '08', (4, 4): '08', (5, 5): '08',
        (6, 6): '08', (7, 7): '08', (8, 8): '08',
        (1, 0): '18', (2, 1): '18', (4, 3): '18', (5, 4): '18', (7, 6): '18', (8, 7): '18',
        (2, 0): '28', (5, 3): '28', (8, 6): '28',
        (3, 2): '36', (6, 5): '36',
        (3, 1): '37', (4, 2): '37', (6, 4): '37', (7, 5): '37',
        (3, 0): '38', (4, 1): '38', (5, 2): '38', (6, 3): '38', (7, 4): '38', (8, 5): '38',
        (4, 0): '48', (5, 1): '48', (7, 3): '48', (8, 4): '48',
        (5, 0): '58', (8, 3): '58',
        (6, 2): '66',
        (6, 1): '67', (7, 2): '67',
        (6, 0): '68', (7, 1): '68', (8, 2): '68',
        (7, 0): '78', (8, 1): '78',
        (8, 0): '88'}

    ptl = generate_pair_tag_lookup()

    for pair, tag in ptl.items():
        assert ptl0[pair] == tag


def test_generate_tag_pair_lookup():
    tpl0 = {
        '00': [(0, 8)],
        '01': [(0, 7), (1, 8)],
        '02': [(0, 6), (1, 7), (2, 8)],
        '12': [(1, 6), (2, 7)],
        '22': [(2, 6)],
        '03': [(0, 5), (3, 8)],
        '04': [(0, 4), (1, 5), (3, 7), (4, 8)],
        '05': [(0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8)],
        '15': [(1, 3), (2, 4), (4, 6), (5, 7)],
        '25': [(2, 3), (5, 6)],
        '06': [(0, 2), (3, 5), (6, 8)],
        '07': [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)],
        '08': [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)],
        '18': [(1, 0), (2, 1), (4, 3), (5, 4), (7, 6), (8, 7)],
        '28': [(2, 0), (5, 3), (8, 6)],
        '36': [(3, 2), (6, 5)],
        '37': [(3, 1), (4, 2), (6, 4), (7, 5)],
        '38': [(3, 0), (4, 1), (5, 2), (6, 3), (7, 4), (8, 5)],
        '48': [(4, 0), (5, 1), (7, 3), (8, 4)],
        '58': [(5, 0), (8, 3)],
        '66': [(6, 2)],
        '67': [(6, 1), (7, 2)],
        '68': [(6, 0), (7, 1), (8, 2)],
        '78': [(7, 0), (8, 1)],
        '88': [(8, 0)]}

    tpl = generate_tag_pair_lookup()

    for tag in tpl:
        for pair1, pair2 in zip(tpl0[tag], tpl[tag]):
            assert pair1[0] == pair2[0] and pair1[1] == pair2[1]


def test_rle_decode():
    data_shape = (768, 768)
    rle = "368419 8 369187 8 369955 8 370723 8 371491 8 372259 8 373027 8 373794 9 374562 9 375330 9 376098 9 376866 9"
    data = rle_decode(rle, data_shape)
    rle_out = rle_encode(data)
    assert rle == rle_out


def test_rle_encode():
    for i in range(3, 100, 3):
        data = np.random.randint(0, 2, (i, i))
        rle = rle_encode(data)
        data_out = rle_decode(rle, data.shape)
        np.testing.assert_allclose(data, data_out)


def test_pad_string():
    x_n_res = [
        ('345', 3, '345'),
        ('345', 4, '0345'),
        ('345', 5, '00345'),
        ('345', 6, '000345')
    ]
    for x, n, res in x_n_res:
        assert pad_string(x, n) == res
