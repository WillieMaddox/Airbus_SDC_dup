import pytest
from pytest import fixture

from sdcdup.utils import overlap_tags
from sdcdup.utils import overlap_tag_pairs
from sdcdup.utils import generate_third_party_overlaps


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


def test_generate_third_party_overlaps():
    third_party_overlaps2 = {
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
        for otp, otp2 in zip(third_party_overlaps[overlap_tag], third_party_overlaps2[overlap_tag]):
            assert otp == otp2, print(overlap_tag, otp, otp2)
