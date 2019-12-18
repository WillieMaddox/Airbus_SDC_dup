import pytest
from pytest import fixture
import json

from sdcdup.rebuild_overlap_groups import PrettyEncoder


def test_PrettyEncoder():
    test_dict = {"filename.jpg": {'d82542ac6.jpg': (0, 0), '7b836bdec.jpg': (1, 0)}}
    true_json = '{\n  "filename.jpg": {\n    "7b836bdec.jpg": [1, 0], \n    "d82542ac6.jpg": [0, 0]\n  }\n}'
    test_json = json.dumps(
        test_dict,
        cls=PrettyEncoder,
        indent=2,
        separators=(', ', ': '),
        sort_keys=True,
    )
    assert test_json == true_json
