import json

import numpy as np
import pytest

from mqns.utils import json_default, json_encodable


class DataClass:
    def __init__(self):
        self.included_attribute = np.array([1], dtype=np.int16)
        self._excluded_attribute = 2

    @property
    def included_property(self) -> int:
        return 3

    @property
    def _excluded_property(self) -> int:
        return 4


@json_encodable
class EncodableClass(DataClass):
    pass


def test_json_default():
    with pytest.raises(TypeError):
        json.dumps(EncodableClass())

    with pytest.raises(TypeError):
        json.dumps(DataClass(), default=json_default)

    wire = json.dumps(EncodableClass(), default=json_default, sort_keys=True)
    assert wire == '{"included_attribute": [1], "included_property": 3}'
