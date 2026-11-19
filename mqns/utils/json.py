from typing import Any, cast

import numpy as np
from pydantic import BaseModel

_MARKER_ATTRIBUTE = "_json_encodable.8a8dd1ed-954e-4ca1-8262-0c8404762fc4"
_MARKER_SENTINEL = object()


def json_encodable[T: type](cls: T) -> T:
    """
    Class decorator to indicate compatibility with :py:func:`json_default`.

    A class instance encodes as a JSON object that contains static attributes and
    ``@property``-decorated properties, except those starting with '_'.
    """
    setattr(cls, _MARKER_ATTRIBUTE, _MARKER_SENTINEL)
    return cls


def json_default(obj: Any) -> Any:
    """
    Custom JSON encoder, passed as ``json.dumps(default=json_default)``.

    It accepts these types:

    * :py:class:`numpy.ndarray`
    * :py:class:`pydantic.BaseModel` subclass
    * class decorated with :py:deco:`json_encodable`
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")

    typ = type(obj)
    if getattr(typ, _MARKER_ATTRIBUTE, None) is not _MARKER_SENTINEL:
        raise TypeError(f"cannot encode {typ}")

    d = {}
    for mem, val in cast(dict[str, Any], vars(obj)).items():
        if mem[:1] != "_":
            d[mem] = val
    for mem in dir(typ):
        if mem[:1] == "_":
            continue
        prop = getattr(typ, mem)
        if isinstance(prop, property) and prop.fget:
            d[mem] = prop.fget(obj)
    return d
