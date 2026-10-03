from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

import yaml

from optimizers.types import CATParams

PathLike = Union[str, Path]


def load_yaml(path: PathLike) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path!s}.")
    return data


def load_cat_params(path: PathLike) -> CATParams:
    data = load_yaml(path)
    if "cat" in data:
        data = data["cat"]
    if not isinstance(data, Mapping):
        raise ValueError("CAT configuration must be a mapping.")
    return CATParams(**dict(data))
