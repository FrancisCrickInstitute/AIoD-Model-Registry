from pathlib import Path
import json
from typing import Optional, Union

from aiod_registry import ModelManifest


def get_manifest_paths():
    json_dir = Path(__file__).parent.parent / "aiod_registry" / "manifests"
    return json_dir.glob("*.json")


def load_manifests(
    paths: Optional[list[Union[Path, str]]] = None
) -> list[ModelManifest]:
    if paths is None:
        paths = get_manifest_paths()
    manifests = []
    for path in paths:
        with open(path, "r") as f:
            json_manifest = json.load(f)
            manifests.append(ModelManifest(**json_manifest))
    return manifests
