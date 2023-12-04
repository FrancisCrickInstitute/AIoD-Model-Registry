import json
from pathlib import Path

from pydantic import ValidationError
import pytest

from aiod_registry import ModelManifest, get_manifest_paths


@pytest.mark.parametrize("json_path", get_manifest_paths())
def test_manifest(json_path):
    with open(json_path, "r") as f:
        json_manifest = json.load(f)
        try:
            ModelManifest.model_validate(json_manifest)
        except ValidationError as e:
            raise e
