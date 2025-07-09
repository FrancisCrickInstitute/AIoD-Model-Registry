import os
import tempfile
import shutil
import pytest
import json
from pathlib import Path
from aiod_registry.utils import (
    load_manifests,
    flatten_manifest,
    filter_location,
    filter_empty_manifests,
    is_accessible,
)
from aiod_registry.schema import ModelManifest

# Example manifest data (based on cellpose.json)
EXAMPLE_MANIFEST = {
    "name": "Cellpose",
    "short_name": "cellpose",
    "metadata": {
        "description": "Cellpose is a generalist model for cell and nucleus segmentation.",
        "url": "https://cellpose.readthedocs.io/en/v3.1.1.1/",
        "repo": "https://github.com/MouseLand/cellpose",
        "pubs": [
            {
                "info": "Cellpose v1",
                "url": "https://doi.org/10.1038/s41592-020-01018-x",
                "title": "Cellpose: a generalist algorithm for cellular segmentation",
                "doi": "10.1038/s41592-020-01018-x",
                "authors": [
                    {
                        "name": "Carsen Stringer",
                        "affiliation": "HHMI Janelia Research Campus, Ashburn, VA, USA",
                    }
                ],
            }
        ],
    },
    "versions": {
        "cyto3": {
            "tasks": {
                "cyto": {
                    "location": [
                        "https://www.cellpose.org/models/cyto3",
                        "file:///nonexistent/path",
                    ]
                }
            }
        },
        "cyto2": {"tasks": {"cyto": {"location": ["file:///nonexistent/path"]}}},
    },
    "params": [
        {
            "name": "Diameter",
            "arg_name": "diameter",
            "value": 0,
            "tooltip": "Diameter of the cells in pixels.",
        }
    ],
}


@pytest.fixture
def temp_manifest_file(tmp_path):
    manifest_path = tmp_path / "test_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(EXAMPLE_MANIFEST, f)
    return manifest_path


def test_load_manifests_basic(temp_manifest_file):
    manifests = load_manifests(paths=[temp_manifest_file])
    assert "cellpose" in manifests
    manifest = manifests["cellpose"]
    assert isinstance(manifest, ModelManifest)
    assert manifest.name == "Cellpose"
    assert "cyto3" in manifest.versions
    assert "cyto" in manifest.versions["cyto3"].tasks


def test_flatten_manifest(temp_manifest_file):
    manifests = load_manifests(paths=[temp_manifest_file])
    manifest = manifests["cellpose"]
    flat = flatten_manifest(manifest)
    # Should have string, not list, for location
    loc = flat.versions["cyto3"].tasks["cyto"].location
    assert isinstance(loc, str)


def test_flatten_manifest_on_raw(temp_manifest_file):
    # Load the manifest as raw JSON and instantiate ModelManifest
    with open(temp_manifest_file) as f:
        raw_json = json.load(f)
    manifest = ModelManifest(**raw_json)
    flat = flatten_manifest(manifest)
    # Should have string, not list, for location
    loc = flat.versions["cyto3"].tasks["cyto"].location
    assert isinstance(loc, str)


def test_filter_location_and_empty_manifests(temp_manifest_file):
    # Load the manifest as raw JSON and instantiate ModelManifest
    with open(temp_manifest_file) as f:
        raw_json = json.load(f)
    manifest = ModelManifest(**raw_json)
    filtered, changed, num_removed = filter_location(manifest)
    # Only the first location for cyto3 is accessible (URL), cyto2 is not accessible (file path)
    assert changed is True
    assert num_removed == 1
    assert "cyto3" in filtered.versions
    assert (
        "cyto2" not in filtered.versions or len(filtered.versions["cyto2"].tasks) == 0
    )
    # Now test filter_empty_manifests
    manifests_dict = {filtered.short_name: filtered}
    filtered_dict = filter_empty_manifests(manifests_dict)
    assert filtered.short_name in filtered_dict


def test_filter_location_and_empty_manifests_on_raw(temp_manifest_file):
    # Load the manifest as raw JSON and instantiate ModelManifest
    with open(temp_manifest_file) as f:
        raw_json = json.load(f)
    manifest = ModelManifest(**raw_json)
    filtered, changed, num_removed = filter_location(manifest)
    # Only the first location for cyto3 is accessible (URL), cyto2 is not accessible (file path)
    assert changed is True
    assert num_removed == 1
    assert "cyto3" in filtered.versions
    assert (
        "cyto2" not in filtered.versions or len(filtered.versions["cyto2"].tasks) == 0
    )
    # Now test filter_empty_manifests
    manifests_dict = {filtered.short_name: filtered}
    filtered_dict = filter_empty_manifests(manifests_dict)
    assert filtered.short_name in filtered_dict


@pytest.mark.parametrize(
    "input_value,expected",
    [
        (None, False),
        ("https://example.com/model", True),
        ("file:///nonexistent/path", False),
        ("/nonexistent/path/to/model.pt", False),
    ],
)
def test_is_accessible_param(input_value, expected):
    assert is_accessible(input_value) is expected


def test_is_accessible_with_tempfile(tmp_path):
    # Create a real file using tmp_path (pytest fixture)
    real_file = tmp_path / "afile.txt"
    real_file.write_text("test")
    assert is_accessible(str(real_file))
    # Nonexistent file in tmp_path
    assert not is_accessible(str(tmp_path / "doesnotexist.txt"))
