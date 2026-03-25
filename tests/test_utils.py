import json
import os
import sys

import pytest
from pydantic import ValidationError

from aiod_registry.schema import ModelManifest, ModelParam
from aiod_registry.utils import (
    filter_empty_manifests,
    filter_location,
    flatten_manifest,
    is_accessible,
    load_manifests,
)

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


def test_filter_location_no_change(tmp_path):
    # Manifest with all accessible locations (URLs)
    manifest_data = {
        "name": "TestModel",
        "short_name": "testmodel",
        "metadata": {
            "description": "Test model with all accessible locations.",
        },
        "versions": {
            "v1": {
                "tasks": {
                    "cyto": {
                        "location": [
                            "https://example.com/model1",
                            "https://example.com/model2",
                        ],
                    }
                }
            }
        },
        "params": [{"name": "Param1", "arg_name": "param1", "value": 1}],
    }
    manifest = ModelManifest(**manifest_data)
    filtered, changed, num_removed = filter_location(manifest)
    assert changed is False
    assert num_removed == 0
    # The task should still exist
    assert "cyto" in filtered.versions["v1"].tasks


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


@pytest.mark.skipif(
    sys.platform == "win32" or os.getuid() == 0,
    reason="Cannot restrict permissions on Windows or as root",
)
def test_is_accessible_permission_denied(tmp_path):
    # Reproduce: file exists but os.stat raises EACCES (errno 13) because the
    # parent directory has its execute/search bit removed.
    # In Python 3.12+, Path.exists() only suppresses ENOENT/ENOTDIR/EBADF/ELOOP;
    # EACCES propagates, so is_accessible raises PermissionError instead of
    # returning False.
    subdir = tmp_path / "restricted"
    subdir.mkdir()
    model_file = subdir / "model.pt"
    model_file.write_text("fake model weights")

    # Confirm the file is accessible before we restrict it
    assert is_accessible(str(model_file))

    # Remove execute (search) bit from the parent directory so that any
    # os.stat() on a path inside it raises PermissionError (errno 13)
    subdir.chmod(0o666)
    try:
        # Bug: PermissionError propagates out of is_accessible instead of
        # being caught and returning False
        result = is_accessible(str(model_file))
        assert result is False
    finally:
        # Restore permissions so that pytest's tmp_path cleanup can delete the dir
        subdir.chmod(0o755)


class TestModelParamDefault:
    def test_list_no_default_uses_first(self):
        """Without `default`, the first list item determines dtype and is the implicit default."""
        p = ModelParam(name="mode", value=["fast", "slow", "accurate"])
        assert p.default is None
        assert p._dtype is str

    def test_list_default_non_first_item(self):
        """Setting `default` to a non-first list item is accepted and reflected in _dtype."""
        p = ModelParam(name="mode", value=["fast", "slow", "accurate"], default="accurate")
        assert p.default == "accurate"
        assert p._dtype is str

    def test_list_default_int(self):
        """Integer default picks the correct dtype."""
        p = ModelParam(name="level", value=[1, 2, 3], default=3)
        assert p.default == 3
        assert p._dtype is int

    def test_default_not_in_list_raises(self):
        """A `default` value that is not in the choices list must raise a ValidationError."""
        with pytest.raises(ValidationError, match="not in the choices list"):
            ModelParam(name="mode", value=["fast", "slow"], default="medium")

    def test_default_on_scalar_raises(self):
        """`default` is only valid for list values; a scalar value must raise a ValidationError."""
        with pytest.raises(ValidationError, match="only be set when `value` is a list"):
            ModelParam(name="thresh", value=0.5, default=0.5)
