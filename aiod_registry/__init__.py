from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aiod-registry")
except PackageNotFoundError:
    __version__ = "unknown"

from aiod_registry.schema import TASK_NAMES, ModelManifest
from aiod_registry.utils import get_manifest_paths, load_manifests

__all__ = ["TASK_NAMES", "ModelManifest", "get_manifest_paths", "load_manifests"]
