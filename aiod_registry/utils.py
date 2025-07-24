from pathlib import Path
import json
from typing import Optional, Union
from urllib.parse import urlparse

from aiod_registry import ModelManifest


def get_manifest_paths():
    json_dir = Path(__file__).parent.parent / "aiod_registry" / "manifests"
    return json_dir.glob("*.json")


def is_accessible(location: str | None) -> bool:
    if location is None:
        return False
    res = urlparse(location)
    if res.scheme in ("file", ""):
        return Path(res.path).exists()
    else:
        return True


def flatten_manifest(manifest: ModelManifest) -> ModelManifest:
    """
    Flatten the manifest by just taking the first location and its type, same for config_path.
    """
    # Make a deep copy of the manifest
    new_manifest = manifest.model_copy(deep=True)
    # Just take the first location and its type, same for config_path
    for v_name, version in manifest.versions.items():
        for task_name, task in version.tasks.items():
            new_manifest.versions[v_name].tasks[task_name].location = task.location[0]
            new_manifest.versions[v_name].tasks[task_name].location_type = (
                task.location_type[0]
            )
            new_manifest.versions[v_name].tasks[task_name].config_path = (
                task.config_path[0] if task.config_path else None
            )
    return new_manifest


def filter_location(manifest: ModelManifest) -> tuple[ModelManifest, bool, int]:
    """
    Filter and flatten the location, location_type, and config_path fields in the manifest.
    We take the first accessible location and its type.
    Then take the first accessible config path.
    If nothing is accessible, set the fields to None.
    """
    num = 0
    changed = False
    # Make a deep copy of the manifest
    new_manifest = manifest.model_copy(deep=True)
    # Loop through the versions and tasks and remove inaccessible ones
    for v_name, version in manifest.versions.items():
        for task_name, task in version.tasks.items():
            # Check model config path and flatten
            for fpath in task.config_path:
                if is_accessible(fpath):
                    res = fpath
                    break
            else:
                res = None
            new_manifest.versions[v_name].tasks[task_name].config_path = res
            # Check which location is accessible and flatten
            for i, loc in enumerate(task.location):
                if is_accessible(loc):
                    # Store the first accessible location
                    new_manifest.versions[v_name].tasks[task_name].location = loc
                    # Flatten the related location type
                    new_manifest.versions[v_name].tasks[task_name].location_type = (
                        task.location_type[i]
                    )
                    # NOTE: Not including config path here in case not paired order
                    break
            # If no location is accessible, remove the task completely
            else:
                del new_manifest.versions[v_name].tasks[task_name]
                changed = True
                num += 1
    return new_manifest, changed, num


def filter_empty_manifests(
    manifests: dict[str, ModelManifest],
) -> dict[str, ModelManifest]:
    # Track whether the whole manifest is empty
    remove = []
    for manifest in manifests.values():
        # Only keep versions that have a task remaining
        manifest.versions = {
            k: v for k, v in manifest.versions.items() if len(v.tasks) > 0
        }
        # If there are no versions, remove the manifest
        if len(manifest.versions) == 0:
            remove.append(True)
        else:
            remove.append(False)
    # Remove the empty manifests
    return {
        manifest.short_name: manifest
        for manifest, remove in zip(manifests.values(), remove)
        if not remove
    }


def load_manifests(
    paths: Optional[list[Union[Path, str]]] = None,
    filter_access: bool = False,
) -> dict[str, ModelManifest]:
    if paths is None:
        paths = get_manifest_paths()
    manifests = {}
    for path in paths:
        with open(path, "r") as f:
            json_manifest = json.load(f)
            manifest = ModelManifest(**json_manifest)
            manifests[manifest.short_name] = manifest
    # Remove those model versions that are not accessible (if a path is provided)
    if filter_access:
        # Track how many versions are removed
        num_versions_removed = 0
        # Dict to store the new manifests
        new_manifests = {}
        # Check that something has been changed, to allow for early return
        changed = False
        for manifest in manifests.values():
            new_manifest, changed_i, num = filter_location(manifest)
            # Needed now as filtering is encapsulated in a function
            if changed_i:
                changed = True
            num_versions_removed += num
            new_manifests[new_manifest.short_name] = new_manifest
        # Check how much of each manifest remains and prune if necessary
        if changed:
            # Print the number of versions removed
            print(f"Removed {num_versions_removed} inaccessible version(s)!")
            new_manifests = filter_empty_manifests(new_manifests)
            if len(new_manifests) != len(manifests):
                print(
                    f"Removed {len(manifests) - len(new_manifests)} empty manifest(s)!"
                )
            return new_manifests
        else:
            # NOTE: Locations etc. at least get flattened so we return the new manifests
            return new_manifests
    else:
        # We still want to flatten the manifests for consistency
        return {k: flatten_manifest(v) for k, v in manifests.items()}
