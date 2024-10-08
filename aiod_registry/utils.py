from pathlib import Path
import json
from typing import Optional, Union
from urllib.parse import urlparse

from aiod_registry import ModelManifest


def get_manifest_paths():
    json_dir = Path(__file__).parent.parent / "aiod_registry" / "manifests"
    return json_dir.glob("*.json")


def is_accessible(location: str) -> bool:
    res = urlparse(location)
    if res.scheme in ("file", ""):
        return Path(res.path).exists()
    else:
        return True


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
                            new_manifest.versions[v_name].tasks[
                                task_name
                            ].location = loc
                            # Flatten the related location type
                            new_manifest.versions[v_name].tasks[
                                task_name
                            ].location_type = task.location_type[i]
                            # NOTE: Not including config path here in case not paired order
                            break
                    # If no location is accessible, remove the task completely
                    else:
                        del new_manifest.versions[v_name].tasks[task_name]
                        changed = True
                        num_versions_removed += 1
            new_manifests[new_manifest.short_name] = new_manifest
        # Check how much of each manifest remains and prune if necessary
        if changed:
            # Print the number of versions removed
            print(f"Removed {num_versions_removed} inaccessible version(s)!")
            # Track whether the whole manifest is empty
            remove = []
            for manifest in new_manifests.values():
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
            new_manifests = {
                manifest.short_name: manifest
                for manifest, remove in zip(new_manifests.values(), remove)
                if not remove
            }
            if len(new_manifests) != len(manifests):
                print(
                    f"Removed {len(manifests) - len(new_manifests)} empty manifest(s)!"
                )
            return new_manifests
        else:
            # NOTE: Locations etc. at least get flattened
            return new_manifests
    else:
        return manifests
