from pathlib import Path
import json
from typing import Optional, Union
from urllib.parse import urlparse
from pydantic import ValidationError

from aiod_registry import ModelManifest
from aiod_registry.schema import ModelVersion


def get_manifest_paths():
    json_dir = Path(__file__).parent.parent / "aiod_registry" / "manifests"
    return json_dir.glob("*.json")


def is_accessible(location: str | None) -> bool:
    if location is None:
        return False
    res = urlparse(location)
    if res.scheme in ("file", ""):
        try:
            return Path(res.path).exists()
        except PermissionError:
            return False
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
            new_manifest.versions[v_name].tasks[task_name].config_path = (
                task.config_path[0] if task.config_path else None
            )
    return new_manifest


def filter_location(manifest: ModelManifest) -> tuple[ModelManifest, bool, int]:
    """
    Filter and flatten the location, and config_path fields in the manifest.
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
    cache_dir: str | Path = None,
) -> dict[str, ModelManifest]:
    if paths is None:
        paths = get_manifest_paths()
    # Default cache_dir to the standard aiod_cache location
    if cache_dir is None:
        cache_dir = Path.home() / ".nextflow" / "aiod" / "aiod_cache"
    manifests = {}
    for path in paths:
        with open(path, "r") as f:
            json_manifest = json.load(f)
            manifest = ModelManifest(**json_manifest)
            manifests[manifest.short_name] = manifest

    # Load and aggregate local manifests from cache.
    # Local manifest files contain only a dict of versions (no top-level
    # manifest metadata), keyed by version name. The filename stem must
    # match the `short_name` of a globally-loaded manifest.
    if cache_dir:
        local_manifests_dir = Path(cache_dir) / "local_manifests"
        if local_manifests_dir.exists():
            for local_path in local_manifests_dir.glob("*.json"):
                short_name = local_path.stem
                if short_name not in manifests:
                    print(
                        f"Skipping local manifest {local_path.name}: "
                        f"no base manifest '{short_name}' found."
                    )
                    continue
                with open(local_path, "r") as f:
                    local_versions = json.load(f)
                manifest_params = manifests[short_name].params
                for v_name, v_data in local_versions.items():
                    try:
                        new_version = ModelVersion(**v_data)
                    except ValidationError as e:
                        raise ValueError(
                            "Mismatch between your local model registry and the global model registry. "
                            "Your local manifest was likely written against an older schema. "
                            "If the cause is unclear, please raise an issue on GitHub and attach the details below.\n"
                            f"Local manifest file: {local_path}\n"
                            f"Version key: {v_name!r}\n"
                            f"Stored data: {v_data}\n"
                            f"Validation errors: {e.errors()}"
                        ) from e
                    for task in new_version.tasks.values():
                        if task.params is None:
                            task.params = manifest_params
                    manifests[short_name].versions[v_name] = new_version

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


def add_model_local(
    model_name: str,
    model_task: str,
    location: str,
    manifest_name: str,
    finetuning_meta_data: dict,
    base_model: str,
    cache_dir: Path | str,
):
    """
    Saves the finetuned model to local cache of finetuned models
    """
    local_manifests_dir = Path(cache_dir) / "local_manifests"
    local_manifests_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_manifests_dir / (manifest_name + ".json")

    if local_path.exists():
        with open(local_path, "r") as f:
            local_data = json.load(f)
    else:
        local_data = {}

    local_data[model_name] = {
        "base_model": base_model,
        "tasks": {
            model_task: {
                "location": location,
                "finetuning_meta_data": finetuning_meta_data,
            }
        },
    }

    with open(local_path, "w") as f:
        json.dump(local_data, f, indent=2)
    print("Saved model to local model registry", str(local_manifests_dir))
