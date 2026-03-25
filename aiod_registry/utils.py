from pathlib import Path
import json
from typing import Optional, Union
from urllib.parse import urlparse

import yaml

from aiod_registry import ModelManifest


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


def _params_to_yaml(params: list) -> str:
    """Serialise a list of ModelParam to a YAML string keyed by arg_name."""
    defaults = {}
    for param in params:
        if isinstance(param.value, list):
            defaults[param.arg_name] = param.default if param.default is not None else param.value[0]
        else:
            defaults[param.arg_name] = param.value
    return yaml.dump(defaults, sort_keys=False, default_flow_style=False, allow_unicode=True)


def generate_default_config(manifest: ModelManifest, version: str, task: str) -> str:
    """Return a YAML string of default parameter values for a given model version and task.

    For list-valued params, the effective default is `param.default` if set, otherwise
    the first element. Returns an empty YAML mapping if the task has no params.

    Raises KeyError if the version or task is not found in the manifest.
    """
    if version not in manifest.versions:
        raise KeyError(
            f"Version '{version}' not found in manifest '{manifest.name}'. "
            f"Available versions: {list(manifest.versions.keys())}"
        )
    version_obj = manifest.versions[version]
    if task not in version_obj.tasks:
        raise KeyError(
            f"Task '{task}' not found in version '{version}' of manifest '{manifest.name}'. "
            f"Available tasks: {list(version_obj.tasks.keys())}"
        )
    task_obj = version_obj.tasks[task]
    if not task_obj.params:
        return yaml.dump({}, default_flow_style=False)
    return _params_to_yaml(task_obj.params)


def save_all_default_configs(output_dir: Union[Path, str] = "default_configs") -> None:
    """Generate and save default parameter config YAML files for all models.

    Saves one shared ``{short_name}.yaml`` for model-level params, plus a
    task-specific ``{short_name}_{version}_{task}.yaml`` only for tasks that
    define their own params (i.e. not inherited from the model level).
    Tasks and models with no params at all are skipped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    header = "# Auto-generated by save_all_default_configs — do not edit manually\n"
    manifests = load_manifests()
    for manifest in manifests.values():
        # Write one shared config for model-level params
        if manifest.params:
            config_str = _params_to_yaml(manifest.params)
            filepath = output_dir / f"{manifest.short_name}.yaml"
            with open(filepath, "w") as f:
                f.write(header)
                f.write(config_str)
            print(f"Saved {filepath}")

        # Write task-specific configs only where the task has its own params
        for version_name, version in manifest.versions.items():
            for task_name, task_obj in version.tasks.items():
                if task_obj._params_inherited:
                    continue  # Already covered by the model-level config
                if not task_obj.params:
                    print(f"Skipping {manifest.short_name}/{version_name}/{task_name} — no params defined.")
                    continue
                config_str = _params_to_yaml(task_obj.params)
                safe_version = version_name.replace(" ", "_")
                filename = f"{manifest.short_name}_{safe_version}_{task_name}.yaml"
                filepath = output_dir / filename
                with open(filepath, "w") as f:
                    f.write(header)
                    f.write(config_str)
                print(f"Saved {filepath}")
