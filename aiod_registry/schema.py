from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator, AnyUrl
from typing_extensions import Annotated

TASK_NAMES = {
    "mito": "Mitochondria",
    "er": "Endoplasmic Reticulum",
    "ne": "Nuclear Envelope",
    "everything": "Everything!",
}
task_names = "|".join(TASK_NAMES.keys())

# Define custom types/fields
# Centralise to make it easier to change later
# Regex pattern to match task names, ignoring case
Task = Annotated[str, Field(..., pattern=rf"^(?i:{task_names})$")]
ModelName = Annotated[str, Field(..., min_length=1, max_length=50)]
ParamName = Annotated[str, Field(..., min_length=1, max_length=50)]
ParamValue = Union[str, int, float, bool, list[Union[str, int, float, bool]]]


def shorten_name(name: str) -> str:
    return "_".join(name.lower().split(" "))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelParam(StrictModel):
    name: ParamName
    short_name: Optional[str] = None
    value: ParamValue
    tooltip: Optional[str] = None

    @model_validator(mode="after")
    def create_short_name(self):
        if self.short_name is None:
            self.short_name = shorten_name(self.name)
        return self


class ModelVersionTask(StrictModel):
    location: str = Field(
        ...,
        description="Either a url or a filepath (will be skipped if the path does not exist/cannot be read!)",
    )
    config_path: Optional[Union[Path, str]] = None
    params: Optional[list[ModelParam]] = None
    location_type: Optional[str] = None

    @model_validator(mode="after")
    def get_location_type(self):
        # Skip if provided
        if self.location_type is not None:
            return self
        # Otherwise, determine the type
        res = urlparse(self.location)
        if res.scheme in ("http", "https"):
            self.location_type = "url"
        elif res.scheme in ("file", ""):
            self.location_type = "file"
        else:
            # NOTE: Because of including "" above, it is unlikely this will be reached
            raise TypeError(
                f"Cannot determine type (file/url) of location: {self.location}!"
            )
        return self


class ModelVersion(StrictModel):
    tasks: dict[Task, ModelVersionTask]


class ModelManifest(StrictModel):
    name: str = Field(..., min_length=1, max_length=50)
    short_name: Optional[str] = None
    versions: dict[ModelName, ModelVersion]
    params: Optional[list[ModelParam]] = None
    config: Optional[Path] = None

    @model_validator(mode="after")
    def create_short_name(self):
        if self.short_name is None:
            self.short_name = shorten_name(self.name)
        return self

    # Embed base model params into each version if not provided
    @model_validator(mode="after")
    def fill_empty_params(self):
        for version in self.versions.values():
            for task in version.tasks.values():
                if task.params is None:
                    task.params = self.params
