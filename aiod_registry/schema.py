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
ParamName = Annotated[
    str,
    Field(
        ...,
        min_length=1,
        max_length=50,
        description="Name of the parameter. If `arg_name` is not provided, this will be used as the argument name to the underlying model.",
    ),
]
ParamValue = Annotated[
    Union[str, int, float, bool, list[Union[str, int, float, bool]]],
    Field(
        ...,
        description="Default parameter value. If a list, the parameters will be treated as dropdown choices, where the first is the default. The type of the first element will be used to determine the type of the parameter.",
    ),
]


def shorten_name(name: str) -> str:
    return "_".join(name.lower().split(" "))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelParam(StrictModel):
    name: ParamName
    arg_name: Optional[str] = None
    value: ParamValue
    tooltip: Optional[str] = None
    _dtype = None

    @model_validator(mode="after")
    def create_arg_name(self):
        if self.arg_name is None:
            self.arg_name = self.name
        return self

    @model_validator(mode="after")
    def extract_arg_type(self):
        if isinstance(self.value, list):
            self._dtype = type(self.value[0])
        else:
            self._dtype = type(self.value)
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


class Author(StrictModel):
    name: str
    affiliation: str
    email: Optional[str] = None
    url: Optional[AnyUrl] = None
    github: Optional[str] = None
    orcid: Optional[str] = None


class Publication(StrictModel):
    info: Annotated[
        str,
        Field(
            ...,
            description="Information on publication, whether it pertains to the model or the underlying data or something else.",
        ),
    ]
    url: AnyUrl
    doi: Optional[str] = None
    authors: Optional[list[Author]] = None


class Metadata(StrictModel):
    description: Annotated[
        str,
        Field(
            ...,
            description="A short description of the model to provide context.",
        ),
    ]
    authors: Optional[list[Author]] = None
    pubs: Optional[list[Publication]] = None
    url: Optional[AnyUrl] = None
    repo: Optional[AnyUrl] = None


class ModelManifest(StrictModel):
    name: str = Field(..., min_length=1, max_length=50)
    short_name: Optional[str] = None
    versions: dict[ModelName, ModelVersion]
    params: Optional[list[ModelParam]] = None
    config: Optional[Path] = None
    metadata: Metadata

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


if __name__ == "__main__":
    import json

    schema_fpath = Path(__file__).parent / "schema.json"

    # Write the schema to file
    with open(schema_fpath, "w") as f:
        f.write(json.dumps(ModelManifest.model_json_schema(), indent=2))
