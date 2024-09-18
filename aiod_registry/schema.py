import builtins
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
    "nuclei": "Nuclei",
    "cyto": "Cytoplasm",
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
    Union[str, int, float, bool, None, list[Union[str, int, float, bool]]],
    Field(
        ...,
        description="Default parameter value. If a list, the parameters will be treated as dropdown choices, where the first is the default. The type of the first element will be used to determine the type of the parameter.",
    ),
]
Usage = Annotated[
    Union[str, Path, AnyUrl],
    Field(
        ...,
        title="Usage Guide",
        description="A path to a file, a URL, or a string containing the usage guide for the model.",
    ),
]


def print_attr(attr, br: bool = True):
    "Shorthand to print something in brackets or not, only if not None."
    if attr is None:
        return ""
    if br:
        return f"({attr})"
    else:
        return f"{attr}"


def shorten_name(name: str) -> str:
    return "_".join(name.lower().split(" "))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelParam(StrictModel):
    name: ParamName
    arg_name: Optional[str] = None
    value: ParamValue
    tooltip: Optional[str] = None
    dtype: Optional[str] = None  # Used of default value is None
    _dtype = None  # Determined from value if given

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
        # If None, we need a dtype to poss cast to when dealing with GUIs
        if self.value is None:
            if self.dtype is None:
                raise ValueError(
                    f"Parameter {self.name} needs a dtype if default value is None!"
                )
            else:
                if getattr(builtins, self.dtype, None) is None:
                    raise ValueError(
                        f"Parameter {self.name} has an invalid dtype ({self.dtype})!"
                    )
        else:
            self.dtype = self._dtype
        return self


class Author(StrictModel):
    name: str
    affiliation: str
    email: Optional[str] = None
    url: Optional[AnyUrl] = None
    github: Optional[str] = None
    orcid: Optional[str] = None


class Publication(StrictModel):
    title: str
    info: Annotated[
        str,
        Field(
            ...,
            description="Information on publication, whether it pertains to the model or the underlying data or something else.",
        ),
    ]
    url: AnyUrl
    year: Optional[int] = None
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

    def __str__(self):
        misc_info = (
            f"{'URL: ' + print_attr(self.url, br=False) if self.url is not None else ''}\n"
            f"{'Repo: ' + print_attr(self.repo, br=False) if self.repo is not None else ''}\n"
        )

        if self.pubs is None:
            all_pubs = ""
        else:
            all_pubs = "\nPublications:\n" + "\n-".join(
                [
                    (
                        f"{pub.title} {print_attr(pub.year)}- {pub.url}"
                        f"{', DOI: ' + print_attr(pub.doi, br=False) if pub.doi is not None else ''}\n"
                    )
                    for pub in self.pubs
                ]
            )
        return f"Description: {self.description}\n{misc_info if len(misc_info) > 0 else ''}{all_pubs}"


class ModelVersionTask(StrictModel):
    location: Union[str, list[str]] = Field(
        ...,
        description="A url or a filepath, or list of alternative locations (skipped if location does not exist/cannot be read!)",
    )
    config_path: Optional[Union[str, list[str]]] = None
    params: Optional[list[ModelParam]] = None
    location_type: Optional[Union[str, list[str]]] = None
    metadata: Optional[Metadata] = None

    @model_validator(mode="after")
    def get_location_type(self):
        # Skip if provided
        if self.location_type is not None:
            # If a single location, convert to list
            if isinstance(self.location_type, str):
                self.location_type = [self.location_type]
            return self
        if not isinstance(self.location, list):
            self.location = [self.location]
        # Create a list to store the type of location
        self.location_type = []
        # Determine the type of location
        for loc in self.location:
            res = urlparse(loc)
            if res.scheme in ("http", "https"):
                self.location_type.append("url")
            elif res.scheme in ("file", ""):
                self.location_type.append("file")
            else:
                # NOTE: Because of including "" above, it is unlikely this will be reached
                raise TypeError(
                    f"Cannot determine type (file/url) of location: {self.location}!"
                )
        return self

    @model_validator(mode="after")
    def get_config_path(self):
        if self.config_path is None:
            self.config_path = [None] * len(self.location)
        elif not isinstance(self.config_path, list):
            self.config_path = [self.config_path]
        return self


class ModelVersion(StrictModel):
    tasks: dict[Task, ModelVersionTask]
    metadata: Optional[Metadata] = None


class ModelManifest(StrictModel):
    name: str = Field(..., min_length=1, max_length=50)
    short_name: Optional[str] = None
    versions: dict[ModelName, ModelVersion]
    params: Optional[list[ModelParam]] = None
    config: Optional[Path] = None
    metadata: Metadata
    usage_guide: Optional[Usage] = None

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
