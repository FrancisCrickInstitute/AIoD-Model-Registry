import json
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator, AnyUrl


def shorten_name(name: str) -> str:
    return "_".join(name.lower().split(" "))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelVersionTask(StrictModel):
    # Regex pattern to match task names, ignoring case
    task: str = Field(..., pattern=r"^(?i:mito|er|ne|everything)$")
    location: Union[Path, AnyUrl, str] = Field(
        ...,
        description="Either a url or a filepath (will be skipped if the path does not exist/cannot be read)",
    )
    config_path: Optional[Union[Path, str]] = None


class ModelVersion(StrictModel):
    name: str = Field(..., min_length=1, max_length=50)
    tasks: list[ModelVersionTask]


class ModelParam(StrictModel):
    name: str = Field(..., min_length=1, max_length=50)
    short_name: Optional[str] = None
    value: Union[str, int, float, bool, list[Union[str, int, float, bool]]]
    tooltip: Optional[str] = None

    @model_validator(mode="after")
    def create_short_name(self):
        if self.short_name is None:
            self.short_name = shorten_name(self.name)
        return self


class ModelManifest(StrictModel):
    name: str = Field(..., min_length=1, max_length=50)
    short_name: Optional[str] = None
    versions: list[ModelVersion]
    params: Optional[list[ModelParam]] = None
    config: Optional[Path] = None

    @model_validator(mode="after")
    def create_short_name(self):
        if self.short_name is None:
            self.short_name = shorten_name(self.name)
        return self
