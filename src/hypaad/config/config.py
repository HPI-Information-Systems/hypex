import os
import typing as t

import jsonschema
import yaml

from .study import Study

__all__ = ["Config"]

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "schema.yml"
)


class Config:
    @classmethod
    def _read_yml(cls, path: str) -> t.Dict[str, t.Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @classmethod
    def load(cls, path: str) -> t.List["Study"]:
        config = cls._read_yml(path)
        schema = cls._read_yml(SCHEMA_PATH)
        jsonschema.validate(config, schema)

        studies = []
        for s in config["studies"]:
            studies.append(Study.from_config(s))
        return studies
