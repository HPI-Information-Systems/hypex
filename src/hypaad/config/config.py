import logging
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
    _logger = logging.getLogger("Config")

    @classmethod
    def read_yml(cls, path: str) -> t.Dict[str, t.Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @classmethod
    def from_yml(cls, path: str) -> t.List[Study]:
        config = cls.read_yml(path)
        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: t.Dict[str, t.Any]) -> t.List[Study]:
        schema = cls.read_yml(SCHEMA_PATH)
        jsonschema.validate(config, schema)

        studies = []
        for s in config["studies"]:
            studies.append(Study.from_config(s))
        cls._logger.info(
            "Found %d studies %s",
            len(studies),
            list(map(lambda s: s.name, studies)),
        )
        return studies
