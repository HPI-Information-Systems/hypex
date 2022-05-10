import logging
import os
import shutil
import typing as t
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

r_source = robjects.r["source"]

__all__ = ["get_RBridge", "RBridge"]


class RBridge:
    _instance: t.Optional["RBridge"] = None
    _logger = logging.getLogger(__name__)

    def __init__(self, RScripts_folder: str) -> None:
        self.RScripts_folder = RScripts_folder
        self._prepare()

    def _prepare(self):
        self._logger.info(
            "Now preparing RScripts folder '%s'", self.RScripts_folder
        )
        package_path = Path(__file__).parent.parent
        target_dir = package_path.parent / "hypaad_RScripts"
        target_dir.mkdir(exist_ok=True)
        with ZipFile(str(package_path)) as zip_obj:
            for file in zip_obj.filelist:
                if file.filename.startswith(self.RScripts_folder):
                    target_path = (
                        target_dir
                        / file.filename[len(self.RScripts_folder) + 1 :]
                    )
                    zip_obj.extract(member=file, path=f"{target_path}.tmp")
                    tmp_path = Path(f"{target_path}.tmp") / file.filename
                    os.rename(src=tmp_path, dst=target_path)
                    shutil.rmtree(f"{target_path}.tmp")
                    self._logger.info(
                        "Extracted file %s to %s", file.filename, target_path
                    )
        self.extracted_RScripts = target_dir

    @classmethod
    def instance(cls, RScripts_folder=Path) -> "RBridge":
        if cls._instance is None:
            cls._logger.info(
                "RBridge was not initialized yet, thus initialzing now..."
            )
            cls._instance = RBridge(RScripts_folder=RScripts_folder)

        if cls._instance.RScripts_folder != RScripts_folder:
            raise ValueError(
                "Cannot handle multiple RScript folders. "
                f"Found '{cls._instance.RScripts_folder}' and '{RScripts_folder}'"
            )
        return cls._instance

    @classmethod
    def transform(cls, obj: t.Any):
        _transform = _converters.get(type(obj), lambda x: x)
        return _transform(obj)

    @classmethod
    def transform_from_pandas_dataframe(
        cls, data: pd.DataFrame
    ) -> robjects.vectors.DataFrame:
        with localconverter(robjects.default_converter + pandas2ri.converter):
            return robjects.conversion.py2rpy(data)

    @classmethod
    def transform_to_pandas_dataframe(
        cls, data: robjects.vectors.DataFrame
    ) -> pd.DataFrame:
        with localconverter(robjects.default_converter + pandas2ri.converter):
            return robjects.conversion.rpy2py(data)

    def call(self, r_file_path: str, r_func_name: str, *args, **kwargs):
        r_source(str(self.extracted_RScripts / r_file_path), chdir=True)
        r_func = robjects.globalenv[r_func_name]
        r_args = [RBridge.transform(obj=arg) for arg in args]
        r_kwargs = {k: RBridge.transform(obj=v) for k, v in kwargs.items()}
        return r_func(*r_args, **r_kwargs)


def get_RBridge():
    return RBridge.instance(RScripts_folder="hypaad/RScripts")


_converters = {
    pd.DataFrame: RBridge.transform_from_pandas_dataframe,
    robjects.vectors.DataFrame: RBridge.transform_to_pandas_dataframe,
}
