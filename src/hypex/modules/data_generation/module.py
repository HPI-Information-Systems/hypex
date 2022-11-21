import logging
import typing as t
from dataclasses import dataclass
from pathlib import Path

import hypex

from ..base_module import BaseModule
from ..base_result import BaseResult
from .generator import DataGenerator

__all__ = ["DataGenerationModule"]

SEED = 1
LOCK_FILENAME = "data-gen.lock"


class DataGenerationModule(BaseModule):

    _logger = logging.getLogger(__name__)

    @dataclass
    class Result(BaseResult):
        data_paths: t.Dict[str, t.Dict[str, str]]
        applied_mutations: t.Dict[str, t.List[t.Dict[str, t.Any]]]
        gutentag_configs: t.Dict[str, t.Any]

        def save(self, study_name: str, base_output_dir: Path) -> None:
            output_dir = base_output_dir / study_name / "data_generation"
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_dict(
                path=output_dir / "data_paths.json",
                data={
                    ts_name: {k: str(v) for k, v in paths.items()}
                    for ts_name, paths in self.data_paths.items()
                },
            )
            self._save_dict(
                path=output_dir / "timeseries_mutations.json",
                data=self.applied_mutations,
            )
            self._save_dict(
                path=output_dir / "gutentag_configs.json",
                data=self.gutentag_configs,
            )

        @classmethod
        def load(
            cls, study_name: str, base_output_dir: Path
        ) -> "DataGenerationModule.Result":
            output_dir = base_output_dir / study_name / "data_generation"
            data_paths = cls._load_dict(path=output_dir / "data_paths.json")
            applied_mutations = cls._load_dict(
                path=output_dir / "timeseries_mutations.json"
            )
            gutentag_configs = cls._load_dict(path=output_dir / "gutentag_configs.json")
            return cls(
                data_paths={
                    ts_name: {k: Path(v) for k, v in paths.items()}
                    for ts_name, paths in data_paths.items()
                },
                applied_mutations=applied_mutations,
                gutentag_configs=gutentag_configs,
            )

    def prepare(self, output_dir: Path) -> None:
        path = output_dir / LOCK_FILENAME
        if path.exists():
            path.unlink(missing_ok=True)

    def _run(
        self,
        output_dir: Path,
        base_data_config: t.Dict[str, t.Any],
        base_timeseries_config: hypex.TimeseriesConfig,
        generator: "hypex.ValueGenerator",
    ) -> t.Tuple[t.List[t.Dict[str, str]], t.Dict[str, t.Any], t.Dict[str, t.Any]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_dir / LOCK_FILENAME, "x"):
                applied_mutations, gutentag_configs = DataGenerator(seed=self.seed).run(
                    base_data_config=base_data_config,
                    base_timeseries_config=base_timeseries_config,
                    output_dir=output_dir,
                    generator=generator,
                )

                return (
                    [
                        {
                            ts_name: {
                                "supervised": output_dir
                                / ts_name
                                / "train_anomaly.csv",
                                "semi-supervised": output_dir
                                / ts_name
                                / "train_no_anomaly.csv",
                                "unsupervised": output_dir / ts_name / "test.csv",
                            }
                        }
                        for ts_name in applied_mutations.keys()
                    ],
                    applied_mutations,
                    gutentag_configs,
                )
        except FileExistsError:
            self._logger.info(
                "Could not aquire lock. There must be another worker generating the data. "
                "Thus skipping the data generation."
            )
            return []

    def run(
        self,
        cluster: hypex.ClusterInstance,
        output_dir: Path,
        base_timeseries_config: hypex.TimeseriesConfig,
        base_data_config: t.Dict[str, t.Any],
        generator: "hypex.ValueGenerator",
    ) -> Result:
        self._logger.info("Preparing data generation.")
        cluster.client.run(
            self.prepare,
            output_dir=output_dir,
        )
        self._logger.info("Running data generation.")
        _retval = list(
            cluster.client.run(
                self._run,
                output_dir=output_dir,
                base_data_config=base_data_config,
                base_timeseries_config=base_timeseries_config,
                generator=generator,
            ).values()
        )
        self._logger.info("Data generation returned %s", _retval)

        data_paths, applied_mutations, gutentag_configs = {}, {}, {}
        for entry in _retval:
            if len(entry) == 3:
                paths, mutations, ts_configs = entry
                for path in paths:
                    data_paths.update(path)
                applied_mutations.update(mutations)
                gutentag_configs.update(ts_configs)

        return self.Result(
            data_paths=data_paths,
            applied_mutations=applied_mutations,
            gutentag_configs=gutentag_configs,
        )
