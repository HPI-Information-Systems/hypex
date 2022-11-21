import logging
import typing as t
from pathlib import Path

from gutenTAG import GutenTAG

import hypex

from .config_mutation import DataConfigMutation

__all__ = []


class DataGenerator:
    _logger = logging.getLogger(__name__)

    def __init__(self, seed: int) -> None:
        self.seed = seed
        hypex.use_seed(seed=self.seed)

    def _generate(
        self,
        gutentag_config: t.Dict[str, t.Any],
        output_dir: Path,
    ) -> t.Tuple[t.Dict[str, t.Any], str]:
        self._logger.info("Now generating data for configuration %s", gutentag_config)

        data_gen = GutenTAG(seed=self.seed, n_jobs=1)
        print("gutentag_config:", gutentag_config)
        data_gen.load_config_dict({"timeseries": [gutentag_config]})
        data_gen.generate(
            output_folder=output_dir,
        )

    def run(
        self,
        base_data_config: t.Dict[str, t.Any],
        base_timeseries_config: hypex.TimeseriesConfig,
        output_dir: Path,
        generator: "hypex.ValueGenerator",
    ) -> t.Tuple[t.Dict[str, t.List[t.Dict[str, t.Any]]], t.Dict[str, t.Any]]:
        applied_mutations, gutentag_configs = {}, {}
        mutation_generator = DataConfigMutation(seed=self.seed)
        for _ in range(base_timeseries_config.n_mutations):
            (
                gutentag_config,
                _applied_mutations,
                ts_name,
            ) = mutation_generator.get_mutation(
                base_data_config=base_data_config,
                base_timeseries_config=base_timeseries_config,
                generator=generator,
            )
            self._generate(gutentag_config=gutentag_config, output_dir=output_dir)
            applied_mutations[ts_name] = _applied_mutations
            gutentag_configs[ts_name] = gutentag_config
        return applied_mutations, gutentag_configs
