import logging
import random
import typing as t

import hypex

__all__ = []


class DataConfigMutation:
    _logger = logging.getLogger(__name__)

    def __init__(self, seed: int) -> None:
        self.seed = seed
        hypex.use_seed(seed=self.seed)

    def get_mutation(
        self,
        base_data_config: t.Dict,
        base_timeseries_config: hypex.TimeseriesConfig,
        timeseries_name: str,
        generator: "hypex.ValueGenerator",
    ) -> t.Tuple[t.Dict[str, t.Any], t.List[t.Dict[str, t.Any]], str]:
        applied_mutations = []
        data_gen_config = list(
            filter(
                lambda ts: "name" in ts.keys()
                and ts["name"] == base_timeseries_config.name,
                base_data_config["timeseries"],
            )
        )
        if len(data_gen_config) != 1:
            raise ValueError(
                f"Did not expect to find {len(data_gen_config)} timeseries with name {base_timeseries_config.name}"
            )
        mutated_data_gen_config = data_gen_config[0].copy()

        for mutation in base_timeseries_config.mutations:
            value = generator.run(**mutation.to_kwargs())

            for path in mutation.paths:
                path_sections = path.split(".")
                is_list_duplication = False
                if path_sections[0] == "__HYPAAD_LIST_DUPLICATON__":
                    is_list_duplication = True
                    path_sections = path_sections[1:]

                mutated_data_gen_config = deep_update(
                    source=mutated_data_gen_config,
                    path_sections=path_sections,
                    value=value,
                    is_list_duplication=is_list_duplication,
                )
            applied_mutations.append(
                {
                    "name": mutation.name,
                    "value": value,
                }
            )
            self._logger.info("mutated_data_gen_config: %s", mutated_data_gen_config)

        mutated_data_gen_config["name"] = timeseries_name
        num_anomalous_points = sum(
            map(
                lambda x: x["length"],
                mutated_data_gen_config.get("anomalies", []),
            )
        )
        ts_length = mutated_data_gen_config["length"]
        applied_mutations.append(
            {
                "name": "contamination",
                "value": num_anomalous_points / ts_length,
            }
        )

        return mutated_data_gen_config, applied_mutations, timeseries_name


def deep_update(
    source: t.Union[t.Dict[str, t.Any], t.List[t.Any]],
    path_sections: t.List[str],
    value: t.Any,
    is_list_duplication: bool = False,
) -> t.Dict[str, t.Any]:
    if len(path_sections) == 0:
        return value

    key = path_sections[0]

    if len(path_sections) == 1 and is_list_duplication:
        if not type(source[key]) is list:
            raise ValueError(
                f"Expected source[key] to be a list, but got {type(source[key])} for key {key} on source {source}"
            )
        source[key] = source[key] * value
        return source

    if type(source) is list:
        if not key.isdigit():
            raise ValueError(
                f"Expected list index to be an integer, but was {key} of type {type(key)}"
            )

        if int(key) < len(source):
            source[int(key)] = deep_update(
                source=source[int(key)],
                path_sections=path_sections[1:],
                value=value,
            )
    else:
        source[key] = deep_update(
            source=source[key],
            path_sections=path_sections[1:],
            value=value,
        )
    return source
