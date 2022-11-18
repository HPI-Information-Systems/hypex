import typing as t

__all__ = ["get_max_anomaly_length", "get_dataset_period_size"]


def get_max_anomaly_length(gutentag_config: t.Dict[str, t.Any]):
    return max([anomaly["length"] for anomaly in gutentag_config["anomalies"]])


def get_dataset_period_size(
    gutentag_config: t.Dict[str, t.Any],
    default: float,
) -> t.Optional[float]:
    frequency = gutentag_config["base-oscillation"].get("frequency", None)
    if frequency is None:
        return default
    return 100 / frequency
