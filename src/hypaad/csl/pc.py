import pandas as pd

from hypaad.r_bridge import get_RBridge

__all__ = ["run_pc"]


def run_pc(
    data: pd.DataFrame,
    alpha: float,
    independence_test: str = "gaussCI",
    subset_size: int = -1,
    skeleton_method: str = "stable.fast",
    cores: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    return get_RBridge().call(
        r_file_path="pc.R",
        r_func_name="cls_pc",
        data=data,
        independence_test=independence_test,
        alpha=alpha,
        cores=cores,
        subset_size=subset_size,
        skeleton_method=skeleton_method,
        verbose=verbose,
    )
