"""Data backend selected by `backend` in config.yaml: ricu (default) or WeavEHR.

Both provide `stay_windows`, `load_concepts` and `hours` with the same interface.
"""
from typing import List

import pandas as pd

from . import conf

backend = conf.get("backend", "ricu")
if backend == "ricu":
    import rpy2.robjects as ro

    from .ricu import concepts, hours, ricu, stay_windows
    from .Rutils import as_data_frame, r_to_pandas

    def load_concepts(x: str | List[str], src: str, **kwargs) -> pd.DataFrame:
        """Load one or more ricu concepts with uniform `stay_id` and `time` columns"""
        if not isinstance(x, list):
            x = [x]
        res = ricu.load_concepts(concepts(x), src, **kwargs)

        # Rename columns uniformly across datasets
        old = ricu.id_var(res)
        new = ['stay_id'] # TODO: only supports loading at the stay_id level
        if ricu.is_ts_tbl(res)[0]:
            old += ricu.index_var(res)
            new += ['time']

        res = ricu.rename_cols(res, ro.StrVector(new), old)

        return r_to_pandas(as_data_frame(res))
elif backend == "weavehr":
    from .weavehr import hours, load_concepts, stay_windows
else:
    raise ValueError(f"backend in config.yaml must be 'ricu' or 'weavehr', got {backend!r}")
