import os
from typing import List

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from . import conf
from .Rutils import as_data_frame, r_to_pandas

# Load ricu
os.environ["RICU_DATA_PATH"] = conf["ricu_data_path"]
ricu = importr('ricu')

ro.r['source']('../ricu-extensions/callbacks/callback-icu-mortality.R')
ro.r['source']('../ricu-extensions/callbacks/callback-kdigo.R')
ro.r['source']('../ricu-extensions/callbacks/callback-sepsis.R')


# ------------------------------------------------------------------------------
# Port existing and often used ricu functions 

rename_cols = ricu.rename_cols

days = ricu.days
hours = ricu.hours
mins = ricu.mins
secs = ricu.secs


# ------------------------------------------------------------------------------
# Wrap or extend other ricu functionality 

def dictionary(dir: str = '../ricu-extensions/configs', **kwargs) -> ro.ListVector:
    """Wrapper around `ricu.load_dictionary` to load the default and custom ricu concepts 

    Args:
        dir: Path to the root folder containing all concept-dicts. Defaults to 'config'.

    Returns:
        rpy2 ListVector of all concepts in the dictionary
    """
    # TODO: make this load all .config files in the subfolders
    folders = [os.path.join(dir, subdir) for subdir in os.listdir(dir)]
    return ricu.load_dictionary(cfg_dirs=folders, **kwargs)

def concepts(x: str | List[str], dict: ro.ListVector | None = None) -> ro.RObject:
    """Get one or more ricu concepts by name

    Args:
        x: name of one or more concepts
        dict: the ricu dictionary object (in R). Defaults to dictionary().

    Returns:
        ricu concepts
    """
    if dict is None: 
        dict = dictionary()

    return dict.rx(ro.StrVector(x))

def stay_windows(src: str, interval: ro.IntVector = hours(1)) -> pd.DataFrame:
    """Load the observation times for all patients in a dataset

    Args:
        src: name of the dataset of interest
        interval: interval in which to provide `start` and `end` times. Defaults to hours(1).

    Returns:
        _description_
    """
    res = ricu.stay_windows(src, interval=interval)
    res = ricu.as_win_tbl(res, index_var="start", dur_var="end", interval=interval)
    res = ricu.rename_cols(res, "stay_id", ricu.id_var(res))
    return r_to_pandas(as_data_frame(res))