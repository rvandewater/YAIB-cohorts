from typing import Callable, List
import numpy as np
import pandas as pd


def stop_window_at(x: pd.DataFrame, end: int | pd.DataFrame) -> pd.DataFrame:
    """Stop observation time at a given end date

    Args:
        x: observation times for each patient
        end: time at which to end observation, can either be a constant scalar
            for all patients or a pandas.DataFrame with a separate time per patient

    Raises:
        ValueError: `x` must contain the three columns `stay_id`, `start`, and `end`
        ValueError: if `end` is a pandas.DataFrame, it must contain a `time` column

    Returns:
        observation times truncated at end
    """
    if np.all(x.columns != ['stay_id', 'start', 'end']):
        raise ValueError('`x` must be pandas.DataFrame with three columns: stay_id, start, end')
    
    if isinstance(end, pd.DataFrame):
        if not 'time' in end.columns:
            raise ValueError('if `end` is a pandas.DataFrame, it must contain a column named `time` specifying the censor time.')
        end = end.groupby('stay_id')['time'].min().reset_index()
        x = x.merge(end, on='stay_id', how='left')
        x['time'] = x['time'].fillna(np.inf)
        x['end'] = x[['end','time']].min(axis=1)
        x = x.drop('time', axis=1)
    else:
        x['end'] = x['end'].clip(upper=end)
    return x

def make_grid_mapper(grid: pd.DataFrame, step_size: int = 1, match_time: bool = True) -> Callable:
    """Return a function that maps data to a grid

    For the use with CustomStep

    Factory Args:
        grid: the grid to map to, for example the observation times for each patient
        step_size: how granular should the grid be. Defaults to 1.
        match_time: should time be matched or should the data be joined on 
           stay_id only. Defaults to True.

    Factory Returns:
        mapper function that takes a single pandas.DataFrame as input to be mapped
    """
    grid = grid.copy()
    grid['time'] = grid.apply(lambda row: list(range(int(row['start']), int(row['end'])+1, step_size)), axis=1)
    grid = grid.drop(['start', 'end'], axis=1)
    grid = grid.explode('time')
    grid['time'] = grid['time'].astype(float)
    grid = grid[~np.isnan(grid['time'])]
    grid['time'] = grid['time'].astype(int)

    def map_to_grid(x: pd.DataFrame) -> pd.DataFrame:
        map_on = ['stay_id']
        if match_time:
            map_on += ['time']
        return x.merge(grid, on=map_on, how='right')
    return map_to_grid 

def make_patient_mapper(pop: pd.DataFrame) -> Callable:
    """Return a function that maps data to a patient population

    Factory Args:
        pop: the population to map to, i.e., a list of patient ids

    Factory Returns:
        mapper function that takes a single pandas.DataFrame as input to be mapped
    """
    pop = pop[['stay_id']].copy()

    def map_to_patients(x: pd.DataFrame) -> pd.DataFrame:
        return x.merge(pop, on=['stay_id'], how='right')
    return map_to_patients

def n_obs_per_row(x: pd.DataFrame) -> pd.DataFrame:
    """Count the number of non-NA observations per row

    Args:
        x: data to be counted

    Returns:
        a pandas.DataFrame with `stay_id`, `time` (if present), and `n`
    """
    x = x.copy()
    ids = x.columns.intersection(['stay_id', 'time'])
    x['n'] = x.notna().sum(axis=1) - len(ids)
    return x[ids.append(pd.Index(['n']))]

def longest_rle(x: pd.Series, value: bool | int | str = False) -> int:
    """Count the longest continuous run of a value

    Args:
        x: values in the order to be counted
        value: which value to look for. Defaults to False.

    Returns:
        run size
    """
    lengths = (x != x.shift()).astype(int).cumsum()
    max_rle = lengths[x == value].value_counts().max()
    if np.isnan(max_rle):
        max_rle = 0
    return max_rle

def make_prevalence_calculator(var: str) -> Callable:
    """Calculate the prevalence of a condition by hospital (e.g., sepsis)

    Factory Args:
        var: name of the ricu concept

    Factory Returns:
        callable prevalence calculator
    
    Args:
        List with two elements: 
            - data for the ricu concept
            - hospital ids for each patient

    Returns:
        prevalence at each patient's hospital
    """
    def calculate_prevalence(x: List[pd.DataFrame]) -> pd.DataFrame:
        data, hospital_ids = x
        cncpt_per_hosp = data.merge(hospital_ids, on='stay_id', how='right')
        cncpt_per_hosp[var] = ~cncpt_per_hosp[var].isnull()
        prev = cncpt_per_hosp.groupby('hospital_id')[var].mean().reset_index()
        prev = prev.rename(columns={var: 'prevalence'})
        res = hospital_ids.merge(prev, on='hospital_id')
        return res.drop('hospital_id', axis=1)
    return calculate_prevalence


def make_outcome_windower(window: int) -> Callable:
        def outcome_window(x: pd.DataFrame):
            x['label'] = x.groupby('stay_id')['label'].ffill(limit=window).bfill(limit=window).fillna(0)
            return x
        return outcome_window