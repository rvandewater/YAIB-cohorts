import os
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from meds_etl.flat import convert_flat_to_meds, convert_meds_to_flat

def output_clairvoyance(data_dir, save_dir, task_type="static"):
    """""Output in clairvoyance format including train test splitting
    Args:
        data_dir: Path to directory where existing parquet data is stored.
        save_dir: Path to directory where output should be saved.
        task_type: Type of task to be performed. Either "static" or "dynamic".

    """

    # Read in parquet files
    outc = pq.read_table(os.path.join(data_dir, 'outc.parquet')).to_pandas()
    dyn = pq.read_table(os.path.join(data_dir, 'dyn.parquet')).to_pandas()
    sta = pq.read_table(os.path.join(data_dir, 'sta.parquet')).to_pandas()

    # Rename as clairvoyance expects id column to be named id
    outc.rename(columns={"stay_id": "id"}, inplace=True)
    dyn.rename(columns={"stay_id": "id"}, inplace=True)
    sta.rename(columns={"stay_id": "id"}, inplace=True)

    os.makedirs(save_dir, exist_ok=True)
    # Turn dynamic data into long format as clairvoyance expects it
    dyn = dyn.melt(id_vars=['id', 'time'])
    # Remove the null values for the missing values
    dyn = dyn[~pd.isnull(dyn['value'])]
    # Split data as clairvoyance expects seperate train and test data
    data = make_train_test({"static": sta, "dynamic": dyn, "outcome": outc}, task_type=task_type, seed=42, train_size=0.8)
    for key, value in data.items():
        # Merge the outcome as Clairvoyance has no seperate outcome file
        if task_type == "static":
            value["static"] = value["static"].merge(value["outcome"], on='id', how='left')
        else:
            # Merge on time as well as id for each dynamic value.
            value["dynamic"] = value["dynamic"].merge(value["outcome"], on=['id',"time"], how='left')
    for key, value in data.items():
        # Save data as csv files
        value["static"].to_csv(os.path.join(save_dir, f'static_{key}_data.csv'), index=False)
        value["dynamic"].to_csv(os.path.join(save_dir, f'temporal_{key}_data.csv'), index=False)

def meds_etl_test(data_dir, save_dir, base_time = datetime(2000,1,1,0)):
    """"
    Test roundtrip from YAIB format -> MEDS_flat -> MEDS -> MEDS_flat
    """

    # Dynamic data
    dyn = pq.read_table(os.path.join(data_dir, 'dyn.parquet')).to_pandas()
    # MEDS does not work well with timedelta, so we need to convert to absolute time
    dyn["time"] = dyn["time"] + base_time
    # print(dyn.head())
    dyn = dyn.melt(id_vars=['stay_id', 'time'])
    dyn = dyn.dropna()
    # print(dyn.head())
    dyn.rename(columns={"stay_id": "patient_id", "variable":"code", "value":"numeric_value"}, inplace=True)

    # Static data
    sta = pq.read_table(os.path.join(data_dir, 'sta.parquet')).to_pandas()
    # print(sta.head())
    sta = sta.melt(id_vars=['stay_id'])
    # print(sta.head())
    sta.rename(columns={"stay_id": "patient_id", "variable":"code", "value":"numeric_value"}, inplace=True)
    # sta["numeric_value"] = sta["numeric_value"].replace({"Male":0, "Female":1})
    sta["text_value"] = None
    sta.loc[sta["numeric_value"]=="Male", "text_value"]="Male"
    sta.loc[sta["numeric_value"]=="Female", "text_value"]="Female"
    sta["numeric_value"] = sta["numeric_value"].replace({"Male":0, "Female":1})
    sta["numeric_value"] = sta["numeric_value"].astype(float)
    sta["time"] = base_time

    # Create folder
    Path(os.path.join(save_dir, 'flat_data')).mkdir(exist_ok=True)
    dyn.to_parquet(os.path.join(save_dir, 'flat_data/dyn.parquet'))
    sta.to_parquet(os.path.join(save_dir, 'flat_data/sta.parquet'))
    # print(dyn)
    # concatenated = pd.concat([dyn, sta])
    # print(concatenated)
    # concatenated.to_parquet(os.path.join(save_dir, 'flat_data/concatenated.parquet'))
    if not os.path.exists(os.path.join(save_dir, 'data')):
        convert_flat_to_meds(source_flat_path=save_dir, target_meds_path=save_dir, num_shards=1)
    if not os.path.exists(os.path.join(save_dir, 'flat_converted_back')):
        convert_meds_to_flat(source_meds_path=save_dir, target_flat_path=os.path.join(save_dir,"flat_converted_back"), format="parquet")

def convert_meds_to_yaib(data_dir, save_dir, base_time = datetime(2000,1,1,0)):
    meds = pq.read_table(os.path.join(data_dir, 'dyn.parquet')).to_pandas()
    meds.pivot(index=['patient_id', 'time'], columns='code', values='numeric_value', inplace=True)
    meds["time"] = meds["time"] - base_time
    meds.to_parquet(os.path.join(save_dir, 'dyn.parquet'))

def make_train_test(
        data: dict[pd.DataFrame],
        train_size=0.8,
        seed: int = 42,
        task_type: str = "static",
) -> dict[dict[pd.DataFrame]]:
    """Randomly split the data into training and validation sets for fitting a full model.

    Args:
        data: dictionary containing data divided int OUTCOME, STATIC, and DYNAMIC.
        vars: Contains the names of columns in the data.
        train_size: Fixed size of train split (including validation data).
        seed: Random seed.
        debug: Load less data if true.
    Returns:
        Input data divided into 'train', 'val', and 'test'.
    """
    # ID variable
    id = "id"
    label = "label"

    # Get stay IDs from outcome segment
    stays = pd.Series(data["outcome"][id].unique(), name=id)

    # If there are labels, and the task is classification, use stratified k-fold
    if task_type == "static":
        # Get labels from outcome data (takes the highest value (or True) in case seq2seq classification)
        labels = data["outcome"].groupby(id).max()[label].reset_index(drop=True)
        train_test = StratifiedShuffleSplit(train_size=train_size, random_state=seed, n_splits=1)
        train, test = list(train_test.split(stays, labels))[0]
    else:
        # If there are no labels or it is a regression task, use random split
        train_test = ShuffleSplit(train_size=train_size, random_state=seed)
        train, test = list(train_test.split(stays))[0]

    split_ids = {
        "train": stays.iloc[train],
        "test": stays.iloc[test]
    }

    data_split = {"train": {}, "test": {}}
    for split in split_ids.keys():  # Loop through splits (train  / test)
        data_split[split]["static"] = data["static"].merge(split_ids[split], on=id, how="right", sort=True)
        data_split[split]["dynamic"] = data["dynamic"].merge(split_ids[split], on=id, how="right", sort=True)
        data_split[split]["outcome"] = data["outcome"].merge(split_ids[split], on=id, how="right", sort=True)
    return data_split
dir = os.path.dirname(__file__)
meds_etl_test(Path("/mnt/c/Users/Robin/Documents/Git/YAIB-cohorts/data/mortality24/mimic_demo/"), Path("/mnt/c/Users/Robin/Documents/Git/YAIB-cohorts/data/mortality24/mimic_demo/flat_data"))