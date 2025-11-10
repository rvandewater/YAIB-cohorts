# Test change
import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
from pandas import DataFrame, Series, option_context, isna

import numpy as np

import rpy2
import rpy2.robjects as robjects

from src.cohort import Cohort, SelectionCriterion
from src.ricu import stay_windows
from src.ricu_utils import (
    longest_rle,
    make_grid_mapper,
    make_outcome_windower,
    make_patient_mapper,
    make_prevalence_calculator,
    n_obs_per_row,
    stop_window_at,
)
from src.steps import (
    AggStep,
    CombineStep,
    CustomStep,
    DropStep,
    FilterStep,
    InputStep,
    LoadStep,
    Pipeline,
    TransformStep,
    RenameStep,
)

class args:
    src = "mimic_demo"
# https://eth-mds.github.io/ricu/reference/callback_sofa.html
outc_vars = ["sofa", "susp_inf_alt", "sep3_alt", "los_icu"]
static_vars = ["age", "sex", "height", "weight"]
dynamic_vars = [
    "alb",
    "alp",
    "alt",
    "ast",
    "be",
    "bicar",
    "bili",
    "bili_dir",
    "bnd",
    "bun",
    "ca",
    "cai",
    "ck",
    "ckmb",
    "cl",
    "crea",
    "crp",
    "dbp",
    "fgn",
    "fio2",
    "glu",
    "hgb",
    "hr",
    "inr_pt",
    "k",
    "lact",
    "lymph",
    "map",
    "mch",
    "mchc",
    "mcv",
    "methb",
    "mg",
    "na",
    "neut",
    "o2sat",
    "pco2",
    "ph",
    "phos",
    "plt",
    "po2",
    "ptt",
    "resp",
    "sbp",
    "temp",
    "tnt",
    "urine",
    "wbc",
]

print("Start creating the SOFA task.")
print("   Preload variables")
sepsis_args = {}

# Load the concepts SOFA, suspected infection and Sepsis3
with robjects.default_converter.context():
    sofa = LoadStep(outc_vars, args.src, cache=True, **sepsis_args).perform()
    sepsis = LoadStep(["sep3_alt"], args.src, cache=True, **sepsis_args).perform()
    los_icu = LoadStep(["los_icu"], args.src, cache=True, **sepsis_args).perform()
    
    static = LoadStep(static_vars, args.src, cache=True).perform()
    dynamic = LoadStep(dynamic_vars, args.src, cache=True).perform()
assert isinstance(sofa, DataFrame), "Could not get SOFA DataFrame"
assert isinstance(dynamic, DataFrame), "Could not get Dynamic DataFrame"
assert isinstance(static, DataFrame), "Could not get Static DataFrame"
csofa = sofa.copy()
print("SOFA", len(sofa), "Sepsis", len(sepsis), "LOS", len(los_icu), "Dyn", len(dynamic), "stat", len(static))
patients = stay_windows(args.src)
patients = stop_window_at(patients, end=24 * 7)

sepsis_add6 = sepsis[sepsis["time"] >= 0].copy()
sepsis_add6["time"] += 6
patients = stop_window_at(patients, end=sepsis_add6)
# SEPSIS based
print("   Define observation times")
patients = stay_windows(args.src)
patients = stop_window_at(patients, end=24 * 7)

print("   Define exclusion criteria")
# General exclusion criteria
excl1 = SelectionCriterion("Invalid length of stay")
excl1.add_step([InputStep(patients), FilterStep("end", lambda x: x < 0)])

excl2 = SelectionCriterion("Length of stay < 6h")
excl2.add_step([InputStep(los_icu), FilterStep("los_icu", lambda x: x < 6 / 24)])

excl3 = SelectionCriterion("Less than 4 hours with any measurement")
excl3.add_step([InputStep(dynamic), AggStep("stay_id", "count"), FilterStep("time", lambda x: x < 4)])

excl4 = SelectionCriterion("More than 12 hour gap between measurements")
excl4.add_step([
    InputStep(dynamic),
    CustomStep(make_grid_mapper(patients, step_size=1)),
    CustomStep(n_obs_per_row),
    TransformStep("n", lambda x: x > 0),
    AggStep("stay_id", longest_rle, "n"),
    FilterStep("n", lambda x: x > 12),
])

excl5 = SelectionCriterion("Aged < 18 years")
excl5.add_step([InputStep(static), FilterStep("age", lambda x: x < 18)])

# Task-specific exclusion criteria
sepsis_add6 = sepsis[sepsis["time"] >= 0].copy()
sepsis_add6["time"] += 6
patients = stop_window_at(patients, end=sepsis_add6)

get_first_sepsis = Pipeline("Get patients")
get_first_sepsis.add_step([InputStep(csofa), AggStep("stay_id", "max")])
load_hospital_id = LoadStep("hospital_id", src=args.src)
excl6 = SelectionCriterion("Low sepsis prevalence")
excl6.add_step([
    CombineStep(steps=[get_first_sepsis, load_hospital_id], func=make_prevalence_calculator("sep3_alt")),
    FilterStep("prevalence", lambda x: x == 0),
])

excl7 = SelectionCriterion("Sepsis onset before 6h in the ICU")
excl7.add_step([InputStep(sepsis), AggStep(["stay_id"], lambda x: x.iloc[0]), FilterStep("time", lambda x: x < 6)])#####################


print("   Select cohort\n")
cohort = Cohort(patients)
cohort.add_criterion(
    [excl1, excl2, excl3, excl4, excl5, excl6]
    if args.src in ["eicu", "eicu_demo"]
    else [excl1, excl2, excl3, excl4, excl5, excl7]
)
print("len csofa", len(csofa))
print(cohort.criteria)
patients, attrition = cohort.select()
print("len patients", len(patients))
print("\n")
outc_formatting = Pipeline("Prepare length of stay")
outc_formatting.add_step([
    InputStep(csofa), 
    CustomStep(lambda x: x.replace({rpy2.rinterface_lib.sexp.NALogicalType(): 0}).fillna(0), "remove nans and intify"),
    CustomStep(make_grid_mapper(patients)),
    CustomStep(make_outcome_windower(1, "sofa")),
    CustomStep(make_outcome_windower(1, "susp_inf_alt")),
    CustomStep(make_outcome_windower(1, "sep3_alt")),
    CustomStep(lambda x: x.drop_duplicates(keep="last", ignore_index=False)),
    TransformStep(["susp_inf_alt", "sofa", "sep3_alt"], lambda x: x.astype(int)),
])

outc = outc_formatting.apply()
sofa_index = outc.set_index(['stay_id', 'time']).index
print("len sofa_index", len(sofa_index))
#print(dynamic)
dyn_formatting = Pipeline("Prepare dynamic variables")
dyn_formatting.add_step([
    InputStep(dynamic),
    CustomStep(make_grid_mapper(patients, step_size=1)),
    CustomStep(lambda x: x.set_index(["stay_id", "time"])),  # Set MultiIndex
    CustomStep(lambda x: x.loc[sofa_index]),  # Filter using tabshe MultiIndex
    CustomStep(lambda x: x.reset_index()),
    CustomStep(make_grid_mapper(patients, step_size=1)),
    CustomStep(lambda x: x.drop_duplicates(keep="last", ignore_index=False)),
])
dyn = dyn_formatting.apply()

combined_index = outc.drop_duplicates(
                    keep="last", ignore_index=False
                 ).index.intersection(dyn.drop_duplicates(keep="last", ignore_index=False).index)
print("len combined_index", len(combined_index))
outc = outc.reindex(combined_index)
dyn = dyn.reindex(combined_index)

sta_formatting = Pipeline("Prepare static variables")
sta_formatting.add_step([InputStep(static), CustomStep(make_patient_mapper(patients))])
sta = sta_formatting.apply()

save_dir = f"../data/sepsis_with_marginals/{args.src}/"
os.makedirs(save_dir, exist_ok=True)
print(f"Data Shapes: {outc.shape}, {dyn.shape}, {sta.shape}")
pq.write_table(pa.Table.from_pandas(outc), os.path.join(save_dir, "outc.parquet"))
pq.write_table(pa.Table.from_pandas(dyn), os.path.join(save_dir, "dyn.parquet"))
pq.write_table(pa.Table.from_pandas(sta), os.path.join(save_dir, "sta.parquet"))
attrition.to_csv(os.path.join(save_dir, "attrition.csv"))

def infection_ramp(arr, X=48, Y=24, peak_val=1.0):
    start_val = 0.1
    end_val = 0.1
    
    # ramps
    increase = start_val + (peak_val - start_val) * (np.arange(1, X + 1) / X)
    decrease = peak_val * (end_val / peak_val) ** (np.arange(1, Y + 1) / Y)
    
    arr_increase = np.zeros_like(arr, dtype=float)
    arr_decrease = np.zeros_like(arr, dtype=float)
    
    # indices of ones
    ones_idx = np.where(arr == 1)[0]
    
    # ramp up
    inc_offsets = -np.arange(X, 0, -1)  # [-X, ..., -1]
    inc_positions = ones_idx[:, None] + inc_offsets
    valid_mask = (inc_positions >= 0)  # filter out negatives
    flat_pos = inc_positions[valid_mask]
    flat_vals = np.tile(increase, (len(ones_idx), 1))[valid_mask]
    np.maximum.at(arr_increase, flat_pos, flat_vals)
    
    # ramp down
    dec_offsets = np.arange(1, Y + 1)  # [1, ..., Y]
    dec_positions = ones_idx[:, None] + dec_offsets
    valid_mask = (dec_positions < len(arr))
    flat_pos = dec_positions[valid_mask]
    flat_vals = np.tile(decrease, (len(ones_idx), 1))[valid_mask]
    np.maximum.at(arr_decrease, flat_pos, flat_vals)
    
    # combine
    result = np.maximum(arr_increase, arr_decrease)
    result[arr == 1] = peak_val
    return result


def apply_infection_ramp(df: DataFrame) -> DataFrame:
    df = df.copy()
    
    def per_stay(g):
        susp = g["susp_inf_alt"].to_numpy()
        
        ramp_main = infection_ramp(susp, X=48, Y=24, peak_val=1.0)
        
        return ramp_main
    
    df["susp_inf_ramp"] = df.groupby("stay_id", group_keys=False).apply(per_stay, include_groups=False).explode().astype(float).values
    return df


print("   Load and format outcome data")
routc_formatting = Pipeline("Prepare length of stay")
routc_formatting.add_step([InputStep(csofa)])
routc_formatting.add_step([CustomStep(apply_infection_ramp)])
routc_formatting.add_step([
    CustomStep(lambda x: x.replace({rpy2.rinterface_lib.sexp.NALogicalType(): 0}).fillna(0), "remove nans and intify"),
    CustomStep(make_grid_mapper(patients)),
    CustomStep(make_outcome_windower(1, "sofa")),
    CustomStep(make_outcome_windower(1, "susp_inf_alt")),
    CustomStep(make_outcome_windower(1, "susp_inf_ramp")),
    CustomStep(make_outcome_windower(1, "sep3_alt")),
    CustomStep(make_outcome_windower(1, "sep3_ramp")),
    CustomStep(lambda x: x.drop_duplicates(keep="last", ignore_index=False)),
    TransformStep(["susp_inf_alt", "sofa", "sep3_alt"], lambda x: x.astype(int)),
])
routc = routc_formatting.apply()

save_dir = f"../data/sepsis_with_marginals_ramp/{args.src}/"
os.makedirs(save_dir, exist_ok=True)
print(f"Data Shapes: {routc.shape}, {dyn.shape}, {sta.shape}")
pq.write_table(pa.Table.from_pandas(routc), os.path.join(save_dir, "outc.parquet"))
pq.write_table(pa.Table.from_pandas(dyn), os.path.join(save_dir, "dyn.parquet"))
pq.write_table(pa.Table.from_pandas(sta), os.path.join(save_dir, "sta.parquet"))
attrition.to_csv(os.path.join(save_dir, "attrition.csv"))

def cut_after_onset(group):
    group = group.sort_values('time')

    # Find the index label of the first True
    onset_label = group.index[group['sep3_alt'] == 1.0].min()
    if isna(onset_label):  # no onset found
        return group
    
    # Convert to position
    onset_pos = group.index.get_loc(onset_label)

    # Keep from start to onset + up to 6 rows after
    cutoff_pos = min(onset_pos + 6, len(group))
    return group.iloc[:cutoff_pos]
    
def apply_cut(df: DataFrame) -> DataFrame:
    cols = df.columns  # all columns
    return df[cols].groupby('stay_id', group_keys=False, exclude_groups=False).apply(cut_after_onset)

print("   Load and format outcome data")
routc_formatting = Pipeline("Prepare length of stay")
routc_formatting.add_step([InputStep(csofa)])
routc_formatting.add_step([CustomStep(apply_infection_ramp, desc="Add infection_ramp column based on susp_inf_alt")])
routc_formatting.add_step([
    CustomStep(lambda x: x.replace({rpy2.rinterface_lib.sexp.NALogicalType(): 0}).fillna(0), "remove nans and intify"),
    CustomStep(make_grid_mapper(patients)),
    CustomStep(make_outcome_windower(1, "sofa")),
    CustomStep(make_outcome_windower(1, "susp_inf_alt")),
    CustomStep(make_outcome_windower(1, "sep3_alt")),
    CustomStep(lambda x: x.drop_duplicates(keep="last", ignore_index=False)),
    TransformStep(["susp_inf_alt", "sofa", "sep3_alt"], lambda x: x.astype(int)),
])
routc_formatting.add_step([CustomStep(apply_cut, desc="Cut sequence after onset")])
routc = routc_formatting.apply()

combined_index = routc.drop_duplicates(
                    keep="last", ignore_index=False
                 ).index.intersection(dyn.drop_duplicates(keep="last", ignore_index=False).index)

routc = routc.reindex(combined_index)
rdyn = dyn.reindex(combined_index)

save_dir = f"../sepsis_with_marginals_ramp_short/{args.src}/"
os.makedirs(save_dir, exist_ok=True)
print(f"Data Shapes: {routc.shape}, {rdyn.shape}, {sta.shape}")
pq.write_table(pa.Table.from_pandas(routc), os.path.join(save_dir, "outc.parquet"))
pq.write_table(pa.Table.from_pandas(rdyn), os.path.join(save_dir, "dyn.parquet"))
pq.write_table(pa.Table.from_pandas(sta), os.path.join(save_dir, "sta.parquet"))
attrition.to_csv(os.path.join(save_dir, "attrition.csv"))
