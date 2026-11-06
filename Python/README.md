## Replicate the cohorts from Python:

The Python scripts derive the cohorts with [WeavEHR](https://github.com/aidh-ms/WeavEHR), a Python framework that harmonises ICU datasets into the MEDS format. They do not need R; the R scripts in [../R](../R/README.md) derive the cohorts with `ricu`.

The following instructions assume that this folder is the current working directory in your Python session. 


### Step 0: Environment and Prerequisites

WeavEHR requires Python >= 3.13. To create the environment with conda, run: 

```bash
conda env create -f environment.yml
conda activate yaib-cohorts
```

[environment.yml](environment.yml) installs the WeavEHR version that the cohorts were checked against. Alternatively, install `pandas`, `numpy`, `polars`, `pyyaml`, `pyarrow`, `scikit-learn` and that WeavEHR version with pip.

> Note: the Docker image in [../docker](../docker) is set up for `ricu` (R and Python 3.10) and does not run these scripts.

### Step 1: Build the WeavEHR project of a dataset

The cohorts are read from a WeavEHR project per dataset, the counterpart of importing a dataset into `ricu`. Configured datasets (`--src`, WeavEHR version): MIMIC-IV (`mimic-iv`, 2.2) and its openly available [demo](https://physionet.org/content/mimic-iv-demo/) (`mimic-iv-demo`, 2.2), eICU-CRD (`eicu-crd`, 2.0) and its [demo](https://physionet.org/content/eicu-crd-demo/) (`eicu-demo`, 2.0), HiRID (`hirid`, 1.1.1) and AmsterdamUMCdb (`aumc`, 1.5.0, the OMOP export).

`weavehr_project_path` in [config.yaml](../config.yaml) is where the WeavEHR project of each dataset (`{src}`) is kept. It contains patient-level data, so keep it outside of this repository. Build the project from the raw data files of the dataset:

```bash
python weavehr_build.py --src mimic-iv-demo --version 2.2 --data-path /path/to/physionet.org/files/mimic-iv-demo/2.2
python weavehr_build.py --src eicu-demo --version 2.0 --data-path /path/to/physionet.org/files/eicu-crd-demo/2.0.1
```


### Step 2: Generate the Cohorts

This repository currently allows for the extraction of five prediction tasks: 

Classification:

1. ICU mortality after 24 hours: [mortality.py](mortality.py)
2. Acute Kidney Injury within the next 6 hours: [aki.py](aki.py)
3. Sepsis within the next 6 hours: [sepsis.py](sepsis.py)

Regression:

4. Kidney function on the second day of ICU admission: [kidney_function.py](kidney_function.py)
5. Remaining length of stay: [los.py](los.py)


Data for a single task like mortality from a single dataset can be extracted via: 
```bash 
python mortality.py --src mimic-iv-demo
```

where `mortality.py` should be replaced with the task file of interest and `mimic-iv-demo` with the dataset of interest (one of `mimic-iv`, `mimic-iv-demo`, `eicu-crd`, `eicu-demo`, `hirid`, `aumc`). 

The output directory for the extracted data can also be set in [config.yaml](../config.yaml).


### Clinical concepts

The clinical concepts are those defined by WeavEHR. [weavehr-extensions/](../weavehr-extensions) is the counterpart of [ricu-extensions/](../ricu-extensions): `config.yml` maps the YAIB concepts to WeavEHR concepts with ricu's value ranges and hourly aggregation, and `datasets/` adds the WeavEHR table configs and concept mappings that YAIB adds to ricu, or that reproduce ricu's items (e.g., non-invasive blood pressure in eICU, MIMIC-IV lab items 53085, 53086 and 53089). [src/weavehr.py](src/weavehr.py) attaches the concepts to ICU stays the way ricu does and computes the outcomes (`death_icu`, `aki`, `sep3_alt`, `los_icu`, `hospital_id`) following [ricu-extensions/callbacks](../ricu-extensions/callbacks).


### Differences from the cohorts extracted with ricu

The cohorts can be compared with those extracted with `ricu` using `weavehr-yaib-compare-cohorts` from [weavehr-yaib](https://github.com/TobiasVeronaLaser/weavehr-yaib). On the MIMIC-IV and eICU demos, the mortality, AKI, kidney function and length of stay cohorts, labels and variables agree with those extracted with ricu, apart from age in MIMIC-IV. Known differences:

- Age (MIMIC-IV): WeavEHR's age is the age at each hospital admission, capped at 90; ricu uses the patient's `anchor_age`.
- Sepsis: SOFA is WeavEHR's score, the sum of the worst value of each component within 24h as in ricu, but evaluated only at component events rather than every hour (WeavEHR emits no event when a score leaves the 24h window), so `sep3_alt` compares the hours with a SOFA event; microbiology samples recorded by date only are timed at the end of that day (ricu: midday).
- `abx_cont` (sepsis) in ricu-extensions adds antibiotic start times in hours and durations in minutes, which R computes in seconds, so its criterion of at most 24h between antibiotics practically never applies. `reproduce_ricu_abx_cont` in `config.yml` reproduces this (default) or applies the criterion as intended.
- ICU stays without a discharge time are left out, and their number is printed (ricu leaves out HiRID patients without observations).
- HiRID and AmsterdamUMCdb have only been tested on synthetic data. AmsterdamUMCdb: WeavEHR reads version 1.5.0 through the OMOP export, ricu version 1.0.2; age, height and weight (age, height and weight groups in ricu) and antibiotics (sepsis) are not available, and ICU deaths assume that the export keeps the discharge destination ("Overleden").
