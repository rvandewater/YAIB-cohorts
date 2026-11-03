## Replicate the cohorts from Python:

The following instructions assume that this folder is the current working directory in your Python session. 


### Step 0: Environment and Prerequisites

An `renv` lock file was created to install all necessary package dependencies. To recreate the environment, start a Terminal and run: 

```bash

python setup_env.py

```

> Update: if you are running into issues setting up your environment, there is now a Docker workflow that should simplify creating the environment. See the subfolder `docker` for more information.

### Step 1: Load the datasets into `ricu`

In order to access the full datasets through R, you need to download them and make them available to `ricu`. Please follow the instructions given by the `ricu` package: `?ricu::import_src`. 

For quick experimentation, `ricu` comes with two demo datasets: `mimic.demo` and `eicu.demo`. These are small, openly available subsets of mimic and eicu that allow for easy prototyping. They should have been installed by `renv`. If they aren't, please see the respective Github pages [here](https://github.com/eth-mds/mimic-demo) and [here](https://github.com/eth-mds/eicu-demo).

Once you have imported the datasets, make sure to set the right data path in the [config.yaml](../config.yaml) file.

```r
ricu_data_path: "/home/ruser/data"
```

> Note: If you are using Docker to run YAIB-cohorts, you don't need to make any changes in [config.yaml](../config.yaml). Instead, you need to define the path in [docker/docker-compose.yml](../docker/docker-compose.yml). 


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
python mortality.py --src mimic_demo
```

where `mortality.py` should be replaced with the task file of interest and `mimic_demo` with the database of interest (one of `mimic_demo`, `eicu_demo`, `aumc`, `hirid`, `eicu`, `mimic`, `miiv`). 

The output directory for the extracted data can also be set in [config.yaml](../config.yaml).


### Alternative: WeavEHR instead of ricu

The cohorts can also be extracted with [WeavEHR](https://github.com/aidh-ms/WeavEHR), a Python framework that harmonises ICU datasets into the MEDS format, instead of ricu. This route does not need R. Configured datasets (`--src`, WeavEHR version): MIMIC-IV (`mimic-iv`, 2.2) and its openly available [demo](https://physionet.org/content/mimic-iv-demo/) (`mimic-iv-demo`, 2.2), eICU-CRD (`eicu-crd`, 2.0) and its [demo](https://physionet.org/content/eicu-crd-demo/) (`eicu-demo`, 2.0), HiRID (`hirid`, 1.1.1) and AmsterdamUMCdb (`aumc`, 1.5.0, the OMOP export).

1. Install WeavEHR (Python >= 3.13, see its README) together with `pandas` and `pyyaml`.
2. Set `backend: "weavehr"` in [config.yaml](../config.yaml). `weavehr_project_path` is where the WeavEHR project of each dataset (`{src}`) is kept. It contains patient-level data, so keep it outside of this repository. To keep the outputs of both backends apart, set `out_dir: "../data/{backend}"`.
3. Build the WeavEHR project of the dataset (the counterpart of importing a dataset into ricu):
   ```bash
   python weavehr_build.py --src mimic-iv-demo --version 2.2 --data-path /path/to/physionet.org/files/mimic-iv-demo/2.2
   python weavehr_build.py --src eicu-demo --version 2.0 --data-path /path/to/physionet.org/files/eicu-crd-demo/2.0.1
   ```
4. Generate the cohorts as above, e.g., `python mortality.py --src mimic-iv-demo`.

The clinical concepts are those defined by WeavEHR. [weavehr-extensions/](../weavehr-extensions) is the counterpart of [ricu-extensions/](../ricu-extensions): `config.yml` maps the YAIB concepts to WeavEHR concepts with ricu's value ranges and hourly aggregation, and `datasets/` adds the WeavEHR table configs and concept mappings that YAIB adds to ricu, or that reproduce ricu's items (e.g., non-invasive blood pressure in eICU, MIMIC-IV lab items 53085, 53086 and 53089). [src/weavehr.py](src/weavehr.py) attaches the concepts to ICU stays the way ricu does and computes the outcomes (`death_icu`, `aki`, `sep3_alt`, `los_icu`, `hospital_id`) following [ricu-extensions/callbacks](../ricu-extensions/callbacks).

The cohorts of both backends can be compared with `weavehr-yaib-compare-cohorts` from [weavehr-yaib](https://github.com/TobiasVeronaLaser/weavehr-yaib). On the MIMIC-IV and eICU demos, the mortality, AKI, kidney function and length of stay cohorts, labels and variables agree with those extracted with ricu. Known differences:

- Sepsis: SOFA is WeavEHR's score (the latest value of each component within 24h, rather than ricu's worst value within 24h), and microbiology samples recorded by date only are timed at the end of that day (ricu: midday).
- `abx_cont` (sepsis) in ricu-extensions adds antibiotic start times in hours and durations in minutes, which R computes in seconds, so its criterion of at most 24h between antibiotics practically never applies. `reproduce_ricu_abx_cont` in `config.yml` reproduces this (default) or applies the criterion as intended.
- ICU stays without a discharge time are left out, and their number is printed (ricu leaves out HiRID patients without observations).
- HiRID and AmsterdamUMCdb have only been tested on synthetic data. AmsterdamUMCdb: WeavEHR reads version 1.5.0 through the OMOP export, ricu version 1.0.2; age, height and weight (age, height and weight groups in ricu) and antibiotics (sepsis) are not available, and ICU deaths assume that the export keeps the discharge destination ("Overleden").
