## Replicate the cohorts from Python:

The following instructions assume that this folder is the current working directory in your Python session.

### Pre-step: Make sure R is installed with conda

#### Install R

```bash
conda install -c conda-forge r-base

```

#### Install rpy2

```bash
conda install -c conda-forge rpy2

```

### Step 0: Environment and Prerequisites

An `renv` lock file was created to install all necessary package dependencies. To recreate the environment, start a
Terminal and run:

```bash

python setup_env.py

```

If you're encountering issues with the units package or working with R projects managed by `renv`, refer to
this [file](fix_ricu.md)

### Step 1: Load the datasets into `ricu`

In order to access the full datasets through R, you need to download them and make them available to `ricu`. Please
follow the instructions given by the `ricu` package: `?ricu::import_src`. Currently, Python code to perform this step is
in development and will be uploaded soon.

For quick experimentation, `ricu` comes with two demo datasets: `mimic.demo` and `eicu.demo`. These are small, openly
available subsets of mimic and eicu that allow for easy prototyping. They should have been installed by `renv`. If they
aren't, please see the respective Github pages [here](https://github.com/eth-mds/mimic-demo)
and [here](https://github.com/eth-mds/eicu-demo).

Once you have imported the datasets, make sure to set the right data path in [.Rprofile](.Rprofile) file in this
directory:

```r

Sys.setenv(RICU_DATA_PATH = "/path/to/your/ricu/data/folder")

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
python mortality.py --src mimic_demo
```

where `mortality.py` should be replaced with the task file of interest and `mimic_demo` with the database of interest (
one of `mimic_demo`, `eicu_demo`, `aumc`, `hirid`, `eicu`, `mimic`, `miiv`).

The output directory for the extracted data can be set in [../config.yaml](../config.yaml)
