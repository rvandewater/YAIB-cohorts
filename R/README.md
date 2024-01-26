## Replicate the cohorts within R:

The following instructions assume that this folder is the current working directory in your R session. 


### Step 0: Environment and Prerequisites

All data extractions were run using R 4.2.2 on an Apple M1 Max with Ventura 13.2.1. An `renv` lock file was created to install all necessary package dependencies. To recreate the environment, start a Terminal and run: 

```bash

Rscript setup_env.R

```
> Note: you might have to install some extra packages for this step to succeed (for WSL Ubuntu: libssl-dev, libcurl4-openssl-dev, libudunits2-dev). Install with `sudo apt install libssl-dev libcurl4-openssl-dev libudunits2-dev` )

> If you get an error prompted by ricu, it might that the `units`, `eicu.demo`, and `mimic.demo` packages are not properly installed. Please check your `.Rprofile` file and stop `ricu` from automatically executing. Then, use the commands in `setup_env.R` to manually install the packages into the Renv environment. Afterwards, you can restore the `.Rprofile` file to its original state to run the commands below.
### Step 1: Load the datasets into `ricu`

In order to access the full datasets through R, you need to download them and make them available to `ricu`. Please follow the instructions given by the `ricu` package: `?ricu::import_src`.

For quick experimentation, `ricu` comes with two demo datasets: `mimic.demo` and `eicu.demo`. These are small, openly available subsets of mimic and eicu that allow for easy prototyping. They should have been installed by `renv`. If they aren't, please see the respective Github pages [here](https://github.com/eth-mds/mimic-demo) and [here](https://github.com/eth-mds/eicu-demo).

Once you have imported the datasets, make sure to set the right data path in [.Rprofile](.Rprofile) file in this directory:

```r

Sys.setenv(RICU_DATA_PATH = "/path/to/your/ricu/data/folder")

```


### Step 2: Generate the Cohorts

This repository currently allows for the extraction of five prediction tasks: 

Classification:

1. ICU mortality after 24 hours: [mortality.R](mortality.R)
2. Acute Kidney Injury within the next 6 hours: [aki.R](aki.R)
3. Sepsis within the next 6 hours: [sepsis.R](sepsis.R)

Regression:

4. Kidney function on the second day of ICU admission: [kidney_function.R](kidney_function.R)
5. Remaining length of stay: [los.R](los.R)


All five tasks rely on a shared data cleaning provided in [base_cohort.R](base_cohort.R), which defines and stores a subset of patients in each dataset with sufficient data quality. [base_cohort.R](base_cohort.R) therefore needs to be called *before* any of the task-specific cohorts can be generated. 

```bash 
Rscript "base_cohort.R" --src mimic_demo
```

Once [base_cohort.R](base_cohort.R) was run, data for a single task like mortality from a single dataset can be extracted via: 
```bash 
Rscript "mortality.R" --src mimic_demo
```

where `mortality.R` should be replaced with the task file of interest and `mimic_demo` with the database of interest (one of `mimic_demo`, `eicu_demo`, `aumc`, `hirid`, `eicu`, `mimic`, `miiv`). Data can be extracted from all datasets simultaneously via `bash gen_cohort.sh mortality`.

The output directory for the extracted data can be set in [../config.yaml](../config.yaml)
