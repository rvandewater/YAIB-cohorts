![YAIB logo](https://github.com/rvandewater/YAIB/blob/development/docs/figures/yaib_logo.png)
# Generating Patient cohorts for 🧪 _Yet Another ICU Benchmark_


This repo uses the `ricu` R package to derive patient cohorts for prediction tasks from the following intensive care databases: 
| **Dataset**                 | [MIMIC-III](https://physionet.org/content/mimiciii/) / [IV](https://physionet.org/content/mimiciv/) | [eICU-CRD](https://physionet.org/content/eicu-crd/) | [HiRID](https://physionet.org/content/hirid/1.1.1/) | [AUMCdb](https://doi.org/10.17026/dans-22u-f8vd) |
|-------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|--------------------------------------------------|
| **Admissions**              | 40k / 73k                                                                                           | 200k                                                | 33k                                                 | 23k                                              |
| **Version**                 | v1.4 / v2.2                                                                                         | v2.0                                                | v1.1.1                                              | v1.0.2                                           |                                                     |                                                     |                                                  |
| **Frequency** (time-series) | 1 hour                                                                                              | 5 minutes                                           | 2 / 5 minutes                                       | up to 1 minute                                   |
| **Originally published**    | 2015  / 2020                                                                                        | 2017                                                | 2020                                                | 2019                                             |                                                                                                     |                                                     |                                                     |                                                  |
| **Origin**                  | USA                                                                                                 | USA                                                 | Switzerland                                         | Netherlands                                      |

New datasets can also be added. We are currently working on a package to make this process as smooth as possible.
<!-- * [AUMCdb](https://github.com/AmsterdamUMC/AmsterdamUMCdb)
* [HiRID](https://hirid.intensivecare.ai/)
* [eICU](https://eicu-crd.mit.edu/)
* [MIMIC IV](https://mimic.mit.edu/) -->
We provide five common tasks for clinical prediction by default:

| No  | Task                 | Frequency        | Type                                | 
|-----|---------------------------|--------------------|-------------------------------------|
| 1   | ICU Mortality             | Once per Stay (after 24H) | Binary Classification  |
| 2   | Acute Kidney Injury (AKI) | Hourly (within 6H) | Binary Classification |
| 3   | Sepsis                    | Hourly (within 6H) | Binary Classification |
| 4   | Kidney Function(KF)       | Once per stay | Regression |
| 5   | Length of Stay (LoS)      | Hourly (within 7D) | Regression |

New tasks can be easily added. 
The following repositories may be relevant as well:
- [YAIB](https://github.com/rvandewater/YAIB): Main repository for YAIB.
- [YAIB-models](https://github.com/rvandewater/YAIB-models): Pretrained models for YAIB.
- [ReciPys](https://github.com/rvandewater/ReciPys): Preprocessing package for YAIB pipelines.

## 📄 Paper

If you use this code in your research, please cite the following publication:

```
@article{vandewaterYetAnotherICUBenchmark2023,
	title = {Yet Another ICU Benchmark: A Flexible Multi-Center Framework for Clinical ML},
	shorttitle = {Yet Another ICU Benchmark},
	url = {http://arxiv.org/abs/2306.05109},
	language = {en},
	urldate = {2023-06-09},
	publisher = {arXiv},
	author = {van de Water, Robin and Schmidt, Hendrik and Elbers, Paul and Thoral, Patrick and Arnrich, Bert and Rockenschaub, Patrick},
	month = jun,
	year = {2023},
	note = {arXiv:2306.05109 [cs]},
	keywords = {Computer Science - Machine Learning},
}
```

This paper can also be found on arxiv: https://arxiv.org/pdf/2306.05109.pdf

## To replicate the cohorts:

Run the following commands to clone this repo:

```
git clone https://github.com/rvandewater/YAIB-cohorts.git
cd YAIB-cohorts
```
Once you have cloned the repo, all cohorts can be created directly from within R or via an interface from python. Instructions for each can be found at: 

- R: [README.md](R/README.md)
- Python: [README.md](Python/README.md)  

Note: due to some recent bug fixes in ricu, the extracted cohorts might differ marginally to those published in the benchmarking paper. 
## Clairvoyance Conversion
To output the cohorts in the Clairvoyance (https://github.com/vanderschaarlab/clairvoyance) format, you can use the following utils.py function
```
output_clairvoyance(data_dir, save_dir, task_type="static")
```
You can specify the size and the type of task ("static": i.e., one outcome label per stay_id (mortality, KF) or "dynamic": (Sepsis, AKI, LOS), i.e., one outcome label per time step) and the train/test split in the `make_train_test` function.

## Acknowledgements

The code in this repository heavily utilises the `ricu` R package, without which deriving these cohorts would have been much more difficult. If you use the code in this repo, please go give their repo a star :)

This repo is based on earlier work by [Rockenschaub et al. (2023)](https://arxiv.org/abs/2303.15354), which can be found at https://github.com/prockenschaub/icuDG-preprocessing

## License
This source code is released under the MIT license, included [here](LICENSE).
