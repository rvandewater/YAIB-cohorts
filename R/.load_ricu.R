library(yaml)
conf <- yaml.load_file("../config.yaml")

Sys.setenv(RICU_DATA_PATH = conf$ricu_data_path)
print(paste0("RICU_DATA_PATH set to ", conf$ricu_data_path))

if(require("ricu", quietly = TRUE)) {
  source("../ricu-extensions/callbacks/callback-icu-mortality.R")
  source("../ricu-extensions/callbacks/callback-kdigo.R")
  source("../ricu-extensions/callbacks/callback-sepsis.R")
  
  concept_path <- file.path("..", "ricu-extensions", "configs", c("chemistry", "circulatory", "demographics", "hematology", "medications", "misc", "outcomes", "output", "vitals"))
  dict <- load_dictionary(cfg_dirs = concept_path)
}