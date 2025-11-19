library(argparser)
library(assertthat)
library(rlang)
library(data.table)
library(vctrs)
library(yaml)

source(".load_ricu.R")
source("src/misc.R")
source("src/steps.R")
source("src/sequential.R")
source("src/obs_time.R")


# Create a parser
p <- arg_parser("Extract and preprocess kidney function data (creatinine)")
p <- add_argument(p, "--src", help="source database", default="mimic_demo")
argv <- parse_args(p)

src <- argv$src 
conf <- yaml.load_file("../config.yaml")
path <- file.path(conf$out_dir, "kidney_function")


cncpt_env <- new.env()

# Task description
time_flow <- "sequential" # static / sequential / continuous
time_unit <- hours
freq <- 1L
max_len <- hours(24L)

static_vars <- c("age", "sex", "height", "weight")

dynamic_vars <- c("alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir",
                  "bnd", "bun", "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", 
                  "dbp", "fgn", "fio2", "glu", "hgb", "hr", "inr_pt", "k", "lact",
                  "lymph", "map", "mch", "mchc", "mcv", "methb", "mg", "na", "neut", 
                  "o2sat", "pco2", "ph", "phos", "plt", "po2", "ptt", "resp", "sbp", 
                  "temp", "tnt", "urine", "wbc")

# cross-sectional vs longitudinal
predictor_type <- "dynamic" # static / dynamic
outcome_type   <- "static" # static / dynamic



patients <- stay_windows(src, interval = time_unit(freq))
patients <- as_win_tbl(patients, index_var = "start", dur_var = "end", interval = time_unit(freq))

# Only keep patients in the base cohort (see base_cohort.R)
base <- arrow::read_parquet(file.path(conf$out_dir, "base", src, "sta.parquet"))
patients <- patients[id_col(patients) %in% id_col(base)]


# Define outcome ----------------------------------------------------------

outc <- load_step(dict["crea"], interval=time_unit(freq))
outc <- filter_step(outc, ~ . > hours(24) & . <= hours(48), col=index_col)
outc <- function_step(outc, function(df) df[, .(crea = median(crea)), by = c(id_var(df))])


# Define observation times ------------------------------------------------

stop_obs_at(patients, offset = ricu:::re_time(max_len, time_unit(freq)), by_ref = TRUE)



# Apply exclusion criteria ------------------------------------------------

# Exclusions 1.-5. are defined in base_cohort.R

# 6. LoS less than 48 hours
x <- load_step(dict["los_icu"])
x <- filter_step(x, ~ . < 48 / 24)

excl6 <- unique(x[, id_vars(x), with = FALSE])

# 7. Had no creatinine measurement between 24 and 48 hours
x <- function_step(outc, function(df) df[patients, on = c(id_var(df)), nomatch = NA][is.na(crea)])

excl7 <- unique(x[, id_vars(x), with = FALSE])


# Apply exclusions
patients <- exclude(patients, mget(paste0("excl", 6:7)))
attrition <- as.data.table(patients[c("incl_n", "excl_n_total", "excl_n")])
patients <- patients[['incl']]
patient_ids <- patients[, .SD, .SDcols = id_var(patients)]


# Prepare data ------------------------------------------------------------

# Get predictors
dyn <- load_step(dict[dynamic_vars], interval=time_unit(freq), cache = TRUE)
sta <- load_step(dict[static_vars], cache = TRUE)

# Transform all variables into the target format
assert_that(outcome_type == "static", time_flow == "sequential")

map_to_patients <- function(x) {
  grid <- patients[, .SD, .SDcols = id_var(patients)]
  merge(grid, x, all.x = TRUE)
}

outc_fmt <- function_step(outc, map_to_patients)

# TODO: make step to add/remove columns
rename_cols(outc_fmt, c("stay_id", "label"), by_ref = TRUE)

dyn_fmt <- function_step(dyn, map_to_grid)
dyn_fmt <- filter_step(dyn_fmt, patients)
rename_cols(dyn_fmt, c("stay_id", "time"), meta_vars(dyn_fmt), by_ref = TRUE)

sta_fmt <- function_step(sta, map_to_patients)
rename_cols(sta_fmt, c("stay_id"), id_vars(sta), by_ref = TRUE)



# Write to disk -----------------------------------------------------------

out_path <- paste0(path, "/", src)

if (!dir.exists(out_path)) {
  dir.create(out_path, recursive = TRUE)
}

arrow::write_parquet(outc_fmt, paste0(out_path, "/outc.parquet"))
arrow::write_parquet(dyn_fmt, paste0(out_path, "/dyn.parquet"))
arrow::write_parquet(sta_fmt, paste0(out_path, "/sta.parquet"))
fwrite(attrition, paste0(out_path, "/attrition.csv"))
