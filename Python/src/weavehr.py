"""WeavEHR backend: replaces the ricu functions used by the cohort scripts.

Reads a WeavEHR project built with `weavehr_build.py` and returns the tables that `ricu::stay_windows()`
and `ricu::load_concepts()` return at an hourly interval: one row per ICU stay (`stay_id`) and hour since
ICU admission (`time`, floored), or one row per stay for static concepts.

As in ricu, each concept row is attached to ICU stays through the most specific ID of its source table:
static concepts to all stays of that ID, time series within that ID to the first stay ending at or after
the row (rows after the last stay go to the last stay). Rows outside the concept's value range are dropped
and the rest is aggregated per hour. Concepts are configured in ../weavehr-extensions/config.yml; los_icu,
hospital_id, death_icu, aki and sep3_alt are ports of ricu and ../ricu-extensions/callbacks.
"""
from functools import cache
from pathlib import Path
from typing import List

import pandas as pd
import polars as pl
import yaml

from . import conf

with open("../weavehr-extensions/config.yml") as stream:  # this path is relative to the calling environment
    ext = yaml.safe_load(stream)

AGGREGATE = {
    "median": lambda x: x.median(),
    "mean": lambda x: x.mean(),
    "sum": lambda x: x.sum(),
    "min": lambda x: x.min(),
    "max": lambda x: x.max(),
    "first": lambda x: x.first(),
}


def hours(x: int) -> pd.Timedelta:
    return pd.Timedelta(hours=x)


def _check_interval(interval) -> None:
    if interval is not None and pd.Timedelta(interval) != hours(1):
        raise NotImplementedError("The WeavEHR backend only supports an interval of hours(1).")


# ------------------------------------------------------------------------------
# Project layout and ICU stays

def project_path(src: str) -> Path:
    return Path(conf["weavehr_project_path"].format(src=src))


@cache
def _source(src: str) -> tuple[dict, Path, str]:
    """Source configuration, extraction output directory and dataset version of a WeavEHR dataset."""
    if src not in ext["sources"]:
        raise ValueError(f"No WeavEHR source {src!r} configured in weavehr-extensions/config.yml.")
    root = project_path(src) / "datasets" / "extraction" / "data" / src
    versions = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if len(versions) != 1:
        raise FileNotFoundError(f"Expected one extracted version of {src} in {root}, found {versions}. "
                                "Build the project with weavehr_build.py.")
    return ext["sources"][src], root / versions[0], versions[0]


def _stay_column(src: str) -> str:
    """Column identifying ICU stays: the first ID column, or the subject if stays have no ID of their own."""
    return (_source(src)[0]["id_columns"] or ["subject_id"])[0]


@cache
def _stays(src: str) -> pl.DataFrame:
    """ICU stays with the IDs of the source, admission (`intime`) and discharge (`outtime`) times.

    Without a discharge event, a stay ends with the last event of its subject in `icu_discharge.last_of`.
    """
    spec, root, _ = _source(src)
    stay = _stay_column(src)
    intime = pl.read_parquet(root / f"{spec['icu_admission']}.parquet",
                             columns=list(dict.fromkeys(["subject_id", *spec["id_columns"], "time"])))
    if isinstance(spec["icu_discharge"], dict):
        outtime = (pl.scan_parquet(root / f"{spec['icu_discharge']['last_of']}.parquet")
                   .group_by("subject_id").agg(pl.col("time").max()).collect())
    else:
        outtime = pl.read_parquet(root / f"{spec['icu_discharge']}.parquet", columns=[stay, "time"])
    return (intime.rename({"time": "intime"})
            .join(outtime.rename({"time": "outtime"}), on=stay, how="left")
            .sort(stay))


def _stay_minutes(src: str) -> pl.DataFrame:
    """Stay ID and length of stay in minutes, as in ricu's stay windows: from minute-floored times, or from the
    admission when stays end with the last event (HiRID)."""
    if isinstance(_source(src)[0]["icu_discharge"], dict):
        minutes = (pl.col("outtime") - pl.col("intime")).dt.total_seconds() // 60
    else:
        minutes = (pl.col("outtime").dt.truncate("1m") - pl.col("intime").dt.truncate("1m")).dt.total_minutes()
    return _stays(src).select(pl.col(_stay_column(src)).alias("stay_id"), minutes.alias("minutes"))


def stay_windows(src: str, interval=None) -> pd.DataFrame:
    """Observation times (in hours since ICU admission) for all ICU stays of a dataset."""
    _check_interval(interval)
    res = _stay_minutes(src).select("stay_id", pl.lit(0).alias("start"), (pl.col("minutes") // 60).alias("end"))
    return res.to_pandas()


# ------------------------------------------------------------------------------
# Attaching concept rows to ICU stays

def _decimal(value: pl.Expr) -> pl.Expr:
    """WeavEHR stores float32 values; their shortest decimal is the source value, as parsed by ricu."""
    return value.cast(pl.Float32).cast(pl.String).cast(pl.Float64)


def _id_column(src: str, table: str) -> str | None:
    """Most specific ID column of the extracted events of a table (ricu: the table's `id_var`)."""
    spec, root, _ = _source(src)
    columns = set()
    for file in (root / table).glob("*.parquet"):
        columns.update(pl.scan_parquet(file).collect_schema().names())
    return next((col for col in spec["id_columns"] if col in columns), None)


def _with_ids(src: str, rows: pl.DataFrame, table: str, event: str | None) -> tuple[pl.DataFrame, str | None]:
    """Add the ID column of the source table to concept rows, matching the extracted events on subject, time
    and, where known, event and code. Rows matching events of several IDs are matched on their value as well;
    rows still matching several IDs get each of them, as each ID has such an event."""
    level = _id_column(src, table)
    if level is None:
        return rows, None
    _, root, _ = _source(src)
    files = [root / table / f"{event}.parquet"] if event else list((root / table).glob("*.parquet"))
    files = [f for f in files if level in pl.scan_parquet(f).collect_schema().names()]
    events = pl.concat([pl.scan_parquet(f).select("subject_id", "time", pl.col("code").alias("source_code"),
                                                  _decimal(pl.col("numeric_value")), level) for f in files])

    def match(rows: pl.DataFrame, keys: List[str]) -> pl.DataFrame:
        ids = (events.join(rows.lazy().select(keys).unique(), on=keys, how="semi", nulls_equal=True)
               .group_by(keys).agg(pl.col(level).drop_nulls().unique().alias("ids")).collect())
        return rows.join(ids, on=keys, how="left", nulls_equal=True)

    keys = ["subject_id", "time"] + (["source_code"] if rows["source_code"].is_not_null().all() else [])
    rows = match(rows, keys)
    several = (pl.col("ids").list.len() > 1).fill_null(False)
    if rows.select(several.any()).item():
        by_value = (match(rows.filter(several).rename({"ids": "candidates"}), keys + ["numeric_value"])
                    .with_columns(pl.when(pl.col("ids").list.len() > 0).then("ids").otherwise("candidates").alias("ids"))
                    .drop("candidates"))
        rows = pl.concat([rows.filter(~several), by_value])
    return rows.explode("ids").rename({"ids": level}), level


def _attach(rows: pl.DataFrame, src: str, level: str | None, static: bool) -> pl.DataFrame:
    """Attach rows with `subject_id`, `time` and the `level` ID column to ICU stays (ricu's `change_id()`).

    Rows without an ID are dropped. Static rows are attached to all stays of their ID; time series rows to
    the first stay of their ID that ends at or after the row, else to the last one, and get `time` in hours
    since ICU admission.
    """
    stay = _stay_column(src)
    key = level or "subject_id"
    keep = [col for col in rows.columns if col not in ("subject_id", key, stay) and not (static and col == "time")]
    stays = _stays(src).select(*dict.fromkeys([key, stay]), "intime", "outtime")
    rows = rows.filter(pl.col(key).is_not_null())

    if static or key == stay:
        res = rows.join(stays, on=key, how="inner")
    else:
        stays = stays.drop_nulls("outtime").sort("outtime")
        rows = rows.drop([stay], strict=False).sort("time")
        res = rows.join_asof(stays, left_on="time", right_on="outtime", by=key, strategy="forward", check_sortedness=False)
        after_last = (res.filter(pl.col(stay).is_null()).select(rows.columns)
                      .join_asof(stays, left_on="time", right_on="outtime", by=key, strategy="backward", check_sortedness=False))
        res = pl.concat([res.filter(pl.col(stay).is_not_null()), after_last.filter(pl.col(stay).is_not_null())])

    if not static:
        res = res.sort("time", maintain_order=True).with_columns(
            ((pl.col("time") - pl.col("intime")).dt.total_microseconds() // 3_600_000_000).alias("time"))
    return res.select(pl.col(stay).alias("stay_id"), *keep)


# ------------------------------------------------------------------------------
# Concepts configured in weavehr-extensions/config.yml

def _concept_cfg(src: str, name: str) -> dict:
    """Configuration of concept `name`, with the settings for the dataset in its `sources` applied."""
    cfg = ext["concepts"][name]
    return {**cfg, **cfg.get("sources", {}).get(src, {})}


def _concept_rows(src: str, name: str) -> pl.DataFrame:
    """Output rows of the WeavEHR concept(s) behind `name`."""
    weavehr = _concept_cfg(src, name)["weavehr"]
    if not isinstance(weavehr, list):
        return _weavehr_rows(src, weavehr, name)
    # pooled concepts: a source row mapped by several of them counts once
    keys = ["subject_id", "time", "numeric_value", "table", "event", "source_code"]
    res = _weavehr_rows(src, weavehr[0], name)
    for concept in weavehr[1:]:
        rows = _weavehr_rows(src, concept, name)
        res = pl.concat([res, rows.join(res.select(keys).unique(), on=keys, how="anti", nulls_equal=True)])
    return res


def _weavehr_rows(src: str, weavehr: str, name: str) -> pl.DataFrame:
    _, _, version = _source(src)
    files = list((project_path(src) / "datasets" / "concept" / "data" / weavehr).glob(f"*/{src}/{version}.parquet"))
    schema = pl.Schema({"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float64,
                        "text_value": pl.String, "table": pl.String, "event": pl.String, "source_code": pl.String})
    if len(files) != 1:
        print(f"WARNING: no unique output of WeavEHR concept {weavehr} for {src} {version}, found {files}.")
        return pl.DataFrame(schema=schema).with_columns(concept=pl.lit(name))
    lf = pl.scan_parquet(files[0])
    present = lf.collect_schema().names()
    return (lf.select(*[pl.col(col) if col in present else pl.lit(None).alias(col) for col in schema])
            .cast(schema).with_columns(_decimal(pl.col("numeric_value")), concept=pl.lit(name)).collect())


def _events(src: str, names: List[str], static: bool) -> pl.DataFrame:
    """Rows of the configured concepts `names`, attached to ICU stays.

    Rows without a source table (e.g., scores computed by WeavEHR) are attached like rows of a patient-level table.
    """
    rows = pl.concat([_concept_rows(src, name) for name in names])
    parts = []
    for (table, event), part in rows.group_by("table", "event"):
        part, level = _with_ids(src, part, table, event) if table is not None else (part, None)
        parts.append(_attach(part, src, level, static))
    res = pl.concat(parts) if parts else _attach(rows, src, None, static)
    return res.drop("table", "event", "source_code")


def _aggregate(src: str, events: pl.DataFrame, name: str, static: bool) -> pl.DataFrame:
    """Values of concept `name` within its range, aggregated per stay (static) or per stay and hour.

    `cumulative`: values are running totals; each record counts its increase (`hirid_urine`).
    `ratio_to`: values are counts, relative to the concept of the same hour in % (`blood_cell_ratio`).
    """
    cfg = _concept_cfg(src, name)
    keys = ["stay_id"] if static else ["stay_id", "time"]
    if "values" in cfg:
        value, fun = pl.col("text_value").replace_strict(cfg["values"], default=None), "first"
    else:
        value, fun = pl.col("numeric_value"), cfg.get("aggregate", "median")
    res = events.filter(pl.col("concept") == name).sort(keys, maintain_order=True).select(*keys, value.alias(name))
    if cfg.get("cumulative"):
        increase = pl.col(name).diff().over("stay_id").fill_null(pl.col(name))
        res = res.with_columns(pl.when(increase < 0).then(0).otherwise(increase).alias(name))
    if "ratio_to" in cfg:
        other = cfg["ratio_to"]
        res = res.join(_load(src, other), on=keys, how="inner").with_columns((100 * pl.col(name) / pl.col(other)).alias(name)).drop(other)
    res = res.drop_nulls(name)
    if "min" in cfg:
        res = res.filter(pl.col(name) >= cfg["min"])
    if "max" in cfg:
        res = res.filter(pl.col(name) <= cfg["max"])
    return res.group_by(keys, maintain_order=True).agg(AGGREGATE[fun](pl.col(name))).sort(keys)


def _load(src: str, name: str) -> pl.DataFrame:
    static = _concept_cfg(src, name).get("static", False)
    return _aggregate(src, _events(src, [name], static), name, static)


# ------------------------------------------------------------------------------
# Outcome concepts (ports of ricu and ricu-extensions/callbacks)

def _los_icu(src: str) -> pl.DataFrame:
    """ICU length of stay in days (ricu `los_callback`)."""
    return _stay_minutes(src).select("stay_id", (pl.col("minutes") / 1440).alias("los_icu")).filter(pl.col("los_icu") >= 0)


def _source_events(src: str, events: str, columns: List[str] | None = None) -> tuple[pl.DataFrame, str | None]:
    """Extracted events (`<table>/<event>`) of a source and the ID column of their table."""
    _, root, _ = _source(src)
    return pl.read_parquet(root / f"{events}.parquet", columns=columns), _id_column(src, events.split("/")[0])


def _hospital_id(src: str) -> pl.DataFrame:
    """Hospital of each stay; hospital 1 for single-hospital datasets."""
    spec = _source(src)[0]
    if "hospital_id" not in spec:
        return _stay_minutes(src).select("stay_id", pl.lit(1).alias("hospital_id"))
    ev, level = _source_events(src, spec["hospital_id"])
    ev = ev.select("subject_id", level, pl.col("numeric_value").alias("hospital_id")).filter(pl.col("hospital_id").is_between(1, 1000))
    return _attach(ev, src, level, static=True).group_by("stay_id").agg(pl.col("hospital_id").first()).sort("stay_id")


def _death_icu(src: str) -> pl.DataFrame:
    """ICU mortality, depending on the dataset.

    MIMIC-IV (`miiv_death_icu`): admissions with a time of death; true for in-hospital deaths whose last ward
    before discharge was an ICU. eICU, AUMC (`aumc_death_icu`): ICU discharge status (in the event code or in
    `status_column`) at ICU discharge; with `death`, false if the patient died more than `within_hours` later.
    HiRID (`hirid_death_icu`): deaths at the last record of the variables in `code`.
    """
    cfg = _source(src)[0]["death_icu"]
    if "discharge_status" in cfg:
        ev, level = _source_events(src, cfg["discharge_status"])
        died = (pl.col(cfg["status_column"]) == cfg["died"]) if "status_column" in cfg else pl.col("code").str.ends_with(f"//{cfg['died']}")
        ev = ev.select("subject_id", level, "time", died.fill_null(False).alias("death_icu"))
        if "death" in cfg:
            death = _source_events(src, cfg["death"], ["subject_id", "time"])[0].group_by("subject_id").agg(pl.col("time").min().alias("death"))
            later = ((pl.col("death") - pl.col("time")) > pl.duration(hours=cfg["within_hours"])).fill_null(False)
            ev = ev.join(death, on="subject_id", how="left").with_columns((pl.col("death_icu") & ~later).alias("death_icu")).drop("death")
        if not ev["death_icu"].any():
            print(f"WARNING: no ICU deaths found for {src}; check death_icu in weavehr-extensions/config.yml.")
        return _attach(ev, src, level, static=False).group_by("stay_id", "time").agg(pl.col("death_icu").any()).sort("stay_id", "time")

    if "last_of" in cfg:
        root = _source(src)[1]
        status = pl.read_parquet(root / f"{cfg['status']}.parquet", columns=["subject_id", "text_value"])
        dead = status.filter(pl.col("text_value") == cfg["died"]).select("subject_id").unique()
        last = (pl.scan_parquet(root / f"{cfg['last_of']}.parquet").filter(pl.col("code").str.contains(cfg["code"]))
                .join(dead.lazy(), on="subject_id", how="semi").group_by("subject_id").agg(pl.col("time").max()).collect())
        level = _id_column(src, cfg["last_of"].split("/")[0])
        return _attach(last.with_columns(pl.lit(True).alias("death_icu")), src, level, static=False).sort("stay_id", "time")

    root = _source(src)[1]
    transfers = (pl.read_parquet(root / f"{cfg['transfers']}.parquet", columns=["subject_id", "hadm_id", "time", "code"])
                 .with_columns(pl.col("code").str.split("//").alias("parts"))
                 .select("subject_id", "hadm_id", "time",
                         pl.col("parts").list.get(1, null_on_oob=True).alias("eventtype"),
                         pl.col("parts").list.get(2, null_on_oob=True).alias("ward"))
                 .sort("hadm_id", "time"))
    icu_wards = (transfers.join(_stays(src).select("subject_id", pl.col("intime").alias("time")),
                                on=["subject_id", "time"], how="semi")["ward"].drop_nulls().unique())
    last_ward = transfers.filter(pl.col("eventtype").shift(-1) == "discharge")
    non_icu = last_ward.filter(~pl.col("ward").is_in(icu_wards.implode()).fill_null(False))["hadm_id"]

    deaths = (pl.read_parquet(root / f"{cfg['deaths']}.parquet", columns=["subject_id", "hadm_id", "time", "numeric_value"])
              .select("subject_id", "hadm_id", "time",
                      ((pl.col("numeric_value") == 1) & ~pl.col("hadm_id").is_in(non_icu.implode())).alias("death_icu")))
    return (_attach(deaths, src, "hadm_id", static=False)
            .group_by("stay_id", "time").agg(pl.col("death_icu").any()).sort("stay_id", "time"))


def _aki(src: str) -> pl.DataFrame:
    """AKI (KDIGO stage >= 1) from creatinine and urine rate (`kdigo_crea`, `urine_rate`, `kdigo_urine`, `aki`)."""
    def slide(x: pl.DataFrame, col: str, hrs: int, fun: str) -> pl.DataFrame:
        # values within (time - hrs, time], like ricu's slide(..., left_closed = FALSE)
        return x.rolling("time", period=f"{hrs}i", group_by="stay_id").agg(AGGREGATE[fun](pl.col(col)).alias(f"{col}_{hrs}hr"))

    crea = _load(src, "crea")
    crea = crea.join(slide(crea, "crea", 48, "min"), on=["stay_id", "time"]).join(slide(crea, "crea", 168, "min"), on=["stay_id", "time"])
    c, c48, c168 = pl.col("crea"), pl.col("crea_48hr"), pl.col("crea_168hr")
    kdigo_crea = crea.select("stay_id", "time", pl.when(c >= 3 * c168).then(3)
                             .when((c >= 4) & ((c - c48 >= 0.3) | (c >= 1.5 * c168))).then(3)
                             .when(c >= 2 * c168).then(2)
                             .when(c >= c48 + 0.3).then(1)
                             .when(c >= 1.5 * c168).then(1)
                             .otherwise(0).alias("kdigo_crea"))

    # urine_rate: each urine record (not aggregated per hour, in recording order) per (hours since the previous
    # record + 1), gaps over 24h count as 1h; the maximum per hour
    cfg = _concept_cfg(src, "urine")
    tm = pl.col("time") - pl.col("time").shift(1).over("stay_id") + 1
    urine_rate = (_events(src, ["urine"], static=False)
                  .filter(pl.col("numeric_value").is_between(cfg["min"], cfg["max"]))
                  .sort("stay_id", "time", maintain_order=True)
                  .with_columns((pl.col("numeric_value") / pl.when(tm.is_null() | (tm > 24)).then(1).otherwise(tm)).alias("urine_rate"))
                  .filter(pl.col("urine_rate").is_between(0, 2000))
                  .group_by("stay_id", "time").agg(pl.col("urine_rate").max())
                  .sort("stay_id", "time"))
    urine = urine_rate
    for hrs in (6, 12, 24):
        urine = urine.join(slide(urine_rate, "urine_rate", hrs, "sum"), on=["stay_id", "time"])
    # as in kdigo_urine, rates are per kg only for stays with a weight
    urine = urine.join(_load(src, "weight"), on="stay_id", how="left").with_columns(
        [pl.when(pl.col("weight").is_not_null()).then(pl.col(col) / pl.col("weight")).otherwise(pl.col(col)).alias(col)
         for col in ("urine_rate_6hr", "urine_rate_12hr", "urine_rate_24hr")])
    t, u6, u12, u24 = pl.col("time"), pl.col("urine_rate_6hr"), pl.col("urine_rate_12hr"), pl.col("urine_rate_24hr")
    kdigo_urine = urine.select("stay_id", "time", pl.when((t >= 24) & (u24 < 0.3)).then(3)
                               .when((t >= 12) & (u12 == 0)).then(3)
                               .when((t >= 12) & (u12 < 0.5)).then(2)
                               .when((t >= 6) & (u6 < 0.5)).then(1)
                               .otherwise(0).alias("kdigo_urine"))

    kdigo = kdigo_crea.join(kdigo_urine, on=["stay_id", "time"], how="full", coalesce=True)
    return (kdigo.filter(pl.max_horizontal("kdigo_crea", "kdigo_urine") >= 1)
            .select("stay_id", "time", pl.lit(True).alias("aki")).sort("stay_id", "time"))


def _abx_duration(src: str) -> pl.DataFrame:
    """Antibiotic administrations with start `time` and duration `dur` in hours (`abx_duration`)."""
    spec = _source(src)[0]
    if "abx_duration" not in spec:
        raise NotImplementedError(f"No antibiotics (abx_duration) configured for {src} in weavehr-extensions/config.yml.")
    parts = []
    for item in spec["abx_duration"]:
        ev, level = _source_events(src, item["events"])
        start = pl.col(item.get("start", "time"))
        if "duration_minutes" in item:
            minutes = pl.col(item["duration_minutes"])
            if "end" in item:
                start = pl.col(item["end"]) - pl.duration(minutes=minutes)
        elif "end" in item:
            minutes = (pl.col(item["end"]) - start).dt.total_minutes()
        else:
            minutes = pl.lit(1)
        ev = (ev.filter(pl.col("code").str.contains(item["code"]))
              .select("subject_id", level, start.alias("time"), (minutes / 60).alias("dur"))
              .drop_nulls("time"))
        parts.append(_attach(ev, src, level, static=False))
    return pl.concat(parts).group_by("stay_id", "time").agg(pl.col("dur").max())


def _abx_cont(src: str, abx_win: int = 72, abx_max_gap: int = 24) -> pl.DataFrame:
    """Start of antibiotic treatment covering the next 72h (or until death) with gaps of at most 24h (`abx_cont`)."""
    abx = _abx_duration(src)
    death = _death_icu(src).filter("death_icu").group_by("stay_id").agg(pl.col("time").min().alias("death"))
    abx = (abx.join(death, on="stay_id", how="left")
           .filter(pl.col("death").is_null() | (pl.col("time") <= pl.col("death")))
           .sort("stay_id", "time"))
    # largest gap between the antibiotics covered so far and the next start (or the end of the window)
    end = pl.min_horizontal(pl.col("death").first(), pl.col("time").first() + abx_win)
    covered = pl.col("time") + pl.col("dur")
    if ext.get("reproduce_ricu_abx_cont", False):
        # ricu-extensions' abx_cont adds start (hours) and duration (minutes), which R does in seconds
        covered = covered * 3600
    gap = pl.coalesce(pl.col("time").shift(-1), end) - covered.cum_max()
    res = abx.rolling("time", period=f"{abx_win}i", offset="0i", closed="both", group_by="stay_id").agg(gap.max().alias("max_gap"))
    return res.filter(pl.col("max_gap") <= abx_max_gap).select("stay_id", "time")


def _susp_inf_alt(src: str, si_mode: str = "and", abx_win: int = 24, samp_win: int = 72) -> pl.DataFrame:
    """Suspected infection from continuous antibiotics and body fluid sampling (`susp_inf_alt`, ricu `susp_inf`)."""
    abx = _abx_cont(src).unique().sort("time")
    if si_mode != "abx":
        samp = _load(src, "samp").select("stay_id", "time").unique().sort("time")

    def followed_by(x: pl.DataFrame, y: pl.DataFrame, win: int) -> pl.DataFrame:
        # rows of x with a row of y within [time, time + win]
        y = y.with_columns(pl.col("time").alias("y_time"))
        res = x.join_asof(y, on="time", by="stay_id", strategy="forward", tolerance=win, check_sortedness=False)
        return res.filter(pl.col("y_time").is_not_null()).select("stay_id", "time")

    if si_mode == "and":
        res = pl.concat([followed_by(abx, samp, abx_win), followed_by(samp, abx, samp_win)])
    elif si_mode == "or":
        res = pl.concat([abx, samp])
    elif si_mode == "abx":
        res = abx
    elif si_mode == "samp":
        res = samp
    else:
        raise ValueError(f"si_mode must be one of 'and', 'or', 'abx', 'samp', got {si_mode!r}.")
    return res.unique().sort("stay_id", "time")


def _sep3_alt(src: str, si_mode: str = "and", si_lwr: int = 48, si_upr: int = 24, sofa_thresh: int = 2) -> pl.DataFrame:
    """Sepsis-3 onset (`sep3_alt`, ricu `sep3`): first SOFA increase >= 2 over its running minimum within
    [-48h, +24h] of the first suspected infection.

    WeavEHR's SOFA is recorded at changes, so it is carried forward between hours.
    """
    sofa = _load(src, "sofa")
    hours_grid = (sofa.group_by("stay_id").agg(pl.int_range(pl.col("time").min(), pl.col("time").max() + 1).alias("time"))
                  .explode("time"))
    sofa = (hours_grid.join(sofa, on=["stay_id", "time"], how="left").sort("stay_id", "time")
            .with_columns(pl.col("sofa").forward_fill().over("stay_id")))

    si = _susp_inf_alt(src, si_mode).group_by("stay_id").agg(pl.col("time").min().alias("si"))
    res = (sofa.join(si, on="stay_id")
           .filter(pl.col("time").is_between(pl.col("si") - si_lwr, pl.col("si") + si_upr))
           .sort("stay_id", "time")
           .with_columns((pl.col("sofa") - pl.col("sofa").cum_min().over("stay_id")).alias("delta_sofa")))
    return (res.filter(pl.col("delta_sofa") >= sofa_thresh)
            .group_by("stay_id").agg(pl.col("time").min())
            .with_columns(pl.lit(True).alias("sep3_alt")).sort("stay_id"))


DERIVED = {
    "los_icu": _los_icu,
    "hospital_id": _hospital_id,
    "death_icu": _death_icu,
    "aki": _aki,
    "sep3_alt": _sep3_alt,
}


# ------------------------------------------------------------------------------
# Counterpart of ricu::load_concepts

def load_concepts(concepts: str | List[str], src: str, interval=None, **kwargs) -> pd.DataFrame:
    """Load one or more concepts, merged on `stay_id` (and `time` for time series).

    Args:
        concepts: name of one or more concepts
        src: name of the WeavEHR dataset
        interval: only hours(1) is supported. Defaults to None (hourly).
        **kwargs: passed to the outcome concepts, e.g., `si_mode` of sep3_alt.

    Returns:
        pandas.DataFrame with `stay_id`, `time` (if any concept is a time series) and one column per concept.
    """
    _check_interval(interval)
    names = [concepts] if isinstance(concepts, str) else list(concepts)
    unknown = [name for name in names if name not in ext["concepts"] and name not in DERIVED]
    if unknown:
        raise ValueError(f"Unknown concepts for the WeavEHR backend: {unknown}")

    tables = {}
    for static in (True, False):
        base = [name for name in names if name in ext["concepts"] and _concept_cfg(src, name).get("static", False) == static]
        if base:
            events = _events(src, base, static)
            tables.update({name: _aggregate(src, events, name, static) for name in base})
    tables.update({name: DERIVED[name](src, **kwargs) for name in names if name in DERIVED})

    res = tables[names[0]]
    for name in names[1:]:
        on = [col for col in ("stay_id", "time") if col in res.columns and col in tables[name].columns]
        res = res.join(tables[name], on=on, how="full", coalesce=True)
    keys = [col for col in ("stay_id", "time") if col in res.columns]
    return res.sort(keys).select(*keys, *names).to_pandas()
