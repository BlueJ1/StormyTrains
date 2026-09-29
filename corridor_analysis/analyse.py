"""Compute the statistics behind the corridor pages. Railway data only.

Reads data/journeys.parquet and data/stops.parquet from extract.py and writes one
JSON file per corridor plus comparison.json to stats/.

Run from the project root: .venv/bin/python corridor_analysis/analyse.py
"""
from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from corridors import COMPARED, CORRIDORS, MUNICH

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STATS = HERE / "stats"
CACHE = HERE / "cache"
DB_STATIONS_VERSION = "5.0.2"  # same pinned version as berlin_munich_weather
COORD_URL = f"https://unpkg.com/db-stations@{DB_STATIONS_VERSION}/data.ndjson"
COVERAGE_CHANGE = "2025-11-03"
DELAY_BINS = [-1e9, 6, 20, 40, 60, 1e9]
DELAY_LABELS = ["<6", "6–19", "20–39", "40–59", "≥60"]
MIN_TRAIN_RUNS = 300
MIN_SEGMENT_JOURNEYS = 300


def station_coordinates() -> dict[str, dict]:
    path = CACHE / f"db_stations_{DB_STATIONS_VERSION}.ndjson"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        # python.org builds on macOS ship without CA certificates.
        context = (ssl.create_default_context(cafile="/etc/ssl/cert.pem")
                   if not ssl.get_default_verify_paths().cafile and Path("/etc/ssl/cert.pem").exists()
                   else ssl.create_default_context())
        with urllib.request.urlopen(COORD_URL, timeout=90, context=context) as response:
            path.write_bytes(response.read())
    records = [json.loads(line) for line in path.read_text().splitlines()]
    # Look up by EVA number first, then by exact name (as fetch_weather.py does).
    return {**{r["name"]: r["location"] for r in records},
            **{str(int(r["id"])): r["location"] for r in records}}


def minutes(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 60


def share_table(frame: pd.DataFrame, key: str) -> dict:
    return pd.crosstab(frame[key], frame.severity_class, normalize="index").round(4).to_dict("index")


def corridor_stats(key: str, j: pd.DataFrame, s: pd.DataFrame, coords: dict) -> dict:
    cfg = CORRIDORS[key]
    j = j.copy()
    j["sched_min"] = minutes(j.destination_arrival_planned_local - j.origin_departure_planned_local)
    j["origin_dep_delay"] = minutes(j.origin_departure_reported_local - j.origin_departure_planned_local)
    j["family"] = j.recorded_route.str.split(" | ", regex=False).map(lambda r: cfg["family"](set(r)))
    ok = ~j.journey_canceled
    d = j.arrival_delay_min
    dep = j.origin_departure_planned_local
    R = {"key": key, "label": cfg["label"], "origin": cfg["origin"], "short": cfg["short"],
         "destination": cfg["destination"], "dest_label": cfg["dest_label"]}
    R["overview"] = dict(
        journeys=len(j), days=int(dep.dt.date.nunique()),
        first=str(dep.min().date()), last=str(dep.max().date()),
        per_day=round(len(j) / dep.dt.date.nunique(), 1),
        train_numbers=int(j.train_number.nunique()), routes=int(j.recorded_route.nunique()),
        ice=int(j.train_type.eq("ICE").sum()), ic=int(j.train_type.eq("IC").sum()),
        sched_median=float(j.sched_min.median()), sched_min=float(j.sched_min.min()),
        sched_p10=float(j.sched_min.quantile(.1)), sched_p90=float(j.sched_min.quantile(.9)),
        delay_mean=round(float(d.mean()), 1), delay_median=float(d.median()),
        p90=float(d.quantile(.9)), p99=float(d.quantile(.99)), dmax=float(d.max()),
        punctual_6=round(float((d[ok] < 6).mean()), 4), under20=round(float((d[ok] < 20).mean()), 4),
        classes={k: int(v) for k, v in j.severity_class.value_counts().items()},
        severe_or_canceled=round(float(j.severe_or_canceled.mean()), 4),
        canceled=round(float(j.journey_canceled.mean()), 4),
        early_5=int((d <= -5).sum()), early_min=float(d.min()))
    bins = list(range(-10, 185, 5))
    h = pd.cut(d[ok].clip(-10, 184.9), bins, right=False).value_counts().sort_index()
    R["hist"] = [dict(lo=int(iv.left), n=int(v)) for iv, v in h.items()]
    R["hist_over180"] = int((d >= 180).sum())
    R["hist_under_minus10"] = int((d < -10).sum())
    # Share of completed journeys arriving within x minutes, for the comparison.
    R["cdf"] = [dict(x=x, p=round(float((d[ok] < x).mean()), 4)) for x in range(-5, 121)]
    j["month"] = dep.dt.strftime("%Y-%m")
    m = j.groupby("month").agg(n=("journey_id", "size"), mean=("arrival_delay_min", "mean"),
                               soc=("severe_or_canceled", "mean"))
    sh = share_table(j, "month")
    R["monthly"] = [dict(month=k, n=int(r.n), mean=round(float(r["mean"]), 1), soc=round(float(r.soc), 4),
                         **{c: sh[k].get(c, 0.0) for c in ["normal_minor", "moderate", "severe", "canceled"]})
                    for k, r in m.iterrows()]
    j["hour"] = dep.dt.hour
    hh = j.groupby("hour").agg(n=("journey_id", "size"), soc=("severe_or_canceled", "mean"),
                               mean=("arrival_delay_min", "mean"), sched=("sched_min", "median"))
    R["hourly"] = [dict(hour=int(k), n=int(r.n), soc=round(float(r.soc), 4),
                        mean=round(float(r["mean"]), 1), sched=float(r.sched)) for k, r in hh.iterrows()]
    j["wd"] = dep.dt.dayofweek
    w = j.groupby("wd").agg(n=("journey_id", "size"), soc=("severe_or_canceled", "mean"),
                            mean=("arrival_delay_min", "mean"), canc=("journey_canceled", "mean"))
    R["weekday"] = [dict(wd=int(k), n=int(r.n), soc=round(float(r.soc), 4), mean=round(float(r["mean"]), 1),
                         canc=round(float(r.canc), 4)) for k, r in w.iterrows()]
    f = j.groupby("family").agg(n=("journey_id", "size"), sched=("sched_min", "median"),
                                mean=("arrival_delay_min", "mean"),
                                median=("arrival_delay_min", "median"),
                                soc=("severe_or_canceled", "mean"), canc=("journey_canceled", "mean"))
    sev = share_table(j, "family")
    R["families"] = [dict(key=k, label=cfg["families"][k][0], via=cfg["families"][k][1], n=int(f.loc[k, "n"]),
                          sched=float(f.loc[k, "sched"]), mean=round(float(f.loc[k, "mean"]), 1),
                          median=float(f.loc[k, "median"]), soc=round(float(f.loc[k, "soc"]), 4),
                          canc=round(float(f.loc[k, "canc"]), 4),
                          **{c: sev[k].get(c, 0.0) for c in ["normal_minor", "moderate", "severe", "canceled"]})
                     for k in cfg["families"] if k in f.index]
    lo = int(j.sched_min.min() // 15 * 15)
    hi = int(np.ceil(j.sched_min.quantile(.998) / 15) * 15)
    R["sched_over"] = int((j.sched_min >= hi).sum())
    fam_sched = j[j.sched_min < hi].groupby(
        [pd.cut(j.sched_min, range(lo, hi + 1, 15), right=False), "family"], observed=False
    ).size().unstack(fill_value=0)
    R["sched_hist"] = [dict(lo=int(iv.left), **{c: int(fam_sched.loc[iv, c]) for c in fam_sched.columns})
                       for iv in fam_sched.index]
    # Delay build-up along the main route family.
    main = j.loc[j.family.eq(cfg["main_family"]), "journey_id"]
    sd = s[s.journey_id.isin(main) & ~s.selected_event_canceled.fillna(False).astype(bool)].copy()
    sd["delay"] = minutes(sd.event_reported_time - sd.event_planned_time)
    sd = sd.merge(j[["journey_id", "origin_departure_planned_local"]], on="journey_id")
    sd["mins"] = minutes(sd.event_planned_time - sd.origin_departure_planned_local)
    st = sd.groupby("station_name").agg(n=("delay", "size"), mean=("delay", "mean"),
                                        p20=("delay", lambda x: (x >= 20).mean()),
                                        p6=("delay", lambda x: (x >= 6).mean()), mins=("mins", "median"))
    st = st[st.n >= 0.12 * len(main)].sort_values("mins")
    R["buildup"] = [dict(station=k, n=int(r.n), mean=round(float(r["mean"]), 1), p20=round(float(r.p20), 4),
                         p6=round(float(r.p6), 4), mins=float(r.mins)) for k, r in st.iterrows()]
    R["main_family"] = cfg["main_family"]
    R["main_n"] = int(len(main))
    outcome = j[["journey_id", "arrival_delay_min", "journey_canceled", "severe_or_canceled"]]

    def conditional(station: str, x: pd.DataFrame) -> dict:
        x = x.merge(outcome, on="journey_id")
        x["b"] = pd.cut(x.dl, DELAY_BINS, right=False, labels=DELAY_LABELS)
        g = x.groupby("b", observed=False).agg(n=("journey_id", "size"), soc=("severe_or_canceled", "mean"),
                                               final=("arrival_delay_min", "median"))
        done = x[~x.journey_canceled]
        return dict(station=station, n=int(len(x)),
                    corr=round(float(done.dl.corr(done.arrival_delay_min)), 3),
                    rows=[dict(b=str(k), n=int(r.n), soc=round(float(r.soc), 4) if r.n else None,
                               final=float(r.final) if pd.notna(r.final) else None) for k, r in g.iterrows()])
    origin_rows = j[~j.origin_departure_canceled.fillna(False).astype(bool)][["journey_id", "origin_dep_delay"]]
    R["enroute"] = [conditional(cfg["origin"], origin_rows.rename(columns={"origin_dep_delay": "dl"}))]
    for station in cfg["enroute"]:
        x = s[s.station_name.eq(station) & ~s.final_destination
              & ~s.selected_event_canceled.fillna(False).astype(bool)].copy()
        x["dl"] = minutes(x.event_reported_time - x.event_planned_time)
        R["enroute"].append(conditional(station, x.drop_duplicates("journey_id")[["journey_id", "dl"]]))
    # Does the run start at the origin (stop number 1) or further back?
    first = s[s.stop_index.eq(0)].set_index("journey_id").train_line_station_num.eq(1)
    starts = origin_rows.assign(starts_here=origin_rows.journey_id.map(first), late=origin_rows.origin_dep_delay >= 6)
    R["origin_start"] = dict(starts_here=round(float(first.mean()), 4),
                             late_if_starts_here=round(float(starts.late[starts.starts_here].mean()), 4),
                             late_if_through=round(float(starts.late[~starts.starts_here].mean()), 4))
    oc = j.origin_departure_canceled.fillna(False).astype(bool)
    dc = j.destination_arrival_canceled.fillna(False).astype(bool)
    R["cancel"] = dict(both=int((oc & dc).sum()), dest_only=int((~oc & dc).sum()),
                       origin_only=int((oc & ~dc).sum()),
                       intermediate_only=int((j.any_selected_stop_canceled.fillna(False).astype(bool)
                                              & ~j.journey_canceled).sum()))
    no_update = s.assign(same=s.event_reported_time.eq(s.event_planned_time)).groupby("journey_id").same.all()
    before = dep < COVERAGE_CHANGE
    R["quality"] = dict(no_update_journeys=int(no_update.sum()), zero_delay_arrivals=int((d == 0).sum()),
                        gaps=int(j.unobserved_sequence_positions.gt(0).sum()),
                        gap_before=round(float(j.unobserved_sequence_positions[before].gt(0).mean()), 4),
                        gap_after=round(float(j.unobserved_sequence_positions[~before].gt(0).mean()), 4))
    t = j.groupby("train_number").agg(
        n=("journey_id", "size"), soc=("severe_or_canceled", "mean"), median=("arrival_delay_min", "median"),
        dep=("origin_departure_planned_local", lambda x: x.dt.strftime("%H:%M").mode().iloc[0]),
        sched=("sched_min", "median"), fam=("family", lambda x: x.mode().iloc[0]),
        type=("train_type", lambda x: x.mode().iloc[0]))
    t = t[t.n >= MIN_TRAIN_RUNS]
    R["trains_count"] = int(len(t))

    def trains(frame: pd.DataFrame) -> list[dict]:
        return [dict(train=str(k), n=int(r.n), soc=round(float(r.soc), 4), median=float(r["median"]),
                     dep=r.dep, sched=float(r.sched), fam=r.fam, type=r.type) for k, r in frame.iterrows()]
    R["trains_worst"] = trains(t.sort_values("soc", ascending=False).head(6))
    R["trains_best"] = trains(t.sort_values("soc").head(6))
    # Map: consecutive recorded stops, straight lines between station coordinates.
    ordered = s.sort_values(["journey_id", "stop_index"])
    nxt = ordered.groupby("journey_id").station_name.shift(-1)
    seg = pd.DataFrame({"a": ordered.station_name, "b": nxt}).dropna().value_counts().reset_index(name="n")
    seg = seg[seg.n >= MIN_SEGMENT_JOURNEYS]
    names = set(seg.a) | set(seg.b)
    evas = s[s.station_name.isin(names)].groupby("station_name").eva.agg(lambda x: x.mode().iloc[0])
    served = s.groupby("station_name").journey_id.nunique()
    where = {n: coords.get(str(int(evas[n]))) or coords[n] for n in names}
    R["map"] = dict(segments=[dict(a=r.a, b=r.b, n=int(r.n)) for r in seg.itertuples()],
                    stations=[dict(name=n, lat=where[n]["latitude"], lon=where[n]["longitude"],
                                   n=int(served[n])) for n in sorted(names)])
    return R


def progress_correlation(j: pd.DataFrame, s: pd.DataFrame) -> list[dict]:
    """Correlation of the delay at a stop with the final arrival delay, by the
    share of the scheduled journey already covered at that stop."""
    x = s[~s.final_destination & ~s.selected_event_canceled.fillna(False).astype(bool)].merge(
        j[~j.journey_canceled][["journey_id", "origin_departure_planned_local",
                                "destination_arrival_planned_local", "arrival_delay_min"]], on="journey_id")
    x["dl"] = minutes(x.event_reported_time - x.event_planned_time)
    x["progress"] = (minutes(x.event_planned_time - x.origin_departure_planned_local)
                     / minutes(x.destination_arrival_planned_local - x.origin_departure_planned_local))
    x["bin"] = pd.cut(x.progress, np.linspace(0, 1, 11), right=False)
    rows = []
    for iv, g in x.groupby("bin", observed=True):
        rows.append(dict(lo=round(float(iv.left), 2), n=int(len(g)),
                         corr=round(float(g.dl.corr(g.arrival_delay_min)), 3),
                         p60_soc=round(float((g.arrival_delay_min[g.dl >= 60] >= 60).mean()), 4)
                         if (g.dl >= 60).any() else None))
    return rows


def main() -> None:
    STATS.mkdir(exist_ok=True)
    journeys = pd.read_parquet(DATA / "journeys.parquet")
    stops = pd.read_parquet(DATA / "stops.parquet")
    coords = station_coordinates()
    audit = json.loads((HERE / "extraction_audit.json").read_text())
    comparison = {"corridors": [], "overlap": {}}
    ids = {key: set(journeys.journey_id[journeys.corridor.eq(key)]) for key in CORRIDORS}
    for key in CORRIDORS:
        j = journeys[journeys.corridor.eq(key)].reset_index(drop=True)
        s = stops[stops.corridor.eq(key)]
        R = corridor_stats(key, j, s, coords)
        R["panel"] = {k: audit[key][k] for k in (
            "stops_before_panel_filter", "stops_removed_outside_panel", "stations_removed_outside_panel")}
        # Runs this corridor shares with the others (one run can serve several pairs).
        R["shared_runs"] = {other: len(ids[key] & ids[other]) for other in CORRIDORS if other != key}
        (STATS / f"{key}.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=str))
        print(key, json.dumps(R["overview"], ensure_ascii=False))
        if key not in COMPARED:
            continue
        R["progress"] = progress_correlation(j, s)
        comparison["corridors"].append({k: R[k] for k in (
            "key", "label", "origin", "short", "overview", "cdf", "monthly", "hourly", "weekday",
            "families", "cancel", "quality", "enroute", "progress", "trains_count", "origin_start")})
    keys = COMPARED
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            comparison["overlap"][f"{a}-{b}"] = len(ids[a] & ids[b])
    monthly = pd.DataFrame({c["key"]: {m["month"]: m["soc"] for m in c["monthly"]}
                            for c in comparison["corridors"]})
    comparison["monthly_corr"] = {f"{a}-{b}": round(float(monthly[a].corr(monthly[b])), 3)
                                  for i, a in enumerate(keys) for b in keys[i + 1:]}
    comparison["destination"] = MUNICH
    (STATS / "comparison.json").write_text(json.dumps(comparison, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: comparison[k] for k in ("overlap", "monthly_corr")}, indent=1))


if __name__ == "__main__":
    main()
