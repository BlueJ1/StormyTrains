"""Add DWD-style precipitation codes to downloaded GFS weather partitions."""

import argparse
import io
import json
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import eccodes
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from download_gfs_weather import KEY, PRECIP, sha256


BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
PHASES = ("CSNOW", "CICEP", "CFRZR", "CRAIN")
COLUMN = "wrtr"
METHOD = "noaa_gfs_hour_and_instantaneous_phase_v2"
GRID = {
    "Ni": 1440,
    "Nj": 721,
    "latitudeOfFirstGridPointInDegrees": 90,
    "longitudeOfFirstGridPointInDegrees": 0,
    "iDirectionIncrementInDegrees": 0.25,
    "jDirectionIncrementInDegrees": 0.25,
    "iScansNegatively": 0,
    "jScansPositively": 0,
    "jPointsAreConsecutive": 0,
    "alternativeRowScanning": 0,
}
http = threading.local()


def session():
    if not hasattr(http, "session"):
        http.session = requests.Session()
        retries = Retry(total=4, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
        http.session.mount("https://", HTTPAdapter(max_retries=retries))
    return http.session


def phase_range(index, lead, read_bucket):
    records = [line.split(":") for line in index.splitlines()]
    bucket_start = ((lead - 1) // 6) * 6
    steps = [(f"{lead} hour fcst", "instant", lead)]
    if read_bucket:
        steps.append((f"{bucket_start}-{lead} hour ave fcst", "avg", bucket_start))
    expected = [(phase, *step) for step in steps for phase in PHASES]
    fields = [(phase, "surface", label) for phase, label, _, _ in expected]
    for i in range(len(records) - len(expected)):
        group = records[i:i + len(expected)]
        if [tuple(record[3:6]) for record in group] == fields:
            return int(group[0][1]), int(records[i + len(expected)][1]) - 1, expected
    raise ValueError(f"Missing contiguous precipitation messages for lead {lead}")


def fetch_phases(init, lead, indices, read_bucket):
    date = init.strftime("%Y%m%d")
    hour = init.strftime("%H")
    url = f"{BUCKET}/gfs.{date}/{hour}/atmos/gfs.t{hour}z.pgrb2.0p25.f{lead:03d}"
    response = session().get(url + ".idx", timeout=(10, 30))
    if response.status_code == 404:
        return None
    response.raise_for_status()
    first, last, messages = phase_range(response.text, lead, read_bucket)
    response = session().get(url, headers={"Range": f"bytes={first}-{last}"}, timeout=(10, 45))
    response.raise_for_status()
    if response.status_code != 206 or len(response.content) != last - first + 1:
        raise ValueError(f"Unexpected GRIB range response for {url}")

    stream = io.BytesIO(response.content)
    phases = []
    for phase, _, step_type, start_step in messages:
        # Read the message length from the fixed GRIB2 indicator section.
        header = stream.read(16)
        if len(header) != 16 or header[:4] != b"GRIB" or header[7] != 2:
            raise ValueError(f"Invalid GRIB2 message for {url}")
        length = int.from_bytes(header[8:], "big")
        body = stream.read(length - len(header))
        if length <= len(header) or len(body) != length - len(header) or body[-4:] != b"7777":
            raise ValueError(f"Incomplete GRIB2 message for {url}")
        handle = eccodes.codes_new_from_message(header + body)
        try:
            expected = {
                "shortName": phase.lower(),
                "stepType": step_type,
                "startStep": start_step,
                "dataDate": int(date),
                "dataTime": int(hour) * 100,
                "endStep": lead,
                "stepUnits": 1,
            }
            for field, value in (GRID | expected).items():
                if eccodes.codes_get(handle, field) != value:
                    raise ValueError(f"Unexpected {field} in {url}")
            phases.append(eccodes.codes_get_double_elements(handle, "values", indices))
        finally:
            eccodes.codes_release(handle)
    if stream.tell() != len(response.content):
        raise ValueError(f"Unexpected extra precipitation messages in {url}")
    return np.asarray(phases).reshape(-1, len(PHASES), len(indices))


def forecast_codes(phases):
    snow, ice, freezing_rain, rain = phases
    solid = (snow == 1) | (ice == 1)
    liquid = (freezing_rain == 1) | (rain == 1)
    usable = np.isin(phases, [0, 1]).all(axis=0) & (solid | liquid)
    codes = np.select([solid & liquid, solid], [8, 7], default=6)
    result = pd.array(codes, dtype="Int8")
    result[~usable] = pd.NA
    return result


def read_forecast(item):
    (init, lead), rows = item
    lat = (90 - rows.grid_latitude) * 4
    lon = rows.grid_longitude.mod(360) * 4
    if not (lat.between(0, 720) & lat.mod(1).eq(0) & lon.mod(1).eq(0)).all():
        raise ValueError("Expected native NOAA GFS 0.25 degree grid coordinates")
    indices = (lat * 1440 + lon).astype(int).tolist()
    use_bucket = rows.only_wet_hour.to_numpy()
    phases = fetch_phases(init, lead, indices, use_bucket.any())
    if phases is None:
        return rows.index, pd.array([pd.NA] * len(rows), dtype="Int8"), 0, True
    codes = forecast_codes(phases[0])
    filled = 0
    if use_bucket.any():
        bucket = forecast_codes(phases[1])
        apply_bucket = use_bucket & ~bucket.isna()
        filled = int((apply_bucket & codes.isna()).sum())
        codes[apply_bucket] = bucket[apply_bucket]
    return rows.index, codes, filled, False


def add_codes(frame, accumulation, workers, label):
    valid_lead = frame.lead_hour.gt(0)
    precip = frame.precipitation_1h_mm
    frame[COLUMN] = pd.array([pd.NA] * len(frame), dtype="Int8")
    frame.loc[valid_lead & precip.eq(0), COLUMN] = 0
    wet = frame.loc[valid_lead & precip.gt(0), KEY].copy()
    # The 1–6 hour bucket describes this hour alone when it had no earlier precipitation.
    wet["only_wet_hour"] = accumulation.eq(precip).loc[wet.index]
    groups = list(wet.groupby(KEY[:2], sort=False))
    missing_forecasts = 0
    filled_from_bucket = 0
    print(f"{label}: reading {len(groups):,} forecasts for {len(wet):,} wet grid rows", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (indices, codes, filled, missing) in enumerate(pool.map(read_forecast, groups), 1):
            frame.loc[indices, COLUMN] = codes
            filled_from_bucket += filled
            missing_forecasts += missing
            if n % 500 == 0:
                print(f"{label}: read {n:,}/{len(groups):,} forecasts", flush=True)
    if groups and missing_forecasts == len(groups):
        raise ValueError(f"{label}: NOAA returned no GRIB indices for this partition")
    return {
        "method": METHOD,
        "source": BUCKET,
        "wet_rows": len(wet),
        "wet_rows_without_code": int(frame.loc[wet.index, COLUMN].isna().sum()),
        "wet_rows_filled_from_bucket": filled_from_bucket,
        "missing_forecast_indices": missing_forecasts,
        "code_counts": {str(k): int(v) for k, v in frame[COLUMN].value_counts().items()},
    }


def backfill(manifest, destination, workers):
    report = json.loads(manifest.read_text())
    files = report["files"]
    paths = {name: manifest.parent / record["file"] for name, record in files.items()}
    for name, path in paths.items():
        if sha256(path) != files[name]["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
    source_hash = files["weather"]["sha256"]
    raw_hash = files["raw"]["sha256"]
    target_manifest = destination / manifest.name
    if target_manifest.exists():
        saved = json.loads(target_manifest.read_text())
        phase = saved.get("precipitation_type", {})
        if (phase.get("source_weather_sha256") == source_hash and phase.get("method") == METHOD
                and saved["files"]["raw"]["sha256"] == raw_hash
                and all(
                    (destination / record["file"]).is_file()
                    and sha256(destination / record["file"]) == record["sha256"]
                    for record in saved["files"].values()
                )):
            print(f"{manifest.stem}: verified existing files", flush=True)
            return

    frame = pd.read_parquet(paths["weather"])
    raw_keys = pd.read_parquet(paths["raw"], columns=KEY + [PRECIP])
    if (len(frame) != report["rows"] or frame.duplicated(KEY).any()
            or not frame[KEY].equals(raw_keys[KEY])):
        raise ValueError(f"Invalid source row count or keys: {paths['weather']}")
    label = manifest.stem.removeprefix("gfs_manifest_")
    report["precipitation_type"] = add_codes(frame, raw_keys[PRECIP], workers, label)
    report["precipitation_type"]["source_weather_sha256"] = source_hash
    raw = destination / paths["raw"].name
    tmp_raw = raw.with_suffix(".tmp")
    shutil.copyfile(paths["raw"], tmp_raw)
    tmp_raw.replace(raw)
    weather = destination / paths["weather"].name
    tmp_weather = weather.with_suffix(".tmp")
    frame.to_parquet(tmp_weather, index=False, compression="zstd")
    tmp_weather.replace(weather)
    files["weather"].update(
        bytes=weather.stat().st_size,
        sha256=sha256(weather),
        nulls=frame.isna().sum().to_dict(),
    )
    tmp_manifest = target_manifest.with_suffix(".tmp")
    tmp_manifest.write_text(json.dumps(report, indent=2) + "\n")
    tmp_manifest.replace(target_manifest)
    print(f"{label}: exported {len(frame):,} rows with WRTR {report['precipitation_type']['code_counts']}",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="Directories of completed exports or manifest files")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", default=12, type=int)
    args = parser.parse_args()
    manifests = sorted({
        path
        for source in args.sources
        for path in (source.glob("gfs_manifest_*.json") if source.is_dir() else [source])
    })
    if not manifests or args.workers < 1:
        parser.error("Provide at least one manifest and a positive worker count")
    if any(path.parent.resolve() == args.output.resolve() for path in manifests):
        parser.error("Use a separate output directory")
    if len({path.name for path in manifests}) != len(manifests):
        parser.error("Source partitions must have unique manifest names")
    args.output.mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        backfill(manifest, args.output, args.workers)


if __name__ == "__main__":
    main()
