"""Reproduce mapping coverage from the saved availability snapshot (Python 3).

Run: python3 verify_mapping.py
Optional original-archive checksum check: python3 verify_mapping.py --archives DIR
This verifies coverage and mapping consistency, not raw-observation quality rules
or the original candidate search.
"""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance_km(first, second):
    lat1, lon1, lat2, lon2 = map(math.radians, (
        first['latitude'], first['longitude'], second['latitude'], second['longitude']))
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371.0088 * 2 * math.asin(math.sqrt(h))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, help='Optional directory of original downloaded DWD ZIPs.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    mapping_path = root / 'station_mapping.json'
    snapshot_path = root / 'coverage_snapshot.json'
    manifest = json.loads((root / 'source_manifest.json').read_text(encoding='utf-8'))
    for path in (mapping_path, snapshot_path):
        require(sha256(path) == manifest['supporting_file_sha256'][path.name], f'File checksum differs: {path.name}')
    document = json.loads(mapping_path.read_text(encoding='utf-8'))
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    metadata, rows = document['metadata'], document['mappings']
    start = datetime.fromisoformat(metadata['period_start'])
    end = datetime.fromisoformat(metadata['period_end_exclusive'])
    expected_days = (end-start).days
    require(snapshot['period_start'] == metadata['period_start'], 'Snapshot start differs.')
    require(snapshot['period_end_exclusive'] == metadata['period_end_exclusive'], 'Snapshot end differs.')
    variables = set(metadata['variables'])
    stations = {row['railway_station'] for row in rows}
    keys = {(row['railway_station'], row['variable']) for row in rows}
    require(len(stations) == metadata['railway_station_count'], 'Unexpected railway station count.')
    require(len(keys) == len(rows) == len(stations)*len(variables), 'Duplicate or missing mapping rows.')
    require(keys == {(station, var) for station in stations for var in variables}, 'Station-variable grid is incomplete.')
    masks = {}
    for key, entry in snapshot['series'].items():
        expected = expected_days if entry['frequency'] == 'daily' else expected_days*24
        require(entry['expected'] == expected, f'Wrong expected count for {key}.')
        missing = entry['unusable_indices']
        require(len(missing) == len(set(missing)), f'Duplicate offsets for {key}.')
        require(all(type(index) is int and 0 <= index < expected for index in missing), f'Invalid offsets for {key}.')
        masks[key] = set(missing)
    remaining_rows = []
    complete_primary = 0
    max_distance = 0.0
    eva_by_name = {}
    for row in rows:
        name, variable = row['railway_station'], row['variable']
        label = f'{name} / {variable}'
        eva = row['railway_eva_id']
        require(bool(re.fullmatch(r'\d{7}', eva)), f'Invalid railway EVA: {label}')
        require(eva_by_name.setdefault(name, eva) == eva, f'Conflicting railway EVA: {name}')
        primary, backup = row['primary_station_id'], row['backup_station_id']
        require(primary != backup, f'Primary and backup are identical: {label}')
        for role, sid in (('primary', primary), ('backup', backup)):
            require(bool(re.fullmatch(r'\d{5}', sid)), f'Invalid DWD ID: {label}')
            distance = distance_km(row['railway_coordinates'], row[f'{role}_coordinates'])
            require(distance <= 50, f'Distance exceeds 50 km: {label} / {role}')
            require(abs(distance-row[f'{role}_distance_km']) <= 0.011, f'Distance differs: {label} / {role}')
            max_distance = max(max_distance, distance)
            metrics = row[f'{role}_metrics']
            require(metrics['time_status'] in ('UTC_verified', 'daily_window_metadata_verified'), f'Time metadata not checked: {label}')
            require(metrics['hard_invalid'] == metrics['unresolved_conflicts'] == 0, f'Unresolved basic QC issue: {label}')
        require(set(row['primary_metrics']['units']) == set(row['backup_metrics']['units']), f'Units differ: {label}')
        if variable == 'SHK_TAG':
            normalize = lambda values: {re.sub(r'\s+', '', value).upper() for value in values}
            require(normalize(row['primary_metrics']['time_references']) == normalize(row['backup_metrics']['time_references']), f'Daily windows differ: {label}')
        primary_bad = masks[f'{variable}:{primary}']
        backup_bad = masks[f'{variable}:{backup}']
        expected = expected_days if variable == 'SHK_TAG' else expected_days*24
        remaining = len(primary_bad & backup_bad)
        recovered = len(primary_bad) - remaining
        require(len(primary_bad) == row['primary_unusable'], f'Primary gap count differs: {label}')
        require(recovered == row['recovered_by_backup'], f'Recoverable count differs: {label}')
        require(remaining == row['remaining_unusable'], f'Remaining count differs: {label}')
        require(abs((expected-remaining)/expected-row['combined_usable_rate']) < 1e-12, f'Joint rate differs: {label}')
        for role, bad in (('primary', primary_bad), ('backup', backup_bad)):
            require(len(bad) == expected-row[f'{role}_metrics']['meaningful'], f'Usable count differs: {label} / {role}')
        require(row['backup_metrics']['meaningful_rate'] >= metadata['backup_minimum_usable_rate'], f'Backup coverage below threshold: {label}')
        complete_primary += not primary_bad
        if remaining:
            remaining_rows.append((name, variable, remaining))
    require(len(set(eva_by_name.values())) == len(eva_by_name), 'Railway EVA IDs are not unique.')
    if args.archives:
        for source in manifest['dwd_archives']:
            path = args.archives / source['url'].rsplit('/', 1)[1]
            require(path.is_file(), f'Missing original archive: {path.name}')
            require(sha256(path) == source['sha256'], f'Archive changed: {path.name}. DWD recent files are updated over time; use the recorded snapshot for this audit.')
        print(f"Original ZIP checksums match: {len(manifest['dwd_archives'])} archives.")
    print(f'Coverage reproduced: {len(stations)} railway stations, {len(rows)} primary/backup mappings.')
    print(f'Maximum station distance: {max_distance:.2f} km.')
    print(f'Complete primary series: {complete_primary}; full joint coverage: {len(rows)-len(remaining_rows)}; remaining gaps: {len(remaining_rows)} mappings.')
    for name, variable, count in remaining_rows:
        print(f'  {name}: {variable}, {count} remaining hours/days')
    print('Scope: saved availability flags and mapping consistency. Raw measurements, extreme values, candidate selection and publication-time availability are not re-audited by this script.')


if __name__ == '__main__':
    main()
