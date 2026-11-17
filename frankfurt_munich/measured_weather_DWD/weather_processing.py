"""Reproducible DWD intake and initial railway-station quality screening.

No network requests and no anomaly-driven substitutions are performed.
"""
from pathlib import Path
import hashlib
import io
import json
import zipfile
import sys
import platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd

BASE_URL = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/'
PRECIP_DOC = BASE_URL + 'hourly/precipitation/DESCRIPTION_obsgermany_climate_hourly_precipitation_en.pdf'
CONFIG = {
    'R1': dict(slug='precipitation', title='Precipitation', column='precipitation_mm', unit='mm',
               qn='QN_8', daily=False, prefix='P', product='hourly/precipitation',
               doc=PRECIP_DOC, rules=dict(high=30., change=30., difference=20., constant=6)),
    'WRTR': dict(slug='precipitation_type', title='Precipitation type', column='precipitation_type_code',
                 unit='code', qn='QN_8', daily=False, prefix='PT', product='hourly/precipitation',
                 doc=PRECIP_DOC, rules={}),
    'F': dict(slug='mean_wind', title='Mean wind', column='mean_wind_ms', unit='m/s',
              qn='QN_3', daily=False, prefix='MW', product='hourly/wind',
              doc=BASE_URL+'hourly/wind/DESCRIPTION_obsgermany_climate_hourly_wind_en.pdf',
              rules=dict(high=25., change=15., difference=15., constant=12)),
    'FX_911': dict(slug='gust', title='Gust', column='gust_ms', unit='m/s',
                   qn='QN_8', daily=False, prefix='GU', product='hourly/extreme_wind',
                   doc=BASE_URL+'hourly/extreme_wind/DESCRIPTION_obsgermany_climate_hourly_extreme_wind_en.pdf',
                   rules=dict(high=35., change=25., difference=25., constant=12)),
    'SHK_TAG': dict(slug='snow_depth', title='Snow depth', column='snow_depth_cm', unit='cm',
                    qn='QN_4', daily=True, prefix='SD', product='daily/kl',
                    doc=BASE_URL+'daily/kl/DESCRIPTION_obsgermany-climate-daily-kl_en.pdf',
                    rules=dict(high=50., change=20., difference=30., constant=30)),
}
TYPE_LABELS = {0: 'none', 1: 'precipitation_only_unspecified', 4: 'unascertainable_or_depositional',
               6: 'liquid', 7: 'solid', 8: 'liquid_and_solid', 9: 'unascertainable'}
KNOWN_TYPES = {0, 6, 7, 8}
ALL_TYPES = set(TYPE_LABELS)
QUALITY_LEVELS = {1, 2, 3, 5, 7, 8, 9, 10}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_table(blob):
    table = pd.read_csv(io.BytesIO(blob), sep=';', encoding='latin1', dtype=str, keep_default_na=False)
    table.columns = table.columns.str.strip()
    for column in table:
        table[column] = table[column].str.strip()
    return table


def normalized(raw):
    numbers = pd.to_numeric(raw, errors='coerce').astype(float)
    return numbers.mask(numbers.eq(-999) | ~np.isfinite(numbers))


def md_table(frame):
    if frame.empty:
        return 'None.'
    frame = frame.copy().fillna('')
    def token(v):
        if isinstance(v, (float, np.floating)):
            return f'{v:.3f}'.rstrip('0').rstrip('.')
        return str(v).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                     '| ' + ' | '.join(['---']*len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(token(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def longest_false(valid):
    arr = np.r_[False, ~np.asarray(valid, dtype=bool), False].astype(int)
    changes = np.diff(arr)
    lengths = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    return int(lengths.max()) if len(lengths) else 0


class WeatherWorkflow:
    def __init__(self, variable, weather_dir, archive_dir):
        self.variable = variable
        self.config = CONFIG[variable].copy()
        self.rules = self.config['rules'].copy()
        self.slug = self.config['slug']
        self.column = self.config['column']
        self.daily = self.config['daily']
        self.time = 'observation_date' if self.daily else 'timestamp_utc'
        self.step = pd.Timedelta(days=1) if self.daily else pd.Timedelta(hours=1)
        self.root = Path(weather_dir).resolve()
        self.archive_dir = Path(archive_dir).resolve()
        for name in ['station_mapping.json', 'source_manifest.json']:
            if not (self.root / name).is_file():
                raise FileNotFoundError(f'Missing {name}. Set WEATHER_DIR to the folder containing the matching mapping and manifest.')
        self.mapping = json.loads((self.root / 'station_mapping.json').read_text())
        self.manifest = json.loads((self.root / 'source_manifest.json').read_text())
        self.maps = [r for r in self.mapping['mappings'] if r['variable'] == variable]
        assert self.maps and len({r['railway_eva_id'] for r in self.maps}) == len(self.maps)
        self.start = pd.Timestamp(self.mapping['metadata']['period_start'], tz='UTC')
        self.end = pd.Timestamp(self.mapping['metadata']['period_end_exclusive'], tz='UTC')
        self.read_start = self.start - pd.Timedelta(hours=48)
        self.grid = pd.date_range(self.read_start, self.end, freq='D' if self.daily else 'h', inclusive='left')
        self.source_key = 'dwd_station_id'
        self.by_station_archives = {}
        for row in self.maps:
            assert str(row['railway_eva_id']).isdigit() and len(row['railway_eva_id']) == 7
            for role in ['primary', 'backup']:
                sid = row[role+'_station_id']
                assert len(sid) == 5 and sid.isdigit() and 0 <= row[role+'_distance_km'] <= 50
                self.by_station_archives.setdefault(sid, set()).update(row[role+'_source_archives'])
            assert row['primary_station_id'] != row['backup_station_id']
        self.archive_names = sorted({n for names in self.by_station_archives.values() for n in names})
        sources = {r['url'].split('/')[-1]: r for r in self.manifest['dwd_archives']}
        self.inventory = []
        for name in self.archive_names:
            path = self.archive_dir / name
            if not path.is_file():
                raise FileNotFoundError(f'Missing original archive: {path}. Use the exact version listed in source_manifest.json.')
            source = sources[name]
            assert digest(path) == source['sha256'], f'Archive version differs from manifest: {name}'
            self.inventory.append(dict(archive=name, sha256=source['sha256'], bytes=path.stat().st_size,
                                       url=source['url'], downloaded_utc=source.get('downloaded_utc')))
        self.inventory = pd.DataFrame(self.inventory)
        self.out = self.root / 'observations'
        self.qc = self.root / 'quality' / self.slug
        self.out.mkdir(exist_ok=True)
        self.qc.mkdir(parents=True, exist_ok=True)
        self.metadata_records = []
        self.invalid_timestamps = []
        self.archive_rows = []
        self.overlaps = []
        self.conflicts = []
        self.source_monthly_quality = pd.DataFrame()
        self.validated = False

    def rule_table(self):
        if self.variable == 'WRTR':
            return pd.DataFrame([
                {'check': 'Usable type codes', 'value': '0 / 6 / 7 / 8'},
                {'check': 'Unknown type codes', 'value': '1 / 4 / 9: traceable, eligible for backup'},
                {'check': 'Strong spatial type disagreement', 'value': 'liquid (6) versus solid (7)'},
                {'check': 'Other categorical differences', 'value': 'separate diagnostic list; no automatic source change'}])
        unit = self.config['unit']
        return pd.DataFrame([
            {'check': 'Physical range', 'value': f'negative values flagged; retained ({unit})'},
            {'check': 'High value', 'value': f'> {self.rules["high"]} {unit}'},
            {'check': 'Adjacent-time absolute change', 'value': f'> {self.rules["change"]} {unit}'},
            {'check': 'Primary/backup absolute difference', 'value': f'> {self.rules["difference"]} {unit}'},
            {'check': 'Identical positive run', 'value': f'>= {self.rules["constant"]} '+('days' if self.daily else 'hours')},
            {'check': 'Zero runs', 'value': 'valid; no stuck-sensor flag'}])

    def show_inputs(self):
        print(f'{self.config["title"]}: {len(self.maps)} railway stations, {len(self.by_station_archives)} DWD stations, '
              f'{len(self.archive_names)} hash-matched ZIPs.')
        print(f'Study: {self.start.date()} to {self.end.date()} exclusive; preceding buffer: 48 hours / 2 dates.')

    def _metadata_for(self, archive, frame, parameters):
        times = frame['_time']
        reference = pd.Series('', index=frame.index, dtype=object)
        unit = pd.Series('', index=frame.index, dtype=object)
        covered = pd.Series(False, index=frame.index)
        for row in parameters.to_dict('records'):
            if row.get('Parameter') != self.variable:
                continue
            begin = pd.to_datetime(row.get('Von_Datum'), format='%Y%m%d', utc=True, errors='coerce')
            finish = pd.to_datetime(row.get('Bis_Datum'), format='%Y%m%d', utc=True, errors='coerce') + pd.Timedelta(days=1)
            mask = times.ge(begin) & times.lt(finish)
            if not mask.any():
                continue
            ref = row.get('Zusatz-Info', '')
            un = row.get('Einheit', '')
            expected_unit = {'mm'} if self.variable == 'R1' else {'numerischer Code'} if self.variable == 'WRTR' else {'cm'} if self.daily else {'m/sec', 'm/s'}
            assert un in expected_unit, (archive, 'Unexpected unit', un)
            assert ref == '06 UTC' if self.daily else 'UTC' in ref and 'MEZ' not in ref, (archive, ref)
            already = mask & covered
            assert not (already & (reference.ne(ref) | unit.ne(un))).any(), (archive, 'Conflicting effective metadata')
            reference.loc[mask] = ref
            unit.loc[mask] = un
            covered |= mask
        assert covered.all(), f'Unverified time/unit metadata in {archive}: {int((~covered).sum())} retained rows'
        frame['time_reference_metadata'] = reference
        frame['source_unit_metadata'] = unit
        frame['time_metadata_verified'] = covered
        return frame

    def _read_archive(self, name, sid):
        archive = self.archive_dir / name
        inventory = self.inventory.set_index('archive').loc[name]
        with zipfile.ZipFile(archive) as z:
            members = z.namelist()
            product_members = [n for n in members if Path(n).name.startswith('produkt_') and n.endswith('.txt')]
            assert len(product_members) == 1, (name, product_members)
            member = product_members[0]
            data = parse_table(z.read(member))
            needed = {'STATIONS_ID', 'MESS_DATUM', self.variable, self.config['qn']}
            assert needed.issubset(data.columns), (name, needed-set(data.columns))
            assert set(data.STATIONS_ID.str.zfill(5)) == {sid}, f'Station ID mismatch: {name}'
            data['source_line'] = np.arange(2, len(data)+2, dtype=np.int64)
            times = pd.to_datetime(data.MESS_DATUM, format='%Y%m%d' if self.daily else '%Y%m%d%H', utc=True, errors='coerce')
            invalid = data.loc[times.isna(), ['STATIONS_ID','MESS_DATUM','source_line']].copy()
            invalid['source_archive'] = name
            self.invalid_timestamps.extend(invalid.to_dict('records'))
            raw_count = len(data)
            data['_time'] = times
            data = data.loc[times.ge(self.read_start) & times.lt(self.end)].copy()
            parameters = []
            for n in members:
                if n.endswith('.txt') and Path(n).name.startswith('Metadaten_'):
                    record = {'dwd_station_id': sid, 'archive': name, 'member': n}
                    if any(token in n for token in ['Parameter', 'Geographie', 'Geraete']):
                        table = parse_table(z.read(n))
                        if 'Parameter' in n:
                            parameters.append(table)
                            table = table.loc[table.Parameter.eq(self.variable)]
                        record['records'] = table.to_dict('records')
                    self.metadata_records.append(record)
            assert parameters, f'Missing parameter metadata: {name}'
            data = self._metadata_for(name, data, pd.concat(parameters, ignore_index=True))
            result = pd.DataFrame({
                self.time: data['_time'], 'dwd_station_id': sid,
                'raw_value_token': data[self.variable], 'quality_code_raw': data[self.config['qn']],
                'raw_timestamp_token': data.MESS_DATUM, 'source_line': data.source_line,
                'source_archive': name, 'source_member': member, 'source_sha256': inventory.sha256,
                'archive_period': 'historical' if name.endswith('_hist.zip') else 'recent',
                'time_reference_metadata': data.time_reference_metadata,
                'source_unit_metadata': data.source_unit_metadata, 'time_metadata_verified': True})
            result['value'] = normalized(result.raw_value_token)
            for aux in ['R1', 'RS_IND', 'WRTR'] if self.variable in {'R1', 'WRTR'} else []:
                result[aux+'_raw_token'] = data[aux]
                result[aux+'_source_value'] = normalized(data[aux])
            result['priority'] = 0 if name.endswith('_hist.zip') else 1
            result['raw_missing_marker'] = result.raw_value_token.isin(['', '-999', '-999.0'])
            result['raw_non_numeric_or_nonfinite'] = result.value.isna() & ~result.raw_missing_marker
            self.archive_rows.append(dict(archive=name, member=member, raw_rows=raw_count,
                                          retained_rows=len(result), invalid_timestamp_rows=len(invalid)))
            return result.reset_index(drop=True)

    def _resolve_station(self, sid):
        raw = pd.concat([self._read_archive(n, sid) for n in sorted(self.by_station_archives[sid])], ignore_index=True)
        raw['value_key'] = raw.value.map(lambda v: '<missing>' if pd.isna(v) else f'{v:.12g}')
        counts = raw.groupby(self.time).agg(raw_rows=('value','size'),
                                            distinct_values=('value_key','nunique'),
                                            distinct_quality_codes=('quality_code_raw','nunique'))
        within = raw.groupby([self.time, 'priority']).agg(
            within_values=('value_key','nunique'), within_quality=('quality_code_raw','nunique'))
        chosen = raw.sort_values([self.time,'priority','source_archive','source_line']).drop_duplicates(self.time).copy()
        within_index = pd.MultiIndex.from_frame(chosen[[self.time,'priority']])
        chosen['duplicate_value_conflict'] = within.within_values.reindex(within_index).to_numpy() > 1
        chosen['duplicate_quality_conflict'] = within.within_quality.reindex(within_index).to_numpy() > 1
        chosen['candidate_value_before_conflict'] = chosen.value
        chosen.loc[chosen.duplicate_value_conflict, 'value'] = np.nan
        chosen['raw_rows_at_time'] = chosen[self.time].map(counts.raw_rows)
        chosen['historical_recent_value_difference'] = chosen[self.time].map(counts.distinct_values).gt(1)
        chosen['historical_recent_quality_difference'] = chosen[self.time].map(counts.distinct_quality_codes).gt(1)
        chosen['row_present'] = True
        repeated = counts.loc[counts.raw_rows.gt(1)].reset_index()
        repeated['dwd_station_id'] = sid
        self.overlaps.append(repeated)
        unresolved_times = set(chosen.loc[chosen.duplicate_value_conflict, self.time])
        unresolved = raw.loc[raw[self.time].isin(unresolved_times)].copy()
        self.conflicts.append(unresolved.drop(columns='value_key'))
        chosen = chosen.set_index(self.time).reindex(self.grid).rename_axis(self.time).reset_index()
        chosen['dwd_station_id'] = sid
        for col in ['row_present','duplicate_value_conflict','duplicate_quality_conflict',
                    'historical_recent_value_difference','historical_recent_quality_difference',
                    'raw_missing_marker','raw_non_numeric_or_nonfinite','time_metadata_verified']:
            chosen[col] = chosen[col].eq(True)
        chosen['raw_rows_at_time'] = chosen.raw_rows_at_time.fillna(0).astype(int)
        chosen = chosen.drop(columns=['value_key', 'priority'])
        chosen['is_buffer'] = chosen[self.time].lt(self.start)
        if self.variable == 'WRTR':
            chosen['usable_observation'] = chosen.value.isin(KNOWN_TYPES)
            chosen['type_label'] = chosen.value.map(TYPE_LABELS).astype('string')
            chosen['unknown_type'] = chosen.value.notna() & chosen.value.isin([1,4,9])
            chosen['invalid_type_code'] = chosen.value.notna() & ~chosen.value.isin(ALL_TYPES)
        else:
            chosen['usable_observation'] = chosen.value.notna()
        if self.daily:
            # This derived timestamp describes measurement reference, not publication.
            chosen['measurement_time_utc'] = chosen[self.time] + pd.Timedelta(hours=6)
        return chosen

    def _screen(self, frame, key, value='value'):
        result = frame.copy()
        groups = result.groupby(key, sort=False)
        values = result[value]
        prior = groups[value].shift()
        prior_time = groups[self.time].shift()
        adjacent = (result[self.time] - prior_time).eq(self.step)
        result['previous_value'] = prior.where(adjacent)
        result['change_from_previous'] = (values-prior).where(adjacent)
        flag = {'missing_or_unknown': ~result.usable_observation}
        flag['duplicate_value_conflict'] = result.duplicate_value_conflict.fillna(False)
        flag['duplicate_quality_conflict'] = result.duplicate_quality_conflict.fillna(False)
        flag['non_numeric_or_nonfinite'] = result.raw_non_numeric_or_nonfinite.fillna(False)
        qn = pd.to_numeric(result.quality_code_raw, errors='coerce')
        flag['unexpected_quality_code'] = result.row_present & values.notna() & (~qn.isin(QUALITY_LEVELS))
        if self.variable == 'WRTR':
            flag['invalid_type_code'] = values.notna() & ~values.isin(ALL_TYPES)
        else:
            flag['negative_value'] = values.lt(0)
            flag['high_value'] = values.gt(self.rules['high'])
            flag['large_change'] = result.change_from_previous.abs().gt(self.rules['change'])
            boundary = result[key].ne(result[key].shift()) | ~adjacent | values.ne(values.shift()) | values.isna()
            run_id = boundary.cumsum()
            run_length = values.groupby(run_id).transform('size')
            flag['constant_positive_run'] = values.gt(0) & run_length.ge(self.rules['constant'])
            result['constant_run_length'] = run_length.where(values.gt(0), 0).astype(int)
        for name, mask in flag.items():
            result['flag_'+name] = mask.fillna(False).astype(bool)
        cols = ['flag_'+c for c in flag]
        result['diagnostic_flagged'] = result[cols].any(axis=1)
        # Vectorized concatenation avoids a row-wise Python call for every grid position.
        issue = np.full(len(result), '', dtype=object)
        for col in cols:
            mask = result[col].to_numpy()
            issue[mask] = issue[mask] + col[5:] + ';'
        result['issues'] = pd.Series(issue, index=result.index).str.rstrip(';')
        return result

    def _summaries(self, frame, key, names=()):
        work = frame.loc[~frame.is_buffer].copy()
        work['month'] = work[self.time].dt.strftime('%Y-%m')
        records = []
        monthly = []
        for keys, sub in work.groupby([key, *names], sort=False):
            if not isinstance(keys, tuple): keys = (keys,)
            ident = dict(zip([key,*names], keys))
            records.append({**ident, **self._summary_row(sub)})
            for month, group in sub.groupby('month', sort=True):
                monthly.append({**ident, 'month': month, **self._summary_row(group)})
        return pd.DataFrame(records), pd.DataFrame(monthly)

    def _summary_row(self, frame):
        finite = frame.value.notna()
        usable = frame.usable_observation
        result = dict(variable=self.variable, expected=len(frame), finite=int(finite.sum()),
                      usable=int(usable.sum()), missing_or_unknown=int((~usable).sum()),
                      availability_pct=100*float(usable.mean()), longest_gap=longest_false(usable),
                      minimum=frame.value.min(), maximum=frame.value.max(),
                      flagged_records=int(frame.diagnostic_flagged.sum()),
                      positive_observations=int(frame.value.gt(0).sum()))
        if 'source_role' in frame:
            for role in ['primary','backup','missing']:
                result[role+'_positions'] = int(frame.source_role.eq(role).sum())
            result['source_switches'] = int(frame.source_switch.sum())
        if 'raw_rows_at_time' in frame:
            result['duplicate_time_positions'] = int(frame.raw_rows_at_time.gt(1).sum())
            result['extra_overlap_rows'] = int((frame.raw_rows_at_time-1).clip(lower=0).sum())
            result['historical_recent_value_differences'] = int(frame.historical_recent_value_difference.sum())
            result['historical_recent_quality_differences'] = int(frame.historical_recent_quality_difference.sum())
        for flag in [c for c in frame if c.startswith('flag_')]:
            result[flag] = int(frame[flag].sum())
        return result

    def _gap_runs(self, frame, key):
        records=[]
        for sid, sub in frame.loc[~frame.is_buffer].groupby(key, sort=False):
            mask = ~sub.usable_observation.to_numpy()
            differences = np.diff(np.r_[False, mask, False].astype(int))
            for begin, end in zip(np.flatnonzero(differences==1), np.flatnonzero(differences==-1)):
                records.append({key: sid, 'start': sub.iloc[begin][self.time],
                                'end_inclusive': sub.iloc[end-1][self.time], 'length': int(end-begin),
                                'frequency': 'daily' if self.daily else 'hourly'})
        return pd.DataFrame(records, columns=[key,'start','end_inclusive','length','frequency'])

    def read_sources(self):
        pieces=[]
        for sid in sorted(self.by_station_archives):
            pieces.append(self._resolve_station(sid))
        source = pd.concat(pieces, ignore_index=True)
        self.source = self._screen(source, 'dwd_station_id')
        self.source_station_summary, self.source_monthly_quality = self._summaries(self.source, 'dwd_station_id')
        self.source_gap_runs = self._gap_runs(self.source, 'dwd_station_id')
        self.overlap_table = pd.concat(self.overlaps, ignore_index=True)
        self.conflict_table = pd.concat(self.conflicts, ignore_index=True)
        study = self.source.loc[~self.source.is_buffer]
        self.source_quality_codes = study.groupby(['dwd_station_id',self.config['qn']] if self.config['qn'] in study else ['dwd_station_id','quality_code_raw'], dropna=False).size().rename('count').reset_index()
        monthly = study.assign(month=study[self.time].dt.strftime('%Y-%m'))
        self.source_monthly_quality_codes = monthly.groupby(
            ['dwd_station_id','month','archive_period','quality_code_raw'],dropna=False).size().rename('count').reset_index()
        self.source_monthly_quality_codes['variable'] = self.variable
        # Per-variable mapping counts are an independent check on ingestion/priority handling.
        by_sid = self.source_station_summary.set_index('dwd_station_id')
        for mapping in self.maps:
            for role in ['primary','backup']:
                metrics = mapping[role+'_metrics']
                n = by_sid.loc[mapping[role+'_station_id'], 'usable']
                expected = metrics['meaningful'] if self.variable == 'WRTR' else metrics['valid']
                assert int(n) == int(expected), (self.variable, mapping[role+'_station_id'], n, expected)
        return self._export_source_view()

    def _export_source_view(self):
        result = self.source.rename(columns={'value': self.column}).copy()
        result = self._stable_export_types(result)
        if self.daily:
            result[self.time] = result[self.time].dt.date
        return result

    def match_and_screen(self):
        frames=[]
        selected_cols = ['value','raw_value_token','quality_code_raw','source_archive','source_member','source_sha256',
                         'source_line','raw_timestamp_token','archive_period','row_present','duplicate_value_conflict',
                         'duplicate_quality_conflict','raw_non_numeric_or_nonfinite','raw_missing_marker',
                         'time_reference_metadata','source_unit_metadata','time_metadata_verified']
        for mapping in self.maps:
            primary = self.source.loc[self.source.dwd_station_id.eq(mapping['primary_station_id'])].reset_index(drop=True)
            backup = self.source.loc[self.source.dwd_station_id.eq(mapping['backup_station_id'])].reset_index(drop=True)
            assert primary[self.time].equals(backup[self.time])
            p_ok = primary.usable_observation.to_numpy()
            b_ok = backup.usable_observation.to_numpy()
            use_b = ~p_ok & b_ok
            available = p_ok | use_b
            # If neither supplies a usable type/value, preserve the primary raw token for diagnosis.
            selected = primary[selected_cols].copy()
            for column in selected_cols:
                selected[column] = primary[column].where(~use_b,backup[column])
            selected['value'] = selected.value.where(available)
            selected[self.time] = primary[self.time]
            selected['is_buffer'] = primary.is_buffer
            selected['railway_eva_id'] = mapping['railway_eva_id']
            selected['railway_station'] = mapping['railway_station']
            selected['usable_observation'] = available
            selected['source_role'] = np.select([p_ok,use_b], ['primary','backup'], default='missing')
            selected['dwd_station_id'] = pd.Series(np.where(use_b, mapping['backup_station_id'], mapping['primary_station_id']), dtype='string').where(available)
            selected['source_distance_km'] = np.where(use_b, mapping['backup_distance_km'], mapping['primary_distance_km'])
            selected.loc[~available, 'source_distance_km'] = np.nan
            for role, candidate in [('primary',primary),('backup',backup)]:
                selected[role+'_dwd_station_id'] = mapping[role+'_station_id']
                selected[role+'_station_name'] = mapping[role+'_station_name']
                selected[role+'_distance_km'] = mapping[role+'_distance_km']
                selected[role+'_value'] = candidate.value
                selected[role+'_available'] = candidate.usable_observation
                selected[role+'_raw_token'] = candidate.raw_value_token
                selected[role+'_quality_code_raw'] = candidate.quality_code_raw
                selected[role+'_archive'] = candidate.source_archive
                selected[role+'_member'] = candidate.source_member
                selected[role+'_line'] = candidate.source_line
                selected[role+'_sha256'] = candidate.source_sha256
                selected[role+'_time_reference'] = candidate.time_reference_metadata
            selected['fallback_reason'] = np.select(
                [p_ok, use_b, ~available],
                ['primary_available', 'primary_missing_or_unknown', 'both_missing_or_unknown'], default='')
            # Empty selected-source provenance when neither source is usable; candidate fields retain evidence.
            for column in ['source_archive','source_member','source_sha256','source_line','archive_period',
                           'quality_code_raw','raw_value_token','raw_timestamp_token',
                           'time_reference_metadata','source_unit_metadata']:
                selected.loc[~available, column] = None
            selected.loc[~available, 'time_metadata_verified'] = False
            if self.daily:
                assert set(primary.time_reference_metadata.dropna()) == {'06 UTC'}
                assert set(backup.time_reference_metadata.dropna()) == {'06 UTC'}
                selected['measurement_time_utc'] = selected[self.time]+pd.Timedelta(hours=6)
                selected['measurement_reference'] = '06 UTC'
            if self.variable == 'WRTR':
                selected['precipitation_type_label'] = selected.value.map(TYPE_LABELS).astype('string')
                selected['solid_or_mixed'] = pd.Series(pd.NA,index=selected.index,dtype='boolean')
                selected.loc[available,'solid_or_mixed'] = selected.loc[available,'value'].isin([7,8])
            frames.append(selected)
        railway = pd.concat(frames, ignore_index=True)
        railway = self._screen(railway,'railway_eva_id')
        both = railway.primary_available & railway.backup_available
        if self.variable == 'WRTR':
            railway['primary_backup_type_disagreement'] = both & railway.primary_value.ne(railway.backup_value)
            railway['flag_solid_liquid_disagreement'] = both & (
                (railway.primary_value.eq(6) & railway.backup_value.eq(7)) |
                (railway.primary_value.eq(7) & railway.backup_value.eq(6)))
            extra = 'flag_solid_liquid_disagreement'
        else:
            railway['primary_backup_difference'] = (railway.primary_value-railway.backup_value).where(both)
            railway['flag_large_primary_backup_difference'] = railway.primary_backup_difference.abs().gt(self.rules['difference'])
            extra = 'flag_large_primary_backup_difference'
        add = railway[extra]
        railway.loc[add,'issues'] = railway.loc[add,'issues'].map(lambda x: (x+';' if x else '')+extra[5:])
        railway['diagnostic_flagged'] |= add
        railway['review_required'] = railway.diagnostic_flagged
        railway['review_status'] = np.where(railway.review_required,'pending','not_flagged')
        prev_id = railway.groupby('railway_eva_id').dwd_station_id.shift()
        current_id = railway.dwd_station_id.fillna('<missing>')
        railway['source_switch'] = current_id.ne(prev_id.fillna('<missing>')) & railway.groupby('railway_eva_id').cumcount().gt(0)
        railway['manual_source_override'] = False
        self.railway = railway
        self.railway_station_summary,self.railway_monthly_quality = self._summaries(railway,'railway_eva_id',('railway_station',))
        self.railway_gap_runs = self._gap_runs(railway,'railway_eva_id')
        study = railway.loc[~railway.is_buffer]
        self.flag_summary = pd.DataFrame([
            dict(check=c[5:], source_records=int(self.source.loc[~self.source.is_buffer,c].sum()) if c in self.source else None,
                 railway_records=int(study[c].sum())) for c in railway if c.startswith('flag_')])
        self.railway_quality_codes = study.groupby(['railway_eva_id','railway_station','quality_code_raw'], dropna=False).size().rename('count').reset_index()
        return self._export_railway_view()

    def _export_railway_view(self):
        result = self.railway.rename(columns={'value':self.column}).copy()
        result = self._stable_export_types(result)
        if self.daily: result[self.time] = result[self.time].dt.date
        return result

    def _public_dates(self, frame):
        frame = frame.copy()
        if self.daily:
            for column in [self.time,'start','end_inclusive']:
                if column in frame:
                    frame[column] = pd.to_datetime(frame[column],utc=True).dt.date
        return frame

    @staticmethod
    def _stable_export_types(frame):
        # Explicit string columns preserve nulls and leading zeros across Parquet round trips.
        frame = frame.infer_objects(copy=False)
        for column in frame.select_dtypes(include='object'):
            nonnull = frame[column].dropna()
            if nonnull.empty or nonnull.map(lambda x: isinstance(x,str)).all():
                frame[column] = frame[column].astype('string')
        return frame

    def locate_review_candidates(self):
        candidates = self.railway.loc[~self.railway.is_buffer & self.railway.review_required].copy()
        candidates['record_id'] = [f'{self.config["prefix"]}-R{i:05d}' for i in range(1,len(candidates)+1)]
        event_ids = []
        events=[]
        contexts=[]
        count=0
        for eva, sub in candidates.groupby('railway_eva_id', sort=False):
            runs = sub[self.time].diff().ne(self.step).cumsum()
            source = self.railway.loc[self.railway.railway_eva_id.eq(eva)].copy()
            for _, group in sub.groupby(runs, sort=False):
                count+=1
                event_id=f'{self.config["prefix"]}-G{count:04d}'
                event_ids.extend([(i,event_id) for i in group.index])
                begin, end = group[self.time].min(),group[self.time].max()
                diff=group.primary_backup_difference.abs().max() if 'primary_backup_difference' in group else np.nan
                events.append(dict(event_id=event_id, railway_station=group.railway_station.iloc[0], railway_eva_id=eva,
                                   start=begin,end_inclusive=end,flagged_records=len(group),
                                   minimum=group.value.min(),maximum=group.value.max(),
                                   max_absolute_change=group.change_from_previous.abs().max(),
                                   max_primary_backup_difference=diff,
                                   primary_dwd_station_id=group.primary_dwd_station_id.iloc[0],
                                   backup_dwd_station_id=group.backup_dwd_station_id.iloc[0],
                                   issues=';'.join(sorted({s for v in group.issues for s in v.split(';')}))))
                window = pd.Timedelta(days=3) if self.daily else pd.Timedelta(hours=6)
                context=source.loc[source[self.time].between(begin-window,end+window)].copy()
                context['event_id']=event_id
                context['is_event_time']=context[self.time].isin(group[self.time])
                contexts.append(context)
        candidates['event_id']=pd.Series(dict(event_ids),dtype='string').reindex(candidates.index)
        self.candidates=candidates
        self.events=pd.DataFrame(events,columns=['event_id','railway_station','railway_eva_id','start','end_inclusive',
                                                'flagged_records','minimum','maximum','max_absolute_change',
                                                'max_primary_backup_difference','primary_dwd_station_id',
                                                'backup_dwd_station_id','issues'])
        self.context = pd.concat(contexts,ignore_index=True) if contexts else pd.DataFrame(columns=[*self.railway.columns,'event_id','is_event_time'])
        # Record IDs are also stored on the exported initial railway table.
        self.railway['record_id'] = pd.Series('',index=self.railway.index,dtype='string')
        self.railway['event_id'] = pd.Series('',index=self.railway.index,dtype='string')
        self.railway.loc[candidates.index,'record_id']=candidates.record_id
        self.railway.loc[candidates.index,'event_id']=candidates.event_id
        return self._public_dates(self.events)

    def show_context_examples(self, limit=8):
        import matplotlib.pyplot as plt
        self.chart_files=[]
        for event in self.events.head(limit).itertuples(index=False):
            sub=self.context.loc[self.context.event_id.eq(event.event_id)]
            fig, ax=plt.subplots(figsize=(10,3.2))
            for column,label in [('primary_value','Primary'),('backup_value','Backup'),('value','Selected')]:
                ax.plot(sub[self.time],sub[column],marker='o',markersize=3,label=label,
                        linestyle='--' if column=='value' else '-')
            ax.axvspan(event.start,event.end_inclusive+self.step/8,color='#ffdf8a',alpha=.5)
            ax.set_title(f'{event.event_id} · {event.railway_station} · {event.issues}',fontsize=9)
            ax.set_ylabel(self.config['unit'])
            ax.set_xlabel('Observation date (06 UTC reference)' if self.daily else 'UTC')
            if self.variable=='WRTR':
                ax.set_yticks([0,1,4,6,7,8,9])
            ax.legend(loc='best',fontsize=8)
            ax.grid(alpha=.2)
            fig.autofmt_xdate()
            fig.tight_layout()
            path=self.qc/(event.event_id+'.png')
            fig.savefig(path,dpi=120,bbox_inches='tight')
            self.chart_files.append(path.name)
            plt.show()
            plt.close(fig)

    def validate_and_export(self):
        source=self.source
        railway=self.railway
        positions=int((self.end-self.read_start)/self.step)
        assert len(source)==positions*len(self.by_station_archives)
        assert len(railway)==positions*len(self.maps)
        assert not source.duplicated(['dwd_station_id',self.time]).any()
        assert not railway.duplicated(['railway_eva_id',self.time]).any()
        assert source.dwd_station_id.str.fullmatch(r'\d{5}').all()
        assert railway.railway_eva_id.str.fullmatch(r'\d{7}').all()
        assert railway.source_distance_km.dropna().le(50).all()
        assert str(source[self.time].dtype)=='datetime64[ns, UTC]'
        assert not source.value.eq(-999).any() and not railway.value.eq(-999).any()
        assert source.loc[source.row_present,'time_metadata_verified'].all()
        assert railway.loc[railway.usable_observation,'time_metadata_verified'].all()
        assert not railway.manual_source_override.any()
        # Check every selected record and provenance against the exact same-time DWD source.
        usable=railway.loc[railway.usable_observation]
        source_index=source.set_index(['dwd_station_id',self.time])
        selected_index=pd.MultiIndex.from_frame(usable[['dwd_station_id',self.time]])
        aligned=source_index.reindex(selected_index)
        np.testing.assert_array_equal(usable.value.to_numpy(),aligned.value.to_numpy())
        for column in ['source_archive','source_member','source_sha256','quality_code_raw','source_line']:
            np.testing.assert_array_equal(usable[column].to_numpy(),aligned[column].to_numpy())
        assert (railway.primary_available.eq(railway.source_role.eq('primary'))).all()
        assert (railway.source_role.eq('backup').eq(~railway.primary_available & railway.backup_available)).all()
        comparisons=[]
        study=railway.loc[~railway.is_buffer]
        for mapping in self.maps:
            sub=study.loc[study.railway_eva_id.eq(mapping['railway_eva_id'])]
            backup=int(sub.source_role.eq('backup').sum())
            missing=int((~sub.usable_observation).sum())
            assert backup==mapping['recovered_by_backup'], (self.variable,mapping['railway_station'],'backup count')
            assert missing==mapping['remaining_unusable'], (self.variable,mapping['railway_station'],'remaining gaps')
            comparisons.append(dict(railway_station=mapping['railway_station'], backup=backup, remaining=missing,
                                    mapping_counts_match=True))
        source_view=self._export_source_view()
        railway_view=self._export_railway_view()
        self.source_path=self.out/(self.slug+'.parquet')
        self.railway_path=self.out/('railway_'+self.slug+'_initial.parquet')
        for path, frame in [(self.source_path,source_view),(self.railway_path,railway_view)]:
            frame.to_parquet(path,index=False,compression='zstd',engine='pyarrow')
            pd.testing.assert_frame_equal(frame,pd.read_parquet(path))
        csvs={
            'source_station_summary.csv':self.source_station_summary,
            'source_monthly_quality.csv':self.source_monthly_quality,
            'source_gap_runs.csv':self.source_gap_runs,
            'source_quality_codes.csv':self.source_quality_codes,
            'source_monthly_quality_codes.csv':self.source_monthly_quality_codes,
            'archive_inventory.csv':self.inventory.merge(pd.DataFrame(self.archive_rows),on='archive'),
            'archive_overlap_diagnostics.csv':self.overlap_table,
            'unresolved_duplicate_records.csv':self.conflict_table,
            'invalid_timestamp_records.csv':pd.DataFrame(self.invalid_timestamps,columns=['STATIONS_ID','MESS_DATUM','source_line','source_archive']),
            'railway_station_summary.csv':self.railway_station_summary,
            'railway_monthly_quality.csv':self.railway_monthly_quality,
            'railway_gap_runs.csv':self.railway_gap_runs,
            'railway_quality_codes.csv':self.railway_quality_codes,
            'flag_summary.csv':self.flag_summary,
            'review_records.csv':self.candidates.rename(columns={'value':self.column}),
            'review_events.csv':self.events,
            'mapping_count_checks.csv':pd.DataFrame(comparisons),
        }
        source_candidates=source.loc[~source.is_buffer & source.diagnostic_flagged].copy()
        source_candidates=source_candidates.rename(columns={'value':self.column})
        csvs['source_flagged_records.csv']=source_candidates
        if self.variable=='WRTR':
            compact_columns=[self.time,'railway_eva_id','railway_station','value','source_role',
                             'primary_dwd_station_id','backup_dwd_station_id','primary_value','backup_value',
                             'primary_distance_km','backup_distance_km','primary_archive','primary_member',
                             'primary_line','backup_archive','backup_member','backup_line',
                             'flag_solid_liquid_disagreement']
            csvs['type_disagreement_diagnostics.csv']=study.loc[
                study.primary_backup_type_disagreement,compact_columns].rename(columns={'value':self.column})
        for name,frame in csvs.items():
            self._public_dates(frame).to_csv(self.qc/name,index=False)
        self._public_dates(self.context.rename(columns={'value':self.column})).to_parquet(self.qc/'review_context.parquet',index=False,compression='zstd')
        (self.qc/'metadata_records.json').write_text(json.dumps(self.metadata_records,ensure_ascii=False,indent=2),encoding='utf-8')
        self.validated=True
        self.validation=pd.DataFrame([
            dict(check='Input archive hashes, station IDs, units and effective time metadata',status='passed'),
            dict(check='Complete source and railway grids, including buffer',status='passed'),
            dict(check='Unique station/time keys and string IDs',status='passed'),
            dict(check='Primary-first / same-time backup selection and exact source provenance',status='passed'),
            dict(check='Saved mapping backup and gap counts for all railway stations',status='passed'),
            dict(check='Parquet round-trip equality',status='passed'),
            dict(check='Anomaly-driven manual changes',status='none'),
        ])
        self.validation.to_csv(self.qc/'validation_checks.csv',index=False)
        manifest={
            'generated_utc':datetime.now(timezone.utc).isoformat(),'variable':self.variable,'column':self.column,
            'frequency':'daily' if self.daily else 'hourly','unit':self.config['unit'],
            'study_start':str(self.start),'study_end_exclusive':str(self.end),'buffer_hours':48,
            'screening_rules':self.rules,'known_type_codes':sorted(KNOWN_TYPES) if self.variable=='WRTR' else None,
            'selection':'primary first; mapped backup at same time/date only when primary unavailable or unknown',
            'historical_priority':'historical row wins, including missing values; same-priority conflicts unavailable',
            'manual_source_changes':0,'pending_study_records':len(self.candidates),
            'source_archives':self.inventory.to_dict('records'),
            'station_mapping_sha256':digest(self.root/'station_mapping.json'),
            'source_manifest_sha256':digest(self.root/'source_manifest.json'),
            'processing_code_sha256':digest(self.root/'weather_processing.py'),
            'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,
            'outputs':{str(p.relative_to(self.root)):dict(sha256=digest(p),rows=len(f))
                       for p,f in [(self.source_path,source_view),(self.railway_path,railway_view)]},
            'limitations':['No publication-time availability check or journey-feature matching.',
                           'Instrument and geography metadata inventoried; independent instrument/site review incomplete.',
                           'Screening candidates retained pending project decisions; flags are retrospective diagnostics.']}
        (self.qc/'workflow_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        return self.validation

    def write_report(self):
        assert self.validated, 'Validate and export before reporting.'
        s=self.source.loc[~self.source.is_buffer]
        r=self.railway.loc[~self.railway.is_buffer]
        flag_table=self.flag_summary.loc[self.flag_summary.railway_records.gt(0)]
        unit='days' if self.daily else 'hours'
        stations=self.railway_station_summary[['railway_station','expected','primary_positions','backup_positions','missing_or_unknown','flagged_records','minimum','maximum']]
        report=f'''# {self.config['title']} quality report

Input snapshot: 16 October 2026. Variable: `{self.variable}`; unit: {self.config['unit']}; frequency: {'daily' if self.daily else 'hourly'}.

## Status and scope

The raw observations have been standardized, primary/backup selection has been applied and the checks below have run. The initial dataset is saved; {len(self.candidates):,} flagged study records in {len(self.events):,} events await project review. No anomaly-driven source substitutions or deletions have been applied.

Study period: {self.start.date()} to {self.end.date()} exclusive. Reported counts exclude the preceding 48-hour buffer. Source observations cover {len(self.by_station_archives)} DWD stations; the railway selection covers {len(self.maps)} railway stations. Shared DWD values at different railway stations are not independent measurements.

## Results

| Measure | Count |
| --- | ---: |
| Original ZIPs verified against recorded SHA-256 | {len(self.archive_names):,} |
| Source study station-{unit} | {len(s):,} |
| Source usable positions | {int(s.usable_observation.sum()):,} |
| Source missing/unknown positions | {int((~s.usable_observation).sum()):,} |
| Railway study station-{unit} | {len(r):,} |
| Railway positions using primary | {int(r.source_role.eq('primary').sum()):,} |
| Railway positions using same-time backup | {int(r.source_role.eq('backup').sum()):,} |
| Railway remaining missing/unknown positions | {int((~r.usable_observation).sum()):,} |
| Source unresolved same-priority value conflicts | {int(s.duplicate_value_conflict.sum()):,} |
| Source historical/recent value differences | {int(s.historical_recent_value_difference.sum()):,} |
| Invalid timestamp rows in original archives | {len(self.invalid_timestamps):,} |
| Railway records flagged for review | {len(self.candidates):,} |

The source grid contains all mapped primary and backup stations. Missing source values can therefore remain even when the selected railway table has complete coverage.

## Rules and screening

`-999`, blanks and non-numeric/non-finite tokens cannot supply a value; raw tokens remain available. Historical rows take priority over recent overlaps, including missing historical rows. Conflicting values at the chosen archive priority are unavailable pending review. Select the primary first; select the same-time backup only if the primary cannot supply a usable observation. No interpolation or zero filling is used.

{md_table(self.rule_table())}

Finite numeric candidates remain in the initial table. Screening thresholds are project review rules, not DWD standards or evidence that an observation is wrong. Differences between nearby stations may reflect local weather. Missing/unknown values are counted separately from numeric anomalies. Quality codes are preserved; a code change alone does not invalidate a value.

{md_table(flag_table)}

## Railway-station results

{md_table(stations)}

## Review locations and supporting checks

- [Review event index and example charts](review_candidates.md); [all flagged records](review_records.csv); [all events](review_events.csv); [context values](review_context.parquet).
- [DWD station summary](source_station_summary.csv), [station × variable × month checks](source_monthly_quality.csv), [continuous source gaps](source_gap_runs.csv), [monthly quality-code counts](source_monthly_quality_codes.csv).
- [Railway monthly checks](railway_monthly_quality.csv), [remaining railway gaps](railway_gap_runs.csv), [screening counts](flag_summary.csv).
- [Archive overlap diagnostics](archive_overlap_diagnostics.csv), [unresolved duplicate records](unresolved_duplicate_records.csv), [metadata inventory and records](metadata_records.json).
- [Original archive URLs and hashes](archive_inventory.csv), [workflow rules and versions](workflow_manifest.json), [validation results](validation_checks.csv).

## Validation and remaining work

Archive hashes, effective metadata, complete grids, unique keys, station IDs, mapped distances and every selected value/provenance record passed automated checks. Backup and remaining-gap counts agree with the saved station-mapping audit for all 19 railway stations. Both Parquet files passed read-back equality checks.

Source flags and railway flags are retrospective audit fields, not model features. Review the listed candidates before treating the initial selection as a reviewed final dataset. Instrument/geography metadata have been inventoried; detailed instrument/site interpretation, observation publication-time availability and journey prediction-time matching remain separate tasks.

## Outputs and sources

- [DWD observations](../../observations/{self.slug}.parquet): DWD station ID plus {'observation date' if self.daily else 'UTC hour'}.
- [Initial railway observations](../../observations/railway_{self.slug}_initial.parquet): railway EVA ID plus {'observation date' if self.daily else 'UTC hour'}; includes both candidate station values and selected-source provenance.
- Original provider: Deutscher Wetterdienst (DWD), Climate Data Center; [official dataset description]({self.config['doc']}). Exact ZIP URLs, download times and hashes are listed in the archive inventory. Parameter/time-reference metadata are read from the original ZIPs.
'''
        if self.variable=='WRTR':
            disagreements=int(r.primary_backup_type_disagreement.sum())
            report+='''\n## Precipitation-type interpretation\n\nOnly codes 0/6/7/8 identify a usable type for this project. Codes 1/4/9 remain traceable as unspecified/unknown and may trigger backup selection. Remaining unknown values stay null, including the `solid_or_mixed` indicator. Codes 7/8 indicate solid or mixed precipitation, which may include forms other than snow.\n\nThe archived reference text explicitly lists particular UTC hours with a one-hour reference interval. That text is preserved; this workflow does not assume that all WRTR observations have an identical one-hour window. Resolve that interpretation before using a rolling snowfall indicator.\n\n'''
            report+=f'Primary/backup type differences: {disagreements:,} study station-hours, saved in [type disagreement diagnostics](type_disagreement_diagnostics.csv). Only direct liquid-versus-solid differences are included as additional review candidates; other category differences remain visible as diagnostics.\n'
        if self.daily:
            report+='''\n## Daily snow-depth timing\n\nAll mapped source metadata used here specify `06 UTC`. `observation_date` remains a date; `measurement_time_utc` records that date at 06:00 UTC. Neither field establishes when DWD published the observation. Backup selection uses the same date and compatible reference metadata. The table is not expanded to hourly values; prediction-time alignment must be decided during integration. Long zero runs are valid snow-free periods.\n'''
        (self.qc/(self.slug+'_report.md')).write_text(report,encoding='utf-8')
        event_index=f'''# {self.config['title']}: records for review

{len(self.candidates):,} flagged study records; {len(self.events):,} consecutive-time events. Event IDs group adjacent flagged times at the same railway station. A flag is not proof of an error. No source treatment has been approved for these candidates.

Use [all records](review_records.csv) for exact values, change magnitudes, primary/backup DWD IDs, UTC time/date, original ZIP/member/line and current source. Use [context values](review_context.parquet) for ±{'3 dates' if self.daily else '6 hours'} around every event. All station IDs must be read as strings.

## Event index (first 100; the CSV contains every event)

{md_table(self._public_dates(self.events.head(100)))}

[Complete event index](review_events.csv) · [Quality report]({self.slug}_report.md)

## Example context charts

'''
        event_index+='\n\n'.join(f'![{Path(n).stem}]({n})' for n in getattr(self,'chart_files',[])) or 'No context charts required.'
        (self.qc/'review_candidates.md').write_text(event_index,encoding='utf-8')
        return report
