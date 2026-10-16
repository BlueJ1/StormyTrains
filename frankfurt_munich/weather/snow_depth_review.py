"""Apply the recorded project decision to retain snow_depth primary values."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd


def apply_snow_depth_review(workflow):
    assert workflow.variable == 'SHK_TAG'
    path = workflow.qc / 'review_decisions.json'
    registry = json.loads(path.read_text())
    initial = workflow.railway
    reviewed = initial.copy()
    reviewed['manual_review_decision'] = pd.Series('', index=reviewed.index, dtype='string')
    reviewed['manual_review_reason'] = pd.Series('', index=reviewed.index, dtype='string')
    indexed = reviewed.set_index(['railway_eva_id', workflow.time], drop=False)
    applied=[]
    for decision in registry['records']:
        key=(decision['railway_eva_id'],pd.Timestamp(decision['timestamp_utc']))
        assert key in indexed.index
        row=indexed.loc[key]
        assert decision['decision']=='keep_primary'
        assert row.record_id==decision['record_id'] and row.source_role=='primary'
        assert row.dwd_station_id==decision['primary_dwd_station_id']
        assert row.value==decision['expected_primary_value']
        assert row.value==row.primary_value and row.diagnostic_flagged
        mask=reviewed.railway_eva_id.eq(key[0]) & reviewed[workflow.time].eq(key[1])
        assert int(mask.sum())==1
        reviewed.loc[mask,'review_required']=False
        reviewed.loc[mask,'review_status']='reviewed_keep_primary'
        reviewed.loc[mask,'manual_review_decision']='keep_primary'
        reviewed.loc[mask,'manual_review_reason']=registry['policy']
        applied.append({**decision,'review_status':'reviewed_keep_primary','value_changed':False})
    expected=set(workflow.candidates.record_id)
    assert {r['record_id'] for r in applied}==expected
    protected=[c for c in initial if c not in ['review_required','review_status']]
    pd.testing.assert_frame_equal(initial[protected],reviewed[protected])
    assert not reviewed.loc[~reviewed.is_buffer,'review_required'].any()
    assert not reviewed.manual_source_override.any()
    workflow.snow_depth_reviewed=reviewed
    workflow.snow_depth_applied=pd.DataFrame(applied)
    workflow.snow_depth_registry=registry
    return reviewed


def export_snow_depth_review(workflow):
    reviewed=workflow.snow_depth_reviewed
    original=workflow._export_railway_view()
    frame=workflow._public_dates(workflow._stable_export_types(reviewed.rename(columns={'value':workflow.column})))
    protected=[c for c in original if c not in ['review_required','review_status']]
    pd.testing.assert_frame_equal(original[protected],frame[protected])
    path=workflow.out/'railway_snow_depth_final.parquet'
    frame.to_parquet(path,index=False,compression='zstd')
    pd.testing.assert_frame_equal(frame,pd.read_parquet(path))
    workflow.snow_depth_applied.to_csv(workflow.qc/'review_decisions.csv',index=False)
    study=frame.loc[~frame.is_buffer]
    study.loc[study.diagnostic_flagged].to_csv(workflow.qc/'retained_diagnostics.csv',index=False)
    study.loc[study.review_required].to_csv(workflow.qc/'new_review_candidates.csv',index=False)
    summary=study.groupby(['railway_eva_id','railway_station'],sort=False,as_index=False).agg(
        expected=('snow_depth_cm','size'),available=('snow_depth_cm','count'),
        primary_positions=('source_role',lambda x:int(x.eq('primary').sum())),
        backup_positions=('source_role',lambda x:int(x.eq('backup').sum())),
        retained_diagnostics=('diagnostic_flagged','sum'),pending_review=('review_required','sum'))
    summary.to_csv(workflow.qc/'reviewed_station_summary.csv',index=False)
    manifest_path=workflow.qc/'workflow_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['initial_flagged_study_records']=len(workflow.candidates)
    manifest['pending_study_records']=int(study.review_required.sum())
    manifest['reviewed_keep_primary_records']=len(workflow.snow_depth_applied)
    manifest['manual_source_changes']=0
    manifest['review_policy']=workflow.snow_depth_registry['policy']
    manifest['review_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['review_decisions_sha256']=hashlib.sha256((workflow.qc/'review_decisions.json').read_bytes()).hexdigest()
    manifest['outputs'][str(path.relative_to(workflow.root))]={
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(frame)}
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(f'Reviewed snow_depth exported: {len(workflow.snow_depth_applied)} records retain primary; '
          f'{int(study.review_required.sum())} pending; snow_depth values and sources unchanged.')
    return path


def finalize_snow_depth_report(workflow,report):
    count=len(workflow.snow_depth_applied)
    old=f'The raw observations have been standardized, primary/backup selection has been applied and the checks below have run. The initial dataset is saved; {count:,} flagged study records in {len(workflow.events):,} events await project review. No anomaly-driven source substitutions or deletions have been applied.'
    new=f'The raw observations have been standardized, primary/backup selection has been applied and the checks below have run. All {count} initial flagged study records have a recorded project decision to retain the primary observation. No study records await review under the current screening rules. Numeric snow_depth values and selected sources are unchanged; diagnostic flags are retained.'
    assert old in report
    report=report.replace(old,new)
    report=report.replace('Review the listed candidates before treating the initial selection as a reviewed final dataset.',
                          'The recorded project review is applied to the separate reviewed railway table; the initial table preserves the original pending-review state. This review records source choices, not independent proof that every measurement is accurate.')
    section=f"\n## Recorded snow_depth review\n\nAll {count} initial flagged records retain their mapped primary station under the recorded project decision. No snow-depth values, sources or quality codes changed. Diagnostic flags remain visible; outstanding study records: zero. Current snow-depth mapping is retained after reviewing station distances and nearby observations. The Plochingen observation remains 25 cm from DWD 02485; nearby Notzingen reports 21 cm. No nearest-available re-mapping has been applied.\n\nCurrent input: [reviewed snow_depth table](../../observations/railway_snow_depth_final.parquet). [Applied decisions](review_decisions.csv) and [decision registry](review_decisions.json) retain the review trail.\n"
    report+=section
    (workflow.qc/'snow_depth_report.md').write_text(report)
    index=workflow.qc/'review_candidates.md'
    text=index.read_text().replace('No source treatment has been approved for these candidates.',
            f'All {count} records have now been reviewed: retain primary. See [applied decisions](review_decisions.csv). Numeric values and sources remain unchanged; the tables below preserve the initial screening evidence.')
    index.write_text(text)
    return report
