from __future__ import annotations

import csv, json, re
from pathlib import Path
from typing import Any
import yaml

from orchestrator.cross_domain import apply_interaction_policy
from orchestrator.decision_baselines import choose_dominant_domain_action
from orchestrator.utility import choose_action_details
from runtime_validation.evidence_adapter import write_evidence
from runtime_validation.evaluate_heldout_runtime import normalize_runtime_evidence, latest_artifact

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=Path(__file__).with_name('interventions_batch3.yaml')


def manifest():
    d=yaml.safe_load(MANIFEST.read_text(encoding='utf-8')); return {c['id']:c for c in d['cases']}

def classify_batch3_config(case_dir:Path,t:dict[str,Any])->dict[str,Any]:
    text=(case_dir/'compose_resolved.yaml').read_text(encoding='utf-8',errors='replace') if (case_dir/'compose_resolved.yaml').exists() else ''
    m=re.search(r'AAF_BATCH3_TEST_POLICY_MARKER:\s*([^\s]+)',text)
    if not m: return t
    marker=m.group(1).strip().strip('"\'')
    deploy=t.setdefault('deploy',{}); dm=deploy.setdefault('_evidence',{})
    deploy['config_drift']=True; dm['config_drift']={'status':'measured','source':'resolved Docker Compose configuration','note':'Test-only Batch 3 configuration marker differs from frozen healthy baseline.'}
    if marker=='exposed_test_secret':
        sec=t.setdefault('sec',{}); sm=sec.setdefault('_evidence',{})
        sec['policy_violation']=True; sm['policy_violation']={'status':'measured','source':'resolved Docker Compose configuration policy check','note':'Test-only secret-like marker violates the prespecified no-secret-in-runtime-config policy.'}
    t.setdefault('_runtime_observables',{})['batch3_config_marker']=marker
    return t

def evaluate(case_id:str,baseline:Path,m:dict[str,dict[str,Any]]):
    case_dir=latest_artifact(case_id,1)
    if case_dir is None: raise FileNotFoundError(case_id)
    t=write_evidence(case_dir,baseline_dir=baseline); t=normalize_runtime_evidence(case_dir,t); t=classify_batch3_config(case_dir,t)
    (case_dir/'batch3_telemetry_evaluated.json').write_text(json.dumps(t,indent=2),encoding='utf-8')
    oracle=set(m[case_id]['admissible_actions'])
    ba,bs,bd=choose_dominant_domain_action(t); ni=choose_action_details(t,(0.4,0.3,0.3))['selected_action']; full=apply_interaction_policy(t,ni); dom=full['interaction_state'].get('dominant_interaction')
    return {'case_id':case_id,'oracle_actions':json.dumps(sorted(oracle)),'baseline_action':ba,'baseline_match':ba in oracle,'baseline_domain':bd,'baseline_severity':round(float(bs),4),'aaf_no_interaction_action':ni,'aaf_no_interaction_match':ni in oracle,'aaf_full_action':full['selected_action'],'aaf_full_match':full['selected_action'] in oracle,'interaction_applied':bool(full['interaction_applied']),'dominant_interaction':None if dom is None else dom['name'],'p95_latency_ms':t.get('sre',{}).get('p95_latency_ms'),'error_rate_pct':t.get('sre',{}).get('error_rate_pct'),'resource_footprint_proxy_pct':t.get('finops',{}).get('cost_spike_pct'),'config_marker':t.get('_runtime_observables',{}).get('batch3_config_marker'),'restart_burst_count':t.get('deploy',{}).get('restart_burst_count')}

def main():
    m=manifest(); baseline=latest_artifact('HRT-01',1)
    if baseline is None: raise FileNotFoundError('HRT-01 baseline required')
    ids=['HRT-08','HRT-09','HRT-10','HRT-11','HRT-12','HRT-13']; rows=[evaluate(i,baseline,m) for i in ids]
    out=ROOT/'results_heldout_runtime_batch3'; out.mkdir(parents=True,exist_ok=True)
    with (out/'runtime_case_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary={'n_independent_scenario_templates':len(rows),'runtime_system':'Sock Shop','evaluation_reference':'frozen Batch 3 held-out intervention oracle','dominant_domain_baseline_action_oracle_agreement':sum(r['baseline_match'] for r in rows)/len(rows),'aaf_no_interaction_action_oracle_agreement':sum(r['aaf_no_interaction_match'] for r in rows)/len(rows),'aaf_full_action_oracle_agreement':sum(r['aaf_full_match'] for r in rows)/len(rows),'aaf_only_wins':sum(r['aaf_full_match'] and not r['baseline_match'] for r in rows),'baseline_only_wins':sum(r['baseline_match'] and not r['aaf_full_match'] for r in rows),'interpretation':'Independent held-out scenario-template analysis; one execution per template; preserved direct/proxy runtime evidence; not production accuracy.'}
    (out/'runtime_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
