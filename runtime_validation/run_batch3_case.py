from __future__ import annotations

"""Execute frozen Batch 3 Sock Shop cases with directly preserved evidence.

The runner never reads oracle actions when choosing behavior. Config/security
cases inject an explicit test-only environment marker through a Docker Compose
override; the resolved Compose file collected by the normal artifact collector
is later classified independently by the Batch 3 evaluator.
"""

import argparse, json, math, subprocess, threading, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("interventions_batch3.yaml")
CMD_TIMEOUT = 90
HTTP_TIMEOUT = 2.0


def utc_now(): return datetime.now(timezone.utc).isoformat()

def run(cmd, *, check=True, capture=False, timeout=CMD_TIMEOUT):
    print('+', ' '.join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, text=True, check=check, capture_output=capture, timeout=timeout)

def load_case(case_id):
    data = yaml.safe_load(MANIFEST.read_text(encoding='utf-8'))
    for c in data['cases']:
        if c['id'] == case_id: return c
    raise SystemExit(f'Unknown Batch 3 case: {case_id}')

def compose_args(base, override=None):
    args = ['docker','compose','-f',base]
    if override: args += ['-f',str(override)]
    return args

def make_override(case, repetition):
    if case['intervention'] not in {'config_security','config_security_load','config_scale_cpu_load','restart_config_security'}:
        return None
    marker = str(case.get('parameters',{}).get('security_marker') or case.get('parameters',{}).get('config_marker') or 'batch3_marker')
    out = ROOT/'runtime_validation'/'batch3_overrides'/case['id']/f'rep-{repetition}.yaml'
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {'services': {'front-end': {'environment': {'AAF_BATCH3_TEST_POLICY_MARKER': marker}}}}
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    return out

def resolve_url(compose):
    cp = run(compose + ['port','edge-router','80'], capture=True)
    port = cp.stdout.strip().splitlines()[0].rsplit(':',1)[-1]
    return f'http://127.0.0.1:{port}/catalogue'

def preflight(url):
    for _ in range(15):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if 200 <= int(r.status) < 400: return int(r.status)
        except Exception: pass
        time.sleep(2)
    raise RuntimeError('HTTP preflight failed')

def worker(url, stop, deadline, state, lock):
    while not stop.is_set() and time.monotonic() < deadline:
        t0=time.perf_counter(); status=None
        try:
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as r:
                r.read(1024); status=int(r.status)
        except urllib.error.HTTPError as e: status=int(e.code)
        except Exception: status=None
        ms=(time.perf_counter()-t0)*1000.0; ok=status is not None and 200<=status<400
        with lock:
            state['requests']+=1; state['successes' if ok else 'failures']+=1; state['latencies'].append(ms)

def percentile(xs,q):
    if not xs: return None
    ys=sorted(xs); return ys[max(0,min(len(ys)-1, math.ceil(q*len(ys))-1))]

def ids(compose, service):
    cp=run(compose+['ps','-q',service],capture=True); return [x.strip() for x in cp.stdout.splitlines() if x.strip()]

def started_at(compose, service):
    xs=ids(compose,service)
    if not xs:return None
    cp=run(['docker','inspect','-f','{{.State.StartedAt}}',xs[0]],capture=True); return cp.stdout.strip() or None

def apply_cpu(compose, service, cpus):
    xs=ids(compose,service)
    for cid in xs: run(['docker','update','--cpus',str(cpus),cid])
    return xs

def latest_artifact(case_id, rep):
    b=ROOT/'runtime_validation'/'artifacts'/case_id/f'rep-{rep}'
    ds=sorted([p for p in b.glob('*') if p.is_dir()]) if b.exists() else []
    return ds[-1] if ds else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('case_id'); ap.add_argument('--repetition',type=int,default=1); ap.add_argument('--compose-file',required=True); args=ap.parse_args()
    case=load_case(args.case_id); p=case.get('parameters',{}) or {}; override=make_override(case,args.repetition)
    compose=compose_args(args.compose_file,override); base_compose=compose_args(args.compose_file)
    cpu_ids=[]; stop=threading.Event(); threads=[]; temporal=None; load_obs=None
    try:
        run(base_compose+['up','-d','--remove-orphans','--scale','front-end=1']); time.sleep(8)
        if override:
            run(compose+['up','-d','--remove-orphans','--scale','front-end=1']); time.sleep(8)
        kind=case['intervention']
        if kind=='scale_service':
            run(compose+['up','-d','--scale',f"front-end={int(p['replicas'])}"]); time.sleep(8)
        if kind=='config_scale_cpu_load':
            run(compose+['up','-d','--scale',f"front-end={int(p['replicas'])}"]); time.sleep(8); cpu_ids=apply_cpu(compose,'front-end',float(p['cpus']))
        if kind=='cpu_load': cpu_ids=apply_cpu(compose,'front-end',float(p['cpus']))

        live = kind in {'config_security_load','config_scale_cpu_load','cpu_load'}
        if live:
            url=resolve_url(compose); status=preflight(url); duration=int(p['duration_seconds']); concurrency=int(p['concurrency'])
            state={'requests':0,'successes':0,'failures':0,'latencies':[]}; lock=threading.Lock(); deadline=time.monotonic()+duration+15
            threads=[threading.Thread(target=worker,args=(url,stop,deadline,state,lock),daemon=True) for _ in range(concurrency)]
            start=utc_now(); [t.start() for t in threads]; time.sleep(min(20,duration/3))
        else: state=lock=start=status=url=duration=concurrency=None

        if kind=='restart_config_security':
            prev=started_at(compose,'front-end'); samples=[prev] if prev else []; observed=0
            for _ in range(int(p['restart_count'])):
                run(compose+['restart','front-end'],timeout=60); time.sleep(float(p['restart_interval_seconds'])); cur=started_at(compose,'front-end')
                if cur:
                    samples.append(cur); observed += int(prev is not None and cur!=prev); prev=cur
            temporal={'case_id':case['id'],'repetition':args.repetition,'observed_restart_events':observed,'started_at_samples':samples,'source':'observed Docker State.StartedAt transitions under test configuration','captured_utc':utc_now()}

        collector=ROOT/'runtime_validation'/'collect_runtime_artifacts.sh'; run(['bash',str(collector),case['id'],str(args.repetition),' '.join(compose)],timeout=90)
        # Collector expects one compose-file argument; if an override is active, replace its resolved compose with the exact multi-file resolution below.
        art=latest_artifact(case['id'],args.repetition)
        if art is None: raise RuntimeError('artifact not created')
        if override:
            cp=run(compose+['config'],capture=True); (art/'compose_resolved.yaml').write_text(cp.stdout,encoding='utf-8')
        if temporal: (art/'temporal_process_observation.json').write_text(json.dumps(temporal,indent=2),encoding='utf-8')

        if live:
            time.sleep(max(0,duration-min(20,duration/3))); stop.set(); [t.join(timeout=3) for t in threads]
            with lock:
                req=state['requests']; suc=state['successes']; fail=state['failures']; lat=list(state['latencies'])
            if req==0 or suc==0: raise RuntimeError('invalid HTTP evidence')
            load_obs={'case_id':case['id'],'repetition':args.repetition,'started_utc':start,'finished_utc':utc_now(),'url':url,'preflight_status':status,'concurrency':concurrency,'duration_seconds':duration,'requests':req,'successes':suc,'failures':fail,'error_rate_pct':100.0*fail/req,'latency_p50_ms':percentile(lat,.5),'latency_p95_ms':percentile(lat,.95),'latency_p99_ms':percentile(lat,.99),'source':'direct concurrent HTTP measurements through Sock Shop edge-router /catalogue'}
            (art/'load_observation.json').write_text(json.dumps(load_obs,indent=2),encoding='utf-8')
        meta=ROOT/'runtime_validation'/'run_metadata_batch3'/case['id']/f'rep-{args.repetition}'; meta.mkdir(parents=True,exist_ok=True)
        (meta/'intervention.json').write_text(json.dumps({'case_id':case['id'],'intervention':kind,'parameters':p,'override_file':str(override) if override else None,'load_observation':load_obs,'temporal_process_observation':temporal,'captured_utc':utc_now()},indent=2),encoding='utf-8')
    finally:
        stop.set(); [t.join(timeout=3) for t in threads]
        for cid in cpu_ids:
            try: run(['docker','update','--cpus','0',cid],check=False,timeout=30)
            except Exception: pass
        try: run(base_compose+['up','-d','--remove-orphans','--scale','front-end=1'],check=False,timeout=60)
        except Exception: pass

if __name__=='__main__': main()
