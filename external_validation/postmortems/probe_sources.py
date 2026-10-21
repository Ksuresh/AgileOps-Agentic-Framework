"""Feasibility probe for a postmortem-based study (read-only, no LLM).

Sources:
  A. danluu/post-mortems (community list of public postmortems; links to many sites)
  B. GitLab's public production-incident tracker (gitlab-com/gl-infra/production, label "incident")

For every document it records only metadata and keyword statistics — reachability, length,
timeline timestamps, and hits for evidence types and for action types that map to AAF actions.
No postmortem text is stored or committed (copyright).
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

OUT = Path(__file__).resolve().parent / "probe_report"
UA = {"User-Agent": "research-feasibility-probe (AgileOps-Agentic-Framework revision)"}

ACTIONS = {
    "rollback": r"\brolled? ?back\b|\brollback\b|\brevert(?:ed|ing)?\b",
    "scale": r"\bscal(?:ed|ing) (?:up|out)\b|\badd(?:ed|ing)? (?:more )?(?:capacity|nodes|instances|replicas|servers)\b|\bincreas(?:ed|ing) (?:capacity|the number of)\b",
    "block_or_patch": r"\b(?:halt|paus|block|freez|stopp)(?:ed|ing)? (?:the |all )?(?:deploy|release|rollout)s?\b|\bdeploy(?:ment)? freeze\b|\bhotfix\b|\bpatch(?:ed|ing)?\b",
    "mitigate": r"\bmitigat(?:e|ed|ion)\b|\bfail(?:ed)? ?over\b|\breroute|\brestart(?:ed|ing)?\b|\brate[- ]limit|\bdisabl(?:e|ed|ing)\b|\bdrain(?:ed|ing)?\b",
}
EVIDENCE = {
    "latency": r"\blatenc|\bp9[059]\b|\bresponse time",
    "error_rate": r"\berror rate|\b5\d\d(?:s| errors)?\b|\berrors?\b",
    "saturation": r"\bcpu\b|\bmemory\b|\boom\b|\bsaturat|\bdisk\b",
    "restarts": r"\brestart|\bcrash ?loop|\bcrash(?:ed|ing)\b",
    "deploy_config": r"\bdeploy|\brelease\b|\bconfig(?:uration)? change|\brollout",
    "security": r"\bcve\b|\bvulnerab|\bexploit|\bcredential",
    "cost": r"\bcost\b|\bbilling|\bspend",
}
TIMESTAMP = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:UTC|GMT|PT|PST|PDT|ET|EST|EDT|Z)?\b")


def stats(text: str) -> dict:
    t = text.lower()
    out = {"chars": len(text), "timestamps": len(TIMESTAMP.findall(text))}
    for k, p in ACTIONS.items():
        out[f"act_{k}"] = len(re.findall(p, t))
    for k, p in EVIDENCE.items():
        out[f"ev_{k}"] = len(re.findall(p, t))
    out["n_action_types"] = sum(1 for k in ACTIONS if out[f"act_{k}"] > 0)
    out["n_evidence_types"] = sum(1 for k in EVIDENCE if out[f"ev_{k}"] > 0)
    return out


def page_text(url: str) -> tuple[bool, str, str]:
    try:
        r = requests.get(url, headers=UA, timeout=25, allow_redirects=True)
        if r.status_code != 200:
            return False, f"http_{r.status_code}", ""
        if "html" not in r.headers.get("content-type", "html"):
            return False, "not_html", ""
        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup(["script", "style", "nav", "header", "footer"]):
            s.decompose()
        return True, "ok", soup.get_text(" ", strip=True)
    except Exception as exc:
        return False, type(exc).__name__, ""


def probe_danluu() -> list[dict]:
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/danluu/post-mortems", "/tmp/pm"], check=True)
    readme = Path("/tmp/pm/README.md").read_text()
    section, rows = "", []
    for line in readme.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        m = re.match(r"\[([^\]]+)\]\((https?://[^)]+)\)", line)
        if m and section not in {"Other lists of postmortems", "Analysis", "Contributors", "Table of Contents"}:
            rows.append({"source": "danluu", "section": section, "org": m.group(1), "url": m.group(2)})
    for r in rows:
        ok, status, text = page_text(r["url"])
        r.update({"domain": urlparse(r["url"]).netloc, "reachable": ok, "status": status, **stats(text)})
        time.sleep(0.5)
    return rows


def probe_gitlab(pages: int = 4) -> tuple[list[dict], dict]:
    base = "https://gitlab.com/api/v4/projects/gitlab-com%2Fgl-infra%2Fproduction/issues"
    rows, meta = [], {}
    for page in range(1, pages + 1):
        r = requests.get(base, params={"labels": "incident", "state": "closed", "per_page": 100, "page": page,
                                       "order_by": "created_at", "sort": "desc"}, headers=UA, timeout=30)
        meta[f"page_{page}_status"] = r.status_code
        if page == 1:
            meta["x_total"] = r.headers.get("X-Total")
        if r.status_code != 200:
            break
        for it in r.json():
            desc = it.get("description") or ""
            sev = next((l for l in it.get("labels", []) if l.lower().startswith("severity::")), "")
            rows.append({"source": "gitlab", "iid": it.get("iid"), "url": it.get("web_url"), "created_at": it.get("created_at"),
                         "severity": sev, "n_labels": len(it.get("labels", [])),
                         "has_timeline_heading": bool(re.search(r"^#+\s*timeline", desc, re.I | re.M)),
                         "has_corrective_heading": bool(re.search(r"^#+\s*(corrective|remediation|mitigation)", desc, re.I | re.M)),
                         **stats(desc)})
        time.sleep(1)
    return rows, meta


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("reachable", True) and r["chars"] > 500]
    usable = [r for r in ok if r["n_action_types"] >= 1 and r["n_evidence_types"] >= 2 and r["timestamps"] >= 3]
    return {
        "documents": len(rows), "reachable_and_substantial": len(ok),
        "with_action_evidence_and_timeline": len(usable),
        "action_type_presence": {k: sum(1 for r in ok if r[f"act_{k}"] > 0) for k in ACTIONS},
        "evidence_type_presence": {k: sum(1 for r in ok if r[f"ev_{k}"] > 0) for k in EVIDENCE},
        "median_chars": sorted(r["chars"] for r in ok)[len(ok) // 2] if ok else None,
        "status_counts": dict(Counter(r.get("status", "api") for r in rows)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = probe_danluu()
    g, gmeta = probe_gitlab()
    for name, rows in (("danluu", d), ("gitlab", g)):
        if rows:
            with (OUT / f"{name}_metadata.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
                w.writeheader(); w.writerows(rows)
    summary = {"danluu": summarize(d), "gitlab": summarize(g) if g else None, "gitlab_meta": gmeta,
               "danluu_by_domain_reachable": dict(Counter(r["domain"] for r in d if r["reachable"]).most_common(15)),
               "gitlab_severity": dict(Counter(r["severity"] for r in g))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
