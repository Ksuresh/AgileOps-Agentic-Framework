"""GitLab extension fetch (PROTOCOL.md amendment A3). Encrypted output only.

Population: closed issues in gitlab-com/gl-infra/production with labels incident AND
severity::1 or severity::2, created 2020-01-01..2025-12-31. Shuffled with seed 20261021; the
first 150 are fetched with their human (non-system) notes. Each incident's timeline text is
"YYYY-MM-DD HH:MM UTC: <note body>" per note, in creation order. Title and description are not used.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
API = "https://gitlab.com/api/v4/projects/gitlab-com%2Fgl-infra%2Fproduction"
UA = {"User-Agent": "research-screening (AgileOps-Agentic-Framework revision)"}
SEED, N = 20261021, 150


def get(url: str, **params) -> requests.Response:
    for attempt in range(5):
        r = requests.get(url, params=params, headers=UA, timeout=40)
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1)); continue
        return r
    return r


def population() -> list[dict]:
    seen: dict[int, dict] = {}
    for sev in ("severity::1", "severity::2"):
        page = 1
        while True:
            r = get(f"{API}/issues", labels=f"incident,{sev}", state="closed", per_page=100, page=page,
                    created_after="2020-01-01T00:00:00Z", created_before="2026-01-01T00:00:00Z",
                    order_by="created_at", sort="asc")
            r.raise_for_status()
            batch = r.json()
            for it in batch:
                seen[it["iid"]] = {"iid": it["iid"], "url": it["web_url"], "created_at": it["created_at"], "severity": sev}
            if len(batch) < 100:
                break
            page += 1
            time.sleep(0.5)
    return [seen[k] for k in sorted(seen)]


GQL = """query($iid: [String!], $after: String) { project(fullPath: "gitlab-com/gl-infra/production") {
  workItems(iids: $iid) { nodes { widgets { ... on WorkItemWidgetNotes {
    discussions(first: 20, after: $after) { pageInfo { hasNextPage endCursor }
      nodes { notes { nodes { system createdAt body } } } } } } } } } }"""


ERRORS: list[dict] = []


def notes_text(iid: int) -> tuple[int, str]:
    """Human notes via GraphQL (the REST notes endpoint requires authentication)."""
    notes, after = [], None
    for _ in range(40):
        r = requests.post("https://gitlab.com/api/graphql", json={"query": GQL, "variables": {"iid": [str(iid)], "after": after}},
                          headers=UA, timeout=40)
        if r.status_code != 200 or r.json().get("errors"):
            ERRORS.append({"iid": iid, "status": r.status_code, "errors": str(r.json().get("errors"))[:200] if r.status_code == 200 else ""})
            break
        nodes = (((r.json().get("data") or {}).get("project") or {}).get("workItems") or {}).get("nodes") or []
        disc = next((w["discussions"] for n in nodes for w in n["widgets"] if w and "discussions" in w), None)
        if not disc:
            break
        for d in disc["nodes"]:
            for n in d["notes"]["nodes"]:
                if not n.get("system"):
                    notes.append(n)
        if not disc["pageInfo"]["hasNextPage"]:
            break
        after = disc["pageInfo"]["endCursor"]
        time.sleep(0.3)
    notes.sort(key=lambda n: n["createdAt"])
    out = [f"{n['createdAt'][:16].replace('T', ' ')} UTC: {' '.join((n.get('body') or '').split())}" for n in notes]
    return len(out), "\n".join(out)[:150_000]


def main() -> None:
    """Batch 1 (default): positions 1..150. Later batches (AAF v3 holdout H3): GL_START/GL_END/GL_NAME env vars.
    If the population changed since batch 1, the current seeded order is used with batch-1 issues removed."""
    start, end = int(os.environ.get("GL_START", 0)), int(os.environ.get("GL_END", N))
    name = os.environ.get("GL_NAME", "gitlab_150")
    pop = population()
    rng = random.Random(SEED)
    order = pop[:]
    rng.shuffle(order)
    if start:
        first = {s["iid"] for s in json.loads((HERE / "gitlab_150.meta.json").read_text())["sample"]}
        same = [o["iid"] for o in order[:N]] == [s["iid"] for s in json.loads((HERE / "gitlab_150.meta.json").read_text())["sample"]]
        rest = [o for o in order if o["iid"] not in first]
        sample = rest[start - N:end - N]
    else:
        same, sample = True, order[:N]
    bundle = []
    for k, it in enumerate(sample, start + 1):
        n, text = notes_text(it["iid"])
        bundle.append({"id": f"GL-{k:03d}", **it, "n_notes": n, "text": text})
        time.sleep(0.3)
    meta = {"population_size": len(pop), "seed": SEED, "fetched": len(bundle), "positions": [start + 1, start + len(bundle)],
            "batch1_order_reproduced": same, "query_errors": ERRORS[:20], "n_query_errors": len(ERRORS),
            "sample": [{k: v for k, v in b.items() if k != "text"} for b in bundle]}
    (HERE / f"{name}.meta.json").write_text(json.dumps(meta, indent=1))
    with tempfile.TemporaryDirectory() as tmp:
        plain, key = Path(tmp) / "b.json", Path(tmp) / "k.bin"
        plain.write_text(json.dumps(bundle)); key.write_bytes(os.urandom(32))
        subprocess.run(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-salt", "-in", str(plain),
                        "-out", str(HERE / f"{name}.bundle.json.enc"), "-pass", f"file:{key}"], check=True)
        subprocess.run(["openssl", "pkeyutl", "-encrypt", "-pubin", "-inkey", str(HERE / "screen_public.pem"),
                        "-in", str(key), "-out", str(HERE / f"{name}.bundle.key.enc")], check=True)
    print(json.dumps({k: meta[k] for k in ("population_size", "seed", "fetched")}))


if __name__ == "__main__":
    main()
