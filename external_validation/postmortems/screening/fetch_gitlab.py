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


def notes_text(iid: int) -> tuple[int, str]:
    out, page = [], 1
    while True:
        r = get(f"{API}/issues/{iid}/notes", sort="asc", order_by="created_at", per_page=100, page=page)
        if r.status_code != 200:
            break
        batch = r.json()
        for n in batch:
            if n.get("system"):
                continue
            ts = n["created_at"][:16].replace("T", " ")
            body = " ".join((n.get("body") or "").split())
            out.append(f"{ts} UTC: {body}")
        if len(batch) < 100:
            break
        page += 1
    return len(out), "\n".join(out)[:150_000]


def main() -> None:
    pop = population()
    rng = random.Random(SEED)
    order = pop[:]
    rng.shuffle(order)
    sample = order[:N]
    bundle = []
    for k, it in enumerate(sample, 1):
        n, text = notes_text(it["iid"])
        bundle.append({"id": f"GL-{k:03d}", **it, "n_notes": n, "text": text})
        time.sleep(0.3)
    meta = {"population_size": len(pop), "seed": SEED, "fetched": len(bundle),
            "sample": [{k: v for k, v in b.items() if k != "text"} for b in bundle]}
    (HERE / "gitlab_150.meta.json").write_text(json.dumps(meta, indent=1))
    with tempfile.TemporaryDirectory() as tmp:
        plain, key = Path(tmp) / "b.json", Path(tmp) / "k.bin"
        plain.write_text(json.dumps(bundle)); key.write_bytes(os.urandom(32))
        subprocess.run(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-salt", "-in", str(plain),
                        "-out", str(HERE / "gitlab_150.bundle.json.enc"), "-pass", f"file:{key}"], check=True)
        subprocess.run(["openssl", "pkeyutl", "-encrypt", "-pubin", "-inkey", str(HERE / "screen_public.pem"),
                        "-in", str(key), "-out", str(HERE / "gitlab_150.bundle.key.enc")], check=True)
    print(json.dumps({k: meta[k] for k in ("population_size", "seed", "fetched")}))


if __name__ == "__main__":
    main()
