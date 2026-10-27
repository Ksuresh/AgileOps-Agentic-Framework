"""Debug GraphQL notes query: prints errors and counts only (no note content)."""
import json, requests, sys
sys.path.insert(0, "external_validation/postmortems/screening")
from fetch_gitlab import GQL, notes_text
UA = {"User-Agent": "research-screening"}
r = requests.post("https://gitlab.com/api/graphql", json={"query": GQL, "variables": {"iid": ["17168"], "after": None}}, headers=UA, timeout=40)
j = r.json()
print("status", r.status_code, "errors", json.dumps(j.get("errors"))[:800])
nodes = (((j.get("data") or {}).get("project") or {}).get("workItems") or {}).get("nodes") or []
print("nodes", len(nodes))
for n in nodes:
    for w in n["widgets"]:
        if w and "discussions" in w:
            d = w["discussions"]["nodes"]
            notes = [x for dd in d for x in dd["notes"]["nodes"]]
            print("discussions", len(d), "notes", len(notes), "human", sum(1 for x in notes if not x["system"]), "body_null", sum(1 for x in notes if x.get("body") is None))
print("notes_text count:", notes_text(17168)[0])
