"""Debug: why do issue notes come back empty? Prints status codes only (no content)."""
import json, requests
API = "https://gitlab.com/api/v4/projects/gitlab-com%2Fgl-infra%2Fproduction"
UA = {"User-Agent": "research-screening"}
for iid in (3654, 17168, 7083):
    r = requests.get(f"{API}/issues/{iid}/notes", headers=UA, timeout=30)
    print("notes", iid, r.status_code, r.headers.get("content-type"), len(r.text), r.text[:120])
    r2 = requests.get(f"{API}/issues/{iid}", headers=UA, timeout=30)
    j = r2.json() if r2.status_code == 200 else {}
    print("issue", iid, r2.status_code, "user_notes_count", j.get("user_notes_count"), "type", j.get("issue_type"))
q = {"query": 'query { project(fullPath: "gitlab-com/gl-infra/production") { workItems(iids: ["3654"]) { nodes { iid widgets { type ... on WorkItemWidgetNotes { discussions(first: 3) { nodes { notes { nodes { system createdAt } } } } } } } } } }'}
r3 = requests.post("https://gitlab.com/api/graphql", json=q, headers=UA, timeout=30)
print("graphql", r3.status_code, r3.text[:600])
