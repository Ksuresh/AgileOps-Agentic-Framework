#!/usr/bin/env python3
"""Generate reviewer-facing AAF manuscript tables and figures from verified aggregate results.

This script does not reconstruct case-level data. Its only input is the committed
aggregate_results.csv, whose rows are mapped to frozen workflow evidence in
paper_results/MANUSCRIPT_RESULT_MAP.md.
"""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "aggregate_results.csv"
OUT = ROOT / "generated"
OUT.mkdir(parents=True, exist_ok=True)

with INPUT.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

required = {"study","block","metric","method","n","successes","rate_pct","paired_p_value","provenance"}
if not rows or set(rows[0]) != required:
    raise SystemExit("aggregate_results.csv schema mismatch")

# Markdown table suitable for direct manuscript cross-checking.
lines = [
    "# Canonical AAF manuscript results",
    "",
    "Generated from `aggregate_results.csv`. See `../MANUSCRIPT_RESULT_MAP.md` for frozen-run provenance.",
    "",
    "| Study | Block | Method | n | Result | Paired test |",
    "|---|---|---|---:|---:|---|",
]
for r in rows:
    result = f"{r['successes']}/{r['n']} ({float(r['rate_pct']):.1f}%)"
    lines.append(f"| {r['study']} | {r['block']} | {r['method']} | {r['n']} | {result} | {r['paired_p_value'] or '—'} |")
(OUT / "master_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Reviewer-readable SVG: no plotting dependency and deterministic text/vector output.
plot_rows = [r for r in rows if r["study"] in {"Controlled", "Runtime", "Learned baseline"}]
width, left, right, row_h = 1000, 360, 80, 34
height = 90 + len(plot_rows) * row_h
scale = width - left - right
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
       '<rect width="100%" height="100%" fill="white"/>',
       '<style>text{font-family:Arial,sans-serif;fill:#111}.label{font-size:13px}.title{font-size:18px;font-weight:bold}.pct{font-size:12px}</style>',
       '<text x="20" y="30" class="title">AAF validation results (verified aggregate evidence)</text>',
       '<text x="20" y="52" class="label">Percent agreement within each frozen evaluation block; denominators differ by block.</text>']
for i, r in enumerate(plot_rows):
    y = 82 + i * row_h
    pct = float(r["rate_pct"])
    label = f"{r['block']} — {r['method']}"
    barw = scale * pct / 100.0
    svg += [f'<text x="20" y="{y+15}" class="label">{label}</text>',
            f'<rect x="{left}" y="{y}" width="{scale}" height="18" fill="#eeeeee"/>',
            f'<rect x="{left}" y="{y}" width="{barw:.1f}" height="18" fill="#555555"/>',
            f'<text x="{left+scale+8}" y="{y+14}" class="pct">{pct:.1f}%</text>']
svg.append('</svg>')
(OUT / "validation_overview.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")

# Compact provenance export used by packaging checks.
with (OUT / "master_provenance.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["study","block","method","provenance"])
    for r in rows:
        w.writerow([r["study"], r["block"], r["method"], r["provenance"]])

print(f"Generated {len(rows)} canonical result rows in {OUT}")
