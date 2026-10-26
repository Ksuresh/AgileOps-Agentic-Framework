"""Generate Part A of the supplementary appendix (evidence for the claims of the paper) from committed artifacts:
A protocols and freeze timeline, B calibration of the confirmatory studies, C per-study statistics, D per-case results,
E robustness, G2 model-independence and containment tables. Output: paper_tables/supp_A.tex ... supp_G2.tex."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from outcome_study.analyze_gpt import load  # noqa: E402
from outcome_study.paper_tables import p_fmt  # noqa: E402

OUT = ROOT / "paper_tables"
S = ("outcome_study14", "outcome_study15", "outcome_study16")
SN = {"outcome_study14": "Study 14", "outcome_study15": "Study 15", "outcome_study16": "Study 16"}
AB = {"Rollback to stable deployment": "RB", "Block release and fix pipeline": "BL", "Patch or block release": "PA",
      "Mitigate and monitor": "MI", "Scale adjustment": "SC", "No action (observe)": "NO", "Review scaling policy": "RV",
      "Escalate for evidence/human review": "ES", None: "--"}
SA = {"rollback": "RB", "patch": "PA", "mitigate": "MI", "scale": "SC", "none": "NO"}
LABEL = {"AAF": "AAF (Qwen investigator)", "AAF-14-GPT": "AAF (GPT investigator)", "B1-14": "Rules v1.4 alone (B1-14)",
         "B2-14": "Rules v1.4 + fetch rule (B2-14)", "B1": "Rules v1.0 (B1)", "B1-12": "Rules v1.2 (B1-12)",
         "B6-GPT": "GPT-5.6 Luna + rules (B6)", "B5-GPT": "GPT-5.6 Luna (B5)", "B6": "Qwen3-8B + rules (B6)", "B5": "Qwen3-8B (B5)",
         "B3-14-GPT": "Agents always on, v1.4 decides (GPT)", "B3-GPT": "Agents always on, v1.0 decides (GPT)",
         "B3-14": "Agents always on, v1.4 decides (Qwen)", "B3": "Agents always on, v1.0 decides (Qwen)",
         "AlwaysRollback": "Always roll back", "AlwaysNone": "Always no action", "Dominant": "Act on most severe domain",
         "BlockAnyCVE": "Block on any vulnerability", "S:D-Sec": "Security expert (rules)", "S:D-DevOps": "DevOps expert (rules)",
         "S:D-FinOps": "FinOps expert (rules)", "S:D-SRE": "SRE expert (rules)", "S:L-Sec": "Security expert (LLM)",
         "S:L-DevOps": "DevOps expert (LLM)", "S:L-SRE": "SRE expert (LLM)", "S:L-FinOps": "FinOps expert (LLM)",
         "D-Sec": "Security expert (v1.0 rules)", "D-DevOps": "DevOps expert (v1.0 rules)", "D-FinOps": "FinOps expert (v1.0 rules)",
         "D-SRE": "SRE expert (v1.0 rules)"}
ORDER = list(LABEL)
GROUP = {**{k: "AAF and its rules" for k in ("AAF", "AAF-14-GPT", "B1-14", "B2-14", "B1", "B1-12", "B3-14-GPT", "B3-GPT", "B3-14", "B3")},
         **{k: "LLM decides" for k in ("B6-GPT", "B5-GPT", "B6", "B5")},
         **{k: "Fixed policies" for k in ("AlwaysRollback", "AlwaysNone", "Dominant", "BlockAnyCVE")}}
GROUP_ORDER = ["AAF and its rules", "LLM decides", "Fixed policies", "Single-domain experts"]


def grouped(keys):
    """Yield (group header or None, key) in a fixed group order."""
    for g in GROUP_ORDER:
        ks = [k for k in keys if GROUP.get(k, "Single-domain experts") == g]
        if ks:
            yield g, None
            for k in ks:
                yield None, k


def esc(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def git_first(path: str) -> str:
    out = subprocess.run(["git", "log", "--reverse", "--format=%h|%cI|%s", "--", path], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    return out[0] if out else "||"


def part_a(cs) -> str:
    rows = [("Rule set v1.4 committed (frozen)", "agentic_experiments/rules_v14.py"),
            ("Study 14 protocol and cases", "outcome_study14/PROTOCOL.md"), ("Study 14 first outcome measured", "outcome_study14/measurements"),
            ("Study 15 protocol, cases, analysis script", "outcome_study15/PROTOCOL.md"), ("Study 15 first outcome measured", "outcome_study15/measurements"),
            ("Study 16 protocol, cases, analysis script", "outcome_study16/PROTOCOL.md"), ("Study 16 first outcome measured", "outcome_study16/measurements"),
            ("GPT-5.6 Luna decider protocol and analysis", "outcome_study/PROTOCOL_GPT.md"), ("GPT-5.6 Luna first decision", "outcome_study14/decisions_gpt")]
    last = subprocess.run(["git", "log", "-1", "--format=%h|%cI", "--", "agentic_experiments/rules_v14.py"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    L = [r"\section*{Appendix A. Protocols, Freeze Timeline and Amendments}",
         r"Every confirmatory study (14--16) and the frontier-model decider runs have a protocol, with hypotheses and analysis script, "
         r"committed to the public repository before the first outcome or decision. Table~A1 lists the commits; the repository's history is the "
         r"authoritative record. Rule set v1.4 has not changed since its first commit (last change: \texttt{" + esc(last.split('|')[0]) + "}).",
         r"\begin{table}[!h]", r"\caption*{Table A1. Commit timeline (UTC)}", r"\centering\scriptsize", r"\begin{tabular}{L{3.6cm}L{1.0cm}L{2.4cm}}", r"\toprule",
         r"\textbf{Event} & \textbf{Commit} & \textbf{Time} \\", r"\midrule"]
    for lab, path in rows:
        h, t, _ = git_first(path).split("|", 2)
        L.append(f"{lab} & \\texttt{{{h}}} & {t[:16].replace('T', ' ')}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}",
          r"\subsection*{A.1 Amendments made before any outcome was measured}",
          r"\textit{Study 15, amendment 1.} The first calibration run showed (i) that the storefront kept failing for minutes after a backend "
          r"recovered, because its gRPC client backs off before reconnecting (healthy currency service measured with 66.6\% errors), and (ii) that "
          r"CPU starvation does not degrade the product-catalog service (p95 $\le$ 498 ms down to a 5m CPU limit). Before any outcome was measured, "
          r"the storefront was set to restart after every backend change, and the degradation of services that CPU starvation does not affect became the "
          r"service's own latency-injection setting, mildest violating value first. The superseded calibration is kept in the repository. The same "
          r"reconnection effect is visible in three Study~14 backend cases (QP-R02, QP-R04, QP-X02), in which the healthy rollback state was measured with "
          r"50--100\% errors; they are reported as non-discriminating and favour no decider.",
          r"\subsection*{A.2 Execution notes}",
          r"Study~14: five decider batches reached the runner time limit and 24 of 864 LLM decisions were completed by a resumed run with identical "
          r"settings. Studies~15 and~16: Qwen3-8B decider batches that reached the runner time limit were resumed with identical settings; completed "
          r"decisions were not rerun (details in each protocol). GPT-5.6 Luna deciders ran once per case under a call and token cap that was not reached "
          r"(Table~F1). In Study~16 no older catalogue image both carried extra critical CVEs and started, so, as pre-specified, its vulnerable-image cases "
          r"carry an exposed secret instead; no low-risk image qualified for any service, so low-risk cases use the stable image, which carries HIGH/MEDIUM findings."]
    comp = cs["completion"]
    L += [r"\subsection*{A.3 Completeness of the Qwen3-8B runs}",
          r"Qwen3-8B runs on CPU-only runners; some decider batches reached the runner's six-hour limit, and completion is uneven across studies "
          r"(Table~A2). Decider jobs are assigned to batches in a fixed pseudo-random order (seed 9) before any decision is made, which makes it "
          r"unlikely that the missing cases were selected by type or difficulty, but it does not prove that they are unbiased: slow cases, for "
          r"example those where the model reasons longer, are more likely to be in a batch that stops. We therefore treat the missingness as a "
          r"limitation. Every Qwen3-8B result uses only completed cases and shows its denominator; the conclusions rest on the complete results with "
          r"the GPT-5.6 Luna investigator and with the rules alone, which agree with the Qwen3-8B investigator on every completed case.",
          r"\begin{table}[!h]", r"\caption*{Table A2. Full-evidence cases with a completed Qwen3-8B decision}", r"\centering\scriptsize",
          r"\begin{tabular}{L{1.4cm}L{1.0cm}L{1.6cm}L{1.0cm}L{1.0cm}}", r"\toprule",
          r"\textbf{Study} & \textbf{Cases} & \textbf{AAF (Qwen investigator)} & \textbf{B5} & \textbf{B6} \\", r"\midrule"]
    for st, v in comp.items():
        L.append(f"{SN[st]} & {v['all_cases']} & {v['AAF_qwen_cases']} & {v['B5_qwen_cases']} & {v['B6_qwen_cases']}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def part_b() -> str:
    L = [r"\section*{Appendix B. Calibration of the Confirmatory Studies}",
         r"Calibration fixed every service-dependent quantity by written rules before any outcome was measured (protocols, Appendix~A).",
         r"\begin{table*}[!h]", r"\caption*{Table B1. Calibration results}", r"\centering\scriptsize",
         r"\begin{tabular}{L{1.3cm}L{2.6cm}L{1.6cm}L{1.6cm}L{4.8cm}L{3.2cm}}", r"\toprule",
         r"\textbf{Study} & \textbf{Service} & \textbf{Healthy p95 / threshold} & \textbf{Degradation} & \textbf{Vulnerable images (critical CVEs)} & \textbf{Stable image (critical / HIGH+MED)} \\", r"\midrule"]
    for st in ("outcome_study15", "outcome_study16"):
        cal = json.loads((ROOT / st / "calibration" / "calibration.json").read_text())
        thr = 2 * cal["healthy_p95_median_ms"]
        per = cal.get("study15_calibration") or cal.get("study16_calibration")
        for svc, r in per.items():
            imgs = []
            for k, v in r["images"].items():
                if v and not k.startswith("N"):
                    crit = next((x.get("critical_cves") for x in r["vuln_scans"] if x["image"] == v), "?")
                    imgs.append(f"{k}: {v.split('/')[-1]} ({crit})")
            imgs = "; ".join(imgs) or "none qualified (exposed secret used)"
            hp = r.get("healthy", {}).get("p95_ms")
            L.append(f"{SN[st]} & \\texttt{{{esc(svc)}}} & {hp:.0f} / {thr:.0f} ms & {esc(r['perf_cpu'])} & {esc(imgs)} & "
                     f"{r['stable']['critical_cves']} / {r['stable']['noncritical_cves']}" + r" \\" if hp else
                     f"{SN[st]} & \\texttt{{{esc(svc)}}} & -- / {thr:.0f} ms & {esc(r['perf_cpu'])} & {esc(imgs)} & {r['stable']['critical_cves']} / {r['stable']['noncritical_cves']}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{2pt}\raggedright\scriptsize Study 14 used fixed CPU limits of 30 and 50 millicores and the "
          r"frontend images E, F and product-catalog images P1, P2 (Appendix~F). Healthy p95 for Study 15 is the per-service calibration measurement; "
          r"the violation threshold is the Study 9 rule (2 $\times$ healthy median of the application). Degradation: CPU limit, or injected latency (lat:).",
          r"\end{table*}"]
    return "\n".join(L) + "\n"


def part_c(cs) -> str:
    L = [r"\section*{Appendix C. Confirmatory Studies 14--16: Statistics per Study}",
         r"For each study and for the pooled data, Table (a) gives each decider's outcome-correct actions on fault cases under $O_1$ and the "
         r"stricter $O_2$, on discriminating cases and on cases with one correct action; Table (b) compares AAF (pre-specified Qwen3-8B investigator) "
         r"with each comparison decider: cases only AAF got right, cases only the other got right, the Holm-adjusted exact McNemar $p$ within the "
         r"pre-specified family, and the same comparison over families of related cases (sign test). A dagger marks configurations whose runs were "
         r"incomplete when this appendix was generated; their denominators are the completed cases."]
    for scope in list(S) + ["pooled"]:
        P = cs["pooled"] if scope == "pooled" else cs["per_study"][scope]
        name = "Pooled, Studies 14--16" if scope == "pooled" else SN[scope]
        full = P["n_faults"]
        mark = lambda v, n: (v + r"$^\dagger$") if v and v.split("/")[1] != str(n) else (v or "--")
        L += [rf"\subsection*{{C.{(list(S) + ['pooled']).index(scope) + 1} {name}}}",
              rf"Fault cases {full}; families {P['n_families']}; discriminating {P['n_discriminating']}; one correct action {P['n_unique_answer']}.",
              r"\begin{longtable}{L{6.4cm}L{1.7cm}L{1.7cm}L{2.2cm}L{2.2cm}}",
              rf"\multicolumn{{5}}{{l}}{{\textbf{{(a) Outcome-correct actions on fault cases, {name}}} (complete unless marked $^\dagger$)}} \\", r"\toprule",
              r"\textbf{Decider} & \textbf{$O_1$} & \textbf{$O_2$ (strict)} & \textbf{Discriminating cases} & \textbf{One-correct-action cases} \\", r"\midrule", r"\endfirsthead",
              rf"\multicolumn{{5}}{{l}}{{\textbf{{(a) continued, {name}}}}} \\", r"\toprule",
              r"\textbf{Decider} & \textbf{$O_1$} & \textbf{$O_2$ (strict)} & \textbf{Discriminating cases} & \textbf{One-correct-action cases} \\", r"\midrule", r"\endhead"]
        for g, d in grouped([d for d in ORDER if d in P["O1"]]):
            if g:
                L.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{g}}}}} \\")
                continue
            L.append(f"\\quad {esc(LABEL[d])} & {mark(P['O1'][d], full)} & {mark(P['O2'].get(d), full)} & {mark(P['O1_discriminating'].get(d), P['n_discriminating'])} & {mark(P['O1_unique_answer'].get(d), P['n_unique_answer'])}" + r" \\")
        L += [r"\bottomrule", r"\end{longtable}"]
        tests = {**P["H1_case_level"], **P["H2_case_level"]}
        hdr = (r"& \multicolumn{3}{l}{\textbf{Case level (exact McNemar)}} & \multicolumn{2}{l}{\textbf{Family level (sign test)}} \\",
               r"\textbf{Comparison decider} & \textbf{AAF only right} & \textbf{Other only right} & \textbf{$p_{\text{Holm}}$} & \textbf{Families AAF / other better} & \textbf{$p$} \\")
        L += [r"\begin{longtable}{L{5.2cm}L{1.5cm}L{1.5cm}L{1.9cm}L{2.2cm}L{1.7cm}}",
              rf"\multicolumn{{6}}{{l}}{{\textbf{{(b) Paired comparisons with AAF (Qwen3-8B investigator, completed cases), {name}}}}} \\", r"\toprule",
              *hdr, r"\midrule", r"\endfirsthead",
              rf"\multicolumn{{6}}{{l}}{{\textbf{{(b) continued, {name}}}}} \\", r"\toprule", *hdr, r"\midrule", r"\endhead"]
        keys = [d for d in ORDER if tests.get(d) or P["family_level"].get(d)]
        for g, d in grouped(keys):
            if g:
                L.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{g}}}}} \\")
                continue
            t_, f = tests.get(d), P["family_level"].get(d)
            L.append("\\quad " + f"{esc(LABEL[d])} & {t_['a_only'] if t_ else '--'} & {t_['b_only'] if t_ else '--'} & {p_fmt(t_.get('p_holm', t_['p_exact'])) if t_ else '--'} & "
                     f"{str(f['families_a_better']) + ' / ' + str(f['families_b_better']) if f else '--'} & {p_fmt(f['sign_test_p']) if f else '--'}" + r" \\")
        L += [r"\bottomrule", r"\end{longtable}"]
        w, h3 = P["withheld"], P["H3_low_risk"]
        L.append(rf"\textit{{Withheld-evidence variants}} ({w['n']}): " + "; ".join(f"{esc(LABEL.get(k, k))} {v}" for k, v in w["O1"].items() if k in LABEL) + ".")
        L.append(rf"\textit{{Low-risk releases}} ({h3['n']}), release continued: " + "; ".join(f"{esc(LABEL.get(k, k))} {v}" for k, v in h3["release_continued"].items()) + ".")
    return "\n\n".join(L) + "\n"


def part_d() -> str:
    L = [r"\section*{Appendix D. Confirmatory Studies 14--16: Per-Case Results}",
         r"Release recipe (previous $\rightarrow$ current), the $O_1$ and $O_2$ sets, and each decider's majority action. RB rollback, PA patch, "
         r"MI mitigate, SC scale, NO no action, ES escalate, BL block. G: GPT-5.6 Luna; Q: Qwen3-8B (as investigator for AAF, as decider for B5/B6). A bold action is outside $O_1$."]
    keys = [("AAF-14", "AAF"), ("AAF-14-GPT", "AAF G"), ("B1-14", "Rules"), ("B5-GPT", "B5 G"), ("B6-GPT", "B6 G"), ("B5", "B5 Q"), ("B6", "B6 Q"), ("AlwaysRollback", "Roll back")]
    for st in S:
        A, table, dec, ok, maj, _ = load(st)
        L += [rf"\subsection*{{D.{S.index(st) + 1} {SN[st]}}}", r"{\setlength\tabcolsep{3pt}\begin{longtable}{L{1.4cm}L{3.9cm}L{2.5cm}L{1.8cm}" + "L{0.66cm}" * len(keys) + "}", r"\toprule",
              r"\textbf{Case} & \textbf{Release} & \textbf{$O_1$} & \textbf{$O_2$} & " + " & ".join(rf"\textbf{{{h}}}" for _, h in keys) + r" \\", r"\midrule", r"\endhead"]
        cases = {}
        for p in sorted((ROOT / st / "measurements" / "rep1").glob("*.json")):
            d = json.loads(p.read_text()); cases[d["case"]["id"]] = d["case"]
        for c in sorted(table):
            cc = cases.get(c, {})
            rec = ("+".join(cc.get("previous") or ["stable"]) + r" $\rightarrow$ " + "+".join(cc.get("current") or ["stable"])) if cc else ""
            cells = []
            for k, _ in keys:
                a = maj(dec[k].get(c) or [])
                s = AB.get(a, "?") if a else "--"
                cells.append(rf"\textbf{{{s}}}" if a and A.STATE_OF.get(a) not in table[c]["O1"] else s)
            L.append(f"{c} & {esc(rec)} & {','.join(SA[x] for x in table[c]['O1'])} & {','.join(SA[x] for x in table[c]['O2'])} & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{longtable}}"]
    return "\n".join(L) + "\n"


def part_e(cs) -> str:
    sc = json.loads((ROOT / "outcome_study/sensitivity_confirm.json").read_text())
    rs = json.loads((ROOT / "outcome_study/rule_sensitivity.json").read_text())["studies"]
    ab = json.loads((ROOT / "outcome_study/ablation.json").read_text())
    L = [r"\section*{Appendix E. Robustness Analyses of the Confirmatory Studies}",
         r"\subsection*{E.1 Oracle thresholds}",
         r"Reliability factor $\{1.5, 2, 3\}\times$ healthy p95, error threshold $\{0.5, 1, 2\}\%$, cost factor $\{2, 3, 4\}\times$ healthy footprint (27 settings)."]
    for st in S:
        v = sc[st]
        L.append(rf"{SN[st]}: rules correct on {v['rules_min']}--{v['rules_max']} of {v['n_faults']} fault cases; settings in which the rules are more "
                 r"accurate than each comparison decider: " + "; ".join(f"{esc(LABEL.get(k, k))} {n}/27" for k, n in v["settings_rules_gt"].items()) + ".")
    L += [r"\subsection*{E.2 AAF's own thresholds}",
          r"The release-regression rule's p95 floor $\{350, 450, 550\}$ ms, factor $\{1.5, 2, 3\}$, error floor $\{5, 8, 12\}\%$, error rise $\{3, 5, 8\}$ points and restart count $\{2, 3, 5\}$ (243 settings)."]
    for st in S:
        v = rs[st]
        L.append(rf"{SN[st]}: correct {v['correct_min']}--{v['correct_max']} of {v['n_faults']} (frozen setting {v['frozen_setting_correct']}); "
                 r"share of settings above each comparison decider: " + "; ".join(f"{esc(LABEL.get(k, k))} {s:.2f}" for k, s in v["share_settings_gt_comparator"].items()) + ".")
    L += [r"\subsection*{E.3 Ablation per study}"]
    for st in S:
        t = ab[st]["tests_full_vs"]
        L.append(rf"{SN[st]} (fault cases {ab[st]['n_faults']}): " + "; ".join(f"{esc(k)} {ab[st]['O1'][k]} ({t[k]['a_only']}/{t[k]['b_only']}, $p_{{\\text{{Holm}}}}$ {p_fmt(t[k]['p_holm'])})" if k in t else f"{esc(k)} {ab[st]['O1'][k]}" for k in ab[st]["O1"]) + ".")
    L += [r"\subsection*{E.4 Measurement replicates}"]
    for st in S:
        v = cs["replicates"][st]
        L.append(rf"{SN[st]}: $O_1$ identical in {v['O1_identical']} of {v['cases_two_replicates']} cases (differing: {esc(', '.join(v['differing']))}); "
                 rf"rules correct under each replicate separately on {v['faults_rules_correct_in_each_replicate_separately']} fault cases.")
    return "\n\n".join(L) + "\n"


def part_g2() -> str:
    m = json.loads((ROOT / "model_replication/analysis_three_models.json").read_text())
    c = json.loads((ROOT / "containment_study/results/containment_summary.json").read_text())
    i = json.loads((ROOT / "containment_study/results/injection_analysis.json").read_text())
    L = [r"\subsection*{G.8 Model independence and containment (Studies 10, 10b, 11, 11b)}",
         r"\textit{Three models.} Same majority action under GPT-5.6 Luna, Qwen3-8B and Llama 3.1 8B: " +
         "; ".join(f"{k} {m[k]['same_all_three']}/{m['n']}" for k in ("AAF", "B3", "B6", "B5")) +
         r". The three AAF cases that differ are incomplete-evidence cases in which one model's investigation retrieved the withheld measurement and another's did not.",
         r"\textit{Simulated LLM faults (Study 11).} Action differs from the rules-only action (AAF / B5): " +
         "; ".join(f"{k} {v['aaf_differs_from_b1']}/{v['case_runs']} / {v['b5_differs_from_b1']}/{v['case_runs']}" for k, v in c.items()) +
         r"; AAF differences without a genuine tool result: " + ", ".join(str(v["aaf_differs_from_b1_without_genuine_tool_result"]) for v in c.values()) + ".",
         r"\textit{Prompt injection (Study 11b, Qwen3-8B).} Injected action adopted: " + "; ".join(f"{k} {v}" for k, v in i["injection_success"].items()) +
         rf"; AAF final action differing from rules-only without genuine evidence: {i['aaf_final_differs_from_B1_without_genuine_tool_result']}."]
    return "\n\n".join(L) + "\n"


def part_f_gpt() -> str:
    L = [r"\subsection*{F.4 Frontier-model decider runs}", r"\begin{table}[!h]", r"\caption*{Table F1. GPT-5.6 Luna runs on the outcome studies}",
         r"\centering\scriptsize", r"\begin{tabular}{L{1.4cm}L{1.0cm}L{1.3cm}L{1.6cm}L{1.6cm}}", r"\toprule",
         r"\textbf{Study} & \textbf{Calls} & \textbf{Tokens} & \textbf{Cap (calls / tokens)} & \textbf{Stopped by cap} \\", r"\midrule"]
    for st in S:
        for b in sorted((ROOT / st / "decisions_gpt").glob("budget_*.json")):
            v = json.loads(b.read_text())
            L.append(f"{SN[st]} & {v['calls']} & {v['tokens']:,} & {v['caps'][0]} / {v['caps'][1]:,} & {'yes' if v['stopped'] else 'no'}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cs = json.loads((ROOT / "outcome_study/confirmatory_stats.json").read_text())
    for name, txt in (("supp_A", part_a(cs)), ("supp_B", part_b()), ("supp_C", part_c(cs)), ("supp_D", part_d()), ("supp_E", part_e(cs)),
                      ("supp_G2", part_g2()), ("supp_F4", part_f_gpt())):
        if name in ("supp_C", "supp_D"):  # long tables need one-column mode in the two-column template
            txt = "\\onecolumn\n" + txt + "\n\\twocolumn\n"
        (OUT / f"{name}.tex").write_text("% generated by outcome_study/paper_supp_partA.py -- do not edit\n" + txt)
        print(name, len(txt))


if __name__ == "__main__":
    main()
