"""Write the LaTeX tables and in-text numbers of the paper's confirmatory results directly from the committed analysis
outputs (confirmatory_stats.json, ablation.json, rule_sensitivity.json, sensitivity_confirm.json, analysis_gpt.json).
Nothing is typed by hand: rerun after any data change. Output: paper_tables/{numbers.tex, tab_confirm.tex,
tab_construct.tex, tab_ablation.tex}. A count is printed as k/n; an incomplete decider (fewer cases than AAF's
denominator) is flagged with a dagger so it cannot be mistaken for a final value."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_tables"
S = ("outcome_study14", "outcome_study15", "outcome_study16")
ROWS = [  # key in confirmatory_stats, label, group
    ("AAF-14-GPT", "AAF, GPT-5.6 Luna investigator", 0),
    ("AAF", "AAF, Qwen3-8B investigator", 0),
    ("B1-14", "AAF rules alone (no investigation)", 0),
    ("B6-GPT", "GPT-5.6 Luna decides, AAF's rules in prompt (B6)", 1),
    ("B5-GPT", "GPT-5.6 Luna decides (B5)", 1),
    ("B6", "Qwen3-8B decides, AAF's rules in prompt (B6)", 1),
    ("B5", "Qwen3-8B decides (B5)", 1),
    ("AlwaysRollback", "Always roll back", 2),
    ("Dominant", "Act on most severe domain", 2),
    ("BlockAnyCVE", "Block on any vulnerability", 2),
    ("S:D-Sec", "Security expert (rules)", 3),
    ("S:D-DevOps", "DevOps expert (rules)", 3),
    ("S:D-FinOps", "FinOps expert (rules)", 3),
    ("S:D-SRE", "SRE expert (rules)", 3),
    ("S:L-Sec", "Security expert (LLM)", 3),
    ("S:L-DevOps", "DevOps expert (LLM)", 3),
    ("S:L-SRE", "SRE expert (LLM)", 3),
    ("S:L-FinOps", "FinOps expert (LLM)", 3),
]


def kn(v: str | None, full: int | None = None) -> str:
    if not v:
        return "--"
    k, n = v.split("/")
    flag = r"$^\dagger$" if full is not None and int(n) < full else ""
    return f"{k}/{n}{flag}"


def p_fmt(p: float) -> str:
    if p >= 0.001:
        return f"{p:.3f}"
    e = f"{p:.1e}"
    m, x = e.split("e")
    return rf"${m}\times10^{{{int(x)}}}$"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cs = json.loads((ROOT / "outcome_study/confirmatory_stats.json").read_text())
    ab = json.loads((ROOT / "outcome_study/ablation.json").read_text())
    rs = json.loads((ROOT / "outcome_study/rule_sensitivity.json").read_text())["studies"]
    sc = json.loads((ROOT / "outcome_study/sensitivity_confirm.json").read_text())
    P, per = cs["pooled"], cs["per_study"]
    nf = {s: per[s]["n_faults"] for s in S}
    tests = {**P["H1_case_level"], **P["H2_case_level"]}

    # Table: confirmatory results
    L = [r"\begin{table*}[!t]", r"\caption{Confirmatory Outcome Studies: Outcome-Correct Actions on Fresh Fault Cases ($O_1$)}",
         r"\label{tab:confirm}", r"\centering\footnotesize",
         r"\begin{tabular}{L{4.3cm}L{1.05cm}L{1.05cm}L{1.05cm}L{1.1cm}L{1.1cm}L{1.5cm}L{1.5cm}L{1.4cm}}", r"\toprule",
         r"\textbf{Decider} & \textbf{S14 (OB, 2 svc)} & \textbf{S15 (OB, 3 svc)} & \textbf{S16 (Sock Shop)} & \textbf{Pooled} & \textbf{Pooled, strict ($O_2$)} & "
         r"\textbf{Cases: AAF only / other only} & \textbf{$p_{\text{Holm}}$ (cases)} & \textbf{Families: AAF / other better} \\", r"\midrule"]
    last = 0
    for key, lab, grp in ROWS:
        if key not in P["O1"]:
            continue
        if grp != last:
            L.append(r"\midrule"); last = grp
        cells = [kn(per[s]["O1"].get(key), nf[s]) for s in S] + [kn(P["O1"].get(key), P["n_faults"]), kn(P["O2"].get(key), P["n_faults"])]
        t = tests.get(key)
        fam = P["family_level"].get(key)
        cells += [f"{t['a_only']} / {t['b_only']}" if t else "--", p_fmt(t.get("p_holm", t["p_exact"])) if t else "--",
                  f"{fam['families_a_better']} / {fam['families_b_better']}" if fam and key != "AAF" else "--"]
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\par\vspace{2pt}\raggedright\scriptsize OB: Online Boutique. Pooled tests: exact McNemar on cases, Holm-adjusted within "
          r"the pre-specified families (single-domain experts; Qwen3-8B deciders and always-rollback; GPT-5.6 Luna deciders); family-level: cases of the same design on "
          r"different services of one study (" + str(P["n_families"]) + r" families), sign test in the text. Single-domain rule experts use AAF's own "
          r"rules on one domain (v1.4 in Studies 15--16; v1.0 in Study 14 as pre-specified). Qwen3-8B: majority of three runs; GPT-5.6 Luna: one run. "
          r"Comparisons use AAF with the pre-specified Qwen3-8B investigator; AAF's action with the GPT-5.6 Luna "
          r"investigator and with the rules alone was identical on every case. "
          r"$O_2$: among the $O_1$-correct actions, those that keep the feature delivered and then use the least resources." + (r" $^\dagger$Fewer cases than the study has: runs incomplete." if any("dagger" in x for x in L) else ""), r"\end{table*}"]
    (OUT / "tab_confirm.tex").write_text("\n".join(L) + "\n")

    # Table: could the benchmark favour AAF by construction?
    keys = [("AAF-14-GPT", "AAF, GPT investigator"), ("B1-14", "AAF rules alone"), ("AAF", "AAF, Qwen investigator"), ("B6-GPT", "GPT-5.6 Luna + rules (B6)"), ("B5-GPT", "GPT-5.6 Luna (B5)"), ("B6", "Qwen3-8B + rules (B6)"),
            ("B5", "Qwen3-8B (B5)"), ("AlwaysRollback", "Always roll back"), ("Dominant", "Act on most severe domain"), ("S:D-Sec", "Security expert (rules)")]
    L = [r"\begin{table}[!t]", r"\caption{Accuracy on Subsets Where the Choice of Action Matters More (Studies 14--16 Pooled, $O_1$)}",
         r"\label{tab:construct}", r"\centering\footnotesize", r"\begin{tabular}{L{2.45cm}L{1.2cm}L{1.55cm}L{1.55cm}}", r"\toprule",
         rf"\textbf{{Decider}} & \textbf{{All faults ({P['n_faults']})}} & \textbf{{Discriminating ({P['n_discriminating']})}} & \textbf{{One correct action ({P['n_unique_answer']})}} \\",
         r"\midrule"]
    for k, lab in keys:
        L.append(f"{lab} & {kn(P['O1'].get(k), P['n_faults'])} & {kn(P['O1_discriminating'].get(k), P['n_discriminating'])} & "
                 f"{kn(P['O1_unique_answer'].get(k), P['n_unique_answer'])}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{2pt}\raggedright\scriptsize Discriminating: not every action is $O_1$-correct. "
          r"One correct action: exactly one of the five actions leaves the fewest domains in violation." + (" $^\\dagger$Runs incomplete." if any("dagger" in x for x in L) else ""), r"\end{table}"]
    (OUT / "tab_construct.tex").write_text("\n".join(L) + "\n")

    # Table: ablation
    names = [("full", "Full v1.4"), ("-R1'", "Without release-regression rule R1"), ("-R2", "Without security-over-cost rule R2"),
             ("-R1'-R2", "Without R1 and R2 (v1.0)"), ("-baseline", "R1 without pre-release baseline (v1.2)"),
             ("-interaction", "Without interaction arbitration"), ("-interaction-R1'-R2", "Without interaction, R1 and R2"),
             ("single-domain(best)", "Best single-domain expert")]
    L = [r"\begin{table}[!t]", r"\caption{Ablation of Rule Set v1.4 (Rules Alone, Fault Cases, $O_1$)}", r"\label{tab:ablation}",
         r"\centering\footnotesize", r"\begin{tabular}{L{3.35cm}L{0.7cm}L{0.7cm}L{0.7cm}L{1.1cm}}", r"\toprule",
         rf"\textbf{{Variant}} & \textbf{{S14}} & \textbf{{S15}} & \textbf{{S16}} & \textbf{{Pooled ({sum(nf.values())})}} \\", r"\midrule"]
    for k, lab in names:
        v = [ab[s]["O1"][k] for s in S]
        L.append(f"{lab} & " + " & ".join(map(str, v)) + f" & {sum(v)}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_ablation.tex").write_text("\n".join(L) + "\n")

    # in-text numbers
    def k_of(v): return v.split("/")[0] if v else "--"
    num = {
        "NFaults": P["n_faults"], "NFamilies": P["n_families"], "NDisc": P["n_discriminating"], "NUnique": P["n_unique_answer"],
        "AAFQwenKN": P["O1"].get("AAF", "--"), "AAFQwenK": k_of(P["O1"].get("AAF")), "AAFQwenN": (P["O1"].get("AAF") or "-/-").split("/")[1],
        "RulesPooled": k_of(P["O1"].get("B1-14")), "AAFGPTPooled": k_of(P["O1"].get("AAF-14-GPT")),
        "AAFQwenComplete": "complete" if (P["O1"].get("AAF") or "0/0").split("/")[1] == str(P["n_faults"]) else "incomplete",
        "UniqueRules": k_of(P["O1_unique_answer"].get("B1-14")), "UniqueAAFGPT": k_of(P["O1_unique_answer"].get("AAF-14-GPT")),
        "QwenDone": sum(v["AAF_qwen_cases"] for v in cs["completion"].values()), "AllCases": sum(v["all_cases"] for v in cs["completion"].values()),
        "BSixGPT": k_of(P["O1"].get("B6-GPT")), "BFiveGPT": k_of(P["O1"].get("B5-GPT")), "BSixQwen": k_of(P["O1"].get("B6")), "BFiveQwen": k_of(P["O1"].get("B5")),
        "Rollback": k_of(P["O1"].get("AlwaysRollback")), "DominantPooled": k_of(P["O1"].get("Dominant")), "SecExpert": k_of(P["O1"].get("S:D-Sec")),
        "RulesSFifteen": k_of(per["outcome_study15"]["O1"].get("B1-14")), "RulesSSixteen": k_of(per["outcome_study16"]["O1"].get("B1-14")),
"UniqueBSixGPT": k_of(P["O1_unique_answer"].get("B6-GPT")),
        "UniqueBFiveGPT": k_of(P["O1_unique_answer"].get("B5-GPT")), "UniqueDominant": k_of(P["O1_unique_answer"].get("Dominant")),
        "UniqueRollback": k_of(P["O1_unique_answer"].get("AlwaysRollback")),
        "NControl": cs["model_independence_outcome"]["all_by_group"]["control"],
        "MIfault": cs["model_independence_outcome"]["by_group"]["fault"], "MIcontrol": cs["model_independence_outcome"]["by_group"]["control"],
        "MIlow": cs["model_independence_outcome"]["by_group"]["low_risk"],
        "MIcases": cs["model_independence_outcome"]["cases"], "MIsame": cs["model_independence_outcome"]["same_action_qwen_vs_gpt_investigator"],
        "RepCases": sum(v["cases_two_replicates"] for v in cs["replicates"].values()), "RepSame": sum(v["O1_identical"] for v in cs["replicates"].values()),
        "RuleSensMinFifteen": rs["outcome_study15"]["correct_min"], "RuleSensMinSixteen": rs["outcome_study16"]["correct_min"],
        "OracleSensSettingsGT": min(min(v["settings_rules_gt"].values()) for v in sc.values()),
        "FreezeCommit": cs["timeline"]["rules_v14_frozen"].split()[0], "FreezeDate": cs["timeline"]["rules_v14_frozen"].split()[1][:16].replace("T", " "),
    }
    fam = P["family_level"]
    for k, m in (("B6-GPT", "FamBSixGPT"), ("B5-GPT", "FamBFiveGPT"), ("AlwaysRollback", "FamRollback"), ("S:D-Sec", "FamSec")):
        if k in fam:
            num[m] = f"{fam[k]['families_a_better']}"
            num[m + "Worse"] = f"{fam[k]['families_b_better']}"
            num[m + "P"] = p_fmt(fam[k]["sign_test_p"])
            num[m + "CI"] = f"{100 * fam[k]['cluster_bootstrap_95'][0]:.0f} to {100 * fam[k]['cluster_bootstrap_95'][1]:.0f}"
    for k, m in (("AAF-14-GPT", "OtwoAAF"), ("B1-14", "OtwoRules"), ("B6-GPT", "OtwoBSixGPT"), ("B5-GPT", "OtwoBFiveGPT"), ("AlwaysRollback", "OtwoRollback"), ("S:D-Sec", "OtwoSec"), ("Dominant", "OtwoDominant")):
        num[m] = k_of(P["O2"].get(k))
    oc = P["O2_case_level"]
    num["OtwoPBSixGPT"] = p_fmt(oc["B6-GPT"].get("p_holm", oc["B6-GPT"]["p_exact"]))
    num["OtwoDiscBSixGPT"] = f"{oc['B6-GPT']['a_only']} vs.\\ {oc['B6-GPT']['b_only']}"
    h3 = P["H3_low_risk"]
    num["LowContinuedAAFGPT"] = h3["release_continued"].get("AAF-14-GPT", "--")
    num["LowOtwoAAFGPT"] = f"{h3['O2'].get('AAF-14-GPT', {}).get('k', '--')}/{h3['O2'].get('AAF-14-GPT', {}).get('n', '--')}"
    num["LowOtwoAAFGPTCI"] = "--".join(str(x) for x in h3["O2"].get("AAF-14-GPT", {}).get("wilson95_pct", []))
    num["LowOtwoNone"] = f"{h3['O2'].get('AlwaysNone', {}).get('k', '--')}/{h3['O2'].get('AlwaysNone', {}).get('n', '--')}"
    num["LowN"] = h3["n"]
    num["LowContinuedAAF"] = h3["release_continued"].get("AAF", "--")
    num["LowOtwoAAF"] = f"{h3['O2'].get('AAF', {}).get('k', '--')}/{h3['O2'].get('AAF', {}).get('n', '--')}"
    w = P["withheld"]
    num["WN"] = w["n"]
    for k, m in (("AAF", "WAAF"), ("AAF-14-GPT", "WAAFGPT"), ("B1-14", "WRules"), ("B2-14", "WFetch")):
        num[m] = w["O1"].get(k, "--")
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in num.items()]
    (OUT / "numbers.tex").write_text("% generated by outcome_study/paper_tables.py -- do not edit\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
