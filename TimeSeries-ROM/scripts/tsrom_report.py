"""Final report: verdict, ladder table, test metrics, verification, warnings, plots, run manifest.
CLI: python tsrom_report.py --config work/config.json
"""
import argparse, os, sys, platform, datetime, json
import numpy as np
from tsrom_common import load, dump

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def plot(wd, meta):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    P = np.load(os.path.join(wd, "test_predictions.npz"))
    t, Y, Yp = P["t_0"], P["true_0"], P["pred_0"]
    ny = Y.shape[1]
    fig, ax = plt.subplots(2, ny, figsize=(6.5 * ny, 6.5), squeeze=False, constrained_layout=True)
    for j, n in enumerate(meta["outputs"]):
        c = PALETTE[j % len(PALETTE)]
        a = ax[0, j]; a.plot(t, Y[:, j], color="#3d3d3a", lw=3, alpha=.35, label="heavy model (data)")
        a.plot(t, Yp[:, j], color=c, lw=1.6, label="ROM free-run"); a.set_title(f"{n} — test run", loc="left")
        a.legend(frameon=False, fontsize=9)
        b = ax[1, j]; b.axhline(0, color="#999", lw=.8); b.plot(t, Yp[:, j] - Y[:, j], color=c, lw=1.6)
        b.set_title(f"{n} — error (ROM − heavy)", loc="left"); b.set_xlabel("time [s]")
        for z in (a, b):
            z.grid(alpha=.25, lw=.6); [z.spines[s].set_visible(False) for s in ("top", "right")]
    path = os.path.join(wd, "test_plot.png"); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def verdict(tr, vr):
    reasons = []
    if tr["status"] != "PASSED":
        reasons.append("accuracy/stability gates not met on validation data")
    if vr and not vr.get("export_fidelity_ok", False):
        reasons.append("exported Modelica does not reproduce the trained model")
    om = (vr or {}).get("omc")
    if isinstance(om, dict):
        if om.get("n_failed_simulations"):
            reasons.append(f"{om['n_failed_simulations']} OMC simulation(s) failed")
        if om.get("errors"):
            reasons.append("OMC reported errors")
    elif vr:
        reasons.append("not yet verified in OpenModelica")
    if not reasons:
        return "READY", reasons
    if reasons == ["not yet verified in OpenModelica"]:
        return "READY_PENDING_OMC", reasons
    return "NOT_READY", reasons


def run(cfg_path):
    cfg = load(cfg_path); wd = os.path.join(os.path.dirname(os.path.abspath(cfg_path)), cfg.get("work_dir", "."))
    import pickle
    meta = pickle.load(open(os.path.join(wd, "prepared.pkl"), "rb"))["meta"]
    pr = load(os.path.join(wd, "prep_report.json")); tr = load(os.path.join(wd, "train_report.json"))
    vr = load(os.path.join(wd, "verify_report.json")) if os.path.exists(os.path.join(wd, "verify_report.json")) else None
    ex = load(os.path.join(wd, "export_info.json")) if os.path.exists(os.path.join(wd, "export_info.json")) else {}
    v, reasons = verdict(tr, vr)
    png = plot(wd, meta)
    f = lambda xs, d=3: ", ".join(f"{x:.{d}g}" for x in xs)
    L = [f"# ROM report — {ex.get('package', cfg['package_name'])}", "",
         f"**Verdict: {v}**" + (f" — {'; '.join(reasons)}" if reasons else ""), "",
         f"- Chosen family: **{tr['chosen']}**  ·  sample time {meta['dt']:g} s  ·  "
         f"time constants {pr['tau_min'] and round(pr['tau_min'], 4)} … {pr['tau_max'] and round(pr['tau_max'], 4)} s",
         f"- Inputs: {', '.join(meta['inputs']) or 'none'}  ·  Outputs: {', '.join(meta['outputs'])}",
         f"- Gates: worst NRMSE ≤ {tr['gates']['nrmse_pct']} %, worst |error| ≤ {tr['gates']['max_abs_pct_range']} % of range, "
         "stable on long and constant-input runs", "",
         "## Model ladder (validation, free-run)", "",
         "| family | worst NRMSE % | worst max|err| % range | seeds (score) | weights | gates | stability |",
         "|---|---|---|---|---|---|---|"]
    for e in tr["ladder"]:
        if "val" not in e:
            L.append(f"| {e['family']} | — | — | — | — | {e.get('status')} | — |"); continue
        st = e.get("stability", {}).get("passed")
        L.append(f"| {e['family']} | {max(e['val']['nrmse_pct']):.3g} | {max(e['val']['max_abs_pct_range']):.3g} | "
                 f"{f(e['seed_scores'])} | {e['n_weights']} | {'pass' if e['passes_gates'] else 'fail'} | "
                 f"{'—' if st is None else ('pass' if st else 'FAIL')} |")
    w = tr["test"]["worst"]
    L += ["", "## Held-out test run (free-run, never seen in training)", "",
          "| output | NRMSE % | max |err| | max |err| % range | R² |", "|---|---|---|---|---|"]
    for j, n in enumerate(meta["outputs"]):
        L.append(f"| {n} | {w['nrmse_pct'][j]:.3g} | {w['max_abs'][j]:.4g} | {w['max_abs_pct_range'][j]:.3g} | {w['r2'][j]:.5f} |")
    L += ["", f"![test]({os.path.basename(png)})", ""]
    if vr:
        pc = vr["python"]["modelica_vs_torch"]
        L += ["## Verification", "",
              f"- Exported Modelica re-simulated vs trained model: worst |err| {max(pc['max_abs_pct_range']):.2g} % of range "
              f"→ {'OK' if vr['export_fidelity_ok'] else 'MISMATCH'}"]
        om = vr.get("omc")
        if isinstance(om, dict):
            L.append(f"- OMC checkModel OK: {', '.join(om.get('checks_ok', [])) or 'none'}; failed simulations: {om.get('n_failed_simulations')}")
            if "omc_rom_vs_heavy_data" in om:
                m = om["omc_rom_vs_heavy_data"]
                L.append(f"- OMC ROM vs heavy data: worst NRMSE {f(m['nrmse_pct'])} %, max |err| {f(m['max_abs'], 4)}")
            if "speedup" in om:
                L.append(f"- Speed: heavy/ROM simulation time = {om['speedup']:.2f}x")
            for e in om.get("errors", [])[:5]:
                L.append(f"  - OMC: `{e}`")
        else:
            L.append(f"- OMC: {om}")
    warn = pr.get("warnings", []) + [f"{e['family']}: {d['issue']} on {d['output']} — {d['detail']} → {d['action']}"
                                      for e in tr["ladder"] for d in e.get("diagnosis", [])]
    if any(e.get("seed_unstable") for e in tr["ladder"]):
        warn.append("Training results vary a lot between random seeds for at least one family.")
    if warn:
        L += ["", "## Warnings & diagnosis", ""] + [f"- {x}" for x in warn]
    if tr.get("next_actions"):
        L += ["", "## Recommended next actions", ""] + [f"- {x}" for x in tr["next_actions"]]
    L += ["", "## Validity", "",
          "Valid only inside the training ranges: " +
          "; ".join(f"{n} ∈ [{a:.4g}, {b:.4g}]" for n, a, b in zip(meta["inputs"], meta["ranges"]["u_min"], meta["ranges"]["u_max"]))
          + ". The Modelica block prints a warning when it leaves them."]
    open(os.path.join(wd, "REPORT.md"), "w").write("\n".join(L) + "\n")
    import torch, scipy, optuna
    dump(dict(created=datetime.datetime.now().isoformat(timespec="seconds"), verdict=v, reasons=reasons,
              config=cfg, data_files=load(os.path.join(wd, "profile.json"))["files"] if os.path.exists(os.path.join(wd, "profile.json")) else cfg["dataset"]["files"],
              versions=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                            scipy=scipy.__version__, optuna=optuna.__version__), chosen=tr["chosen"], params=tr["chosen_params"]),
         os.path.join(wd, "manifest.json"))
    print(f"VERDICT {v}" + (f": {'; '.join(reasons)}" if reasons else "") + f"   report: {os.path.join(wd, 'REPORT.md')}")
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    run(ap.parse_args().config)
