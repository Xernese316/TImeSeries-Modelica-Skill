"""Verification.
  python  : re-read the exported .mo constants, re-implement them in numpy, re-simulate the test run with an
            adaptive solver (ODE) / the sampled update (GRU), compare to the PyTorch model and to the data.
  omc     : write verify.mos, run OpenModelica (checkModel + simulate every example + reference timing),
            parse results: Modelica ROM vs heavy-model data, speed-up.

CLI: python tsrom_verify.py --config work/config.json [--omc auto|skip|only-script]
"""
import argparse, os, re, glob, shutil, subprocess, sys, time
import numpy as np, pandas as pd
from scipy.integrate import solve_ivp
from tsrom_common import load, dump, metrics


def parse_constants(path):
    txt = open(path).read(); C = {}
    for name, val in re.findall(r"constant Real (\w+)(?:\[[^\]]*\])? = ([^;]+);", txt):
        C[name] = np.array(eval(val.split('"')[0].replace("{", "[").replace("}", "]")), dtype=float)
    return C, txt


def python_check(info, meta, wd):
    root = info["root"]; net = os.path.join(root, "Networks")
    C, txt = parse_constants(os.path.join(net, "SurrogateBlock.mo"))
    pred = np.load(os.path.join(wd, "test_predictions.npz"))
    t, Y, U, Ptorch = pred["t_0"], pred["true_0"], pred["u_0"], pred["pred_0"]
    ny, nu = Y.shape[1], U.shape[1]
    tanh, sig = np.tanh, lambda x: 1 / (1 + np.exp(-x))
    if info["family"] != "gru":
        nhid = len(re.findall(r"\bh\d+ := ", txt))
        nx = C["Wl"].shape[0]
        def f(x, u):
            parts = [(x[:ny] - C["y_mean"]) / C["y_std"]]
            if nx > ny: parts.append(x[ny:])
            if nu: parts.append((u - C["u_mean"]) / C["u_std"])
            v = np.concatenate(parts); g = C["Wl"] @ v + C["bl"]
            if nhid or "Wo" in C:
                h = v
                for i in range(1, nhid + 1): h = tanh(C[f"W{i}"] @ h + C[f"b{i}"])
                g = g + C["Wo"] @ h
            d = g / float(C["dt"]); d[:ny] *= C["y_std"]; return d
        x = Y[0].copy()
        if nx > ny:
            CI, _ = parse_constants(os.path.join(net, "SurrogateInit.mo"))
            z = CI["Ey"] @ ((Y[0] - CI["y_mean"]) / CI["y_std"]) + CI["e"]
            if nu: z = z + CI["Eu"] @ ((U[0] - CI["u_mean"]) / CI["u_std"])
            x = np.concatenate([x, tanh(z)])
        out = [x[:ny]]
        for k in range(len(t) - 1):
            x = solve_ivp(lambda _, s: f(s, U[k]), (t[k], t[k + 1]), x, rtol=1e-8, atol=1e-8).y[:, -1]
            out.append(x[:ny])
        Pmo = np.array(out)
    else:
        CO, _ = parse_constants(os.path.join(net, "SurrogateOutput.mo"))
        CI, _ = parse_constants(os.path.join(net, "SurrogateInit.mo"))
        v0 = (Y[0] - CI["y_mean"]) / CI["y_std"]
        if nu: v0 = np.concatenate([v0, (U[0] - CI["u_mean"]) / CI["u_std"]])
        h = tanh(CI["E"] @ v0 + CI["e"]); out = []
        for k in range(len(t)):
            out.append(CO["y_mean"] + CO["y_std"] * (CO["Cw"] @ h + CO["d"]))
            v = (U[k] - C["u_mean"]) / C["u_std"] if nu else np.zeros(1)
            r = sig(C["Wir"] @ v + C["Whr"] @ h + C["br"]); z = sig(C["Wiz"] @ v + C["Whz"] @ h + C["bz"])
            n = tanh(C["Win"] @ v + C["bin"] + r * (C["Whn"] @ h + C["bhn"])); h = (1 - z) * n + z * h
        Pmo = np.array(out)
    names = meta["outputs"]; R = np.array(meta["ranges"]["y_max"]) - np.array(meta["ranges"]["y_min"])
    return dict(modelica_vs_torch=metrics(Pmo, Ptorch, names, R), modelica_vs_data=metrics(Pmo, Y, names, R))


# ------------------------------------------------------------------ OpenModelica
def find_omc():
    cands = []
    if os.environ.get("OPENMODELICAHOME"):
        cands.append(os.path.join(os.environ["OPENMODELICAHOME"], "bin", "omc.exe" if os.name == "nt" else "omc"))
    w = shutil.which("omc")
    if w: cands.append(w)
    cands += sorted(glob.glob(r"C:\Program Files\OpenModelica*\bin\omc.exe"), reverse=True)
    cands += ["/usr/bin/omc", "/opt/openmodelica/bin/omc"]
    return next((c for c in cands if c and os.path.exists(c)), None)


def write_mos(info, cfg, vdir):
    pkg, root = info["package"], info["root"]
    lines = ['loadModel(Modelica, {"4.0.0"}); getErrorString();',
             f'loadFile("{os.path.relpath(os.path.join(root, "package.mo"), vdir).replace(os.sep, "/")}"); getErrorString();']
    # paths are relative to the verify folder: run omc with that folder as working directory (portable work dirs)
    for ex in info["examples"]:
        lines.append(f'checkModel({pkg}.Examples.{ex}); getErrorString();')
    sims = [("ValidateAgainstData", "rom_validate"), ("RomOnly", "rom_only")]
    if "CompareToReference" in info["examples"]:
        sims.append(("CompareToReference", "rom_compare"))
    if "PortsDemo" in info["examples"]:
        sims.append(("PortsDemo", "rom_ports"))
    for ex, prefix in sims:
        lines.append(f'simulate({pkg}.Examples.{ex}, outputFormat="csv", fileNamePrefix="{prefix}"); getErrorString();')
    ref = cfg.get("reference")
    if ref:
        lines.append(f'simulate({ref["model"]}, {ref.get("simulate_args", "")}outputFormat="csv", '
                     f'fileNamePrefix="heavy_only"); getErrorString();')
    path = os.path.join(vdir, "verify.mos")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return path


def parse_omc_log(log):
    recs = {}
    for m in re.finditer(r'resultFile = "([^"]*)".*?timeSimulation = ([0-9.eE+-]+).*?timeTotal = ([0-9.eE+-]+)', log, re.S):
        recs[os.path.basename(m.group(1))] = dict(timeSimulation=float(m.group(2)), timeTotal=float(m.group(3)))
    checks = re.findall(r'"Check of ([\w.]+) completed successfully', log)
    failed_sims = re.findall(r'resultFile = ""', log)
    errors = [l for l in log.splitlines() if "Error:" in l]
    warnings = [l for l in log.splitlines() if "WARNING (" in l]
    return dict(timings=recs, checks_ok=checks, n_failed_simulations=len(failed_sims), errors=errors[:30],
                range_warnings=warnings[:30])


def omc_results(info, meta, vdir):
    out = {}
    f = os.path.join(vdir, "rom_validate_res.csv")
    if os.path.exists(f):
        df = pd.read_csv(f); df = df.groupby("time", as_index=False).last()
        P = df[[f"rom.{o}" for o in meta["outputs"]]].values
        E = df[[f"expected.y[{i+1}]" for i in range(len(meta["outputs"]))]].values
        R = np.array(meta["ranges"]["y_max"]) - np.array(meta["ranges"]["y_min"])
        out["omc_rom_vs_heavy_data"] = metrics(P, E, meta["outputs"], R)
    f = os.path.join(vdir, "rom_compare_res.csv")
    if os.path.exists(f):
        df = pd.read_csv(f); df = df.groupby("time", as_index=False).last()
        errs = [c for c in df.columns if c.startswith("err_")]
        out["omc_compare_to_reference_max_abs_err"] = {c: float(df[c].abs().max()) for c in errs}
    return out


def run(cfg_path, omc_mode="auto"):
    cfg = load(cfg_path)
    wd = os.path.join(os.path.dirname(os.path.abspath(cfg_path)), cfg.get("work_dir", "."))
    info = load(os.path.join(wd, "export_info.json"))
    import pickle
    meta = pickle.load(open(os.path.join(wd, "prepared.pkl"), "rb"))["meta"]
    rep = dict(python=python_check(info, meta, wd))
    pc = rep["python"]["modelica_vs_torch"]
    rep["export_fidelity_ok"] = bool(max(pc["max_abs_pct_range"]) < 0.5)
    print(f"python check: exported Modelica vs PyTorch worst max|err| % range = {max(pc['max_abs_pct_range']):.4f} "
          f"-> {'OK' if rep['export_fidelity_ok'] else 'MISMATCH'}")
    vdir = os.path.join(wd, "omc_verify"); os.makedirs(vdir, exist_ok=True)
    mos = write_mos(info, cfg, os.path.abspath(vdir)); rep["mos"] = mos
    omc = find_omc() if omc_mode != "skip" else None
    if omc_mode == "only-script" or omc is None:
        rep["omc"] = "not run (" + ("script only" if omc_mode == "only-script" else "omc not found") + f"): run  omc {mos}"
        print(rep["omc"])
    else:
        t0 = time.time()
        p = subprocess.run([omc, mos], cwd=vdir, capture_output=True, text=True, timeout=cfg.get("omc_timeout", 1800))
        log = p.stdout + p.stderr
        open(os.path.join(vdir, "verify.log"), "w").write(log)
        rep["omc"] = parse_omc_log(log); rep["omc"]["wall_secs"] = round(time.time() - t0, 1)
        rep["omc"].update(omc_results(info, meta, vdir))
        tm = rep["omc"]["timings"]
        rom_t = next((v["timeSimulation"] for k, v in tm.items() if k.startswith("rom_only")), None)
        heavy_t = next((v["timeSimulation"] for k, v in tm.items() if k.startswith("heavy_only")), None)
        if rom_t and heavy_t:
            rep["omc"]["speedup"] = heavy_t / max(rom_t, 1e-9)
        print("omc:", {k: rep["omc"][k] for k in ("checks_ok", "n_failed_simulations") if k in rep["omc"]})
    dump(rep, os.path.join(wd, "verify_report.json"))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--omc", default="auto", choices=["auto", "skip", "only-script"])
    a = ap.parse_args(); r = run(a.config, a.omc)
    sys.exit(0 if r["export_fidelity_ok"] else 3)
