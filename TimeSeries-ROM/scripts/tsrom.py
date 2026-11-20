"""One entry point for the agent.
  python tsrom.py env                                   environment check (packages, omc, MSL 4.0.0)
  python tsrom.py profile --data <files/globs> --out work/profile.json
  python tsrom.py run --config work/config.json [--omc auto|skip|only-script]   prepare -> train -> export -> verify -> report
  python tsrom.py <prepare|train|export|verify|report> --config work/config.json
Exit codes: 0 ok, 1 data error, 2 gates failed (model still exported, marked FAILED_GATES), 3 export mismatch.
"""
import argparse, importlib, os, shutil, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REQ = ["numpy", "scipy", "pandas", "torch", "optuna", "matplotlib"]


def env():
    ok = True
    print(f"python {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        print("  ERROR: Python >= 3.9 required"); ok = False
    for m in REQ:
        try:
            mod = importlib.import_module(m); print(f"  {m:11s} {getattr(mod, '__version__', '?')}")
        except Exception:
            print(f"  {m:11s} MISSING  ->  pip install {m}"); ok = False
    from tsrom_verify import find_omc
    omc = find_omc()
    if not omc:
        print("  omc        NOT FOUND (set OPENMODELICAHOME or add omc to PATH). Training/export work; OMC verification will be skipped.")
    else:
        print(f"  omc        {omc}")
        try:
            p = subprocess.run([omc, "--version"], capture_output=True, text=True, timeout=60); print("             " + p.stdout.strip())
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                mos = os.path.join(d, "t.mos"); open(mos, "w").write('loadModel(Modelica, {"4.0.0"});\ngetErrorString();\n')
                p = subprocess.run([omc, mos], capture_output=True, text=True, timeout=300, cwd=d)
                good = "true" in p.stdout.splitlines()[0] if p.stdout else False
                print(f"  MSL 4.0.0  {'OK' if good else 'NOT LOADABLE: ' + p.stdout.strip()[:200]}")
        except Exception as e:
            print(f"  omc check failed: {e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["env", "profile", "prepare", "train", "export", "verify", "report", "run"])
    ap.add_argument("--config"); ap.add_argument("--data", nargs="*"); ap.add_argument("--out")
    ap.add_argument("--omc", default="auto", choices=["auto", "skip", "only-script"])
    a = ap.parse_args()
    if a.cmd == "env":
        sys.exit(0 if env() else 1)
    if a.cmd == "profile":
        import tsrom_data; from tsrom_common import dump
        p = tsrom_data.profile(tsrom_data.expand(a.data)); dump(p, a.out)
        print(f"profile: {a.out}  runs={p['n_runs']}"); [print("WARNING:", w) for w in p["warnings"]]; return
    steps = ["prepare", "train", "export", "verify", "report"] if a.cmd == "run" else [a.cmd]
    code = 0
    for s in steps:
        print(f"===== {s} =====", flush=True)
        if s == "prepare":
            import tsrom_data
            r = tsrom_data.prepare(a.config)
            [print("WARNING:", w) for w in r["warnings"]]; [print("ERROR:", e) for e in r["errors"]]
            print(f"sample time {r['sample_time']}  tau [{r['tau_min']}, {r['tau_max']}]")
            if r["errors"]:
                sys.exit(1)
        elif s == "train":
            import tsrom_train
            if tsrom_train.run(a.config)["status"] != "PASSED":
                code = 2
        elif s == "export":
            import tsrom_export; tsrom_export.export(a.config)
        elif s == "verify":
            import tsrom_verify
            if not tsrom_verify.run(a.config, a.omc)["export_fidelity_ok"]:
                code = 3
        elif s == "report":
            import tsrom_report; tsrom_report.run(a.config)
    sys.exit(code)


if __name__ == "__main__":
    main()
