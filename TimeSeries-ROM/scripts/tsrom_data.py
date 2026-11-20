"""Data layer: load (CSV / OpenModelica .mat), clean, profile, prepare (resample, split, coverage, time constants).

CLI:
  python tsrom_data.py profile --data "<glob or file list>" --out work/profile.json
  python tsrom_data.py prepare --config work/config.json
"""
import argparse, glob, os, sys, warnings
import numpy as np, pandas as pd
from tsrom_common import safe_name, dump, load, file_hash, Scaler

TIME_NAMES = ("time", "Time", "TIME", "t", "t_s", "time_s")
SOURCE_HINTS = ("loss", "power", "input", "u_", "set", "ref", "amb", "table", "source", "command", "cmd", "load",
                "voltage_in", "flow_in", "q_in", "torque", "speed", "y[")


# ----------------------------------------------------------------- loading
def read_om_mat(path):
    """OpenModelica / Dymola result file (MATLAB v4, 'binTrans' or 'binNormal')."""
    from scipy.io import loadmat
    m = loadmat(path, chars_as_strings=False)
    aclass = ["".join(r).strip() for r in m["Aclass"]] if "Aclass" in m else ["", "", "", "binNormal"]
    trans = len(aclass) > 3 and aclass[3] == "binTrans"
    nm = m["name"].T if trans else m["name"]
    names = ["".join(c for c in row if c not in ("\x00",)).rstrip() for row in nm]
    info = m["dataInfo"] if trans else m["dataInfo"].T
    if info.shape[0] != 4 and info.shape[1] == 4:
        info = info.T
    d1, d2 = m.get("data_1"), m["data_2"]
    # orient so that axis 0 = variables (use dataInfo to know how many variables each block holds)
    n2 = int(np.max(np.abs(info[1][info[0] == 2]))) if np.any(info[0] == 2) else d2.shape[0]
    if d2.shape[0] != n2 and d2.shape[1] == n2:
        d2 = d2.T
    if d1 is not None:
        n1 = int(np.max(np.abs(info[1][info[0] == 1]))) if np.any(info[0] == 1) else d1.shape[0]
        if d1.shape[0] != n1 and d1.shape[1] == n1:
            d1 = d1.T
    t = d2[0]
    cols = {"time": t}
    for i, n in enumerate(names):
        src, idx = int(info[0, i]), int(info[1, i])
        if n == "time" or idx == 0:
            continue
        sgn = -1.0 if idx < 0 else 1.0
        row = abs(idx) - 1
        if src == 2:
            cols[n] = sgn * d2[row]
        elif src in (0, 1) and d1 is not None:
            cols[n] = np.full_like(t, sgn * d1[row][0])
    return pd.DataFrame(cols)


def read_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        return read_om_mat(path)
    if ext in (".csv", ".txt"):
        df = pd.read_csv(path, sep=None, engine="python")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    return df


def expand(data):
    files = []
    for item in (data if isinstance(data, list) else [data]):
        hits = sorted(glob.glob(item))
        files += hits if hits else [item]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing data files: {missing}")
    return files


def find_time_col(df, hint=None):
    if hint and hint in df.columns:
        return hint
    for n in TIME_NAMES:
        if n in df.columns:
            return n
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce").values
        if np.isfinite(v).all() and np.all(np.diff(v) >= 0) and v[-1] > v[0]:
            return c
    raise ValueError("No time column found (no monotonic numeric column). Specify dataset.time_column.")


def load_runs(files, time_col=None, run_col=None):
    """Returns list of (run_name, DataFrame). One file = one run, unless run_col splits a file."""
    runs = []
    for f in files:
        df = read_any(f)
        base = os.path.splitext(os.path.basename(f))[0]
        if run_col and run_col in df.columns:
            for k, g in df.groupby(run_col, sort=False):
                runs.append((f"{base}:{k}", g.drop(columns=[run_col]).reset_index(drop=True)))
        else:
            runs.append((base, df))
    tc = find_time_col(runs[0][1], time_col)
    return runs, tc


def clean_run(df, tc):
    """Numeric only, sorted, duplicate timestamps (event points) collapsed to the post-event value, NaNs handled."""
    rep = {}
    df = df.apply(pd.to_numeric, errors="coerce")
    all_nan = [c for c in df.columns if df[c].isna().all()]
    if all_nan:
        rep["dropped_empty_columns"] = all_nan
        df = df.drop(columns=all_nan)
    df = df.sort_values(tc, kind="mergesort")
    n0 = len(df)
    df = df.groupby(tc, sort=True, as_index=False).last()
    if len(df) < n0:
        rep["duplicate_timestamps_removed"] = n0 - len(df)
    nan_rows = int(df.isna().any(axis=1).sum())
    if nan_rows:
        rep["rows_with_nan_interpolated"] = nan_rows
        df = df.interpolate(limit_direction="both")
    return df.reset_index(drop=True), rep


# ----------------------------------------------------------------- profiling
def unit_guess(name, v):
    n = name.lower()
    if "degc" in n or "celsius" in n:
        return "degC"
    if n.endswith(".t") or n.startswith("t") and ("temp" in n or len(n) < 12):
        if 180 < np.nanmedian(v) < 2500:
            return "K?"
        if -60 < np.nanmedian(v) < 180:
            return "degC?"
    return ""


def profile(files, time_col=None, run_col=None):
    runs, tc = load_runs(files, time_col, run_col)
    cleaned, crep = [], {}
    for name, df in runs:
        d, r = clean_run(df, tc)
        cleaned.append((name, d)); crep[name] = r
    cols = [c for c in cleaned[0][1].columns if c != tc]
    common = [c for c in cols if all(c in d.columns for _, d in cleaned)]
    allv = {c: np.concatenate([d[c].values for _, d in cleaned]) for c in common}
    dts = np.concatenate([np.diff(d[tc].values) for _, d in cleaned])
    prof = dict(files=[dict(path=f, sha256_16=file_hash(f)) for f in files], time_column=tc, n_runs=len(cleaned),
                run_names=[n for n, _ in cleaned], rows_per_run=[len(d) for _, d in cleaned],
                duration_per_run=[float(d[tc].iloc[-1] - d[tc].iloc[0]) for _, d in cleaned],
                dt=dict(min=float(dts.min()), median=float(np.median(dts)), max=float(dts.max()),
                        uniform=bool(np.allclose(dts, np.median(dts), rtol=1e-3))),
                cleaning=crep, columns={}, lookalikes=[], warnings=[])
    for c in common:
        v = allv[c]
        dv = np.diff(v)
        rng = float(np.nanmax(v) - np.nanmin(v))
        const = rng < 1e-9 * max(1.0, abs(float(np.nanmean(v))))
        pwc = float(np.mean(np.abs(dv) < 1e-12 * max(1.0, rng))) if len(dv) else 1.0
        hint = any(h in c.lower() for h in SOURCE_HINTS)
        role = "constant" if const else ("input?" if (pwc > 0.85 or hint) else "output?")
        per_run_const = all(np.ptp(d[c].values) < 1e-9 * max(1, abs(d[c].values[0])) for _, d in cleaned)
        prof["columns"][c] = dict(min=float(np.nanmin(v)), max=float(np.nanmax(v)), mean=float(np.nanmean(v)),
                                  std=float(np.nanstd(v)), constant=bool(const), piecewise_constant_frac=round(pwc, 3),
                                  constant_within_each_run=bool(per_run_const and not const),
                                  unit_guess=unit_guess(c, v), suggested_role=role,
                                  safe_name=safe_name(c))
    # lookalikes: same signal with an offset/scale (e.g. K vs degC sensor, alias variables)
    keys = [c for c in common if not prof["columns"][c]["constant"]]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            x, y = allv[a], allv[b]
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                continue
            r = np.corrcoef(x, y)[0, 1]
            if r > 0.999999:
                k, off = np.polyfit(x, y, 1)
                kind = "alias (identical)" if abs(k - 1) < 1e-6 and abs(off) < 1e-6 else \
                       ("same temperature, K vs degC" if abs(k - 1) < 1e-6 and abs(abs(off) - 273.15) < 1e-3 else
                        f"linear copy y = {k:.6g}*x + {off:.6g}")
                prof["lookalikes"].append(dict(a=a, b=b, relation=kind))
    if not prof["dt"]["uniform"]:
        prof["warnings"].append("Non-uniform time grid: data will be resampled to a uniform sample time.")
    if prof["lookalikes"]:
        prof["warnings"].append("Lookalike columns found: pick ONE of each pair (prefer SI units, e.g. Kelvin).")
    if len(cleaned) == 1:
        prof["warnings"].append("Only one run: train/val/test will be contiguous time blocks; results are less trustworthy. "
                                "More runs with varied inputs/initial states are strongly recommended.")
    names = [safe_name(c) for c in common]
    if len(set(names)) < len(names):
        prof["warnings"].append("Some column names collide after sanitizing; unique suffixes will be added.")
    return prof


# ----------------------------------------------------------------- preparation
def estimate_time_constants(runs, y_cols, u_cols, tc):
    """Least-squares linear fit dy/dt = A y + B u + c on raw data -> eigenvalues -> time constants."""
    Ys, Us, dY = [], [], []
    for _, d in runs:
        t = d[tc].values; Y = d[y_cols].values; U = d[u_cols].values if u_cols else np.zeros((len(t), 0))
        dt = np.diff(t); ok = dt > 0
        dy = np.diff(Y, axis=0)[ok] / dt[ok, None]
        Ys.append(Y[:-1][ok]); Us.append(U[:-1][ok]); dY.append(dy)
    Y, U, dY = np.vstack(Ys), np.vstack(Us), np.vstack(dY)
    sy = Scaler(Y); su = Scaler(U)
    X = np.hstack([sy.f(Y), su.f(U), np.ones((len(Y), 1))])
    coef, *_ = np.linalg.lstsq(X, dY / sy.s, rcond=None)
    A = coef[:len(y_cols)].T
    lam = np.linalg.eigvals(A)
    re = np.abs(lam.real); re = re[re > 1e-12]
    if len(re) == 0:
        return None, None, lam
    taus = 1.0 / re
    return float(taus.min()), float(taus.max()), lam


def resample(d, tc, y_cols, u_cols, dt, input_hold=True):
    t0, t1 = d[tc].values[0], d[tc].values[-1]
    tg = np.arange(t0, t1 + dt * 1e-6, dt)
    t = d[tc].values
    Y = np.column_stack([np.interp(tg, t, d[c].values) for c in y_cols])
    if u_cols:
        if input_hold:   # zero-order hold: value at or before tg (post-event value at switching instants)
            idx = np.clip(np.searchsorted(t, tg + 1e-9 * dt, side="right") - 1, 0, len(t) - 1)
            U = np.column_stack([d[c].values[idx] for c in u_cols])
        else:
            U = np.column_stack([np.interp(tg, t, d[c].values) for c in u_cols])
    else:
        U = np.zeros((len(tg), 0))
    return tg - t0, U, Y


def choose_dt(runs, tc, ins, data_dt, tmin):
    """Coarsest multiple k of the data step such that (a) >= 10 samples per fastest time constant and
    (b) >= 99% of input switching instants fall on the coarse grid (no aliasing of input steps)."""
    if not tmin:
        return data_dt
    kmax = max(1, int(tmin / 20.0 / data_dt))   # >= 20 samples per fastest time constant
    events = []
    for _, d in runs:
        t = d[tc].values
        if ins:
            ch = np.where(np.any(np.abs(np.diff(d[ins].values, axis=0)) > 0, axis=1))[0] + 1
            events.append(np.round((t[ch] - t[0]) / data_dt).astype(int))
    ev = np.concatenate(events) if events else np.zeros(0, int)
    steps = [int(round((d[tc].values[-1] - d[tc].values[0]) / data_dt)) for _, d in runs]
    ok = []
    for k in range(kmax, 0, -1):
        if len(ev) == 0 or np.mean(ev % k == 0) >= 0.99 or len(ev) > 0.2 * sum(len(d) for _, d in runs):
            ok.append(k)
    if not ok:
        return data_dt
    divides = [k for k in ok if all(s % k == 0 for s in steps)]   # prefer a step that ends exactly on the run end
    k = divides[0] if divides and divides[0] >= ok[0] / 2 else ok[0]
    return float(np.round(k * data_dt, 12))


def nice_dt(target, data_dt):
    k = max(1, int(round(target / data_dt)))
    return k * data_dt


def prepare(cfg_path):
    cfg = load(cfg_path)
    base = os.path.dirname(os.path.abspath(cfg_path))
    wd = os.path.join(base, cfg.get("work_dir", "."))
    ds = cfg["dataset"]
    files = expand([f if os.path.isabs(f) else os.path.join(base, f) for f in ds["files"]])
    runs, tc = load_runs(files, ds.get("time_column"), ds.get("run_column"))
    rep = dict(warnings=[], errors=[], cleaning={})
    cl = []
    for name, df in runs:
        d, r = clean_run(df, tc); rep["cleaning"][name] = r
        for conv in cfg.get("unit_conversions", []):          # e.g. {"column":"Tsensor1.T","op":"degC_to_K"}
            if conv["column"] in d.columns:
                d[conv["column"]] = d[conv["column"]] + (273.15 if conv["op"] == "degC_to_K" else -273.15)
        cl.append((name, d))
    ins = [c["column"] for c in cfg["inputs"]]
    outs = [c["column"] for c in cfg["outputs"]]
    for name, d in cl:
        miss = [c for c in ins + outs if c not in d.columns]
        if miss:
            raise ValueError(f"Run {name} is missing columns {miss}")
    overlap = set(ins) & set(outs)
    if overlap:
        raise ValueError(f"Columns cannot be both input and output: {overlap}")
    # constant checks
    allY = np.vstack([d[outs].values for _, d in cl])
    allU = np.vstack([d[ins].values for _, d in cl]) if ins else np.zeros((len(allY), 0))
    for j, c in enumerate(outs):
        if np.ptp(allY[:, j]) < 1e-9 * max(1, abs(allY[:, j].mean())):
            rep["errors"].append(f"Output '{c}' is constant in all data - nothing to learn. Remove it.")
    for j, c in enumerate(ins):
        if np.ptp(allU[:, j]) < 1e-9 * max(1, abs(allU[:, j].mean())):
            rep["warnings"].append(f"Input '{c}' never varies - the ROM cannot learn its effect (only valid at this value).")
    # time constants & sample time
    data_dt = float(np.median(np.concatenate([np.diff(d[tc].values) for _, d in cl])))
    tmin, tmax, lam = estimate_time_constants(cl, outs, ins, tc)
    rep["data_dt"] = data_dt
    rep["linear_fit_eigenvalues"] = [complex(x).__repr__() for x in lam]
    rep["tau_min"], rep["tau_max"] = tmin, tmax
    if cfg.get("sample_time"):
        dt = float(cfg["sample_time"]); rep["sample_time_source"] = "config"
    else:
        dt = choose_dt(cl, tc, ins, data_dt, tmin); rep["sample_time_source"] = \
            "auto: coarsest multiple of the data step with >= 20 samples per fastest time constant and inputs switching on the grid"
    rep["sample_time"] = dt
    if tmin and tmax and tmax / tmin > 1000:
        rep["warnings"].append(f"Multi-time-scale / stiff system (tau_max/tau_min = {tmax/tmin:.0f}). Consider separate "
                               "ROMs per time scale or treating the fast part as quasi-static.")
    # resample + names
    used = set()
    in_names = [c.get("name") or safe_name(c["column"]) for c in cfg["inputs"]]
    out_names = [c.get("name") or safe_name(c["column"]) for c in cfg["outputs"]]
    in_names = [safe_name(n, used) for n in in_names]; out_names = [safe_name(n, used) for n in out_names]
    hold = cfg.get("input_interpolation", "hold") == "hold"
    data = {}
    for name, d in cl:
        t, U, Y = resample(d, tc, outs, ins, dt, hold)
        data[name] = (t, U, Y)
    # splits
    names = [n for n, _ in cl]
    test = cfg.get("test_runs") or ([names[-1]] if len(names) > 2 else [])
    rest = [n for n in names if n not in test]
    if len(names) <= 2:
        rep["warnings"].append("<=2 runs: using contiguous blocks of each run for val/test.")
        split = {"train": [], "val": [], "test": []}
        for n in names:
            t, U, Y = data[n]; N = len(t); a, b = int(.7 * N), int(.85 * N)
            split["train"].append((n + "#train", t[:a], U[:a], Y[:a]))
            split["val"].append((n + "#val", t[a:b] - t[a], U[a:b], Y[a:b]))
            split["test"].append((n + "#test", t[b:] - t[b], U[b:], Y[b:]))
    else:
        rng = np.random.default_rng(cfg.get("seed", 0))
        perm = list(rng.permutation(rest)); nv = max(1, int(round(0.2 * len(rest))))
        val = cfg.get("val_runs") or perm[:nv]
        train = [n for n in rest if n not in val]
        split = {k: [(n, *data[n]) for n in v] for k, v in (("train", train), ("val", val), ("test", test))}
    # coverage
    trY = np.vstack([s[3] for s in split["train"]]); trU = np.vstack([s[2] for s in split["train"]])
    cov = {}
    for part in ("val", "test"):
        for (n, t, U, Y) in split[part]:
            for j, c in enumerate(out_names):
                lo, hi = trY[:, j].min(), trY[:, j].max(); r = hi - lo
                frac = float(np.mean((Y[:, j] < lo - .02 * r) | (Y[:, j] > hi + .02 * r)))
                if frac > 0:
                    cov.setdefault(n, {})[c] = round(frac, 4)
            for j, c in enumerate(in_names):
                lo, hi = trU[:, j].min(), trU[:, j].max(); r = max(hi - lo, 1e-9)
                frac = float(np.mean((U[:, j] < lo - .02 * r) | (U[:, j] > hi + .02 * r)))
                if frac > 0:
                    cov.setdefault(n, {})[c] = round(frac, 4)
    rep["coverage_outside_training_range"] = cov
    if cov:
        rep["warnings"].append("Some val/test samples are outside the training range (extrapolation): " +
                               "; ".join(f"{k}: {v}" for k, v in cov.items()) + ". Add training runs that cover them.")
    dur = sum(s[1][-1] - s[1][0] for s in split["train"])
    rep["train_duration_over_tau_max"] = float(dur / tmax) if tmax else None
    if tmax and dur / tmax < 20:
        rep["warnings"].append(f"Training data spans only {dur/tmax:.1f} x the slowest time constant (want >= 20).")
    for k in ("train", "val", "test"):
        if not split[k]:
            rep["errors"].append(f"No {k} data after splitting.")
    rep["splits"] = {k: [s[0] for s in v] for k, v in split.items()}
    rep["n_samples"] = {k: int(sum(len(s[1]) for s in v)) for k, v in split.items()}
    meta = dict(inputs=in_names, outputs=out_names, input_columns=ins, output_columns=outs, dt=dt,
                tau_min=tmin, tau_max=tmax,
                ranges=dict(u_min=trU.min(0).tolist() if ins else [], u_max=trU.max(0).tolist() if ins else [],
                            y_min=trY.min(0).tolist(), y_max=trY.max(0).tolist()),
                units=dict(inputs=[c.get("unit", "") for c in cfg["inputs"]],
                           outputs=[c.get("unit", "") for c in cfg["outputs"]]))
    import pickle
    with open(os.path.join(wd, "prepared.pkl"), "wb") as f:
        pickle.dump(dict(meta=meta, split=split), f)
    dump(rep, os.path.join(wd, "prep_report.json"))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["profile", "prepare"])
    ap.add_argument("--data", nargs="*"); ap.add_argument("--out"); ap.add_argument("--config")
    ap.add_argument("--time-column"); ap.add_argument("--run-column")
    a = ap.parse_args()
    if a.cmd == "profile":
        p = profile(expand(a.data), a.time_column, a.run_column)
        dump(p, a.out); print(f"profile written: {a.out}  runs={p['n_runs']}  warnings={len(p['warnings'])}")
    else:
        r = prepare(a.config)
        print(f"prepared: dt={r['sample_time']}  tau=[{r['tau_min']}, {r['tau_max']}]  "
              f"warnings={len(r['warnings'])}  errors={len(r['errors'])}")
        for w in r["warnings"]: print("WARNING:", w)
        for e in r["errors"]: print("ERROR:", e)
        sys.exit(1 if r["errors"] else 0)
