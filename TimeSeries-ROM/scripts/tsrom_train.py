"""Model ladder: linear -> Neural ODE -> latent Neural ODE -> GRU, trained on free-run (rollout) loss,
tuned with Optuna on free-run validation error, gated, diagnosed, stability-tested, seed-checked.

CLI: python tsrom_train.py --config work/config.json
"""
import argparse, os, sys, time, pickle, copy
import numpy as np, torch, torch.nn as nn, optuna
from tsrom_common import Scaler, metrics, dump, load

optuna.logging.set_verbosity(optuna.logging.WARNING)
DEFAULT_GATES = dict(nrmse_pct=1.0, max_abs_pct_range=5.0)
LADDER = ["linear", "node", "node_latent", "gru"]


# ------------------------------------------------------------------ models
class ODEFunc(nn.Module):
    def __init__(self, nx, nu, hidden, layers):
        super().__init__()
        self.lin = nn.Linear(nx + nu, nx)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)
        self.mlp = None
        if hidden:
            mods, d = [], nx + nu
            for _ in range(layers):
                mods += [nn.Linear(d, hidden), nn.Tanh()]; d = hidden
            last = nn.Linear(d, nx, bias=False)          # bias lives in self.lin only (no redundant biases)
            nn.init.zeros_(last.weight)
            self.mlp = nn.Sequential(*mods, last)
    def forward(self, x, u):
        z = torch.cat([x, u], -1)
        return self.lin(z) + (self.mlp(z) if self.mlp is not None else 0)


class ODEModel(nn.Module):
    """Continuous time, states x = [y (outputs, scaled) ; z (latent)], time unit = one sample step.
    Latent init z0 = tanh(E [y history; u history] + e) from the last K samples (padded with the first sample)."""
    kind = "ode"
    def __init__(self, ny, nu, hidden=0, layers=1, nz=0, K=1):
        super().__init__()
        self.ny, self.nu, self.nz, self.K = ny, nu, nz, K
        self.f = ODEFunc(ny + nz, nu, hidden, layers)
        self.enc = nn.Linear(K * (ny + nu), nz) if nz else None
    def init_state(self, yh, uh):           # yh (B,K,ny), uh (B,K,nu)
        y0 = yh[:, -1]
        if not self.nz:
            return y0
        z0 = torch.tanh(self.enc(torch.cat([yh.flatten(1), uh.flatten(1)], -1)))
        return torch.cat([y0, z0], -1)
    def rollout(self, yh, uh, U):           # U (B,H,nu) -> (B,H+1,ny)
        x = self.init_state(yh, uh); ys = [x[:, :self.ny]]
        for k in range(U.shape[1]):
            u = U[:, k]
            k1 = self.f(x, u); k2 = self.f(x + .5 * k1, u); k3 = self.f(x + .5 * k2, u); k4 = self.f(x + k3, u)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            ys.append(x[:, :self.ny])
        return torch.stack(ys, 1)


class GRUModel(nn.Module):
    """Discrete time (sample time dt). h0 = tanh(E [y0; u0] + e), h[k+1] = GRU(u[k], h[k]), y = C h + d."""
    kind = "gru"
    def __init__(self, ny, nu, hidden=16, **_):
        super().__init__()
        self.ny, self.nu, self.K = ny, nu, 1
        self.enc = nn.Linear(ny + nu, hidden)
        self.cell = nn.GRUCell(max(nu, 1), hidden)
        self.out = nn.Linear(hidden, ny)
    def rollout(self, yh, uh, U):
        h = torch.tanh(self.enc(torch.cat([yh[:, -1], uh[:, -1]], -1))); ys = [self.out(h)]
        for k in range(U.shape[1]):
            u = U[:, k] if self.nu else torch.zeros(U.shape[0], 1)
            h = self.cell(u, h); ys.append(self.out(h))
        return torch.stack(ys, 1)


def ls_init(model, trajs):
    """Least-squares warm start of the linear part: dy/dt (per step, scaled) ~ A y + B u + c, from central differences."""
    if getattr(model, "kind", "") != "ode":
        return model
    X, D = [], []
    for U, Y in trajs:
        if len(Y) < 3:
            continue
        D.append((Y[2:] - Y[:-2]) / 2.0); X.append(np.hstack([Y[1:-1], U[1:-1]]))
    X, D = np.vstack(X), np.vstack(D)
    ny = model.ny; nz = model.nz
    A = np.hstack([X, np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(A, D, rcond=None)           # (ny+nu+1, ny)
    Wfull = np.zeros((ny + nz, ny + nz + model.nu)); b = np.zeros(ny + nz)
    Wfull[:ny, :ny] = coef[:ny].T
    Wfull[:ny, ny + nz:] = coef[ny:ny + model.nu].T
    b[:ny] = coef[-1]
    if nz:
        Wfull[ny:, ny:ny + nz] = -0.1 * np.eye(nz)            # latent states start as slow stable modes
    with torch.no_grad():
        model.f.lin.weight.copy_(T(Wfull)); model.f.lin.bias.copy_(T(b))
    return model


def build(family, ny, nu, p):
    if family == "linear":
        return ODEModel(ny, nu)
    if family == "node":
        return ODEModel(ny, nu, p["hidden"], p["layers"])
    if family == "node_latent":
        return ODEModel(ny, nu, p["hidden"], p["layers"], p["nz"], p["K"])
    if family == "gru":
        return GRUModel(ny, nu, p["hidden"])
    raise ValueError(family)


# ------------------------------------------------------------------ data windows
T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)


def history(U, Y, s, K):
    idx = np.clip(np.arange(s - K + 1, s + 1), 0, None)
    return Y[idx], U[idx]


def windows(trajs, H, n, K, rng, p_start=0.25):
    Yh, Uh, Us, Ys = [], [], [], []
    for i in rng.integers(0, len(trajs), n):
        U, Y = trajs[i]; Hc = min(H, len(Y) - 1)
        s = 0 if rng.random() < p_start else int(rng.integers(0, len(Y) - Hc))
        yh, uh = history(U, Y, s, K)
        Yh.append(yh); Uh.append(uh)
        u = U[s:s + Hc]; y = Y[s:s + Hc + 1]
        if Hc < H:   # pad short trajectories (rare) by repeating last sample; masked by equal values
            u = np.vstack([u, np.repeat(u[-1:], H - Hc, 0)]); y = np.vstack([y, np.repeat(y[-1:], H - Hc, 0)])
        Us.append(u); Ys.append(y)
    return T(Yh), T(Uh), T(Us), T(Ys)


@torch.no_grad()
def free_run(model, U, Y, K):
    yh, uh = history(U, Y, 0, K)
    return model.rollout(T(yh)[None], T(uh)[None], T(U[:-1])[None])[0].numpy()


def train(model, trajs, p, rng, iters, val_fn=None, trial=None):
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p.get("wd", 0.0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    Hmax = p["horizon"]
    stages = [h for h in (4, 16, 64, 256, 1024) if h < Hmax] + [Hmax]
    K = getattr(model, "K", 1)
    for it in range(iters):
        H = stages[min(it * len(stages) // iters, len(stages) - 1)]
        yh, uh, U, Y = windows(trajs, H, p["batch"], K, rng)
        if p["noise"] > 0:
            yh = yh + p["noise"] * torch.randn_like(yh)
        pred = model.rollout(yh, uh, U)
        loss = ((pred - Y) ** 2).mean()
        if not torch.isfinite(loss):
            raise optuna.TrialPruned() if trial else FloatingPointError("non-finite loss")
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if trial is not None and val_fn is not None and (it + 1) % max(1, iters // 4) == 0 and it + 1 < iters:
            v = val_fn(model); trial.report(v, it)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return model


# ------------------------------------------------------------------ evaluation, diagnostics, stability
REF_RANGE = None   # set in run(): output ranges over the training set


def evaluate(model, trajs, sy, K, names):
    res = []
    for U, Y in trajs:
        Yp = sy.inv(free_run(model, U, Y, K)); Yt = sy.inv(Y)
        res.append((Yp, Yt, metrics(Yp, Yt, names, REF_RANGE)))
    return res


def score(evals):
    """Scalar objective: mean over runs of the worst output NRMSE (penalise any output being bad)."""
    vals = [max(m["nrmse_pct"]) if m["finite"] else 1e6 for _, _, m in evals]
    return float(np.mean(vals))


def agg(evals):
    """Worst case over runs, per output."""
    ms = [m for _, _, m in evals]
    return dict(finite=all(m["finite"] for m in ms),
                nrmse_pct=np.max([m["nrmse_pct"] for m in ms], 0).tolist(),
                max_abs=np.max([m["max_abs"] for m in ms], 0).tolist(),
                max_abs_pct_range=np.max([m["max_abs_pct_range"] for m in ms], 0).tolist(),
                r2=np.min([m["r2"] for m in ms], 0).tolist(),
                mean_nrmse_pct=np.mean([m["nrmse_pct"] for m in ms], 0).tolist())


def passes(a, gates):
    return bool(a["finite"] and max(a["nrmse_pct"]) <= gates["nrmse_pct"]
                and max(a["max_abs_pct_range"]) <= gates["max_abs_pct_range"])


def diagnose(evals, trajs_u, su, names, in_names, run_names=None):
    """Explain WHERE the error is and suggest the fix (bias / drift / initialization / regime / specific runs)."""
    out = []
    err = np.vstack([Yp - Yt for Yp, Yt, _ in evals]); Ut = np.vstack([su.inv(U) for U in trajs_u])
    for j, n in enumerate(names):
        e = err[:, j]; rmse = np.sqrt((e ** 2).mean()) + 1e-12
        if abs(e.mean()) / rmse > 0.5:
            out.append(dict(output=n, issue="bias", detail=f"largely a constant offset ({e.mean():.4g})",
                            action="add latent states (node_latent) or check for a missing input"))
        grow, early = [], []
        for Yp, Yt, _ in evals:
            ee = np.abs(Yp[:, j] - Yt[:, j]); N = len(ee); q = max(2, N // 10)
            early.append(np.sqrt((ee[:q] ** 2).mean()) / (np.sqrt((ee[q:] ** 2).mean()) + 1e-12))
            grow.append(ee[-N // 4:].mean() / (ee[:N // 2].mean() + 1e-12))
        if np.median(grow) > 2:
            out.append(dict(output=n, issue="drift", detail=f"error grows ~{np.median(grow):.1f}x towards the end of runs",
                            action="longer rollout horizon / more iterations / check stability"))
        if np.median(early) > 2:
            out.append(dict(output=n, issue="initialization",
                            detail=f"error concentrated in the first 10% of runs (~{np.median(early):.1f}x the rest)",
                            action="the initial internal state is not identifiable from the outputs: log the missing "
                                   "state(s) as extra outputs, or start training/usage runs from a steady state"))
        for k, iname in enumerate(in_names):
            u = Ut[:, k]
            if np.ptp(u) < 1e-12:
                continue
            bins = np.quantile(u, [0, .25, .5, .75, 1])
            r = [np.sqrt((e[(u >= bins[b]) & (u <= bins[b + 1])] ** 2).mean()) for b in range(4)]
            r = [x for x in r if np.isfinite(x)]
            if r and max(r) / (min(r) + 1e-12) > 3:
                out.append(dict(output=n, issue="regime", detail=f"error concentrated in one range of {iname}",
                                action=f"more training data covering that range of {iname}"))
        per_run = np.array([m["nrmse_pct"][j] for _, _, m in evals])
        if len(per_run) > 2 and per_run.max() > 3 * np.median(per_run):
            worst = int(per_run.argmax())
            nm = run_names[worst] if run_names else f"#{worst}"
            out.append(dict(output=n, issue="specific runs",
                            detail=f"run {nm} is {per_run.max()/np.median(per_run):.1f}x worse than the median run",
                            action="inspect that run (unusual operating point / initial state); add similar runs"))
    if not out:
        out.append(dict(output="all", issue="underfit", detail="error is spread over all runs and times",
                        action="more capacity / iterations (next ladder rung); if latent models also fail, the "
                               "outputs probably do not carry enough information: log the missing internal states"))
    return out


def stability_tests(model, meta, sy, su, K, dt, val_trajs):
    res = {}
    ymin, ymax = np.array(meta["ranges"]["y_min"]), np.array(meta["ranges"]["y_max"]); R = ymax - ymin
    # 1) long run: longest val inputs repeated 3x
    U, Y = max(val_trajs, key=lambda t: len(t[0]))
    U3 = np.vstack([U] * 3); Y3 = np.vstack([Y] + [np.repeat(Y[-1:], len(Y), 0)] * 2)
    Yp = sy.inv(free_run(model, U3, Y3, K))
    ok = bool(np.isfinite(Yp).all() and (Yp >= ymin - .5 * R).all() and (Yp <= ymax + .5 * R).all())
    res["long_run_3x"] = dict(passed=ok, min=Yp.min(0).tolist(), max=Yp.max(0).tolist())
    # 2) constant input hold: must settle (no oscillation / runaway)
    tau = meta.get("tau_max") or dt * 100
    n = int(min(20000, max(200, 10 * tau / dt)))
    u_mid = su.f(np.array([(a + b) / 2 for a, b in zip(meta["ranges"]["u_min"], meta["ranges"]["u_max"])])) \
        if meta["inputs"] else np.zeros(0)
    Uc = np.repeat(u_mid[None], n, 0); Yc = np.repeat(sy.f(((ymin + ymax) / 2))[None], n, 0)
    Yp = sy.inv(free_run(model, Uc, Yc, K))
    tail = Yp[-max(5, n // 20):]
    settled = bool(np.isfinite(Yp).all() and (np.ptp(tail, 0) < 0.005 * R + 1e-9).all())
    bounded = bool(np.isfinite(Yp).all() and (Yp >= ymin - .5 * R).all() and (Yp <= ymax + .5 * R).all())
    res["constant_input_settles"] = dict(passed=settled and bounded, settled=settled, bounded=bounded,
                                        final=Yp[-1].tolist())
    res["passed"] = all(v["passed"] for v in res.values() if isinstance(v, dict))
    return res


# ------------------------------------------------------------------ ladder
def search_space(family, trial, Tlen):
    hz = [h for h in (32, 64, 128, 256) if h < Tlen] or [Tlen - 1]
    p = dict(lr=trial.suggest_float("lr", 1e-3, 2e-2, log=True),
             horizon=trial.suggest_categorical("horizon", hz),
             noise=trial.suggest_float("noise", 0.0, 0.05))
    if family in ("node", "node_latent"):
        p["hidden"] = trial.suggest_categorical("hidden", [8, 16, 32, 64])
        p["layers"] = trial.suggest_int("layers", 1, 2)
    if family == "node_latent":
        p["nz"] = trial.suggest_categorical("nz", [1, 2, 4])
        p["K"] = trial.suggest_categorical("K", [4, 8, 16])
    if family == "gru":
        p["hidden"] = trial.suggest_categorical("hidden", [8, 16, 32])
    return p


def run(cfg_path):
    cfg = load(cfg_path)
    wd = os.path.join(os.path.dirname(os.path.abspath(cfg_path)), cfg.get("work_dir", "."))
    torch.set_num_threads(cfg.get("threads", max(1, (os.cpu_count() or 2))))
    P = pickle.load(open(os.path.join(wd, "prepared.pkl"), "rb"))
    meta, split = P["meta"], P["split"]
    gates = dict(DEFAULT_GATES, **cfg.get("gates", {}))
    budget = dict(dict(trials=8, iters=800, seeds=3), **cfg.get("budget", {}))
    families = cfg.get("families", LADDER)
    ny, nu = len(meta["outputs"]), len(meta["inputs"])
    allY = np.vstack([s[3] for s in split["train"]]); allU = np.vstack([s[2] for s in split["train"]])
    sy, su = Scaler(allY), Scaler(allU)
    global REF_RANGE
    REF_RANGE = allY.max(0) - allY.min(0)
    sc = lambda part: [(su.f(U), sy.f(Y)) for (_, t, U, Y) in split[part]]
    tr, va, te = sc("train"), sc("val"), sc("test")
    Tlen = min(len(Y) for _, Y in tr)
    names, in_names = meta["outputs"], meta["inputs"]
    report = dict(gates=gates, budget=budget, ladder=[], dt=meta["dt"])
    t_start = time.time()

    def fit(family, p, seed, trial=None):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        m = ls_init(build(family, ny, nu, p), tr)
        K = getattr(m, "K", 1)
        iters = budget["iters"] * (2 if family == "gru" else 1)
        val_fn = lambda mm: score(evaluate(mm, va, sy, K, names))
        init_state = copy.deepcopy(m.state_dict()) if m.kind == "ode" else None
        init_ev = evaluate(m, va, sy, K, names) if init_state else None
        try:
            m = train(m, tr, dict(dict(batch=64, wd=0.0), **p), rng, iters, val_fn, trial)
            ev = evaluate(m, va, sy, K, names)
        except FloatingPointError:
            ev = None
        if init_ev is not None and (ev is None or score(init_ev) < score(ev)):   # never worse than the LS warm start
            m.load_state_dict(init_state); ev = init_ev
        if ev is None:
            return m, 1e6, None
        return m, score(ev), ev

    chosen, best_any = None, None
    for family in families:
        t0 = time.time()
        if family == "linear":
            best_p = dict(lr=2e-3, horizon=min(128, Tlen - 1), noise=0.0, hidden=0, layers=1)
            n_done = 1
        else:
            st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0),
                                     pruner=optuna.pruners.MedianPruner(n_startup_trials=3))
            st.optimize(lambda tri: fit(family, search_space(family, tri, Tlen), 0, tri)[1],
                        n_trials=budget["trials"], catch=(RuntimeError, ValueError))
            done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not done:
                report["ladder"].append(dict(family=family, status="all trials failed"))
                continue
            best_p = dict(st.best_trial.params); n_done = len(st.trials)
        # seed check on the best configuration
        runs = []
        for s in range(budget["seeds"]):
            m, v, ev = fit(family, best_p, s)
            if ev is not None:
                runs.append((v, s, m, ev))
        if not runs:
            report["ladder"].append(dict(family=family, status="diverged for every seed")); continue
        runs.sort(key=lambda r: r[0])
        vbest, seed, m, ev = runs[0]
        spread = [r[0] for r in runs]
        a = agg(ev); ok = passes(a, gates)
        entry = dict(family=family, params=best_p, trials=n_done, seed=seed, val=a, val_score=vbest,
                     seed_scores=spread, seed_unstable=bool(len(spread) > 1 and np.median(spread) > 0 and
                                                          (max(spread) - min(spread)) > 0.5 * np.median(spread)),
                     passes_gates=ok, n_weights=int(sum(q.numel() for q in m.parameters())),
                     secs=round(time.time() - t0, 1))
        entry["diagnosis"] = diagnose(ev, [U for U, _ in va], su, names, in_names, [s[0] for s in split["val"]]) if not ok else []
        report["ladder"].append(entry)
        print(f"[{family}] val worst-NRMSE%={max(a['nrmse_pct']):.3f} maxabs%={max(a['max_abs_pct_range']):.2f} "
              f"pass={ok} seeds={np.round(spread,3).tolist()} ({entry['secs']}s)", flush=True)
        for d in entry["diagnosis"]:
            print("   diagnosis:", d, flush=True)
        if best_any is None or vbest < best_any[1]:
            best_any = (family, vbest, m, best_p, entry)
        if ok:
            stab = stability_tests(m, meta, sy, su, getattr(m, "K", 1), meta["dt"], va)
            entry["stability"] = stab
            if stab["passed"]:
                chosen = (family, vbest, m, best_p, entry)
                break
            print("   stability FAILED:", {k: v for k, v in stab.items() if k != 'passed'}, flush=True)
    status = "PASSED"
    if chosen is None:
        if best_any is None:
            raise RuntimeError("Every family failed to train (diverged). Check data scaling / sample time.")
        status = "FAILED_GATES"
        chosen = best_any
        chosen[4]["stability"] = stability_tests(chosen[2], meta, sy, su, getattr(chosen[2], "K", 1), meta["dt"], va)
    family, vbest, m, best_p, entry = chosen
    K = getattr(m, "K", 1)
    tev = evaluate(m, te, sy, K, names)
    report.update(status=status, chosen=family, chosen_params=best_p,
                  test=dict(per_run={n: e[2] for (n, *_), e in zip(split["test"], tev)}, worst=agg(tev)),
                  total_secs=round(time.time() - t_start, 1))
    if status != "PASSED":
        report["next_actions"] = sorted({d["action"] for e in report["ladder"] for d in e.get("diagnosis", [])})
    torch.save(dict(family=family, params=best_p, state=m.state_dict(), sy=sy.to_dict(), su=su.to_dict(),
                    meta=meta, K=K), os.path.join(wd, "model.pt"))
    np.savez(os.path.join(wd, "test_predictions.npz"),
             **{f"pred_{i}": e[0] for i, e in enumerate(tev)}, **{f"true_{i}": e[1] for i, e in enumerate(tev)},
             **{f"t_{i}": s[1] for i, s in enumerate(split["test"])},
             **{f"u_{i}": su.inv(U) for i, (U, _) in enumerate(te)})
    dump(report, os.path.join(wd, "train_report.json"))
    w = report["test"]["worst"]
    print(f"STATUS {status}: {family}  test worst NRMSE% {np.round(w['nrmse_pct'],3).tolist()}  "
          f"max|err| {np.round(w['max_abs'],4).tolist()}", flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    r = run(ap.parse_args().config)
    sys.exit(0 if r["status"] == "PASSED" else 2)
