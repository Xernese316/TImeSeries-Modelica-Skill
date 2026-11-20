"""Shared helpers: safe Modelica names, robust scaling, metrics, JSON I/O."""
import json, re, hashlib, os
import numpy as np

MODELICA_KEYWORDS = set("""algorithm and annotation block break class connect connector constant constrainedby der
discrete each else elseif elsewhen encapsulated end enumeration equation expandable extends external false final flow
for function if import impure in initial inner input loop model not operator or outer output package parameter partial
protected public pure record redeclare replaceable return stream then true type when while within time
Real Integer Boolean String sample pre edge change reinit terminal assert size cat fill zeros ones abs sign sqrt
sin cos tan asin acos atan atan2 sinh cosh tanh exp log log10 min max sum product transpose""".split())


def safe_name(raw, used=None):
    """Turn any column name into a legal, unique Modelica identifier."""
    s = re.sub(r"[^0-9A-Za-z_]", "_", str(raw).strip())
    s = re.sub(r"_+", "_", s).strip("_") or "var"
    if s[0].isdigit():
        s = "v_" + s
    if s in MODELICA_KEYWORDS:
        s = s + "_"
    if used is not None:
        base, k = s, 2
        while s in used:
            s = f"{base}_{k}"; k += 1
        used.add(s)
    return s


def safe_package_name(raw):
    s = safe_name(raw)
    return s[0].upper() + s[1:]


class Scaler:
    """Mean/std scaler that never divides by ~0 (near-constant columns get scale 1)."""
    def __init__(self, X=None, m=None, s=None):
        if X is not None:
            X = np.asarray(X, float)
            if X.shape[1] == 0:
                m, s = np.zeros(0), np.ones(0)
            else:
                m = X.mean(0); s = X.std(0)
                tiny = s < 1e-6 * np.maximum(1.0, np.abs(m))
                s = np.where(tiny, 1.0, s)
        self.m, self.s = np.asarray(m, float), np.asarray(s, float)
    def f(self, X): return (X - self.m) / self.s
    def inv(self, X): return X * self.s + self.m
    def to_dict(self): return dict(m=self.m.tolist(), s=self.s.tolist())
    @classmethod
    def from_dict(cls, d): return cls(m=np.array(d["m"]), s=np.array(d["s"]))


def metrics(Yp, Y, names=None, ref_range=None):
    """Per-output free-run metrics. NRMSE and max error are relative to ref_range (the output's range over the
    whole training set) so that weakly excited runs do not inflate percentages; falls back to this run's range."""
    Yp, Y = np.asarray(Yp, float), np.asarray(Y, float)
    err = Yp - Y
    rng = np.asarray(ref_range, float) if ref_range is not None else Y.max(0) - Y.min(0)
    rng = np.where(rng < 1e-9, np.maximum(np.abs(Y).max(0), 1.0), rng)
    finite = bool(np.isfinite(Yp).all())
    if not finite:
        err = np.nan_to_num(err, nan=1e6, posinf=1e6, neginf=-1e6)
    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    sst = np.where(sst < 1e-12, 1.0, sst)
    out = dict(
        finite=finite,
        nrmse_pct=(100 * np.sqrt((err ** 2).mean(0)) / rng).tolist(),
        max_abs=np.abs(err).max(0).tolist(),
        max_abs_pct_range=(100 * np.abs(err).max(0) / rng).tolist(),
        r2=(1 - (err ** 2).sum(0) / sst).tolist(),
        bias=err.mean(0).tolist(),
        steady_state_err=err[-max(1, len(err) // 20):].mean(0).tolist(),
    )
    if names:
        out["outputs"] = list(names)
    return out


def dump(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load(path):
    with open(path) as f:
        return json.load(f)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
