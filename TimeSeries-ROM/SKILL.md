---
name: timeseries-rom
description: Build a time-series reduced order model (ROM) that replaces a heavy physics model in OpenModelica. Takes simulation data (CSV/.mat) or a heavy Modelica model, generates excitation data with OMC if needed, trains and selects the best dynamic model (linear state-space, Neural ODE, latent Neural ODE, GRU) on free-run error with Optuna, exports a Modelica 4.0.0 library (.mo package + zip) with Layers/Networks/Examples, and verifies it in OMC. Use when the user wants a time-series surrogate, ROM, digital-twin block or "next-value predictor" for Modelica/OpenModelica.
---

# Time-series ROM for OpenModelica

You turn a heavy dynamic model (or its simulation data) into a small, fast, **verified** Modelica block:
`der(y) = f(y, u)` (or a sampled GRU) learned from data, exported as `Networks.SurrogateWrapper`
with RealInputs on the left and RealOutputs on the right.

All numeric work is done by the tested scripts in `scripts/`. **You decide, the scripts execute.**
Your job: understand the data, ask the user ONE confirmation gate, write `config.json`, run the
pipeline, read the reports, fix failures with the playbook, and report the verdict honestly.

## Hard rules

1. **Never edit the scripts** in `scripts/`. Every decision goes into `config.json` / `datagen.json`.
   If a script crashes, report the traceback and the probable cause; do not patch the code.
2. **One confirmation gate** with the user (Step 3). Everything else (model family, hyperparameters,
   sample time, retries) is your decision. Do not ask the user about ML details.
3. **Never loosen the quality gates** (`gates` in config) without the user's explicit approval, and
   never call a ROM ready unless `REPORT.md` says `Verdict: READY` (or `READY_PENDING_OMC` when OMC
   is genuinely unavailable — then say so plainly).
4. Quote numbers only from the generated reports (`REPORT.md`, `train_report.json`,
   `verify_report.json`). Never estimate accuracy or speed-up yourself.
5. Free-run (closed-loop) error is the only accuracy that counts. One-step error is not a result.
6. Ask before installing packages or running anything that writes outside the work folder.

## Workflow

Create a work folder next to the user's data, e.g. `<project>/rom_work/<case>/`. Use forward slashes
in JSON paths. Relative paths in configs are relative to the config file.

### Step 0 — Environment
```
python <skill>/scripts/tsrom.py env
```
Needs Python ≥ 3.9 and `scripts/requirements.txt` (ask, then `pip install -r requirements.txt`).
If `omc` is not found, training/export still work; tell the user OMC verification will be pending and
how to fix it (set `OPENMODELICAHOME` or put `omc` on PATH).

### Step 1 — Get data (two entry modes)

**A. The user gives data files** (CSV, OpenModelica/Dymola `.mat`, XLSX, parquet; one file per run, or a
run-id column). Go to Step 2.

**B. The user gives a heavy Modelica model (or only data from a single simulation).** Generate rich data
with OMC. Read `references/omc-data-generation.md`, write `datagen.json` (which tables/parameters/start
values to randomise and in what ranges), show the ranges in the confirmation gate, then run:
```
python <skill>/scripts/tsrom_datagen.py --spec rom_work/<case>/datagen.json
```
One simulation is almost never enough: ask for / generate ≥ 20 runs that cover the operating envelope,
with ~1/3 starting from rest. The unmodified model is kept as the held-out test run (`original_res`).

### Step 2 — Profile
```
python <skill>/scripts/tsrom.py profile --data "<files or glob>" --out rom_work/<case>/profile.json
```
Read `profile.json`: time column, uniform grid?, duplicate event rows, constant columns, columns constant
within each run (parameters), **lookalike columns** (same signal in K and °C, aliases), suggested roles,
unit guesses, warnings. Decide your proposal:
- **Outputs** = the quantities the ROM must predict (states the heavy model integrates: temperatures,
  speeds, charges, flows). Include every internal state the data exposes that the outputs depend on —
  hidden states are the #1 cause of failure (see playbook).
- **Inputs** = signals the surroundings impose (sources, loads, set-points, ambient conditions, modes).
  An input must not be computed from the outputs inside the heavy model.
- Prefer SI columns (K over °C) and drop duplicates/aliases.

### Step 3 — Confirmation gate (the only user stop)
Present ONE compact table and wait for an explicit "yes" or edits:

| item | proposal |
|---|---|
| Data | N runs, duration, time column, test run |
| Inputs → RealInput names | column → name [unit] (range) for each |
| Outputs → RealOutput names | column → name [unit] (range) for each |
| Modelica package name | e.g. `MotorThermalROM` |
| Interface | plain signals (default) or heat ports on temperature outputs |
| Issues found | lookalikes dropped, constant columns, coverage gaps, hidden-state risk, single-run data |
| (mode B) Excitation | ranges and hold times per randomised signal / parameter / start value |

Say in one line what happens next (automatic training, model choice and verification).

### Step 4 — Write `config.json`
Schema and every option: `references/config.md`. Start from `examples/<case>/config.json`.
Set `reference` when the heavy model is available in OMC — it generates `Examples.CompareToReference`
and the speed-up measurement. Leave `budget`, `families`, `gates` at defaults unless the playbook says otherwise.

### Step 5 — Run the pipeline
```
python <skill>/scripts/tsrom.py run --config rom_work/<case>/config.json
```
Runs prepare → train → export → verify → report. Exit codes: `0` ok · `1` data error (read the ERROR
lines; fix the config or the data) · `2` gates failed (model exported but marked FAILED_GATES) ·
`3` exported Modelica does not reproduce the trained model (stop; report as a bug).
Training prints one line per ladder rung; a typical small case takes 5–30 min on CPU.

### Step 6 — If gates failed: diagnose and retry (max 3 rounds)
Read `train_report.json` → `ladder[*].diagnosis` and `next_actions`, then apply
`references/failure-playbook.md`. Allowed automatic fixes: larger `budget`, different `families`
order, explicit `sample_time`, more/wider data via datagen (mode B), adding exposed states as outputs
(re-confirm with the user only if the RealOutput list changes). After 3 rounds, stop and report the
best model, its numbers, the diagnosis and the concrete data the user should provide.

### Step 7 — Verify in OpenModelica
`run` already calls verify. Check `verify_report.json`:
- `export_fidelity_ok` true (Modelica re-simulation equals the trained model);
- `omc.checks_ok` lists every example; `omc.n_failed_simulations` is 0; no `omc.errors`;
- `omc.omc_rom_vs_heavy_data` within the gates; `omc.omc_compare_to_reference_max_abs_err` small;
- `omc.speedup` (heavy / ROM simulation time). A ROM slower than the heavy model is not useful:
  report it and suggest a smaller family (the ladder's first passing rung is already the cheapest).
If OMC is not available here, give the user `omc_verify/verify.mos` and the one command to run it.

### Step 8 — Deliver
Give the user: the package folder and `.zip` (`rom_work/<case>/package/`), the verdict and the key numbers
from `REPORT.md` (chosen family, worst free-run NRMSE and max |error| per output on the held-out run,
OMC check, speed-up), the validity ranges, and how to use it:
File > Load Library > `<Pkg>/package.mo`; drag `Networks.SurrogateWrapper` in place of the heavy model;
run `Examples.ValidateAgainstData` and plot `err_*`. Mention any warnings honestly.

## What the scripts already handle (do not re-implement)
Duplicate event timestamps, non-uniform grids, NaNs, empty/constant columns, unsafe Modelica names
(dots, brackets, keywords, leading digits), near-zero scaling, automatic sample time (≥ 20 samples per
fastest time constant, input switching kept on the grid), stiffness warning, coverage/extrapolation
check, split by run, least-squares warm start, rollout curriculum, noise injection, Optuna with pruning,
3 seeds, gates, stability tests (3× long run, constant-input settling), diagnosis, latent-state
initialisation, range warnings inside the Modelica block, heat-port adapter, data-driven validation
example, OMC check/simulate/timing, report + manifest.

## References
- `references/config.md` — config.json and datagen.json schema
- `references/model-selection.md` — the ladder, why, search spaces, gates, when to deviate
- `references/failure-playbook.md` — symptom → cause → fix, including OMC errors
- `references/modelica-package.md` — package layout, wrapper semantics, using/replacing the heavy block
- `references/omc-data-generation.md` — how to excite a heavy model so the ROM generalises
- `examples/motor`, `examples/twomasses` — complete configs for the MSL examples
