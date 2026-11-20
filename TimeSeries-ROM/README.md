# timeseries-rom — GitHub Copilot agent skill

Turns a heavy OpenModelica model (or its simulation data) into a verified time-series reduced order model
packaged as a Modelica 4.0.0 library.

## Install
Copy this folder (the one containing `SKILL.md`) to one of:
- `%USERPROFILE%\.copilot\skills\timeseries-rom\`  (global, all workspaces)
- `<repo>\.github\skills\timeseries-rom\`         (one repository)

Then in VS Code (Copilot Chat, Agent mode) ask e.g. *"Create a time-series ROM from these simulation
results"* or *"Replace Modelica.Thermal.HeatTransfer.Examples.Motor with a ROM"*.

Requirements: Python ≥ 3.9 with `pip install -r scripts/requirements.txt`, OpenModelica with MSL 4.0.0
(`omc` found via `OPENMODELICAHOME`, PATH or `C:\Program Files\OpenModelica*`).

## Run the examples by hand
```
cd examples\motor
python ..\..\scripts\tsrom.py env
python ..\..\scripts\tsrom_datagen.py --spec datagen.json      # 31 OMC simulations -> work\omc_data
python ..\..\scripts\tsrom.py run --config config.json         # train, export, verify, report
```
Results: `REPORT.md`, `package\MotorThermalROM\` (+ .zip), `omc_verify\`.
Same for `examples\twomasses`.

## Layout
```
SKILL.md                    agent instructions (workflow, gate, rules)
scripts/                    tested pipeline (agent never edits these)
  tsrom.py                  entry point: env | profile | run | prepare | train | export | verify | report
  tsrom_datagen.py          heavy-model data generation with OMC from a JSON spec
  tsrom_data.py             loading (.csv/.mat/.xlsx), cleaning, profiling, preparation
  tsrom_train.py            model ladder, Optuna, gates, diagnosis, stability tests
  tsrom_export.py           Modelica package writer
  tsrom_verify.py           export fidelity check + OMC verification
  tsrom_report.py           REPORT.md, plot, manifest
references/                 config schema, model selection, failure playbook, package docs, data generation
examples/                   motor, twomasses (MSL Thermal.HeatTransfer examples)
```
