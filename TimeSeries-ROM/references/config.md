# config.json

Relative paths are relative to the config file. Use forward slashes.

```json
{
  "work_dir": ".",                                  // outputs go here (default: config folder)
  "dataset": {
    "files": ["work/omc_data/run_*_res.csv", "work/omc_data/original_res.csv"],   // globs ok; 1 file = 1 run
    "time_column": "time",                          // optional, auto-detected
    "run_column": null                              // optional: split one file into runs by this column
  },
  "test_runs": ["original_res"],                    // run names = file name without extension; held out
  "val_runs": null,                                 // optional; default: 20% of the remaining runs (random, seed)
  "unit_conversions": [{"column": "Tsensor.T", "op": "degC_to_K"}],   // optional (or "K_to_degC")
  "inputs":  [{"column": "lossTable.y[1]", "name": "P_winding", "unit": "W"}],
  "outputs": [{"column": "winding.T", "name": "T_winding", "unit": "K", "port": "heat"}],
  "package_name": "MotorThermalROM",
  "input_interpolation": "hold",                    // "hold" (steps, tables with ConstantSegments) | "linear"
  "sample_time": null,                              // optional; default automatic
  "start_values": null,                             // optional wrapper defaults; default = test run initial outputs
  "families": ["linear", "node", "node_latent", "gru"],   // ladder order (stop at first rung passing gates)
  "budget": {"trials": 8, "iters": 800, "seeds": 3},       // Optuna trials per rung, iterations per fit, seeds
  "gates": {"nrmse_pct": 1.0, "max_abs_pct_range": 5.0},  // worst output, worst val run, % of training range
  "threads": null,                                  // torch threads (default: all cores)
  "seed": 0,
  "reference": {                                    // optional: heavy model available in OMC
    "model": "Modelica.Thermal.HeatTransfer.Examples.Motor",
    "outputs": ["winding.T", "core.T"],             // variables in the heavy model matching the ROM outputs
    "start_values": [293.15, 293.15],               // ROM start values in the comparison
    "drivers": "  <Modelica declarations producing the same excitation as the heavy model>\n",
    "connects": ["lossTable.y[1], rom.P_winding"],  // connect() arguments for the ROM inputs
    "experiment": "StopTime=7200, Interval=5",
    "simulate_args": "stopTime=7200, numberOfIntervals=1440, "   // for timing the heavy model alone
  },
  "omc_timeout": 1800
}
```

Notes
- `name` values become Modelica identifiers (sanitised automatically if needed). Keep them short and physical.
- `"port": "heat"` on temperature outputs generates `Networks.SurrogatePorts` (temperatures imposed on
  `HeatPort_b` connectors) and `Examples.PortsDemo`.
- Input/output units are written as `unit=` on the connectors; give SI units.
- If the heavy model is a library of the user's, put its `package.mo` path in the reference via
  `drivers`/`model` and make sure OMC can load it (see `load_files` in datagen).

# datagen.json (tsrom_datagen.py)

```json
{
  "model": "Modelica.Thermal.HeatTransfer.Examples.Motor",   // heavy model (full Modelica path)
  "load_files": [],                                // extra .mo libraries to load (user's own models)
  "out_dir": "work/omc_data",
  "stop_time": 7200, "intervals": 1440,            // output grid; choose >= 20 points per fastest time constant
  "n_runs": 30, "seed": 0,
  "tables": [{"component": "lossTable", "time_step": 5, "smoothness": "ConstantSegments",
              "columns": [{"lo": 0, "hi": 1200, "hold_min": 60, "hold_max": 900}]}],   // one entry per table column
  "parameters": [{"name": "TAmb", "lo": 273.15, "hi": 318.15}],                         // parameters varied per run
  "start_values": [{"name": "winding.T", "relative_to": "TAmb", "lo_offset": 0, "hi_offset": 80, "cold_fraction": 0.33},
                   {"name": "mass1.T", "lo": 253.15, "hi": 413.15}],                    // absolute range form
  "extra_declarations": ["Real TAmbSig = TAmb;"],  // expose parameters as variables so they appear in results
  "variable_filter": "time|winding.T|core.T|lossTable.y.1.|lossTable.y.2.|TAmbSig",
  "include_original": true                         // also simulate the unmodified model -> test run "original"
}
```
Result files: `<out_dir>/run_XXX_res.csv`, `original_res.csv`, plus `runs.json` (what each run used).
