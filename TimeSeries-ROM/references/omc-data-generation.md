# Generating training data from the heavy model

Data quality decides ROM quality more than any hyperparameter. The ROM can only be trusted inside the
envelope the data covers.

## What to randomise
1. **Every input the ROM will have** (tables, sources, set-points): piecewise-constant random levels
   (APRBS) via the source's `CombiTimeTable` (`tables` in datagen.json). Ranges = the real operating range
   plus ~10 %. Hold times from ~0.2× the fastest to ~1–2× the slowest time constant, so both transients
   and near-steady states appear.
2. **Parameters that represent operating conditions** (ambient temperature, supply voltage): one value
   per run (`parameters`). Expose them with `extra_declarations` (e.g. `Real TAmbSig = TAmb;`) and use
   that variable as a ROM input.
3. **Initial states** (`start_values`): ~1/3 of runs from rest/ambient (`cold_fraction`) — that is how the
   ROM is usually started — and the rest from varied states so the model learns the whole state space.

## How much
- ≥ 20 runs (30 is a good default for 2–5 inputs); each run ≥ 3–5× the slowest time constant.
- Output grid (`intervals`) with ≥ 20 points per fastest time constant; input switching times on that grid
  (`time_step`).
- Keep the unmodified model (`include_original`) as the held-out test run — it is the scenario the user cares about.

## Finding what to vary in an unknown model
Read the model's source (`list(Model)` in OMC or the .mo file): sources/tables connected to the component
being replaced are inputs; `parameter` declarations describing conditions are candidates; components with
`fixed=true` start values define the states. If the heavy component is embedded in a larger model, build a
harness model that instantiates only that component and drives its connectors with tables (signals) or
prescribed sources (physical ports), and use it as `model` in datagen.json.

## Variable filter
Include `time`, every input signal, every output (prefer SI, e.g. `winding.T` not the °C sensor) and any
extra internal states (they can become outputs if the hidden-state diagnosis appears). Regex is matched
against full variable names; `.` matches any char, so `lossTable.y.1.` matches `lossTable.y[1]`.
