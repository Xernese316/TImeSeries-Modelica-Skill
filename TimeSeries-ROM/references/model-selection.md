# Model selection

## The problem being solved
System identification: learn `der(y) = f(y, u)` (continuous) or `h[k+1] = g(h[k], u[k])` (discrete) so that
the model, **fed its own predictions** inside Modelica, tracks the heavy model. Generic forecasting models
(Transformers, PatchTST, TCN, gradient boosting on lags) are built for one-step/benchmark forecasting, not
closed-loop simulation with exogenous inputs, and are hard or impossible to export as small Modelica
functions. They are intentionally not in the ladder.

## The ladder (cheapest first; stop at the first rung that passes gates AND stability)
| rung | model | Modelica form | strengths | weaknesses |
|---|---|---|---|---|
| `linear` | `der(y) = A y + B u + c` | continuous states = outputs | exact for linear systems (e.g. TwoMasses), tiny, fastest, interpretable | no nonlinearity |
| `node` | linear + MLP correction (tanh) | continuous states = outputs | nonlinear (e.g. temperature-dependent losses), still small, solver-friendly | needs outputs = full state |
| `node_latent` | node + `nz` latent states, initial latent from a K-sample history encoder | continuous states = outputs + latent | handles unlogged internal states | hidden initial state must be identifiable; harder to train |
| `gru` | GRU cell, `y = C h + d` | discrete, `when sample(0, dt)` | long memory, hysteresis-like effects | fixed sample time, solver stops every dt, invented initial hidden state, more weights |

LSTM is not in the default ladder: a GRU is the same idea with fewer gates/weights and in practice equal
accuracy; it was worse than the Neural ODE on the reference cases. Add it only if a GRU fails and memory
effects are clearly the cause (would need a new export template — do not hand-write one).

## Why the training is set up this way
- **Rollout (multiple-shooting) loss with a curriculum** (horizon 4 → 16 → 64 → … → chosen): one-step
  loss looks great in Python and drifts in OMEdit. Windows start from true data, so long data sets are cheap.
- **Least-squares warm start** of the linear part; a training run is never allowed to end worse than it.
- **Noise injection** on the initial state: the model learns to recover from its own errors.
- **Optuna TPE + median pruning**, objective = free-run validation error (worst output, mean over runs).
- **3 seeds** of the best configuration; large spread = unreliable result (reported).
- Metrics are relative to each output's **training range** (so weakly excited runs don't inflate %).

## Search spaces
- all: `lr` 1e-3…2e-2 (log), `horizon` ∈ {32, 64, 128, 256} (< run length), `noise` 0…0.05
- `node`: `hidden` ∈ {8, 16, 32, 64}, `layers` ∈ {1, 2}
- `node_latent`: + `nz` ∈ {1, 2, 4}, `K` ∈ {4, 8, 16}
- `gru`: `hidden` ∈ {8, 16, 32}
Hidden ≤ 64 keeps the Modelica function small (fast compile, cheap Jacobian).

## Gates (defaults; only the user may loosen them)
- worst free-run NRMSE ≤ 1 % of the training range, for every output, on the worst validation run;
- worst |error| ≤ 5 % of the training range;
- finite; 3× long run stays within ±50 % of the training range; constant inputs settle (no oscillation/runaway).
Typical engineering targets: thermal ±1–2 K, electrical ±1–2 % of rated values. Tighten gates if the user
states a tolerance (convert it to % of range).

## When to deviate from the default order (`families`)
- Known strongly nonlinear system → skip `linear` only to save time (it is cheap; normally keep it).
- Outputs are clearly not the full state (e.g. only one of several temperatures logged) → put
  `node_latent` before `node`, and first ask for the missing states.
- Hysteresis, backlash, rate-dependent memory → allow `gru` earlier.
- Very large data (> 1e6 samples): raise `iters`, lower `trials` (e.g. 4), keep seeds 2.
