# Failure playbook (symptom → cause → fix)

Apply at most 3 retry rounds, one change set per round, and record what you changed in your reply.

## Data / preparation (exit code 1 or prep warnings)
| symptom | cause | fix |
|---|---|---|
| `Output X is constant` | wrong column / nothing excited | pick another column or excite the model (datagen) |
| `Input X never varies` | parameter not varied | vary it in datagen, or drop it (ROM then only valid at that value) |
| lookalike pair in profile | same signal twice (K/°C, alias) | keep ONE (SI) — never both as outputs, never one as input of the other |
| `outside the training range` | test/val operating point not covered | add runs covering it (wider datagen ranges / cold & warm starts) |
| `spans only N x the slowest time constant` | runs too short | longer `stop_time` or more runs |
| `Multi-time-scale / stiff` | tau_max/tau_min > 1000 | separate ROMs per time scale, or treat the fast part as quasi-static (algebraic) |
| `Only one run` | single simulation | generate ≥ 20 runs (mode B); results from 1 run are not trustworthy |
| missing columns in a run | inconsistent files / variableFilter | regenerate with the same filter; check names in profile.json |

## Training (exit code 2: FAILED_GATES) — read `ladder[*].diagnosis`
| diagnosis | meaning | fix (in order) |
|---|---|---|
| `bias` | constant offset | missing input? add latent states (`node_latent`); check units/offsets |
| `drift` | error grows along the run | the ladder already tries longer horizons; raise `budget.iters` ×2; check stability section |
| `initialization` | error at the start of runs | an internal state at t=0 is unknown: expose it as an output, or make runs start from rest/steady state |
| `regime` | error in one input range | more runs in that range (datagen ranges/hold times) |
| `specific runs` | one run much worse | look at that run (runs.json): unusual start or extreme input → add similar runs |
| `underfit` | error everywhere | next rung / larger budget; if latent models also fail → outputs lack information: log the missing states |
| `seed_unstable` | seeds disagree | raise `budget.iters`; more data |
| stability FAILED | blow-up or oscillation | more data near equilibrium, longer runs; the ladder continues to the next rung automatically |
| all families diverged | scaling/time step problem | set explicit `sample_time` (smaller); check for huge outliers in data |

Never "fix" a failure by loosening gates, removing the hard test run, or training on the test run.

## Export / verification
| symptom | cause | fix |
|---|---|---|
| exit code 3 / `MISMATCH` | export bug | stop, report the traceback/reports; do not edit scripts |
| OMC `Class Modelica not found` / MSL version | MSL 4.0.0 not installed | install MSL 4.0.0 via OMEdit library manager |
| OMC check fails on `CompareToReference` | reference model/drivers wrong in config | fix `reference` (model path, outputs, drivers, connects) — ROM itself is fine if ValidateAgainstData passes |
| `omc_rom_vs_heavy_data` much worse than Python test | input interpolation mismatch | set `input_interpolation` to match how the heavy model's inputs behave (steps → hold) |
| `WARNING (...) left the training range` in OMC log | extrapolation in the example | expected only if the example leaves the envelope; otherwise widen training data |
| speed-up < 1 | heavy model is already cheap, or ROM too big / GRU sample events | prefer continuous families; smaller hidden size; tell the user honestly |
| `omc not found` | PATH / OPENMODELICAHOME | `set OPENMODELICAHOME=C:\Program Files\OpenModelica1.xx.x-64bit` or run `omc verify.mos` in `omc_verify/` |

## Integration in a larger model
- ROM inputs must be signals the surroundings supply; if the heavy component was connected acausally
  (heat ports / pins), use `SurrogatePorts` (thermal) or sensors/sources around the wrapper.
- If ROM outputs feed back into its own inputs elsewhere, test that closed loop explicitly in OMC; prefer
  continuous families (they are solver-friendly in feedback loops).
