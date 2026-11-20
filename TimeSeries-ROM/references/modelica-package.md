# Generated Modelica package

```
<Pkg>/
  package.mo                 uses(Modelica(version="4.0.0")), documentation with gate status
  package.order              Networks, Layers, Examples
  Layers/                    Dense (y = W*x + b), Tanh, Sigmoid — pure, inlined functions
  Networks/
    SurrogateWrapper.mo      block: RealInputs left, RealOutputs right (drop-in replacement)
    SurrogateBlock.mo        function with all learned constants:
                               continuous families: der(x) = SurrogateBlock(x, u)
                               GRU: h_next = SurrogateBlock(h, u)
    SurrogateInit.mo         latent / GRU initial state from start values and initial inputs (if needed)
    SurrogateOutput.mo       GRU read-out (GRU only)
    SurrogatePorts.mo        optional heat-port adapter (outputs with "port": "heat")
  Examples/
    ValidateAgainstData      ROM driven by the held-out heavy-model inputs; err_* = ROM - heavy data
    RomOnly                  same, ROM only (timing)
    CompareToReference       ROM next to the original heavy model (when config.reference is set)
    PortsDemo                adapter connected to a thermal network (when ports are generated)
  Resources/Data/validation.txt   tables "inputs" and "outputs" of the test run
<Pkg>.zip
```

## Wrapper semantics
- Parameters `<output>_start` (initial values, default = test run start) and `warnOutOfRange`.
- Continuous families: states `x = [outputs; latent]`, `fixed` by initial equations; the solver chooses
  the step (no sample events). GRU: `discrete` hidden state updated by `when sample(0, dt)`; inputs are
  sampled at the start of each step (zero-order hold), matching training.
- Range checks: a message is printed once when an input leaves the training range (±2 %) or an output
  leaves the trained range (±10 %). The ROM keeps running; results outside the range are unreliable.
- Connector units are set from the config (`unit=`).

## Using it
1. OMEdit: File > Load Library > `<Pkg>/package.mo` (or unzip the zip first).
2. Simulate `Examples.ValidateAgainstData`, plot `err_*` (should stay within the reported max |error|).
3. Replace the heavy block: delete it, drag `Networks.SurrogateWrapper` in, connect the same signals to
   the inputs, take the outputs from the right side. For heat-port networks use `SurrogatePorts`.
4. Keep inputs inside the documented ranges (also listed in the block documentation and input comments).

## Conventions (keep stable for downstream users)
- Networks holds the function(s) and the wrapper; Examples holds only test models; Layers holds generic functions.
- Weights are `protected constant` arrays inside the functions (no external files at runtime except the
  example's validation table).
- No FMU, no external C code: pure Modelica 4.0.0, opens in OpenModelica and Dymola.
