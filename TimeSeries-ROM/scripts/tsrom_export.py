"""Export the trained ROM to a Modelica 4.0.0 library (directory structure + zip).

<Pkg>/package.mo, package.order
<Pkg>/Layers/      Dense, Tanh, Sigmoid           (pure functions)
<Pkg>/Networks/    SurrogateBlock                 (function: learned dynamics, all weights as constants)
                   SurrogateInit                  (function: latent/hidden initial state; latent & GRU only)
                   SurrogateOutput                (function: GRU read-out; GRU only)
                   SurrogateWrapper               (block: RealInputs left, RealOutputs right)
                   SurrogatePorts                 (optional: heat-port adapter around the wrapper)
<Pkg>/Examples/    ValidateAgainstData, RomOnly, [CompareToReference]
<Pkg>/Resources/Data/validation.txt               (test-run inputs + expected outputs)

CLI: python tsrom_export.py --config work/config.json
"""
import argparse, os, shutil, zipfile, datetime
import numpy as np, torch
from tsrom_common import Scaler, load, dump, safe_package_name
from tsrom_train import build


def arr(a):
    a = np.asarray(a, dtype=float)
    if a.ndim == 0: return f"{float(a):.10g}"
    if a.ndim == 1: return "{" + ", ".join(f"{v:.10g}" for v in a) + "}"
    return "{" + ", ".join(arr(r) for r in a) + "}"


def W(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write(text)


def places(n, side):
    fmt = "{{-140,%g},{-100,%g}}" if side == "in" else "{{100,%g},{140,%g}}"
    return [fmt % (round(100 - (i + .5) * 200 / n - 20, 3), round(100 - (i + .5) * 200 / n + 20, 3)) for i in range(n)]


def np_(t): return t.detach().cpu().numpy().astype(float)


def write_layers(root, pkg):
    W(f"{root}/Layers/package.mo", f'within {pkg};\npackage Layers "Neural-network layer functions"\n'
      f'  extends Modelica.Icons.FunctionsPackage;\nend Layers;\n')
    W(f"{root}/Layers/package.order", "Dense\nTanh\nSigmoid\n")
    W(f"{root}/Layers/Dense.mo", f'''within {pkg}.Layers;
function Dense "Fully connected layer y = W*x + b"
  extends Modelica.Icons.Function;
  input Real x[:];
  input Real W[:, size(x, 1)];
  input Real b[size(W, 1)];
  output Real y[size(W, 1)];
algorithm
  y := W*x + b;
  annotation (Inline=true);
end Dense;
''')
    for name, expr, doc in (("Tanh", "Modelica.Math.tanh(x[i])", "tanh"),
                            ("Sigmoid", "1/(1 + Modelica.Math.exp(-x[i]))", "logistic sigmoid")):
        W(f"{root}/Layers/{name}.mo", f'''within {pkg}.Layers;
function {name} "Element-wise {doc}"
  extends Modelica.Icons.Function;
  input Real x[:];
  output Real y[size(x, 1)];
algorithm
  for i in 1:size(x, 1) loop
    y[i] := {expr};
  end for;
  annotation (Inline=true);
end {name};
''')


def table_file(path, tables):
    lines = ["#1"]
    for name, M in tables:
        lines.append(f"double {name}({M.shape[0]},{M.shape[1]})")
        lines += [" ".join(f"{v:.10g}" for v in row) for row in M]
    W(path, "\n".join(lines) + "\n")


def export(cfg_path):
    cfg = load(cfg_path)
    wd = os.path.join(os.path.dirname(os.path.abspath(cfg_path)), cfg.get("work_dir", "."))
    ck = torch.load(os.path.join(wd, "model.pt"), weights_only=False)
    rep = load(os.path.join(wd, "train_report.json"))
    meta, fam, p = ck["meta"], ck["family"], ck["params"]
    sy, su = Scaler.from_dict(ck["sy"]), Scaler.from_dict(ck["su"])
    ins, outs, dt = meta["inputs"], meta["outputs"], float(meta["dt"])
    ny, nu = len(outs), len(ins)
    m = build(fam, ny, nu, p); m.load_state_dict(ck["state"]); m.eval()
    pkg = safe_package_name(cfg["package_name"])
    outdir = os.path.join(wd, cfg.get("output_dir", "package"))
    root = os.path.join(outdir, pkg)
    if os.path.exists(root):
        shutil.rmtree(root)
    write_layers(root, pkg)
    u_units = meta["units"]["inputs"]; y_units = meta["units"]["outputs"]
    ulo, uhi = np.array(meta["ranges"]["u_min"]), np.array(meta["ranges"]["u_max"])
    ylo, yhi = np.array(meta["ranges"]["y_min"]), np.array(meta["ranges"]["y_max"])
    status = rep["status"]; tw = rep["test"]["worst"]
    files_order = ["SurrogateWrapper"]

    def cblock(name, a): return f"  constant Real {name}[{', '.join(str(s) for s in np.shape(a))}] = {arr(a)};"
    scal = [cblock("y_mean", sy.m), cblock("y_std", sy.s)]
    if nu:
        scal += [cblock("u_mean", su.m), cblock("u_std", su.s)]
    uin = f'  input Real u[{nu}] "inputs: {", ".join(ins)}";\n' if nu else ""
    ucall = "{" + ", ".join(ins) + "}" if nu else None
    uz = "(u - u_mean) ./ u_std" if nu else None

    # -------------------------------------------------------------- continuous-time families
    if m.kind == "ode":
        nz = m.nz; nx = ny + nz
        f = m.f
        C = [*scal, f'  constant Real dt = {dt:.10g} "training sample time (time unit of the learned dynamics) [s]";',
             cblock("Wl", np_(f.lin.weight)), cblock("bl", np_(f.lin.bias))]
        loc, body = [f"  Real v[{nx + nu}] \"normalized [y; latent; u]\";", f"  Real g[{nx}];"], []
        parts = ["(x[1:%d] - y_mean) ./ y_std" % ny] + ([f"x[{ny+1}:{nx}]"] if nz else []) + ([uz] if nu else [])
        body.append("  v := " + (f"cat(1, {', '.join(parts)});" if len(parts) > 1 else parts[0] + ";"))
        body.append("  g := Layers.Dense(v, Wl, bl);")
        if f.mlp is not None:
            lins = [l for l in f.mlp if isinstance(l, torch.nn.Linear)]
            prev = "v"
            for i, l in enumerate(lins[:-1]):
                C += [cblock(f"W{i+1}", np_(l.weight)), cblock(f"b{i+1}", np_(l.bias))]
                loc.append(f"  Real h{i+1}[{l.out_features}];")
                body.append(f"  h{i+1} := Layers.Tanh(Layers.Dense({prev}, W{i+1}, b{i+1}));"); prev = f"h{i+1}"
            C.append(cblock("Wo", np_(lins[-1].weight)))
            body.append(f"  g := g + Wo*{prev};")
        body.append(f"  dxdt[1:{ny}] := y_std .* g[1:{ny}] / dt;")
        if nz:
            body.append(f"  dxdt[{ny+1}:{nx}] := g[{ny+1}:{nx}] / dt;")
        W(f"{root}/Networks/SurrogateBlock.mo", f'''within {pkg}.Networks;
function SurrogateBlock "Learned ROM dynamics der(x) = f(x, u); x = [outputs; {nz} latent states]"
  extends Modelica.Icons.Function;
  input Real x[{nx}] "states: {", ".join(outs)}{' + latent' if nz else ''}";
{uin}  output Real dxdt[{nx}];
protected
{chr(10).join(C)}
{chr(10).join(loc)}
algorithm
{chr(10).join(body)}
end SurrogateBlock;
''')
        files_order.append("SurrogateBlock")
        if nz:   # fold the K-sample history encoder for a constant (at-rest) history into two matrices
            E = np_(m.enc.weight); K = m.K
            Ey = sum(E[:, k * ny:(k + 1) * ny] for k in range(K))
            Eu = sum(E[:, K * ny + k * nu: K * ny + (k + 1) * nu] for k in range(K)) if nu else None
            CI = [cblock("y_mean", sy.m), cblock("y_std", sy.s), cblock("Ey", Ey), cblock("e", np_(m.enc.bias))]
            expr = "Ey*((y0 - y_mean) ./ y_std) + e"
            if nu:
                CI += [cblock("u_mean", su.m), cblock("u_std", su.s), cblock("Eu", Eu)]
                expr = "Ey*((y0 - y_mean) ./ y_std) + Eu*((u0 - u_mean) ./ u_std) + e"
            W(f"{root}/Networks/SurrogateInit.mo", f'''within {pkg}.Networks;
function SurrogateInit "Initial latent states from initial outputs/inputs (system assumed at its initial condition)"
  extends Modelica.Icons.Function;
  input Real y0[{ny}];
{('  input Real u0[%d];' % nu + chr(10)) if nu else ''}  output Real z0[{nz}];
protected
{chr(10).join(CI)}
algorithm
  z0 := Layers.Tanh({expr});
end SurrogateInit;
''')
            files_order.append("SurrogateInit")
        state_decl = f'  Real x[{nx}] "ROM states (outputs{" + latent" if nz else ""})";'
        init = [f"  x[1:{ny}] = {{{', '.join(o + '_start' for o in outs)}}};"]
        if nz:
            init.append(f"  x[{ny+1}:{nx}] = SurrogateInit({{{', '.join(o + '_start' for o in outs)}}}"
                        + (f", {ucall});" if nu else ");"))
        eqs = [f"  der(x) = SurrogateBlock(x{', ' + ucall if nu else ''});"] + \
              [f"  {o} = x[{i+1}];" for i, o in enumerate(outs)]
        family_doc = "continuous-time Neural ODE" if fam != "linear" else "continuous-time linear state-space"
    # -------------------------------------------------------------- GRU (discrete, sample time dt)
    else:
        nh = m.cell.hidden_size
        Wih, Whh = np_(m.cell.weight_ih), np_(m.cell.weight_hh); bih, bhh = np_(m.cell.bias_ih), np_(m.cell.bias_hh)
        sl = lambda a, k: a[k * nh:(k + 1) * nh]
        C = [*scal, cblock("Wir", sl(Wih, 0)), cblock("Wiz", sl(Wih, 1)), cblock("Win", sl(Wih, 2)),
             cblock("Whr", sl(Whh, 0)), cblock("Whz", sl(Whh, 1)), cblock("Whn", sl(Whh, 2)),
             cblock("br", sl(bih, 0) + sl(bhh, 0)), cblock("bz", sl(bih, 1) + sl(bhh, 1)),
             cblock("bin", sl(bih, 2)), cblock("bhn", sl(bhh, 2))]
        uvec = uz if nu else "{0.0}"
        W(f"{root}/Networks/SurrogateBlock.mo", f'''within {pkg}.Networks;
function SurrogateBlock "One learned GRU step: h_next = GRU(u, h)"
  extends Modelica.Icons.Function;
  input Real h[{nh}];
{uin}  output Real h_next[{nh}];
protected
{chr(10).join(C)}
  Real v[{max(nu,1)}];
  Real r[{nh}];
  Real z[{nh}];
  Real n[{nh}];
algorithm
  v := {uvec};
  r := Layers.Sigmoid(Wir*v + Whr*h + br);
  z := Layers.Sigmoid(Wiz*v + Whz*h + bz);
  n := Layers.Tanh(Win*v + bin + r .* (Whn*h + bhn));
  h_next := (ones({nh}) - z) .* n + z .* h;
end SurrogateBlock;
''')
        W(f"{root}/Networks/SurrogateOutput.mo", f'''within {pkg}.Networks;
function SurrogateOutput "GRU read-out y = C h + d (physical units)"
  extends Modelica.Icons.Function;
  input Real h[{nh}];
  output Real y[{ny}];
protected
{cblock("y_mean", sy.m)}
{cblock("y_std", sy.s)}
{cblock("Cw", np_(m.out.weight))}
{cblock("d", np_(m.out.bias))}
algorithm
  y := y_mean + y_std .* Layers.Dense(h, Cw, d);
end SurrogateOutput;
''')
        CI = [cblock("y_mean", sy.m), cblock("y_std", sy.s), cblock("E", np_(m.enc.weight)), cblock("e", np_(m.enc.bias))]
        if nu:
            CI += [cblock("u_mean", su.m), cblock("u_std", su.s)]
        vin = "cat(1, (y0 - y_mean) ./ y_std, (u0 - u_mean) ./ u_std)" if nu else "(y0 - y_mean) ./ y_std"
        W(f"{root}/Networks/SurrogateInit.mo", f'''within {pkg}.Networks;
function SurrogateInit "Initial GRU hidden state from initial outputs/inputs"
  extends Modelica.Icons.Function;
  input Real y0[{ny}];
{('  input Real u0[%d];' % nu + chr(10)) if nu else ''}  output Real h0[{nh}];
protected
{chr(10).join(CI)}
algorithm
  h0 := Layers.Tanh(E*{vin} + e);
end SurrogateInit;
''')
        files_order += ["SurrogateBlock", "SurrogateOutput", "SurrogateInit"]
        state_decl = (f'  discrete Real h[{nh}] "GRU hidden state";\n'
                      + (f'  discrete Real u_held[{nu}] "inputs sampled at the start of each step";' if nu else ""))
        init = [f"  h = SurrogateInit({{{', '.join(o + '_start' for o in outs)}}}" + (f", {ucall});" if nu else ");")]
        if nu:
            init.append(f"  u_held = {ucall};")
        eqs = [f"  when sample(0, {dt:.10g}) then"]
        if nu:
            eqs.append(f"    u_held = {ucall};")
        eqs.append(f"    h = if time > {dt/2:.10g} then SurrogateBlock(pre(h){', pre(u_held)' if nu else ''}) else pre(h);")
        eqs.append("  end when;")
        eqs.append(f"  {{{', '.join(outs)}}} = SurrogateOutput(h);")
        family_doc = f"discrete-time GRU, sample time {dt:g} s"

    # -------------------------------------------------------------- wrapper block
    pin, pout = (places(nu, "in") if nu else []), places(ny, "out")
    unit_mod = lambda u: '(unit="%s")' % u if u else ""
    ports = [f'  Modelica.Blocks.Interfaces.RealInput {n}{unit_mod(u_units[i])} '
             f'"training range [{ulo[i]:.6g}, {uhi[i]:.6g}]" annotation (Placement(transformation(extent={pin[i]})));'
             for i, n in enumerate(ins)]
    ports += [f'  Modelica.Blocks.Interfaces.RealOutput {n}{unit_mod(y_units[i])} '
              f'annotation (Placement(transformation(extent={pout[i]})));' for i, n in enumerate(outs)]
    pred0 = np.load(os.path.join(wd, "test_predictions.npz"))["true_0"][0]
    starts = [float(v) for v in (cfg.get("start_values") or pred0)]
    params = [f'  parameter Real {o}_start = {starts[i]:.10g} "initial value of {o}";' for i, o in enumerate(outs)]
    params.append('  parameter Boolean warnOutOfRange = true "print a warning when inputs/outputs leave the training range";')
    mg = 0.02
    warn = []
    for i, n in enumerate(ins):
        lo, hi = ulo[i] - mg * (uhi[i] - ulo[i]), uhi[i] + mg * (uhi[i] - ulo[i])
        warn.append(f'  when warnOutOfRange and ({n} < {lo:.10g} or {n} > {hi:.10g}) then\n'
                    f'    Modelica.Utilities.Streams.print("WARNING ({pkg}): input {n} = " + String({n}) + '
                    f'" left the training range [{ulo[i]:.6g}, {uhi[i]:.6g}] at time " + String(time) + " - ROM is extrapolating.");\n'
                    f'  end when;')
    for i, n in enumerate(outs):
        r = yhi[i] - ylo[i]; lo, hi = ylo[i] - .1 * r, yhi[i] + .1 * r
        warn.append(f'  when warnOutOfRange and ({n} < {lo:.10g} or {n} > {hi:.10g}) then\n'
                    f'    Modelica.Utilities.Streams.print("WARNING ({pkg}): output {n} = " + String({n}) + '
                    f'" is outside the trained range [{ylo[i]:.6g}, {yhi[i]:.6g}] at time " + String(time) + " - results unreliable.");\n'
                    f'  end when;')
    W(f"{root}/Networks/SurrogateWrapper.mo", f'''within {pkg}.Networks;
block SurrogateWrapper "Reduced order model ({family_doc}) - drop-in replacement block"
  extends Modelica.Blocks.Icons.Block;
{chr(10).join(params)}
{chr(10).join(ports)}
protected
{state_decl}
initial equation
{chr(10).join(init)}
equation
{chr(10).join(eqs)}
algorithm
{chr(10).join(warn) if warn else "  // no range checks"}
  annotation (Icon(graphics={{Text(extent={{{{-90,30}},{{90,-30}}}}, textString="ROM"),
    Text(extent={{{{-90,-40}},{{90,-80}}}}, textString="{fam}", textColor={{128,128,128}})}}),
    Documentation(info="<html><p>Learned reduced order model. Family: {fam}. Training sample time {dt:g} s.</p>
<p>Inputs (left): {", ".join(f"{n} [{ulo[i]:.4g}..{uhi[i]:.4g}]" for i, n in enumerate(ins)) or "none"}.<br>
Outputs (right): {", ".join(outs)}.</p>
<p>Test free-run error: worst NRMSE {max(tw["nrmse_pct"]):.3g} %, worst |error| {", ".join(f"{v:.4g}" for v in tw["max_abs"])}.
Gate status: {status}.</p></html>"));
end SurrogateWrapper;
''')

    # -------------------------------------------------------------- optional heat-port adapter
    port_outs = [o for o, c in zip(outs, cfg["outputs"]) if c.get("port") == "heat"]
    if port_outs:
        pp = places(len(port_outs), "out")
        decl = [f"  Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_b port_{o} annotation (Placement(transformation(extent={pp[i]})));"
                f"\n  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature prescribed_{o};" for i, o in enumerate(port_outs)]
        inp = [f"  Modelica.Blocks.Interfaces.RealInput {n} annotation (Placement(transformation(extent={pin[i]})));"
               for i, n in enumerate(ins)]
        con = [f"  connect({n}, rom.{n});" for n in ins] + \
              [f"  connect(rom.{o}, prescribed_{o}.T);\n  connect(prescribed_{o}.port, port_{o});" for o in port_outs]
        W(f"{root}/Networks/SurrogatePorts.mo", f'''within {pkg}.Networks;
model SurrogatePorts "ROM with heat ports: temperature outputs are imposed on HeatPort_b connectors"
  SurrogateWrapper rom;
{chr(10).join(inp)}
{chr(10).join(decl)}
equation
{chr(10).join(con)}
  annotation (Icon(graphics={{Rectangle(extent={{{{-100,100}},{{100,-100}}}}, lineColor={{191,0,0}}, fillColor={{255,255,255}},
    fillPattern=FillPattern.Solid), Text(extent={{{{-90,30}},{{90,-30}}}}, textString="ROM"),
    Text(extent={{{{-150,150}},{{150,110}}}}, textString="%name", textColor={{0,0,255}})}}));
end SurrogatePorts;
''')
        files_order.append("SurrogatePorts")
    W(f"{root}/Networks/package.mo", f'within {pkg};\npackage Networks "Learned ROM"\n  extends Modelica.Icons.Package;\nend Networks;\n')
    W(f"{root}/Networks/package.order", "\n".join(files_order) + "\n")

    # -------------------------------------------------------------- examples + data
    pred = np.load(os.path.join(wd, "test_predictions.npz"))
    t, Y, Uin = pred["t_0"], pred["true_0"], pred["u_0"]
    tables = [("outputs", np.column_stack([t, Y]))]
    if nu:
        tables.insert(0, ("inputs", np.column_stack([t, Uin])))
    table_file(f"{root}/Resources/Data/validation.txt", tables)
    hold = cfg.get("input_interpolation", "hold") == "hold"
    res = f'Modelica.Utilities.Files.loadResource("modelica://{pkg}/Resources/Data/validation.txt")'
    starts_mod = ", ".join(f"{o}_start={Y[0, i]:.10g}" for i, o in enumerate(outs))
    drv = ""
    if nu:
        drv = (f'  Modelica.Blocks.Sources.CombiTimeTable inputs(tableOnFile=true, tableName="inputs", fileName={res},\n'
               f'    columns=2:{nu+1}, smoothness=Modelica.Blocks.Types.Smoothness.'
               f'{"ConstantSegments" if hold else "LinearSegments"},\n'
               f'    extrapolation=Modelica.Blocks.Types.Extrapolation.HoldLastPoint)\n'
               f'    annotation (Placement(transformation(extent={{{{-80,-10}},{{-60,10}}}})));\n')
    con = "".join(f"  connect(inputs.y[{i+1}], rom.{n});\n" for i, n in enumerate(ins))
    exp_ = f"StopTime={t[-1]:.10g}, Interval={dt:.10g}, Tolerance=1e-06"
    romd = f"  Networks.SurrogateWrapper rom({starts_mod}) annotation (Placement(transformation(extent={{{{-20,-20}},{{20,20}}}})));\n"
    errs = "".join(f'  Real err_{o} = rom.{o} - expected.y[{i+1}] "ROM minus heavy-model data";\n' for i, o in enumerate(outs))
    W(f"{root}/Examples/ValidateAgainstData.mo", f'''within {pkg}.Examples;
model ValidateAgainstData "Drive the ROM with the held-out test inputs; err_* = ROM - heavy model"
  extends Modelica.Icons.Example;
{drv}{romd}  Modelica.Blocks.Sources.CombiTimeTable expected(tableOnFile=true, tableName="outputs", fileName={res},
    columns=2:{ny+1}, extrapolation=Modelica.Blocks.Types.Extrapolation.HoldLastPoint)
    annotation (Placement(transformation(extent={{{{40,40}},{{60,60}}}})));
{errs}equation
{con}  annotation (experiment({exp_}));
end ValidateAgainstData;
''')
    W(f"{root}/Examples/RomOnly.mo", f'''within {pkg}.Examples;
model RomOnly "ROM alone, same inputs (use for timing)"
  extends Modelica.Icons.Example;
{drv}{romd}equation
{con}  annotation (experiment({exp_}));
end RomOnly;
''')
    ex_order = ["ValidateAgainstData", "RomOnly"]
    ref = cfg.get("reference")
    if ref:
        rerrs = "".join(f'  Real err_{o} = rom.{o} - reference.{ref["outputs"][i]} "ROM minus heavy model";\n'
                        for i, o in enumerate(outs))
        rstarts = ", ".join(f"{o}_start={v}" for o, v in zip(outs, ref.get("start_values", Y[0])))
        rcon = "".join(f"  connect({c});\n" for c in ref.get("connects", []))
        W(f"{root}/Examples/CompareToReference.mo", f'''within {pkg}.Examples;
model CompareToReference "ROM next to the original heavy model, same excitation"
  extends Modelica.Icons.Example;
  {ref["model"]} reference "original physics model" annotation (Placement(transformation(extent={{{{40,40}},{{80,80}}}})));
  Networks.SurrogateWrapper rom({rstarts}) annotation (Placement(transformation(extent={{{{-20,-20}},{{20,20}}}})));
{ref.get("drivers", "")}{rerrs}equation
{rcon}  annotation (experiment({ref.get("experiment", exp_)}));
end CompareToReference;
''')
        ex_order.append("CompareToReference")
    if port_outs:
        cons = "".join(f"  connect(inputs.y[{i+1}], romPorts.{n});\n" for i, n in enumerate(ins))
        cons += "".join(f"  connect(romPorts.port_{o}, sink_{o}.port_a);\n  connect(sink_{o}.port_b, ambient.port);\n" for o in port_outs)
        W(f"{root}/Examples/PortsDemo.mo", f'''within {pkg}.Examples;
model PortsDemo "Heat-port adapter connected to a thermal network"
  extends Modelica.Icons.Example;
{drv}  Networks.SurrogatePorts romPorts(rom({starts_mod}));
{"".join(f"  Modelica.Thermal.HeatTransfer.Components.ThermalConductor sink_{o}(G=1);{chr(10)}" for o in port_outs)}  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature ambient(T=293.15);
equation
{cons}  annotation (experiment({exp_}));
end PortsDemo;
''')
        ex_order.append("PortsDemo")
    W(f"{root}/Examples/package.mo", f'within {pkg};\npackage Examples "Validation and test models"\n'
      f'  extends Modelica.Icons.ExamplesPackage;\nend Examples;\n')
    W(f"{root}/Examples/package.order", "\n".join(ex_order) + "\n")
    W(f"{root}/package.mo", f'''package {pkg} "Time-series reduced order model ({fam})"
  annotation (uses(Modelica(version="4.0.0")), version="1.0.0",
    Documentation(info="<html>
<p>Generated by the time-series ROM process on {datetime.date.today().isoformat()}. Gate status: <b>{status}</b>.</p>
<p>Use <code>Networks.SurrogateWrapper</code> to replace the heavy model. Check it with <code>Examples.ValidateAgainstData</code>
(plot the err_* variables).</p>
<p>Valid only inside the training range (see the input comments); the block prints a warning when it extrapolates.</p>
</html>"));
end {pkg};
''')
    W(f"{root}/package.order", "Networks\nLayers\nExamples\n")
    zpath = os.path.join(outdir, f"{pkg}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, fs in os.walk(root):
            for fn in fs:
                full = os.path.join(d, fn); z.write(full, os.path.relpath(full, outdir))
    info = dict(package=pkg, root=root, zip=zpath, family=fam, status=status, examples=ex_order)
    dump(info, os.path.join(wd, "export_info.json"))
    print(f"exported {pkg} ({fam}, {status}) -> {root}")
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    export(ap.parse_args().config)
