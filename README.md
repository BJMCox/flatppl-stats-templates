# flatppl-stats-templates

Statistical template combinations, response projection, observation laws, finite simulation statistics, and scalar grid interpolation for FlatPPL 0.1. Library version: **0.1**.

Templates have **bins as rows and components as columns** (`[N,K]`). Responses have `[measured,true]` axes. Every fitted parameter is explicit; the library adds no prior, floor, profiling, response normalization, or extrapolation.

```flatppl
flatppl_compat = "0.1"
t = load_module("main.flatppl")
shapes = rowstack([[0.6, 0.1], [0.3, 0.3], [0.1, 0.6]])
means = t.combine(shapes, [25.0, 15.0]) .+ [1.0, 1.0, 1.0]
model = t.poisson_model(shapes, [25.0, 15.0], [1.0, 1.0, 1.0])
log_likelihood = logdensityof(model, [17, 14, 12])
```

See the [linked mathematical reference](docs/reference.md) for signatures, axes, domains, and normalized laws. [The synthetic laboratory example](examples/histogram-fit.flatppl) exposes nonnegative strengths as its input and returns the log likelihood and expected counts. Its unconstrained-by-prior maximum occurs near strengths `[25.1656, 14.8234]`, with log likelihood `-6.8075651`. A host optimizer supplies the fitting algorithm.

With existing NumPy/SciPy and FlatPPL engines, run:

```sh
FLATPPL_JS_DIR=/path/to/flatppl-js python tests/check_math.py
FLATPPL_JS_DIR=/path/to/flatppl-js python examples/fit.py
flatppl stablehlo examples/histogram-fit.flatppl --dtype f64 -o /tmp/histogram-fit.mlir
flatppl stablehlo tests/runtime.flatppl --dtype f64 -o /tmp/template-runtime.mlir
```

The tests use the real JS module loader and worker, with independent NumPy/SciPy values. `tests/runtime.flatppl` and the matching `runtime_oracle`/`RUNTIME_POINTS` in `tests/check_math.py` support compiled value and derivative checks. Interpolation derivatives are meaningful inside cells; they are generally undefined at interior grid knots.

The mathematical suite includes the [fixed-total fixture](tests/fixed-total.flatppl), which checks the normalized multinomial MC and Poisson data joint law against an independent SciPy oracle.
