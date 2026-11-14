# flatppl-stats-templates

[Function reference](docs/reference.md) · [Live example](https://live.flatppl.org/#model=https%3A%2F%2Fraw.githubusercontent.com%2Fflatppl-community%2Fflatppl-stats-templates%2Fmain%2Fexamples%2Fplayground.flatppl&target=Pi_post)

Template combinations, response projection, observation laws, finite simulation statistics, and scalar grid interpolation for FlatPPL. Library version **0.1**.

```flatppl
flatppl_compat = "0.1"
t = load_module("main.flatppl")
shapes = rowstack([[0.6, 0.1], [0.3, 0.3], [0.1, 0.6]])
means = t.combine(shapes, [25.0, 15.0]) .+ [1.0, 1.0, 1.0]
model = t.poisson_model(shapes, [25.0, 15.0], [1.0, 1.0, 1.0])
log_likelihood = logdensityof(model, [17, 14, 12])
```

Templates use `[bins, components]` axes. Responses use `[measured, true]` axes. Parameters are explicit. The library adds no priors, floors, profiling, response normalization, or extrapolation.

The [histogram example](examples/histogram-fit.flatppl) exposes component strengths, expected counts, and a log likelihood for host-side optimization.

See the reference for definitions, domains, numerical limits, and validation commands.
