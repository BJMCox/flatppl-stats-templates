# Statistical template reference — 0.1

Load `main.flatppl` and access the functions below. The source uses FlatPPL language compatibility `0.1`. This document is standalone Markdown; its display equations use GitHub math fences.

| Function | Meaning |
|---|---|
| [`combine(templates, weights)`](#combine) | Linear component sum |
| [`normalize_columns(templates, quadrature_weights)`](#normalize_columns) | Weighted column normalization |
| [`project(response, values, quadrature_weights)`](#project) | Apply a measured/true response |
| [`fold_templates(response, templates, quadrature_weights)`](#fold_templates) | Project all component columns |
| [`poisson_model(templates, weights, background)`](#poisson_model) | Independent Poisson observation law |
| [`gaussian_model(templates, weights, offset, covariance)`](#gaussian_model) | Correlated Gaussian observation law |
| [`gaussian_marginal(templates, mean_weights, covariance_weights, offset, covariance)`](#gaussian_marginal) | Exact unrestricted Gaussian coefficient marginal |
| [`barlow_beeston(expected_mc, scales, background)`](#barlow_beeston) | Full unweighted Poisson MC and data joint law |
| [`fixed_total_templates(probabilities, mc_totals, strengths, background)`](#fixed_total_templates) | Fixed-total multinomial MC and data joint law |
| [`log_deform(templates, log_shift)`](#log_deform) | Explicit multiplicative log deformation |
| [`linear_interpolate(nodes, values, x)`](#linear_interpolate) | Scalar interpolation on a one-dimensional grid |
| [`bilinear_interpolate(x_nodes, y_nodes, values, x, y)`](#bilinear_interpolate) | Scalar interpolation on a rectangular grid |

## Conventions

All arrays are real unless integer counts or totals are specified. Shapes are fixed before compilation. Templates `T` have shape `[N,K]`: bins first, components second. A response `R` has shape `[M,N]`: measured bins first, true bins second. Backgrounds, offsets, strengths, and all other parameters are explicit inputs.

The caller establishes physical units, selected domain, and reference measure. Positive quadrature weights `q[N]` integrate density-valued inputs. Already integrated bin values use unit weights to avoid a second integration. Response factors such as exposure and efficiency enter once. A lossy response retains its lost mass.

Strengths are not implicitly a simplex. Signed bases are valid for Gaussian models; a Poisson model additionally requires nonnegative predicted means over its parameter support. Functions have the stated mathematical domains. There is no clipping of invalid means, floor in zero bins, implicit prior, regularization, or likelihood profiling.

<a id="combine"></a>
## `combine(templates, weights)`

Arguments: `templates=T[N,K]` and `weights=a[K]`. Returns the vector `T*a[N]`.

```math
f_i = \sum_{k=1}^K T_{ik}a_k.
```

Domain: finite real entries with aligned dimensions. Weights may be unrestricted for a signed linear model. Positive component models choose nonnegative shapes and strengths explicitly. Rescaling column `k` by `c_k>0` and its strength by `1/c_k` preserves the prediction; Bayesian priors must transform too.

<a id="normalize_columns"></a>
## `normalize_columns(templates, quadrature_weights)`

Arguments: `templates=U[N,K]`, `quadrature_weights=q[N]`. Returns a `[N,K]` matrix with unit weighted integral in every column.

```math
Z_k=\sum_i q_iU_{ik},\qquad T_{ik}=U_{ik}/Z_k.
```

Domain: finite real `U`, strictly positive finite `q`, and all `Z_k>0`. Physical shapes additionally require nonnegative `U`. All-zero columns cannot be normalized. Exact zero entries stay zero; no floor is inserted.

<a id="project"></a>
## `project(response, values, quadrature_weights)`

Arguments: `response=R[M,N]`, `values=f[N]`, `quadrature_weights=q[N]`. Returns a measured-grid vector `[M]`.

```math
p_j=\sum_i R_{ji}q_if_i.
```

Domain: finite aligned arrays and strictly positive finite `q`. A physical nonnegative response additionally uses nonnegative `R` and `f`. Columns are never renormalized after selection or tail loss.

<a id="fold_templates"></a>
## `fold_templates(response, templates, quadrature_weights)`

Arguments: `response=R[M,N]`, `templates=T[N,K]`, `quadrature_weights=q[N]`. Returns the projected template matrix `H[M,K]`.

```math
H_{jk}=\sum_iR_{ji}q_iT_{ik},\qquad Ha=R(q\mathbin{\odot}(Ta)).
```

Domain: as for `project`, with aligned component columns. Precomputation is valid while both response and templates are fixed. Fitted shifts, widths, or calibration require recomputing their affected projection.

<a id="poisson_model"></a>
## `poisson_model(templates, weights, background)`

Arguments: `templates=T[N,K]`, `weights=a[K]`, `background=b[N]`. Returns the normalized independent count-array law on nonnegative integer vectors `[N]`, with counting reference measure.

```math
n_i\sim\mathrm{Poisson}(\mu_i),\qquad \mu=Ta+b.
```

Domain: aligned finite arrays and all `mu_i>=0`. A zero-mean bin is a point mass at zero. Zero observed counts remain part of the likelihood. Factorial constants are retained by the law. For positive means its log likelihood has gradient and Hessian

```math
\nabla_a\ell=T^T(n/\mu-1),\qquad
\nabla_a^2\ell=-T^T\mathrm{diag}(n/\mu^2)T.
```

Rank loss can leave weights unidentified; this library adds no regularization.

<a id="gaussian_model"></a>
## `gaussian_model(templates, weights, offset, covariance)`

Arguments: `templates=T[N,K]`, `weights=a[K]`, `offset=b[N]`, `covariance=C[N,N]`. Returns the normalized multivariate Gaussian law on real `[N]` vectors, with Lebesgue reference measure.

```math
y\sim N(Ta+b,C).
```

Domain: finite aligned arrays, symmetric positive-definite `C`. Coefficients and templates may be signed. The law retains the determinant normalization even when `C` depends on fitted parameters. Selecting bins also requires selecting the corresponding covariance submatrix.

<a id="gaussian_marginal"></a>
## `gaussian_marginal(templates, mean_weights, covariance_weights, offset, covariance)`

Arguments: `templates=T[N,K]`, `mean_weights=m[K]`, `covariance_weights=V[K,K]`, `offset=b[N]`, `covariance=C[N,N]`. Returns a normalized multivariate Gaussian law on real `[N]` vectors.

```math
a\sim N(m,V),\quad y\mid a\sim N(Ta+b,C)
\quad\Longrightarrow\quad y\sim N(Tm+b,C+TVT^T).
```

Domain: unrestricted real Gaussian coefficients, finite aligned arrays, symmetric positive-semidefinite `V`, and positive-definite `C` (hence positive-definite returned covariance). A degenerate coefficient Gaussian is permitted through its covariance image. This formula does not integrate nonnegative or simplex-constrained coefficients. It adds no other prior.

<a id="barlow_beeston"></a>
## `barlow_beeston(expected_mc, scales, background)`

Arguments: `expected_mc=xi[N,K]`, `scales=c[K]`, `background=b[N]`. Returns a normalized joint law with fields `data[N]` and `mc[N,K]`, all nonnegative integers.

```math
m_{ik}\sim\mathrm{Poisson}(\xi_{ik}),\qquad
n_i\sim\mathrm{Poisson}\left(b_i+\sum_kc_k\xi_{ik}\right).
```

Domain: finite nonnegative `xi`, `c`, and `b`. All MC bins and all data bins are independent conditional on these explicit parameters. The same `xi` enters both terms. Its nuisance expectations have no hidden prior and are neither profiled nor integrated by this constructor.

This is the full unweighted independent Poisson simulation model. Fixed simulation totals require [the multinomial law](#fixed_total_templates). Arbitrary event weights, negative weights, correlated MC templates, and effective-count approximations do not automatically follow this sampling design.

### Shared simulation observations enter once

For two data arrays sharing one MC sample, form a joint law with two data fields and a single raw MC field:

```flatppl
shared_law = joint(
    data_a = t.poisson_model(expected_mc, scales_a, background_a),
    data_b = t.poisson_model(expected_mc, scales_b, background_b),
    mc = Poisson.(expected_mc)
)
```

Score `record(data_a=counts_a, data_b=counts_b, mc=counts_mc)`. Calling `barlow_beeston` once for each data array would count the same auxiliary observation twice. For shared fixed-total MC, likewise combine the separate data laws with a single multinomial column law.

<a id="fixed_total_templates"></a>
## `fixed_total_templates(probabilities, mc_totals, strengths, background)`

Arguments: `probabilities=p[N,K]`, `mc_totals=M[K]` nonnegative integers, `strengths=A[K]`, `background=b[N]`. Returns a normalized joint law with fields `data[N]` and `mc[N,K]`.

```math
m_{\cdot k}\sim\mathrm{Multinomial}(M_k,p_{\cdot k}),\qquad
n_i\sim\mathrm{Poisson}\left(b_i+\sum_k A_kp_{ik}\right).
```

Domain: each column of `p` is a nonnegative probability vector summing to one, all `A_k>=0`, `b_i>=0`, and finite aligned arrays. Each observed MC column must sum to its declared fixed total. MC columns are mutually independent; their bins are dependent under the fixed-total constraint. Conditional data bins are independent, and the data array is independent of MC given `p,A,b`.

Probabilities describe integrated bins; density templates require an explicit quadrature conversion before using this law. `A` gives integrated expected data yields, not probability fractions. Zero totals and structural zero probabilities have their ordinary multinomial meanings. Weighted events require a separately declared sampling law.

The implementation rearranges the independent multinomial column vectors into a `[N,K]` matrix by an explicit counting-space bijection with unit volume. It retains the multinomial coefficients. See [current engine support](#engine-support).

<a id="log_deform"></a>
## `log_deform(templates, log_shift)`

Arguments: matching `[N,K]` arrays `templates=T` and `log_shift=D`. Returns the deformed matrix of the same shape.

```math
U_{ik}=T_{ik}\exp(D_{ik}).
```

Domain: finite real arrays for which exponentiation is representable. Nonnegative shapes remain nonnegative and structural zeros remain zero. The caller supplies all deformation modes and their distributions. This operation changes both rate and shape. A shape-only model explicitly calls `normalize_columns(U,q)` afterward. Normalization generally changes the nominal pointwise median of a Gaussian log deformation; no median-preservation claim is made.

<a id="linear_interpolate"></a>
## `linear_interpolate(nodes, values, x)`

Arguments: finite strictly increasing `nodes=u[L]`, finite aligned scalar `values=v[L]`, and scalar `x`. Requires `L>=2`. Returns one scalar.

```math
f(x)=(1-t)v_l+tv_{l+1},\qquad
t=\frac{x-u_l}{u_{l+1}-u_l},\qquad u_l\le x\le u_{l+1}.
```

Domain: the closed hull `[u_1,u_L]`. All knots and endpoints reproduce their values exactly up to arithmetic rounding. Outside the hull the source returns NaN; it provides no extrapolation, clamping, or zero fill. Nodes and their number are fixed for compilation. Grid coordinates can be physical or logarithmic only by the caller's explicit choice.

The implementation uses continuous ramp/hat weights and no integer cell search. Derivatives inside cells equal the segment slope. At interior knots a derivative generally does not exist; an AD system's branch convention is not a mathematical derivative there. A grid-refinement or held-out-forward calculation checks interpolation error separately from AD correctness.

<a id="bilinear_interpolate"></a>
## `bilinear_interpolate(x_nodes, y_nodes, values, x, y)`

Arguments: finite strictly increasing `x_nodes=u[Nx]`, `y_nodes=v[Ny]`, finite `values=F[Nx,Ny]`, and scalar coordinates `x,y`. Each axis has at least two nodes. Returns one scalar.

```math
f(x,y)=(1-s)(1-t)F_{ij}+s(1-t)F_{i+1,j}
       +(1-s)tF_{i,j+1}+stF_{i+1,j+1},
```
```math
s=\frac{x-u_i}{u_{i+1}-u_i},\qquad
t=\frac{y-v_j}{v_{j+1}-v_j}.
```

Domain: the closed rectangular hull, including all edges and corners. Outside it the source returns NaN. The array axes follow the two grid vectors in the order shown; there is no automatic transposition. Cell-interior derivatives are valid; cell-boundary derivatives generally are not. Convex interpolation preserves positivity and any common linear normalization of interpolated template entries, but interpolation in log values is a different model.

<a id="engine-support"></a>
## Engine support

The independent tests evaluate linear operations, Poisson and Gaussian laws, the unrestricted Gaussian marginal, full Barlow–Beeston, and both interpolation functions through the JS module loader and worker. The compiled runtime fixture and the synthetic fit example emit StableHLO with the Rust CLI. Backend value/gradient checks should use `tests/runtime.flatppl` and its independent Python oracle.

The mathematical suite includes the fixed-total fixture `tests/fixed-total.flatppl`, which checks the normalized multinomial MC and Poisson data joint law. Its JS score agrees with the independent SciPy joint-log-PMF oracle `-5.895049151094491`, and the Rust CLI emits the fixture as StableHLO.
