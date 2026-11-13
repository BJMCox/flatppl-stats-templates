"""Independent NumPy/SciPy oracles; no engine implementation supplies answers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from scipy import optimize, stats
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_POINTS = np.array([[2.0, 3.0, 0.2, 0.4, 0.3], [0.7, 1.4, -0.3, 2.2, 1.1]])
T = np.array([[0.3, 0.1], [0.1, 0.6], [0.2, 0.2]])
Q = np.array([1.0, 2.0, 0.5])
R = np.array([[0.7, 0.1, 0.0], [0.1, 0.5, 0.6]])
B = np.array([0.5, 0.7, 0.2])
C = np.array([[1.2, 0.1, 0.0], [0.1, 1.8, 0.2], [0.0, 0.2, 0.9]])
V = np.array([[0.2, -0.04], [-0.04, 0.3]])
XI = np.array([[0.0, 2.0], [3.0, 4.0], [1.0, 0.0]])
MC = np.array([[0, 1], [4, 2], [1, 0]])
N = np.array([3, 4, 0])


def runtime_oracle(x):
    """The eleven outputs of runtime.flatppl, using independent libraries."""
    w = np.asarray(x[:2])
    deformation = np.array([[x[2], 0.0], [0.0, -x[2]], [x[2], x[2]]])
    deformed = T * np.exp(deformation)
    normalized = deformed / (Q @ deformed)
    mu = T @ w + B
    covariance = np.exp(x[2]) * C
    grid = RegularGridInterpolator(
        ([0.0, 1.0, 3.0], [-1.0, 2.0]), [[0.0, 2.0], [3.0, 6.0], [4.0, 10.0]]
    )
    return np.array(
        [
            (T @ w)[0],
            normalized[1, 0],
            (R @ (Q[:, None] * T))[0, 1],
            (R @ (Q * (T @ w)))[1],
            deformed[0, 0],
            stats.poisson.logpmf(N, mu).sum(),
            stats.multivariate_normal.logpdf([2.1, 3.2, 0.5], mu, covariance),
            stats.multivariate_normal.logpdf(
                [2.1, 3.2, 0.5], mu, covariance + T @ V @ T.T
            ),
            stats.poisson.logpmf(N, XI @ w + B).sum()
            + stats.poisson.logpmf(MC, XI).sum(),
            np.interp(x[3], [0.0, 1.0, 3.0], [1.0, 3.0, 2.0]),
            grid([[x[3], x[4]]]).item(),
        ]
    )


def literal(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return "[" + ", ".join(literal(v) for v in value) + "]"
    return repr(float(value))


def decode(value):
    if isinstance(value, str):
        return float(value)
    if not isinstance(value, dict):
        return np.asarray(value)
    data = value["data"]
    if isinstance(data, dict):
        data = list(data.values())
    return np.asarray([float(v) for v in data]).reshape(value["shape"])


def evaluate(request):
    result = subprocess.run(
        ["node", str(ROOT / "tests/evaluate.cjs")],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
    )
    return [decode(v) for v in json.loads(result.stdout)]


def close(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual).ravel(), np.asarray(expected).ravel(), rtol=2e-12, atol=2e-12
    )


def run():
    # Each assertion locks a different mathematical contract or boundary.
    matrix = "rowstack([[0.3,0.1],[0.1,0.6],[0.2,0.2]])"
    response = "rowstack([[0.7,0.1,0.0],[0.1,0.5,0.6]])"
    expressions = [
        f"a.combine({matrix},[2.,3.])",
        f"a.normalize_columns({matrix},[1.,2.,0.5])",
        f"a.project({response},[0.9,2.,1.],[1.,2.,0.5])",
        f"a.fold_templates({response},{matrix},[1.,2.,0.5])",
        "a.log_deform(rowstack([[0.,2.],[3.,4.]]),rowstack([[1.,-0.2],[0.3,0.]]))",
        "logdensityof(a.poisson_model(rowstack([[0.],[1.]]),[2.],[0.,0.]),[0,3])",
        "a.linear_interpolate([0.,1.,3.],[1.,3.,2.],0.)",
        "a.linear_interpolate([0.,1.,3.],[1.,3.,2.],1.)",
        "a.linear_interpolate([0.,1.,3.],[1.,3.,2.],3.)",
        "a.bilinear_interpolate([0.,1.,3.],[-1.,2.],rowstack([[0.,2.],[3.,6.],[4.,10.]]),3.,-1.)",
        "a.linear_interpolate([0.,1.,3.],[1.,3.,2.],-0.1)",
        "a.bilinear_interpolate([0.,1.],[-1.,2.],rowstack([[0.,2.],[3.,6.]]),0.3,2.1)",
        "logdensityof(a.gaussian_marginal(rowstack([[1.,-0.2],[0.4,1.]]),[-1.,2.],rowstack([[0.3,0.1],[0.1,0.2]]),[0.,1.],rowstack([[1.,0.1],[0.1,2.]])),[0.2,-0.5])",
    ]
    expected = [
        T @ [2.0, 3.0],
        T / (Q @ T),
        R @ (Q * [0.9, 2.0, 1.0]),
        R @ (Q[:, None] * T),
        np.array([[0.0, 2.0], [3.0, 4.0]]) * np.exp([[1.0, -0.2], [0.3, 0.0]]),
        stats.poisson.logpmf([0, 3], [0, 2]).sum(),
        1.0,
        3.0,
        2.0,
        4.0,
    ]
    values = evaluate(expressions)
    for actual, want in zip(values[:10], expected):
        close(actual, want)
    assert np.isnan(values[10]) and np.isnan(values[11])
    signed = np.array([[1.0, -0.2], [0.4, 1.0]])
    close(
        values[12],
        stats.multivariate_normal.logpdf(
            [0.2, -0.5],
            signed @ [-1.0, 2.0] + [0.0, 1.0],
            np.array([[1.0, 0.1], [0.1, 2.0]])
            + signed @ [[0.3, 0.1], [0.1, 0.2]] @ signed.T,
        ),
    )
    source = (ROOT / "tests/runtime.flatppl").read_text()
    for x in RUNTIME_POINTS:
        expressions = [
            "prediction[1]",
            "normalized[2,1]",
            "folded[1,2]",
            "projected[2]",
            "deformed[1,1]",
            "poisson_score",
            "gaussian_score",
            "marginal_score",
            "bb_score",
            "linear_value",
            "bilinear_value",
        ]
        request = {
            "source": source.replace("elementof(cartpow(reals, 5))", literal(x))
            + "\n"
            + "\n".join(f"result{i} = {expr}" for i, expr in enumerate(expressions)),
            "bindings": [f"result{i}" for i in range(len(expressions))],
        }
        close(
            [float(np.asarray(v).item()) for v in evaluate(request)], runtime_oracle(x)
        )
    # Column/strength rescaling and folded-vs-direct predictions use public calls.
    close(
        evaluate([f"a.combine({matrix},[2.,3.])"])[0],
        evaluate(["a.combine(rowstack([[0.6,0.05],[0.2,0.3],[0.4,0.1]]),[1.,6.])"])[0],
    )
    close(
        evaluate(
            [f"a.combine(a.fold_templates({response},{matrix},[1.,2.,0.5]),[2.,3.])"]
        )[0],
        R @ (Q * (T @ [2.0, 3.0])),
    )
    # Shared MC observations contribute one auxiliary factor across two data sets.
    shared = """flatppl_compat = "0.1"
a = load_module("../main.flatppl")
xi = rowstack([[0.,2.],[3.,4.]])
law = joint(data_a = a.poisson_model(xi,[2.,0.5],[1.,1.]),
            data_b = a.poisson_model(xi,[0.3,1.],[0.2,0.5]), mc = Poisson.(xi))
score = logdensityof(law,record(data_a=[1,7],data_b=[2,4],mc=rowstack([[0,1],[4,2]])))
"""
    xi = np.array([[0.0, 2.0], [3.0, 4.0]])
    shared_oracle = (
        stats.poisson.logpmf([1, 7], xi @ [2.0, 0.5] + 1.0).sum()
        + stats.poisson.logpmf([2, 4], xi @ [0.3, 1.0] + [0.2, 0.5]).sum()
        + stats.poisson.logpmf([[0, 1], [4, 2]], xi).sum()
    )
    close(evaluate({"source": shared, "bindings": ["score"]})[0], shared_oracle)
    # An independent fit checks the example's score and predictions at its optimum.
    fit_templates = np.array([[0.6, 0.1], [0.3, 0.3], [0.1, 0.6]])
    objective = lambda w: (
        -stats.poisson.logpmf([17, 14, 12], fit_templates @ w + 1.0).sum()
    )
    fit = optimize.minimize(
        objective,
        [20.0, 20.0],
        bounds=[(0.0, None), (0.0, None)],
        method="L-BFGS-B",
        options={"ftol": 1e-13, "gtol": 1e-8},
    )
    example = (
        (ROOT / "examples/histogram-fit.flatppl")
        .read_text()
        .replace("elementof(cartpow(nonnegreals, 2))", literal(fit.x))
    )
    example_score, example_mean = evaluate(
        {"source": example, "bindings": ["log_likelihood", "expected"]}
    )
    close(example_score, -objective(fit.x))
    close(example_mean, fit_templates @ fit.x + 1.0)
    print(
        json.dumps(
            {
                "math_checks": "passed",
                "runtime_points": RUNTIME_POINTS.tolist(),
                "fit_strengths": fit.x.tolist(),
                "fit_log_likelihood": -objective(fit.x),
            }
        )
    )


def check_fixed_total():
    source = (ROOT / "tests/fixed-total.flatppl").read_text()
    expected = stats.poisson.logpmf(
        [2, 5], np.array([[0.2, 0.6], [0.8, 0.4]]) @ [2.0, 4.0] + 1.0
    ).sum()
    expected += stats.multinomial.logpmf([2, 8], 10, [0.2, 0.8])
    expected += stats.multinomial.logpmf([3, 2], 5, [0.6, 0.4])
    close(evaluate({"source": source, "bindings": ["outputs"]})[0], expected)
    print(json.dumps({"fixed_total": "passed", "oracle": expected}))


if __name__ == "__main__":
    run()
    check_fixed_total()
