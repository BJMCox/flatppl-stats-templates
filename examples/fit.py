"""Fit the synthetic histogram through the FlatPPL JS observation law."""

import json
import subprocess
from pathlib import Path

from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "examples/histogram-fit.flatppl").read_text()


def evaluate(strengths):
    source = SOURCE.replace(
        "elementof(cartpow(nonnegreals, 2))",
        "[" + ", ".join(repr(float(v)) for v in strengths) + "]",
    )
    result = subprocess.run(
        ["node", str(ROOT / "tests/evaluate.cjs")],
        input=json.dumps(
            {"source": source, "bindings": ["log_likelihood", "expected"]}
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    score, prediction = json.loads(result.stdout)
    return score["data"][0], prediction["data"]


if __name__ == "__main__":
    fit = minimize(
        lambda strengths: -evaluate(strengths)[0],
        [20.0, 20.0],
        method="L-BFGS-B",
        bounds=[(0.0, None), (0.0, None)],
        options={"ftol": 1e-12, "gtol": 1e-7},
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    score, prediction = evaluate(fit.x)
    print(
        json.dumps(
            {
                "input_strengths": fit.x.tolist(),
                "log_likelihood": score,
                "expected_counts": prediction,
                "observed_counts": [17, 14, 12],
            }
        )
    )
