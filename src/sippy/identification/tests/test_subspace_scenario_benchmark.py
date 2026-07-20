import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_subspace_scenario_grid_runs_every_method_without_hidden_failures():
    root = Path(__file__).parents[4]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "benchmark_subspace.py"),
            "--scenario-grid",
            "--samples",
            "600",
            "--seeds",
            "1",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    results = report["results"]
    methods = {
        "SSARX",
        "N4SID",
        "MOESP",
        "CVA",
        "PARSIM-K",
        "PARSIM-S",
        "PARSIM-P",
    }
    scenarios = {
        "open-loop-siso",
        "open-loop-mimo",
        "closed-loop-mathworks",
        "closed-loop-two-excitation",
    }

    assert report["configuration"] == {
        "fixed_true_order": True,
        "future_horizon": 12,
        "past_horizon": 24,
    }
    assert len(results) == len(methods) * len(scenarios)
    assert {(result["scenario"], result["method"]) for result in results} == {
        (scenario, method) for scenario in scenarios for method in methods
    }
    for result in results:
        assert result["successful_seeds"] == 1
        assert result["median_seconds"] > 0.0
        for metric in (
            "median_held_out_nrmse",
            "median_frequency_response_error",
            "median_pole_error",
            "median_markov_parameter_error",
        ):
            assert np.isfinite(result[metric])

    by_case = {(result["scenario"], result["method"]): result for result in results}
    mathworks_ssarx = by_case[("closed-loop-mathworks", "SSARX")]
    mathworks_parsim_k = by_case[("closed-loop-mathworks", "PARSIM-K")]
    assert mathworks_ssarx["median_held_out_nrmse"] < 0.25
    assert mathworks_ssarx["median_frequency_response_error"] < 0.25
    assert mathworks_parsim_k["median_held_out_nrmse"] < 0.3
