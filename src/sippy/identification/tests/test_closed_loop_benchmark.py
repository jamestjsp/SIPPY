import json
import subprocess
import sys
from pathlib import Path


def test_closed_loop_benchmark_exercises_predictor_and_reference_routes():
    root = Path(__file__).parents[4]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "benchmark_closed_loop.py"),
            "--skip-motor",
            "--samples",
            "2400",
            "--repeat",
            "1",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    results = json.loads(completed.stdout)["results"]
    by_case = {
        (result["dataset"], result["reference_mode"]): result for result in results
    }
    mathworks_reference = by_case[("mathworks-ssarx", "measured")]
    predictor = by_case[("two-independent-excitations", "unavailable")]
    ort = by_case[("two-independent-excitations", "measured")]

    assert mathworks_reference["estimator_route"] == "predictor"
    assert mathworks_reference["reference_status"] == "fallback"
    assert mathworks_reference["reference_reason"] == (
        "reference_deterministic_regressor_rank_deficient"
    )
    assert predictor["estimator_route"] == "predictor"
    assert predictor["reference_status"] == "not-provided"
    assert ort["estimator_route"] == "two-stage-ort"
    assert ort["reference_status"] == "used"
    assert predictor["selected_order"] == ort["selected_order"] == 2
    assert predictor["stable"]
    assert ort["stable"]
    assert predictor["validation_nrmse"] < 0.35
    assert ort["validation_nrmse"] < 0.35
    assert predictor["frequency_response_error"] < 0.35
    assert ort["frequency_response_error"] < 0.35
    assert predictor["pole_error"] < 0.25
    assert ort["pole_error"] < 0.5
    assert predictor["median_seconds"] > 0.0
    assert ort["median_seconds"] > 0.0
