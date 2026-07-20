import argparse
import gc
import itertools
import json
import platform
import statistics
import timeit
import tracemalloc
from dataclasses import asdict, dataclass

import numpy as np
from benchmark_closed_loop import (
    mathworks_closed_loop_dataset,
    two_excitation_closed_loop_dataset,
)

from sippy.identification.factory import create_algorithm

COMPARISON_METHODS = (
    "SSARX",
    "N4SID",
    "MOESP",
    "CVA",
    "PARSIM-K",
    "PARSIM-S",
    "PARSIM-P",
)


@dataclass(frozen=True)
class BenchmarkResult:
    dataset: str
    method: str
    median_seconds: float
    peak_python_bytes: int
    selected_order: int
    estimator_route: str
    refit_on_full_record: bool
    held_out_nrmse: float
    frequency_response_error: float
    pole_error: float
    markov_parameter_error: float


@dataclass(frozen=True)
class BenchmarkDataset:
    outputs: np.ndarray
    inputs: np.ndarray
    validation_outputs: np.ndarray
    validation_inputs: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray


@dataclass(frozen=True)
class ScenarioBenchmarkResult:
    scenario: str
    method: str
    successful_seeds: int
    median_seconds: float
    median_held_out_nrmse: float
    median_frequency_response_error: float
    median_pole_error: float
    median_markov_parameter_error: float


def _simulate_response(A, B, C, D, inputs):
    outputs = np.zeros((C.shape[0], inputs.shape[1]))
    state = np.zeros(A.shape[0])
    for sample in range(inputs.shape[1]):
        outputs[:, sample] = C @ state + D @ inputs[:, sample]
        state = A @ state + B @ inputs[:, sample]
    return outputs


def _simulate_dataset(
    kind: str, sample_count: int, seed: int | None = None
) -> BenchmarkDataset:
    default_seed = 20260720 if kind == "siso" else 20260721
    rng = np.random.default_rng(default_seed if seed is None else seed)
    if kind == "siso":
        A = np.array([[0.72, 0.12], [-0.08, 0.84]])
        B = np.array([[0.35], [0.18]])
        C = np.array([[1.0, -0.25]])
    else:
        A = np.array(
            [
                [0.72, 0.08, 0.0],
                [-0.04, 0.81, 0.06],
                [0.02, -0.05, 0.66],
            ]
        )
        B = np.array([[0.35, 0.08], [0.12, 0.28], [0.18, -0.15]])
        C = np.array([[1.0, 0.2, -0.1], [0.15, -0.25, 0.9]])
    D = np.zeros((C.shape[0], B.shape[1]))
    inputs = rng.normal(size=(B.shape[1], sample_count))
    outputs = _simulate_response(A, B, C, D, inputs)
    outputs += rng.normal(scale=0.01, size=outputs.shape)
    validation_inputs = rng.normal(size=(B.shape[1], 400))
    validation_outputs = _simulate_response(A, B, C, D, validation_inputs)
    return BenchmarkDataset(
        outputs=outputs,
        inputs=inputs,
        validation_outputs=validation_outputs,
        validation_inputs=validation_inputs,
        A=A,
        B=B,
        C=C,
        D=D,
    )


def _closed_loop_fixture(kind: str, sample_count: int, seed: int) -> BenchmarkDataset:
    if kind == "closed-loop-mathworks":
        dataset = mathworks_closed_loop_dataset(
            sample_count=sample_count,
            seed=seed,
        )
    elif kind == "closed-loop-two-excitation":
        dataset = two_excitation_closed_loop_dataset(
            sample_count=sample_count,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown closed-loop benchmark scenario: {kind}")
    if dataset.reference_system is None:
        raise ValueError(f"{kind} must provide a ground-truth state-space model")
    A, B, C, D = dataset.reference_system
    return BenchmarkDataset(
        outputs=dataset.outputs,
        inputs=dataset.inputs,
        validation_outputs=dataset.validation_outputs,
        validation_inputs=dataset.validation_inputs,
        A=A,
        B=B,
        C=C,
        D=D,
    )


def _scenario_fixture(kind: str, sample_count: int, seed: int) -> BenchmarkDataset:
    if kind == "open-loop-siso":
        return _simulate_dataset("siso", sample_count, seed)
    if kind == "open-loop-mimo":
        return _simulate_dataset("mimo", sample_count, seed)
    return _closed_loop_fixture(kind, sample_count, seed)


def _relative_error(expected, actual):
    scale = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(expected - actual) / scale)


def _frequency_response(A, B, C, D):
    frequencies = np.linspace(0.0, np.pi, 128)
    identity = np.eye(A.shape[0])
    return np.stack(
        [
            C @ np.linalg.solve(np.exp(1j * omega) * identity - A, B) + D
            for omega in frequencies
        ]
    )


def _markov_parameters(A, B, C, D, count=12):
    parameters = [D]
    state_power = np.eye(A.shape[0])
    for _ in range(1, count):
        parameters.append(C @ state_power @ B)
        state_power = state_power @ A
    return np.stack(parameters)


def _accuracy_metrics(dataset, model):
    predicted = _simulate_response(
        model.A,
        model.B,
        model.C,
        model.D,
        dataset.validation_inputs,
    )
    centered = dataset.validation_outputs - np.mean(
        dataset.validation_outputs, axis=1, keepdims=True
    )
    output_errors = np.sqrt(
        np.sum((dataset.validation_outputs - predicted) ** 2, axis=1)
        / np.maximum(
            np.sum(centered**2, axis=1),
            np.finfo(np.float64).tiny,
        )
    )
    expected_poles = np.linalg.eigvals(dataset.A)
    actual_poles = np.linalg.eigvals(model.A)
    if expected_poles.shape != actual_poles.shape:
        pole_error = np.inf
    else:
        pole_error = min(
            _relative_error(expected_poles, actual_poles[list(order)])
            for order in itertools.permutations(range(actual_poles.size))
        )
    return {
        "held_out_nrmse": float(np.max(output_errors)),
        "frequency_response_error": _relative_error(
            _frequency_response(dataset.A, dataset.B, dataset.C, dataset.D),
            _frequency_response(model.A, model.B, model.C, model.D),
        ),
        "pole_error": pole_error,
        "markov_parameter_error": _relative_error(
            _markov_parameters(dataset.A, dataset.B, dataset.C, dataset.D),
            _markov_parameters(model.A, model.B, model.C, model.D),
        ),
    }


def _workload(
    method: str,
    y: np.ndarray,
    u: np.ndarray,
    order: int,
    *,
    horizon: int = 15,
    past_horizon: int | None = None,
):
    options = {"tsample": 1.0}
    if method in {"N4SID", "MOESP", "CVA"}:
        options.update({"ss_f": horizon, "ss_fixed_order": order})
    elif method in {"SSARX", "PARSIM-K", "PARSIM-S", "PARSIM-P"}:
        options.update(
            {
                "ss_f": horizon,
                "ss_p": horizon if past_horizon is None else past_horizon,
                "ss_fixed_order": order,
            }
        )
    return lambda: create_algorithm(method).identify(y=y, u=u, **options)


def _measure(workload, repeat: int) -> tuple[float, int, object]:
    model = workload()
    seconds = statistics.median(timeit.repeat(workload, number=1, repeat=repeat))
    gc.collect()
    tracemalloc.start()
    workload()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return seconds, peak_bytes, model


def _run_scenario_grid(
    *,
    sample_count: int,
    seed_count: int,
    horizon: int,
    past_horizon: int,
) -> list[ScenarioBenchmarkResult]:
    scenarios = (
        "open-loop-siso",
        "open-loop-mimo",
        "closed-loop-mathworks",
        "closed-loop-two-excitation",
    )
    results = []
    for scenario_index, scenario in enumerate(scenarios):
        timings = {method: [] for method in COMPARISON_METHODS}
        metrics = {method: [] for method in COMPARISON_METHODS}
        for seed_index in range(seed_count):
            seed = 20260720 + 100 * scenario_index + seed_index
            fixture = _scenario_fixture(scenario, sample_count, seed)
            order = fixture.A.shape[0]
            for method in COMPARISON_METHODS:
                workload = _workload(
                    method,
                    fixture.outputs,
                    fixture.inputs,
                    order,
                    horizon=horizon,
                    past_horizon=past_horizon,
                )
                start = timeit.default_timer()
                model = workload()
                timings[method].append(timeit.default_timer() - start)
                metrics[method].append(_accuracy_metrics(fixture, model))

        for method in COMPARISON_METHODS:
            method_metrics = metrics[method]
            results.append(
                ScenarioBenchmarkResult(
                    scenario=scenario,
                    method=method,
                    successful_seeds=len(method_metrics),
                    median_seconds=float(statistics.median(timings[method])),
                    median_held_out_nrmse=float(
                        statistics.median(
                            metric["held_out_nrmse"] for metric in method_metrics
                        )
                    ),
                    median_frequency_response_error=float(
                        statistics.median(
                            metric["frequency_response_error"]
                            for metric in method_metrics
                        )
                    ),
                    median_pole_error=float(
                        statistics.median(
                            metric["pole_error"] for metric in method_metrics
                        )
                    ),
                    median_markov_parameter_error=float(
                        statistics.median(
                            metric["markov_parameter_error"]
                            for metric in method_metrics
                        )
                    ),
                )
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare canonical and named compact subspace estimators."
    )
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--scenario-grid", action="store_true")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    if args.samples < 64:
        parser.error("--samples must be at least 64")
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    repeat = 2 if args.quick else 5
    horizon = 15
    if args.scenario_grid:
        grid_horizon = 12
        grid_past_horizon = 24
        grid_results = _run_scenario_grid(
            sample_count=args.samples,
            seed_count=args.seeds,
            horizon=grid_horizon,
            past_horizon=grid_past_horizon,
        )
        print(
            json.dumps(
                {
                    "environment": {
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                        "platform": platform.platform(),
                        "samples": args.samples,
                        "seeds": args.seeds,
                    },
                    "configuration": {
                        "fixed_true_order": True,
                        "future_horizon": grid_horizon,
                        "past_horizon": grid_past_horizon,
                    },
                    "results": [asdict(result) for result in grid_results],
                },
                indent=2,
            )
        )
        return
    results = []
    structural_bounds = {}

    for dataset in ("siso", "mimo"):
        fixture = _simulate_dataset(dataset, args.samples)
        y = fixture.outputs
        u = fixture.inputs
        order = fixture.A.shape[0]
        channel_count = y.shape[0] + u.shape[0]
        usable_columns = args.samples - 2 * horizon + 1
        compact_lq_rows = 2 * horizon * channel_count
        reusable_qr_rows = (horizon + 1) * channel_count
        structural_bounds[dataset] = {
            "hankel_columns": usable_columns,
            "compact_lq_max_dimension": compact_lq_rows,
            "reusable_predictor_qr_max_rows": reusable_qr_rows,
            "avoids_sample_square_projector": usable_columns > compact_lq_rows,
        }
        for method in ("SUBSPACE", "N4SID", "PARSIM-K"):
            seconds, peak_bytes, model = _measure(
                _workload(method, y, u, order), repeat
            )
            accuracy = _accuracy_metrics(fixture, model)
            results.append(
                BenchmarkResult(
                    dataset=dataset,
                    method=method,
                    median_seconds=seconds,
                    peak_python_bytes=peak_bytes,
                    selected_order=model.n,
                    estimator_route=model.identification_info.get(
                        "estimator_route", "named"
                    ),
                    refit_on_full_record=model.identification_info.get(
                        "refit_on_full_record", False
                    ),
                    **accuracy,
                )
            )

    print(
        json.dumps(
            {
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "platform": platform.platform(),
                    "samples": args.samples,
                    "repeat": repeat,
                },
                "structural_bounds": structural_bounds,
                "results": [asdict(result) for result in results],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
