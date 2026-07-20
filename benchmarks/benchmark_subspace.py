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

from sippy.identification.factory import create_algorithm


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


def _simulate_response(A, B, C, D, inputs):
    outputs = np.zeros((C.shape[0], inputs.shape[1]))
    state = np.zeros(A.shape[0])
    for sample in range(inputs.shape[1]):
        outputs[:, sample] = C @ state + D @ inputs[:, sample]
        state = A @ state + B @ inputs[:, sample]
    return outputs


def _simulate_dataset(kind: str, sample_count: int) -> BenchmarkDataset:
    rng = np.random.default_rng(20260720 if kind == "siso" else 20260721)
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


def _workload(method: str, y: np.ndarray, u: np.ndarray, order: int):
    options = {"tsample": 1.0}
    if method == "N4SID":
        options.update({"ss_f": 15, "ss_fixed_order": order})
    elif method == "PARSIM-K":
        options.update({"ss_f": 15, "ss_p": 15, "ss_fixed_order": order})
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare canonical and named compact subspace estimators."
    )
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    repeat = 2 if args.quick else 5
    horizon = 15
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
