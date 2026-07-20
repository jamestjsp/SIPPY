import argparse
import gc
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


def _simulate_dataset(
    kind: str, sample_count: int
) -> tuple[np.ndarray, np.ndarray, int]:
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
    inputs = rng.normal(size=(B.shape[1], sample_count))
    outputs = np.zeros((C.shape[0], sample_count))
    state = np.zeros(A.shape[0])
    for sample in range(sample_count):
        outputs[:, sample] = C @ state
        state = A @ state + B @ inputs[:, sample]
    outputs += rng.normal(scale=0.01, size=outputs.shape)
    return outputs, inputs, A.shape[0]


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
        y, u, order = _simulate_dataset(dataset, args.samples)
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
