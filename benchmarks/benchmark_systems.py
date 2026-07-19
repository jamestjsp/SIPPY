import argparse
import json
import platform
import statistics
import timeit
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from sippy import systems
from sippy.identification.algorithms.arx import ARXAlgorithm
from sippy.identification.algorithms.n4sid import N4SIDAlgorithm


@dataclass(frozen=True)
class Benchmark:
    name: str
    current: Callable[[], object]
    baseline: Callable[[], object] | None
    number: int


@dataclass(frozen=True)
class Result:
    name: str
    current_seconds: float
    baseline_seconds: float | None
    baseline_over_current: float | None


def _measure(function: Callable[[], object], number: int, repeat: int) -> float:
    function()
    samples = timeit.repeat(function, number=number, repeat=repeat)
    return statistics.median(samples) / number


def _identification_data(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arx_samples = 20_000
    u_arx = rng.normal(size=(1, arx_samples))
    y_arx = np.zeros((1, arx_samples))
    noise = rng.normal(scale=0.01, size=arx_samples)
    for sample in range(2, arx_samples):
        y_arx[0, sample] = (
            1.35 * y_arx[0, sample - 1]
            - 0.46 * y_arx[0, sample - 2]
            + 0.30 * u_arx[0, sample - 1]
            + 0.10 * u_arx[0, sample - 2]
            + noise[sample]
        )

    n4sid_samples = 2_500
    A = np.array(
        [
            [0.82, 0.10, 0.0, 0.0],
            [-0.05, 0.75, 0.08, 0.0],
            [0.0, 0.0, 0.65, 0.12],
            [0.02, 0.0, -0.08, 0.70],
        ]
    )
    B = np.array([[0.25, 0.05], [0.10, 0.20], [0.15, -0.10], [0.0, 0.18]])
    C = np.array([[1.0, 0.0, 0.3, 0.0], [0.0, 0.7, 0.0, 1.0]])
    u_n4sid = rng.normal(size=(2, n4sid_samples))
    y_n4sid = np.zeros((2, n4sid_samples))
    state = np.zeros(4)
    for sample in range(n4sid_samples):
        y_n4sid[:, sample] = C @ state
        state = A @ state + B @ u_n4sid[:, sample]
    y_n4sid += rng.normal(scale=0.005, size=y_n4sid.shape)
    return u_arx, y_arx, u_n4sid, y_n4sid


def _build_benchmarks(control: object | None) -> tuple[list[Benchmark], dict]:
    rng = np.random.default_rng(20260719)
    siso_num = [0.2, 0.1, 0.05]
    siso_den = [1.0, -1.2, 0.55, -0.08]
    mimo_num = [
        [[0.2, 0.1], [0.1]],
        [[0.3], [0.4, 0.2]],
    ]
    mimo_den = [
        [[1.0, -0.5, 0.1], [1.0, -0.4]],
        [[1.0, -0.3], [1.0, -0.2, 0.05]],
    ]
    shared_denominator = np.poly(np.linspace(0.2, 0.8, 6)).real.tolist()
    shared_num = [
        [[0.01 * (1 + output + input_), 0.02 * (1 + output)] for input_ in range(4)]
        for output in range(4)
    ]
    shared_den = [[shared_denominator for _ in range(4)] for _ in range(4)]

    current_siso_tf = systems.tf(siso_num, siso_den, 0.1)
    current_mimo_tf = systems.tf(mimo_num, mimo_den, 0.1)
    current_shared_tf = systems.tf(shared_num, shared_den, 0.1)
    baseline_siso_tf = control.tf(siso_num, siso_den, 0.1) if control else None
    baseline_mimo_tf = control.tf(mimo_num, mimo_den, 0.1) if control else None
    baseline_shared_tf = control.tf(shared_num, shared_den, 0.1) if control else None

    state_count = 40
    A = np.diag(np.linspace(0.2, 0.9, state_count))
    A += rng.normal(scale=0.002, size=(state_count, state_count))
    spectral_radius = np.max(np.abs(np.linalg.eigvals(A)))
    A *= 0.9 / spectral_radius
    B = rng.normal(scale=0.1, size=(state_count, 2))
    C = rng.normal(scale=0.1, size=(2, state_count))
    D = rng.normal(scale=0.01, size=(2, 2))
    current_ss = systems.ss(A, B, C, D, 0.1)
    baseline_ss = control.ss(A, B, C, D, 0.1) if control else None
    frequencies = np.geomspace(0.01, 20.0, 512)
    short_input = rng.normal(size=(2, 10))
    long_input = rng.normal(size=(2, 20_000))
    u_arx, y_arx, u_n4sid, y_n4sid = _identification_data(rng)

    benchmarks = [
        Benchmark(
            "tf2ss_siso",
            lambda: systems.tf2ss(current_siso_tf),
            (lambda: control.tf2ss(baseline_siso_tf, method="slycot"))
            if control
            else None,
            100,
        ),
        Benchmark(
            "tf2ss_mimo",
            lambda: systems.tf2ss(current_mimo_tf),
            (lambda: control.tf2ss(baseline_mimo_tf, method="slycot"))
            if control
            else None,
            50,
        ),
        Benchmark(
            "tf2ss_shared_4x4",
            lambda: systems.tf2ss(current_shared_tf),
            (lambda: control.tf2ss(baseline_shared_tf, method="slycot"))
            if control
            else None,
            20,
        ),
        Benchmark(
            "ss2tf_40_state",
            lambda: systems.ss2tf(current_ss),
            (lambda: control.ss2tf(baseline_ss)) if control else None,
            20,
        ),
        Benchmark(
            "frequency_response_tf_siso_512",
            lambda: systems.frequency_response(current_siso_tf, frequencies).frdata,
            (lambda: control.frequency_response(baseline_siso_tf, frequencies).frdata)
            if control
            else None,
            10,
        ),
        Benchmark(
            "frequency_response_tf_mimo_512",
            lambda: systems.frequency_response(current_mimo_tf, frequencies).frdata,
            (lambda: control.frequency_response(baseline_mimo_tf, frequencies).frdata)
            if control
            else None,
            5,
        ),
        Benchmark(
            "frequency_response_ss_40_state_512",
            lambda: systems.frequency_response(current_ss, frequencies).frdata,
            (lambda: control.frequency_response(baseline_ss, frequencies).frdata)
            if control
            else None,
            3,
        ),
        Benchmark(
            "forced_response_short",
            lambda: systems.forced_response(
                current_ss, U=short_input, squeeze=False
            ).outputs,
            (
                lambda: control.forced_response(
                    baseline_ss, U=short_input, squeeze=False
                ).outputs
            )
            if control
            else None,
            100,
        ),
        Benchmark(
            "forced_response_long",
            lambda: systems.forced_response(
                current_ss, U=long_input, squeeze=False
            ).outputs,
            (
                lambda: control.forced_response(
                    baseline_ss, U=long_input, squeeze=False
                ).outputs
            )
            if control
            else None,
            3,
        ),
        Benchmark(
            "identify_arx_20000",
            lambda: ARXAlgorithm().identify(
                y=y_arx, u=u_arx, na=8, nb=8, nk=1, tsample=0.1
            ),
            None,
            3,
        ),
        Benchmark(
            "identify_n4sid_2500",
            lambda: N4SIDAlgorithm().identify(
                y=y_n4sid,
                u=u_n4sid,
                ss_f=15,
                ss_fixed_order=4,
                ss_d_required=True,
                tsample=0.1,
            ),
            None,
            1,
        ),
    ]

    metadata = {
        "current_shared_states": systems.tf2ss(current_shared_tf).nstates,
        "baseline_shared_states": (
            control.tf2ss(baseline_shared_tf, method="slycot").nstates
            if control
            else None
        ),
    }
    if control:
        current_frequency = systems.frequency_response(current_ss, frequencies).frdata
        baseline_frequency = control.frequency_response(baseline_ss, frequencies).frdata
        current_output = systems.forced_response(
            current_ss, U=short_input, squeeze=False
        ).outputs
        baseline_output = control.forced_response(
            baseline_ss, U=short_input, squeeze=False
        ).outputs
        metadata["frequency_max_abs_error"] = float(
            np.max(np.abs(current_frequency - baseline_frequency))
        )
        metadata["simulation_max_abs_error"] = float(
            np.max(np.abs(current_output - baseline_output))
        )
    return benchmarks, metadata


def _load_control(enabled: bool) -> object | None:
    if not enabled:
        return None
    try:
        import control
        import slycot  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "Comparison requires control and slycot. Run with "
            "`uv run --with control==0.10.2 --with slycot==0.7.0 python "
            "benchmarks/benchmark_systems.py --compare-control`."
        ) from error
    return control


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SIPPY's ctrlsys-backed model operations."
    )
    parser.add_argument(
        "--compare-control",
        action="store_true",
        help="compare with python-control and Slycot in the same environment",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use fewer repetitions for a fast smoke benchmark",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="run benchmarks whose names contain this value",
    )
    args = parser.parse_args()

    control = _load_control(args.compare_control)
    benchmarks, metadata = _build_benchmarks(control)
    if args.only:
        benchmarks = [
            benchmark
            for benchmark in benchmarks
            if any(pattern in benchmark.name for pattern in args.only)
        ]
    repeat = 3 if args.quick else 7
    results = []
    for benchmark in benchmarks:
        number = max(1, benchmark.number // 5) if args.quick else benchmark.number
        current_seconds = _measure(benchmark.current, number, repeat)
        baseline_seconds = (
            _measure(benchmark.baseline, number, repeat)
            if benchmark.baseline is not None
            else None
        )
        results.append(
            Result(
                name=benchmark.name,
                current_seconds=current_seconds,
                baseline_seconds=baseline_seconds,
                baseline_over_current=(
                    baseline_seconds / current_seconds
                    if baseline_seconds is not None
                    else None
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
                },
                "metadata": metadata,
                "results": [asdict(result) for result in results],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
