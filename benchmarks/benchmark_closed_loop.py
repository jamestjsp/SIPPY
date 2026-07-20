import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import statistics
import tempfile
import timeit
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import sippy

MATHWORKS_SSARX_URL = "https://www.mathworks.com/help/ident/ref/n4sid.html"
MOTOR_DATASET_URL = "https://data.mendeley.com/public-api/zip/5xvg43r9r8/download/1"
MOTOR_DATASET_DOI = "10.17632/5xvg43r9r8.1"
MOTOR_DATASET_SHA256 = (
    "0f1781b7443dc5f6cfaab8e8cb473ca84832f8d8e4ad8e0b368da09d1a1a06df"
)
MOTOR_DATASET_MEMBER_SUFFIX = (
    "05_discrete_controller_validation/"
    "Case_c_integrator_with_added_zero_at_0.66_gain_of_9.9/raw_data.txt"
)


@dataclass(frozen=True)
class ClosedLoopDataset:
    name: str
    outputs: np.ndarray
    inputs: np.ndarray
    references: np.ndarray
    sample_time: float
    validation_outputs: np.ndarray
    validation_inputs: np.ndarray
    reference_system: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
    comparison_target: str
    validation_context: str


@dataclass(frozen=True)
class ClosedLoopBenchmarkResult:
    dataset: str
    algorithm: str
    reference_mode: str
    estimator_route: str
    reference_status: str
    reference_reason: str | None
    selected_order: int
    selected_horizon: int
    stable: bool
    median_seconds: float
    comparison_target: str
    validation_context: str
    validation_nrmse: float
    frequency_response_error: float | None
    pole_error: float | None


def _simulate(A, B, C, D, inputs, initial_state=None):
    state = (
        np.zeros(A.shape[0])
        if initial_state is None
        else np.asarray(initial_state, dtype=float).reshape(-1).copy()
    )
    outputs = np.empty((C.shape[0], inputs.shape[1]))
    for sample in range(inputs.shape[1]):
        outputs[:, sample] = C @ state + D @ inputs[:, sample]
        state = A @ state + B @ inputs[:, sample]
    return outputs


def mathworks_closed_loop_dataset(
    sample_count=3000,
    validation_count=600,
    seed=20260720,
):
    rng = np.random.default_rng(seed)
    A = np.array([[1.5, -0.7], [1.0, 0.0]])
    B = np.array([[1.0], [0.0]])
    C = np.array([[1.0, 0.5]])
    D = np.zeros((1, 1))
    excitation = rng.normal(size=(1, sample_count))
    noise = 0.8 * lfilter(
        [1.0, 0.5],
        [1.0, 1.5, 0.7],
        rng.normal(size=sample_count),
    )
    state = np.zeros(2)
    outputs = np.empty((1, sample_count))
    inputs = np.empty((1, sample_count))
    for sample in range(sample_count):
        clean_output = (C @ state).item()
        outputs[0, sample] = clean_output + noise[sample]
        inputs[0, sample] = -0.5 * outputs[0, sample] + excitation[0, sample]
        state = A @ state + B[:, 0] * inputs[0, sample]

    validation_inputs = rng.normal(size=(1, validation_count))
    validation_outputs = _simulate(A, B, C, D, validation_inputs)
    return ClosedLoopDataset(
        name="mathworks-ssarx",
        outputs=outputs,
        inputs=inputs,
        references=excitation,
        sample_time=1.0,
        validation_outputs=validation_outputs,
        validation_inputs=validation_inputs,
        reference_system=(A, B, C, D),
        comparison_target="known-ground-truth",
        validation_context="held-out-open-loop-simulation",
    )


def two_excitation_closed_loop_dataset(
    sample_count=3000,
    validation_count=600,
    seed=20260721,
):
    rng = np.random.default_rng(seed)
    A = np.array([[0.72, 0.12], [-0.08, 0.84]])
    B = np.array([[0.35], [0.18]])
    C = np.array([[1.0, -0.25]])
    D = np.zeros((1, 1))
    reference = rng.normal(size=(1, sample_count))
    dither = 0.1 * rng.normal(size=(1, sample_count))
    disturbance = lfilter(
        [0.06],
        [1.0, -0.7],
        rng.normal(size=sample_count),
    )
    state = np.zeros(2)
    outputs = np.empty((1, sample_count))
    inputs = np.empty((1, sample_count))
    for sample in range(sample_count):
        outputs[0, sample] = (C @ state).item() + disturbance[sample]
        inputs[0, sample] = (
            1.1 * (reference[0, sample] - outputs[0, sample]) + dither[0, sample]
        )
        state = A @ state + B[:, 0] * inputs[0, sample]

    validation_inputs = rng.normal(size=(1, validation_count))
    validation_outputs = _simulate(A, B, C, D, validation_inputs)
    return ClosedLoopDataset(
        name="two-independent-excitations",
        outputs=outputs,
        inputs=inputs,
        references=np.vstack((reference, dither)),
        sample_time=1.0,
        validation_outputs=validation_outputs,
        validation_inputs=validation_inputs,
        reference_system=(A, B, C, D),
        comparison_target="known-ground-truth",
        validation_context="held-out-open-loop-simulation",
    )


def _read_motor_frame(archive_path):
    archive = Path(archive_path)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if MOTOR_DATASET_SHA256 and digest != MOTOR_DATASET_SHA256:
        raise ValueError(
            f"motor dataset checksum mismatch: expected {MOTOR_DATASET_SHA256}, "
            f"got {digest}"
        )
    with zipfile.ZipFile(archive) as zipped:
        matches = [
            name
            for name in zipped.namelist()
            if name.endswith(MOTOR_DATASET_MEMBER_SUFFIX)
        ]
        if len(matches) != 1:
            raise ValueError(
                "motor dataset archive must contain exactly one case-C raw record"
            )
        with zipped.open(matches[0]) as raw_file:
            text = io.TextIOWrapper(raw_file, encoding="utf-8")
            frame = pd.read_csv(text, skiprows=2)
    required = {"REF", "MEAS", "DT_ms", "PWM"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"motor dataset is missing columns: {sorted(missing)}")
    if not np.all(np.isfinite(frame[list(required)].to_numpy(dtype=float))):
        raise ValueError("motor identification channels must be finite")
    return frame


def openmct_motor_dataset(archive_path):
    frame = _read_motor_frame(archive_path)
    split = int(0.7 * len(frame))
    train = frame.iloc[:split]
    validation = frame.iloc[split:]
    input_center = float(train["PWM"].mean())
    output_center = float(train["MEAS"].mean())
    reference_center = float(train["REF"].mean())
    sample_time = float(np.median(frame["DT_ms"])) / 1000.0
    reference_A = np.array([[0.63763]])
    reference_B = np.array([[1.0]])
    reference_C = np.array([[0.41488]])
    reference_D = np.zeros((1, 1))
    return ClosedLoopDataset(
        name="openmct-dc-motor-case-c",
        outputs=(train["MEAS"].to_numpy(dtype=float) - output_center)[None, :],
        inputs=(train["PWM"].to_numpy(dtype=float) - input_center)[None, :],
        references=(train["REF"].to_numpy(dtype=float) - reference_center)[None, :],
        sample_time=sample_time,
        validation_outputs=(validation["MEAS"].to_numpy(dtype=float) - output_center)[
            None, :
        ],
        validation_inputs=(validation["PWM"].to_numpy(dtype=float) - input_center)[
            None, :
        ],
        reference_system=(
            reference_A,
            reference_B,
            reference_C,
            reference_D,
        ),
        comparison_target="published-open-loop-fit",
        validation_context="held-out-closed-loop-record",
    )


def _download_motor_archive():
    temporary = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    temporary.close()
    path = Path(temporary.name)
    try:
        request = urllib.request.Request(
            MOTOR_DATASET_URL,
            headers={"Accept": "*/*", "User-Agent": "curl/8.7.1"},
        )
        with urllib.request.urlopen(request) as response, path.open("wb") as output:
            shutil.copyfileobj(response, output)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _validation_initial_state(model, inputs, outputs):
    state_count = model.A.shape[0]
    if state_count == 0:
        return np.empty(0)
    sample_count = min(outputs.shape[1], max(4 * state_count, 20))
    zero_state = _simulate(
        model.A,
        model.B,
        model.C,
        model.D,
        inputs[:, :sample_count],
    )
    observability = []
    state_power = np.eye(state_count)
    for _ in range(sample_count):
        observability.append(model.C @ state_power)
        state_power = state_power @ model.A
    matrix = np.vstack(observability)
    target = (outputs[:, :sample_count] - zero_state).T.reshape(-1)
    return np.linalg.lstsq(matrix, target, rcond=None)[0]


def _nrmse(expected, actual, burn_in=0):
    expected = expected[:, burn_in:]
    actual = actual[:, burn_in:]
    centered = expected - np.mean(expected, axis=1, keepdims=True)
    numerator = np.sum((expected - actual) ** 2, axis=1)
    denominator = np.maximum(
        np.sum(centered**2, axis=1),
        np.finfo(np.float64).tiny,
    )
    return float(np.max(np.sqrt(numerator / denominator)))


def _frequency_response(A, B, C, D):
    frequencies = np.linspace(0.0, np.pi, 256)
    identity = np.eye(A.shape[0])
    return np.stack(
        [
            C @ np.linalg.solve(np.exp(1j * omega) * identity - A, B) + D
            for omega in frequencies
        ]
    )


def _relative_error(expected, actual):
    scale = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(expected - actual) / scale)


def _pole_error(expected, actual):
    expected = np.sort_complex(np.linalg.eigvals(expected))
    actual = np.sort_complex(np.linalg.eigvals(actual))
    if expected.shape != actual.shape:
        return float("inf")
    return _relative_error(expected, actual)


def benchmark_dataset(dataset, *, method, use_reference, order, horizon, repeat):
    options = {
        "method": method,
        "ss_f": horizon,
        "ss_fixed_order": order,
        "tsample": dataset.sample_time,
    }
    if method == "SSARX":
        options["ss_p"] = 2 * horizon
    elif use_reference:
        options["reference"] = dataset.references

    def identify():
        return sippy.identify(dataset.outputs, dataset.inputs, **options)

    model = identify()
    timings = timeit.repeat(identify, number=1, repeat=repeat)
    initial_state = _validation_initial_state(
        model,
        dataset.validation_inputs,
        dataset.validation_outputs,
    )
    prediction = _simulate(
        model.A,
        model.B,
        model.C,
        model.D,
        dataset.validation_inputs,
        initial_state=initial_state,
    )
    burn_in = min(max(2 * model.n, 10), prediction.shape[1] // 4)
    reference_projection = model.identification_info.get(
        "reference_projection",
        {"status": "not-applicable", "reason": None},
    )
    frequency_error = None
    pole_error = None
    if dataset.reference_system is not None:
        A, B, C, D = dataset.reference_system
        frequency_error = _relative_error(
            _frequency_response(A, B, C, D),
            _frequency_response(model.A, model.B, model.C, model.D),
        )
        pole_error = _pole_error(A, model.A)
    return ClosedLoopBenchmarkResult(
        dataset=dataset.name,
        algorithm=model.method,
        reference_mode="measured" if use_reference else "unavailable",
        estimator_route=model.identification_info["estimator_route"],
        reference_status=reference_projection["status"],
        reference_reason=reference_projection["reason"],
        selected_order=model.n,
        selected_horizon=(
            model.identification_info["selected_horizon"]
            if "selected_horizon" in model.identification_info
            else model.identification_info["future_horizon"]
        ),
        stable=bool(np.all(np.abs(np.linalg.eigvals(model.A)) < 1.0)),
        median_seconds=float(statistics.median(timings)),
        comparison_target=dataset.comparison_target,
        validation_context=dataset.validation_context,
        validation_nrmse=_nrmse(
            dataset.validation_outputs,
            prediction,
            burn_in=burn_in,
        ),
        frequency_response_error=frequency_error,
        pole_error=pole_error,
    )


def run_synthetic_benchmark(sample_count=3000, repeat=3):
    datasets = (
        mathworks_closed_loop_dataset(sample_count=sample_count),
        two_excitation_closed_loop_dataset(sample_count=sample_count),
    )
    subspace_results = [
        benchmark_dataset(
            dataset,
            method="SUBSPACE",
            use_reference=use_reference,
            order=2,
            horizon=12,
            repeat=repeat,
        )
        for dataset in datasets
        for use_reference in (False, True)
    ]
    ssarx_results = [
        benchmark_dataset(
            dataset,
            method="SSARX",
            use_reference=False,
            order=2,
            horizon=12,
            repeat=repeat,
        )
        for dataset in datasets
    ]
    return subspace_results + ssarx_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SIPPY closed-loop SUBSPACE and SSARX identification."
    )
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--skip-motor", action="store_true")
    args = parser.parse_args()

    archive = args.archive
    downloaded_archive = False
    results = run_synthetic_benchmark(args.samples, args.repeat)
    try:
        if not args.skip_motor:
            if archive is None:
                archive = _download_motor_archive()
                downloaded_archive = True
            motor = openmct_motor_dataset(archive)
            for use_reference in (False, True):
                results.append(
                    benchmark_dataset(
                        motor,
                        method="SUBSPACE",
                        use_reference=use_reference,
                        order=1,
                        horizon=8,
                        repeat=args.repeat,
                    )
                )
            results.append(
                benchmark_dataset(
                    motor,
                    method="SSARX",
                    use_reference=False,
                    order=1,
                    horizon=8,
                    repeat=args.repeat,
                )
            )
    finally:
        if downloaded_archive and archive is not None:
            archive.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "platform": platform.platform(),
                    "samples": args.samples,
                    "repeat": args.repeat,
                },
                "sources": {
                    "synthetic": {
                        "name": "MathWorks n4sid closed-loop SSARX example",
                        "url": MATHWORKS_SSARX_URL,
                    },
                    "experimental": {
                        "doi": MOTOR_DATASET_DOI,
                        "record": MOTOR_DATASET_MEMBER_SUFFIX,
                        "sha256": MOTOR_DATASET_SHA256,
                    },
                },
                "results": [asdict(result) for result in results],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
