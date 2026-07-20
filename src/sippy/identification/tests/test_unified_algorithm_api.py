import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from sippy.identification.base import StateSpaceModel
from sippy.identification.factory import AlgorithmFactory, create_algorithm
from sippy.identification.iddata import IDData
from sippy.identification.parameters import (
    ALGORITHM_OPTIONS,
    normalize_identification_options,
)

from .simulation_scenarios import (
    simulate_scenario,
    stable_mimo_plant,
    stable_siso_plant,
)

ALGORITHM_NAMES = [
    "SUBSPACE",
    "SSARX",
    "N4SID",
    "MOESP",
    "CVA",
    "PARSIM-K",
    "PARSIM-S",
    "PARSIM-P",
    "ARX",
    "ARARX",
    "ARARMAX",
    "FIR",
    "ARMAX",
    "OE",
    "ARMA",
    "BJ",
    "GEN",
    "FD",
]

SUBSPACE_NAMES = [
    "SUBSPACE",
    "SSARX",
    "N4SID",
    "MOESP",
    "CVA",
    "PARSIM-K",
    "PARSIM-S",
    "PARSIM-P",
]
CANONICAL_NAMES = [
    "SUBSPACE",
    "SSARX",
    "N4SID",
    "MOESP",
    "CVA",
    "PARSIM-K",
    "PARSIM-S",
    "PARSIM-P",
    "ARX",
    "ARARX",
    "ARARMAX",
    "FIR",
    "ARMAX",
    "OE",
    "ARMA",
    "BJ",
    "GEN",
    "FD",
]
STOCHASTIC_SUBSPACE_NAMES = ["N4SID", "MOESP", "CVA"]
PREDICTOR_GAIN_NAMES = ["SUBSPACE", "SSARX", "PARSIM-K", "PARSIM-S", "PARSIM-P"]
PARSIM_NAMES = ["PARSIM-K", "PARSIM-S", "PARSIM-P"]


def test_parameter_aliases_normalize_to_one_vocabulary():
    with pytest.warns(DeprecationWarning, match="theta.*nk"):
        options = normalize_identification_options("ARMAX", {"theta": 2})

    assert options == {"nk": 3}


def test_conflicting_parameter_alias_is_rejected():
    with pytest.warns(DeprecationWarning, match="theta.*nk"):
        with pytest.raises(ValueError, match="Conflicting values"):
            normalize_identification_options("ARMAX", {"theta": 1, "nk": 1})


@pytest.mark.parametrize(
    "method,bundle_name,bundle,expected",
    [
        ("ARX", "arx_orders", [2, 3, 0], {"na": 2, "nb": 3, "nk": 1}),
        (
            "ARMAX",
            "armax_orders",
            [2, 3, 1, 1],
            {"na": 2, "nb": 3, "nc": 1, "nk": 2},
        ),
        (
            "ARARX",
            "ararx_orders",
            [[2], [[3]], [1], [[1]]],
            {"na": [2], "nb": [[3]], "nd": [1], "nk": [[2]]},
        ),
        (
            "ARARMAX",
            "ararmax_orders",
            [2, 3, 1, 1, 0],
            {"na": 2, "nb": 3, "nc": 1, "nd": 1, "nk": 1},
        ),
        (
            "BJ",
            "bj_orders",
            [3, 1, 2, 4, 2],
            {"nb": 3, "nc": 1, "nd": 2, "nf": 4, "nk": 3},
        ),
    ],
)
def test_master_order_bundles_translate_to_canonical_options(
    method, bundle_name, bundle, expected
):
    with pytest.warns(DeprecationWarning, match=bundle_name):
        options = normalize_identification_options(method, {bundle_name: bundle})

    assert options == expected


def test_invalid_master_order_bundle_is_rejected_instead_of_using_defaults():
    with pytest.raises(ValueError, match="ararmax_orders.*five"):
        normalize_identification_options("ARARMAX", {"ararmax_orders": [2, 2, 1]})


def test_master_order_bundle_conflict_is_rejected():
    with pytest.warns(DeprecationWarning, match="arx_orders"):
        with pytest.raises(ValueError, match="Conflicting values.*na"):
            normalize_identification_options("ARX", {"arx_orders": [2, 2, 0], "na": 3})


def test_master_order_bundle_warning_can_be_suppressed():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        options = normalize_identification_options(
            "ARARMAX",
            {"ararmax_orders": [2, 2, 1, 1, 0]},
            warn_deprecated=False,
        )

    assert options["nk"] == 1


@pytest.fixture(scope="module")
def siso_data():
    scenario = simulate_scenario(
        stable_siso_plant(dt=0.25),
        n_train=180,
        n_validation=80,
        input_kind="white",
        snr_db=40,
        seed=101,
    )
    frame = pd.DataFrame(
        {
            "u": scenario.u_train[0],
            "y": scenario.y_train[0],
        }
    )
    return scenario, IDData(frame, inputs=["u"], outputs=["y"], tsample=0.25)


@pytest.fixture(scope="module")
def mimo_data():
    return simulate_scenario(
        stable_mimo_plant(dt=0.2),
        n_train=240,
        n_validation=80,
        input_kind="white",
        snr_db=35,
        input_correlation=0.25,
        seed=102,
    )


def test_registered_algorithm_set_is_complete():
    assert set(AlgorithmFactory.list_algorithms()) == set(ALGORITHM_NAMES)


@pytest.mark.parametrize("alias", ["FREQUENCY_DOMAIN", "FREQ_DOMAIN"])
def test_frequency_domain_aliases_resolve_without_duplicate_registration(alias):
    assert type(create_algorithm(alias)) is type(create_algorithm("FD"))


@pytest.mark.parametrize("method", ALGORITHM_NAMES)
def test_every_algorithm_exposes_the_unified_identify_signature(method):
    signature = inspect.signature(type(create_algorithm(method)).identify)
    parameters = list(signature.parameters.values())

    assert [parameter.name for parameter in parameters[:4]] == [
        "self",
        "y",
        "u",
        "iddata",
    ]
    assert parameters[-1].kind is inspect.Parameter.VAR_KEYWORD


@pytest.mark.parametrize("method", ALGORITHM_NAMES)
def test_every_algorithm_rejects_mixed_data_sources(method, siso_data):
    scenario, iddata = siso_data
    algorithm = create_algorithm(method)

    with pytest.raises(ValueError, match="either|Either|Provide"):
        algorithm.identify(
            y=scenario.y_train,
            u=scenario.u_train,
            iddata=iddata,
            ss_f=8,
            ss_p=8,
            ss_fixed_order=2,
        )


@pytest.mark.parametrize("method", SUBSPACE_NAMES)
def test_subspace_algorithms_accept_iddata_and_preserve_sample_time(method, siso_data):
    _, iddata = siso_data
    options = {"ss_f": 8, "ss_fixed_order": 2}
    if method in PARSIM_NAMES:
        options["ss_p"] = 8
    model = create_algorithm(method).identify(
        iddata=iddata,
        **options,
    )

    assert isinstance(model, StateSpaceModel)
    assert model.ts == pytest.approx(iddata.sample_time)


@pytest.mark.parametrize("method", CANONICAL_NAMES)
def test_every_identification_result_obeys_the_common_result_contract(
    method, siso_data
):
    scenario, _ = siso_data
    algorithm = create_algorithm(method)
    options = {
        "ss_f": 8,
        "ss_p": 8,
        "ss_fixed_order": 2,
        "na": 1,
        "nb": 1,
        "nc": 1,
        "nd": 1,
        "nf": 1,
        "nk": 1,
        "max_iterations": 40,
        "tsample": 0.25,
    }
    options = {
        name: value
        for name, value in options.items()
        if name == "tsample" or name in ALGORITHM_OPTIONS[method]
    }
    if method == "ARMA":
        model = algorithm.identify(y=scenario.y_train, **options)
    else:
        model = algorithm.identify(y=scenario.y_train, u=scenario.u_train, **options)

    assert isinstance(model, StateSpaceModel)
    assert model.method == method
    assert model.noutputs == scenario.y_train.shape[0]
    assert model.ninputs == (0 if method == "ARMA" else scenario.u_train.shape[0])
    assert model.ts == pytest.approx(0.25)
    assert set(model.identification_info) >= {
        "method",
        "model_type",
        "n_inputs",
        "n_outputs",
        "sample_time",
        "provenance",
        "options",
    }
    for operation in (
        "frequency_response",
        "uncertainty",
        "simulation",
        "prediction",
        "one_step_prediction",
        "residuals",
        "fit",
        "stability",
        "modal_properties",
        "time_response",
        "innovations_response",
        "stochastic_state_space",
    ):
        assert isinstance(model.supports(operation), bool)

    if method == "FD":
        assert model.Yid is None
        assert model.residual_covariance is None
        assert not model.supports("simulation")
    else:
        assert model.Yid.shape == scenario.y_train.shape
        assert model.residuals().shape == scenario.y_train.shape
        assert model.residual_covariance.shape == (model.noutputs, model.noutputs)
        assert np.isfinite(model.Vn)
        assert model.fit()["nrmse"].shape == (model.noutputs,)

    if method == "ARMA":
        assert model.supports("one_step_prediction")
        assert not model.supports("simulation")
        assert model.predict(y=scenario.y_validation).shape == (
            1,
            scenario.y_validation.shape[1],
        )

    if method in STOCHASTIC_SUBSPACE_NAMES:
        assert model.supports("stochastic_state_space")
    elif method in PREDICTOR_GAIN_NAMES:
        assert model.K is not None
        assert model.Q is None and model.R is None and model.S is None
    else:
        assert model.K is None
        assert model.Q is None and model.R is None and model.S is None


@pytest.mark.parametrize("method", [name for name in CANONICAL_NAMES if name != "ARMA"])
def test_input_output_algorithms_preserve_the_common_contract_for_mimo(
    method, mimo_data
):
    orders = {
        "na": 1,
        "nb": [[1, 1], [1, 1]] if method == "ARARX" else 1,
        "nc": 1,
        "nd": 1,
        "nf": 1,
        "nk": 1,
    }
    options = {
        "ss_f": 8,
        "ss_p": 8,
        "ss_fixed_order": 3,
        "max_iterations": 20,
        "tsample": mimo_data.sample_time,
        **orders,
    }
    options = {
        name: value
        for name, value in options.items()
        if name == "tsample" or name in ALGORITHM_OPTIONS[method]
    }
    model = create_algorithm(method).identify(
        y=mimo_data.y_train,
        u=mimo_data.u_train,
        **options,
    )

    assert model.ninputs == 2
    assert model.noutputs == 2
    assert model.identification_info["n_inputs"] == 2
    assert model.identification_info["n_outputs"] == 2
    assert model.frequency_response(np.array([0.1, 0.5])).frdata.shape == (2, 2, 2)
    if method == "FD":
        assert not model.supports("simulation")
    else:
        assert model.Yid.shape == mimo_data.y_train.shape
        assert model.predict(u=mimo_data.u_validation).shape == (
            2,
            mimo_data.u_validation.shape[1],
        )
