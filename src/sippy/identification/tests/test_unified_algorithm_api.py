import inspect

import pandas as pd
import pytest

from sippy.identification.base import StateSpaceModel
from sippy.identification.factory import AlgorithmFactory, create_algorithm
from sippy.identification.iddata import IDData

from .simulation_scenarios import simulate_scenario, stable_siso_plant

ALGORITHM_NAMES = [
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
    "FREQUENCY_DOMAIN",
    "FREQ_DOMAIN",
]

SUBSPACE_NAMES = ["N4SID", "MOESP", "CVA", "PARSIM-K", "PARSIM-S", "PARSIM-P"]


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


def test_registered_algorithm_set_is_complete():
    assert set(AlgorithmFactory.list_algorithms()) == set(ALGORITHM_NAMES)


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
    model = create_algorithm(method).identify(
        iddata=iddata,
        ss_f=8,
        ss_p=8,
        ss_fixed_order=2,
    )

    assert isinstance(model, StateSpaceModel)
    assert model.ts == pytest.approx(iddata.sample_time)
