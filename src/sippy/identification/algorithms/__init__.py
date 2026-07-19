"""Concrete algorithm implementations and registrations."""

from ..factory import AlgorithmFactory
from .ararmax import ARARMAXAlgorithm
from .ararx import ARARXAlgorithm
from .arma import ARMAAlgorithm
from .armax import ARMAXAlgorithm
from .arx import ARXAlgorithm
from .bj import BJAlgorithm
from .cva import CVAAlgorithm
from .fir import FIRAlgorithm
from .frequency_domain import FrequencyDomainAlgorithm
from .gen import GENAlgorithm
from .moesp import MOESPAlgorithm
from .n4sid import N4SIDAlgorithm
from .oe import OEAlgorithm
from .parsim_k import PARSIMKAlgorithm
from .parsim_p import PARSIMPAlgorithm
from .parsim_s import PARSIMSAlgorithm

AlgorithmFactory.register("N4SID", N4SIDAlgorithm)
AlgorithmFactory.register("MOESP", MOESPAlgorithm)
AlgorithmFactory.register("CVA", CVAAlgorithm)
AlgorithmFactory.register("PARSIM-K", PARSIMKAlgorithm)
AlgorithmFactory.register("PARSIM-S", PARSIMSAlgorithm)
AlgorithmFactory.register("PARSIM-P", PARSIMPAlgorithm)
AlgorithmFactory.register("ARX", ARXAlgorithm)
AlgorithmFactory.register("ARARX", ARARXAlgorithm)
AlgorithmFactory.register("ARARMAX", ARARMAXAlgorithm)
AlgorithmFactory.register("FIR", FIRAlgorithm)
AlgorithmFactory.register("ARMAX", ARMAXAlgorithm)
AlgorithmFactory.register("OE", OEAlgorithm)
AlgorithmFactory.register("ARMA", ARMAAlgorithm)
AlgorithmFactory.register("BJ", BJAlgorithm)
AlgorithmFactory.register("GEN", GENAlgorithm)
AlgorithmFactory.register("FD", FrequencyDomainAlgorithm)

__all__ = [
    "ARARMAXAlgorithm",
    "ARARXAlgorithm",
    "ARMAAlgorithm",
    "ARMAXAlgorithm",
    "ARXAlgorithm",
    "BJAlgorithm",
    "CVAAlgorithm",
    "FIRAlgorithm",
    "FrequencyDomainAlgorithm",
    "GENAlgorithm",
    "MOESPAlgorithm",
    "N4SIDAlgorithm",
    "OEAlgorithm",
    "PARSIMKAlgorithm",
    "PARSIMPAlgorithm",
    "PARSIMSAlgorithm",
]
