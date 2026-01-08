from abc import ABC
from dataclasses import dataclass, replace
from enum import Enum
from typing import Self, TypeVar


@dataclass(frozen=True)
class MachineSpecs:
    """
    This class is a dataclass holding machine specs

    Parameters
    ----------
    max_gantry_speed : float
        The maximum gantry speed in deg/sec
    max_mlc_position : float
        The max mlc position in mm
    max_mlc_overtravel : float
        The maximum distance in mm the MLC leaves can overtravel from each other as well as the jaw size (for tail exposure protection).
    max_mlc_speed : float
        The maximum speed of the MLC leaves in mm/s.
    """

    max_gantry_speed: float
    max_mlc_position: float
    max_mlc_overtravel: float
    max_mlc_speed: float

    def replace(self, **overrides) -> Self:
        return replace(self, **overrides)


class MachineBase(ABC):
    """This is a base class that represents a generic machine (TrueBeam or Halcyon)"""

    specs: MachineSpecs


TMachine = TypeVar("TMachine", bound=MachineBase)


class GantryDirection(Enum):
    CLOCKWISE = "CW"
    COUNTER_CLOCKWISE = "CC"
    NONE = "NONE"


class GantrySpeedTransition(Enum):
    LEADING = "leading"
    TRAILING = "trailing"


class FluenceMode(Enum):
    STANDARD = "STANDARD"
    FFF = "FFF"
    SRS = "SRS"
