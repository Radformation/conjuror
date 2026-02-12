from abc import ABC
from collections.abc import Sequence
from enum import StrEnum

from pydantic import Field

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence

from conjuror.plans.mlc import MLCModulator
from conjuror.plans.plan_generator import QAProcedureBase
from conjuror.plans.machine import MachineSpecs, MachineBase, FluenceMode
from conjuror.plans.beam import Beam as BeamBase

MLC_BOUNDARIES_HAL_DIST = tuple(np.arange(-140, 140 + 1, 10).astype(float))
MLC_BOUNDARIES_HAL_PROX = tuple(np.arange(-145, 145 + 1, 10).astype(float))

DEFAULT_SPECS_HAL = MachineSpecs(
    max_gantry_speed=24.0,
    max_mlc_position=140,
    max_mlc_overtravel=140,
    max_mlc_speed=25,
)


class HalcyonMachine(MachineBase):
    """A class that represents a TrueBeam machine."""

    def __init__(self, machine_specs: MachineSpecs | None = None):
        self.machine_specs = machine_specs or DEFAULT_SPECS_HAL
        self.mlc_boundaries_dist = MLC_BOUNDARIES_HAL_DIST
        self.mlc_boundaries_prox = MLC_BOUNDARIES_HAL_PROX


class Stack(StrEnum):
    DISTAL = "distal"
    PROXIMAL = "proximal"
    BOTH = "both"


class Beam(BeamBase[HalcyonMachine]):
    """A class that represents a Halcyon beam."""

    def __init__(
        self,
        beam_name: str,
        metersets: Sequence[float],
        gantry_angles: float | Sequence[float],
        distal_mlc_positions: list[list[float]],
        proximal_mlc_positions: list[list[float]],
        coll_angle: float,
        couch_vrt: float,
        couch_lat: float,
        couch_lng: float,
    ):
        """
        Parameters
        ----------
        beam_name : str
            The name of the beam. Must be less than 16 characters.
        metersets : Sequence[float]
            The meter sets for each control point. The length must match the number of control points in mlc_positions.
        gantry_angles : Union[float, Sequence[float]]
            The gantry angle(s) of the beam. If a single number, it's assumed to be a static beam. If multiple numbers, it's assumed to be a dynamic beam.
        distal_mlc_positions : list[list[float]]
            The distal MLC positions for each control point. This is the x-position of each leaf for each control point.
        proximal_mlc_positions : list[list[float]]
            The proximal MLC positions for each control point. This is the x-position of each leaf for each control point.
        coll_angle : float
            The collimator angle.
        couch_vrt : float
            The couch vertical position.
        couch_lat : float
            The couch lateral position.
        couch_lng : float
            The couch longitudinal position.
        """
        jaw_x = Dataset()
        jaw_x.RTBeamLimitingDeviceType = "X"
        jaw_x.NumberOfLeafJawPairs = 1
        jaw_y = Dataset()
        jaw_y.RTBeamLimitingDeviceType = "Y"
        jaw_y.NumberOfLeafJawPairs = 1
        mlc_x1 = Dataset()
        mlc_x1.RTBeamLimitingDeviceType = "MLCX1"
        mlc_x1.NumberOfLeafJawPairs = 28
        mlc_x1.LeafPositionBoundaries = list(MLC_BOUNDARIES_HAL_DIST)
        mlc_x2 = Dataset()
        mlc_x2.RTBeamLimitingDeviceType = "MLCX2"
        mlc_x2.NumberOfLeafJawPairs = 29
        mlc_x2.LeafPositionBoundaries = list(MLC_BOUNDARIES_HAL_PROX)
        bld_sequence = DicomSequence((jaw_x, jaw_y, mlc_x1, mlc_x2))

        beam_limiting_device_positions = {
            "X": [[-140, 140]],
            "Y": [[-140, 140]],
            "MLCX1": distal_mlc_positions,
            "MLCX2": proximal_mlc_positions,
        }

        super().__init__(
            beam_limiting_device_sequence=bld_sequence,
            beam_name=beam_name,
            energy=6,
            fluence_mode=FluenceMode.FFF,
            dose_rate=600,
            metersets=metersets,
            gantry_angles=gantry_angles,
            beam_limiting_device_positions=beam_limiting_device_positions,
            coll_angle=coll_angle,
            couch_vrt=couch_vrt,
            couch_lat=couch_lat,
            couch_lng=couch_lng,
            couch_rot=0,
        )

    @staticmethod
    def create_mlc(machine: HalcyonMachine) -> tuple[MLCModulator, MLCModulator]:
        """Create 2 MLC shaper objects, one for each stack."""
        proximal_mlc = MLCModulator(
            leaf_y_positions=machine.mlc_boundaries_prox,
            max_mlc_position=machine.machine_specs.max_mlc_position,
            max_overtravel_mm=machine.machine_specs.max_mlc_overtravel,
            sacrifice_gap_mm=None,
            sacrifice_max_move_mm=None,
        )
        distal_mlc = MLCModulator(
            leaf_y_positions=machine.mlc_boundaries_dist,
            max_mlc_position=machine.machine_specs.max_mlc_position,
            max_overtravel_mm=machine.machine_specs.max_mlc_overtravel,
            sacrifice_gap_mm=None,
            sacrifice_max_move_mm=None,
        )
        return proximal_mlc, distal_mlc


class QAProcedure(QAProcedureBase[HalcyonMachine], ABC):
    pass


class PicketFence(QAProcedure):
    """Add a picket fence beam to the plan. The beam will be delivered with the MLCs stacked on top of each other."""

    stack: Stack = Field(
        title="Stack",
        description="Which MLC stack to use for the beam. The other stack will be parked.",
    )
    strip_width_mm: float = Field(
        default=3,
        title="Strip Width",
        description="The width of the strips.",
        json_schema_extra={"units": "mm"},
    )
    strip_positions_mm: tuple[float, ...] = Field(
        default=(-45, -30, -15, 0, 15, 30, 45),
        title="Strip Positions",
        description="The positions of the strips.",
        json_schema_extra={"units": "mm"},
    )
    gantry_angle: float = Field(
        default=0,
        title="Gantry Angle",
        description="The gantry angle of the beam.",
        json_schema_extra={"units": "degrees"},
    )
    coll_angle: float = Field(
        default=0,
        title="Collimator Angle",
        description="The collimator angle of the beam.",
        json_schema_extra={"units": "degrees"},
    )
    couch_vrt: float = Field(
        default=0, title="Couch Vertical", description="The couch vertical position."
    )
    couch_lng: float = Field(
        default=1000,
        title="Couch Longitudinal",
        description="The couch longitudinal position.",
    )
    couch_lat: float = Field(
        default=0, title="Couch Lateral", description="The couch lateral position."
    )
    mu: int = Field(
        default=200, title="Monitor Units", description="The monitor units of the beam."
    )
    beam_name: str = Field(
        default="PF", title="Beam Name", description="The name of the beam."
    )

    def compute(self, machine: HalcyonMachine) -> None:
        prox_mlc, dist_mlc = Beam.create_mlc(machine)

        # we prepend the positions with an initial starting position 2mm from the first strip
        # that way, each picket is the same cadence where the leaves move into position dynamically.
        # If you didn't do this, the first picket might be different as it has the advantage
        # of starting from a static position vs the rest of the pickets being dynamic.
        strip_positions = [self.strip_positions_mm[0] - 2, *self.strip_positions_mm]
        metersets = [
            0,
            *[1 / len(self.strip_positions_mm) for _ in self.strip_positions_mm],
        ]

        for strip, meterset in zip(strip_positions, metersets):
            if self.stack in (Stack.DISTAL, Stack.BOTH):
                dist_mlc.add_strip(
                    position_mm=strip,
                    strip_width_mm=self.strip_width_mm,
                    meterset_at_target=meterset,
                )
                if self.stack == Stack.DISTAL:
                    prox_mlc.park(meterset=meterset)
            if self.stack in (Stack.PROXIMAL, Stack.BOTH):
                prox_mlc.add_strip(
                    position_mm=strip,
                    strip_width_mm=self.strip_width_mm,
                    meterset_at_target=meterset,
                )
                if self.stack == Stack.PROXIMAL:
                    dist_mlc.park(meterset=meterset)

        beam = Beam(
            beam_name=self.beam_name,
            gantry_angles=self.gantry_angle,
            coll_angle=self.coll_angle,
            couch_vrt=self.couch_vrt,
            couch_lat=self.couch_lat,
            couch_lng=self.couch_lng,
            proximal_mlc_positions=prox_mlc.as_control_points(),
            distal_mlc_positions=dist_mlc.as_control_points(),
            # can use either MLC for metersets
            metersets=[self.mu * m for m in prox_mlc.as_metersets()],
        )
        self.beams.append(beam)
