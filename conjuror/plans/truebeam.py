import math
import warnings
from abc import ABC
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

import numpy as np
from plotly import graph_objects as go
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from scipy.interpolate import make_interp_spline

from conjuror.images.simulators import Imager
from conjuror.plans.mlc import MLCModulator, MLCShaper, Rectangle, RectangleMode, Strip
from conjuror.plans.plan_generator import QAProcedureBase
from conjuror.plans.machine import (
    MachineSpecs,
    MachineBase,
    GantryDirection,
    FluenceMode,
)
from conjuror.plans.beam import Beam as BeamBase
from conjuror.utils import wrap360


# MLC boundaries are immutable by design, so they are stored as tuples.
# However, pydicom expects lists, so these are converted to lists when necessary (Beam).
MLC_BOUNDARIES_TB_MIL120 = (
    tuple(np.arange(-200, -100 + 1, 10).astype(float))
    + tuple(np.arange(-95, 95 + 1, 5).astype(float))
    + tuple(np.arange(100, 200 + 1, 10).astype(float))
)
MLC_BOUNDARIES_TB_HD120 = (
    tuple(np.arange(-110, -40 + 1, 5).astype(float))
    + tuple(np.arange(-37.5, 37.5 + 1, 2.5).astype(float))
    + tuple(np.arange(40, 110 + 1, 5).astype(float))
)

DEFAULT_SPECS_TB = MachineSpecs(
    max_gantry_speed=6.0, max_mlc_position=200, max_mlc_overtravel=200, max_mlc_speed=25
)


class MLCLeafBoundaryAlignmentMode(StrEnum):
    EXACT = RectangleMode.EXACT
    ROUND = RectangleMode.ROUND
    INWARD = RectangleMode.INWARD
    OUTWARD = RectangleMode.OUTWARD


@dataclass
class WinstonLutzField:
    gantry: float
    collimator: float
    couch: float
    name: str | None = None


class TrueBeamMachine(MachineBase):
    """A class that represents a TrueBeam machine."""

    def __init__(self, mlc_is_hd: bool, specs: MachineSpecs | None = None):
        self.mlc_is_hd = mlc_is_hd
        self.specs = specs or DEFAULT_SPECS_TB

    @property
    def mlc_boundaries(self) -> tuple[float, ...]:
        return MLC_BOUNDARIES_TB_HD120 if self.mlc_is_hd else MLC_BOUNDARIES_TB_MIL120


class Beam(BeamBase[TrueBeamMachine]):
    """A class that represents a TrueBeam beam."""

    def __init__(
        self,
        mlc_is_hd: bool,
        beam_name: str,
        energy: float,
        fluence_mode: FluenceMode,
        dose_rate: int,
        metersets: Sequence[float],
        gantry_angles: float | Sequence[float],
        x1: float,
        x2: float,
        y1: float,
        y2: float,
        mlc_positions: list[list[float]],
        coll_angle: float,
        couch_vrt: float,
        couch_lat: float,
        couch_lng: float,
        couch_rot: float,
    ):
        """
        Parameters
        ----------
        mlc_is_hd : bool
            Whether the MLC type is HD or Millennium
        beam_name : str
            The name of the beam. Must be less than 16 characters.
        energy : float
            The energy of the beam.
        fluence_mode : FluenceMode
            The fluence mode of the beam.
        dose_rate : int
            The dose rate of the beam.
        metersets : Sequence[float]
            The meter sets for each control point. The length must match the number of control points in mlc_positions.
        gantry_angles : Union[float, Sequence[float]]
            The gantry angle(s) of the beam. If a single number, it's assumed to be a static beam. If multiple numbers, it's assumed to be a dynamic beam.
        x1 : float
            The left jaw position.
        x2 : float
            The right jaw position.
        y1 : float
            The bottom jaw position.
        y2 : float
            The top jaw position.
        mlc_positions : list[list[float]]
            The MLC positions for each control point. This is the x-position of each leaf for each control point.
        coll_angle : float
            The collimator angle.
        couch_vrt : float
            The couch vertical position.
        couch_lat : float
            The couch lateral position.
        couch_lng : float
            The couch longitudinal position.
        couch_rot : float
            The couch rotation.
        """
        jaw_x = Dataset()
        jaw_x.RTBeamLimitingDeviceType = "X"
        jaw_x.NumberOfLeafJawPairs = 1
        jaw_y = Dataset()
        jaw_y.RTBeamLimitingDeviceType = "Y"
        jaw_y.NumberOfLeafJawPairs = 1
        jaw_asymx = Dataset()
        jaw_asymx.RTBeamLimitingDeviceType = "ASYMX"
        jaw_asymx.NumberOfLeafJawPairs = 1
        jaw_asymy = Dataset()
        jaw_asymy.RTBeamLimitingDeviceType = "ASYMY"
        jaw_asymy.NumberOfLeafJawPairs = 1
        mlc = Dataset()
        mlc.RTBeamLimitingDeviceType = "MLCX"
        mlc.NumberOfLeafJawPairs = 60
        mlc.LeafPositionBoundaries = list(
            MLC_BOUNDARIES_TB_HD120 if mlc_is_hd else MLC_BOUNDARIES_TB_MIL120
        )

        bld_sequence = DicomSequence((jaw_x, jaw_y, jaw_asymx, jaw_asymy, mlc))

        beam_limiting_device_positions = {
            "ASYMX": [[x1, x2]],
            "ASYMY": [[y1, y2]],
            "MLCX": mlc_positions,
        }

        super().__init__(
            beam_limiting_device_sequence=bld_sequence,
            beam_name=beam_name,
            energy=energy,
            fluence_mode=fluence_mode,
            dose_rate=dose_rate,
            metersets=metersets,
            gantry_angles=gantry_angles,
            beam_limiting_device_positions=beam_limiting_device_positions,
            coll_angle=coll_angle,
            couch_vrt=couch_vrt,
            couch_lat=couch_lat,
            couch_lng=couch_lng,
            couch_rot=couch_rot,
        )

    @staticmethod
    def create_shaper(
        machine: TrueBeamMachine,
    ) -> MLCShaper:
        """Utility to create MLC shaper instances."""
        return MLCShaper(
            machine.mlc_boundaries,
            machine.specs.max_mlc_position,
            machine.specs.max_mlc_overtravel,
        )

    @staticmethod
    def create_modulator(
        machine: TrueBeamMachine,
        sacrifice_gap_mm: float = None,
        sacrifice_max_move_mm: float = None,
    ) -> MLCModulator:
        """Utility to create MLC modulator instances."""
        return MLCModulator(
            leaf_y_positions=machine.mlc_boundaries,
            max_mlc_position=machine.specs.max_mlc_position,
            max_overtravel_mm=machine.specs.max_mlc_overtravel,
            sacrifice_gap_mm=sacrifice_gap_mm,
            sacrifice_max_move_mm=sacrifice_max_move_mm,
        )


class QAProcedure(QAProcedureBase[TrueBeamMachine], ABC):
    pass


@dataclass
class OpenField(QAProcedure):
    """Create an open field beam.

    Parameters
    ----------
    x1 : float
        The left edge position.
    x2 : float
        The right edge position.
    y1 : float
        The bottom edge position.
    y2 : float
        The top edge position.
    mu : float
        The monitor units of the beam.
    defined_by_mlc : bool
        Whether the field edges are defined by the MLCs or the jaws.
    mlc_mode : MLCLeafBoundaryAlignmentMode
        Controls how the open field aligns with MLC leaf boundaries along the y-axis.

        * EXACT -- Both ``y1`` and ``y2`` must coincide with an MLC leaf boundary.
          If either edge does not align exactly, an error is raised.
        * ROUND -- If ``y1`` or ``y2`` falls between boundaries, the limits are rounded to the nearest boundary.
        * INWARD -- If ``y1`` or ``y2`` falls between boundaries, the leaf band is treated as "outfield."
          This results in a smaller field in the y-direction.
        * OUTWARD -- If ``y1`` or ``y2`` falls between boundaries, the leaf band is treated as "infield."
          This results in a larger field in the y-direction.
    energy : float
        The energy of the beam.
    fluence_mode : FluenceMode
        The fluence mode of the beam.
    dose_rate : int
        The dose rate of the beam.
    gantry_angle : float
        The gantry angle of the beam.
    coll_angle : float
        The collimator angle of the beam.
    couch_vrt : float
        The couch vertical position.
    couch_lng : float
        The couch longitudinal position.
    couch_lat : float
        The couch lateral position.
    couch_rot : float
        The couch rotation.
    padding : float
        The padding to add to the jaws or MLCs.
    beam_name : str
        The name of the beam.
    outside_strip_width : float
        The width of the strip of MLCs outside the field. The MLCs will be placed to the
        left, under the X1 jaw by 20mm.
    """

    x1: float
    x2: float
    y1: float
    y2: float
    mu: float = 100.0
    defined_by_mlc: bool = True
    mlc_mode: MLCLeafBoundaryAlignmentMode = MLCLeafBoundaryAlignmentMode.OUTWARD
    energy: float = 6
    fluence_mode: FluenceMode = FluenceMode.STANDARD
    dose_rate: int = 600
    gantry_angle: float = 0
    coll_angle: float = 0
    couch_vrt: float = 0
    couch_lng: float = 1000
    couch_lat: float = 0
    couch_rot: float = 0
    padding: float = 5
    beam_name: str = "Open"
    outside_strip_width: float = 5

    def compute(self, machine: TrueBeamMachine) -> None:
        if self.defined_by_mlc:
            mlc_padding = 0
            jaw_padding = self.padding
            y_mode = self.mlc_mode
        else:
            mlc_padding = self.padding
            jaw_padding = 0
            y_mode = MLCLeafBoundaryAlignmentMode.OUTWARD

        shaper = Beam.create_shaper(machine)
        shape = Rectangle(
            x_min=self.x1 - mlc_padding,
            x_max=self.x2 + mlc_padding,
            y_min=self.y1 - mlc_padding,
            y_max=self.y2 + mlc_padding,
            y_mode=RectangleMode(y_mode),
            outer_strip_width=self.outside_strip_width,
            x_outfield_position=self.x1 - jaw_padding - 20,
        )
        mlc = shaper.get_shape(shape)
        beam = Beam(
            beam_name=self.beam_name,
            energy=self.energy,
            dose_rate=self.dose_rate,
            x1=self.x1 - jaw_padding,
            x2=self.x2 + jaw_padding,
            y1=self.y1 - jaw_padding,
            y2=self.y2 + jaw_padding,
            gantry_angles=self.gantry_angle,
            coll_angle=self.coll_angle,
            couch_vrt=self.couch_vrt,
            couch_lat=self.couch_lat,
            couch_lng=self.couch_lng,
            couch_rot=self.couch_rot,
            mlc_positions=2 * [mlc],
            metersets=[0, self.mu],
            fluence_mode=self.fluence_mode,
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(beam)


@dataclass
class MLCTransmission(QAProcedure):
    """Add MLC transmission beams to the plan.
    The beam is delivered with the MLCs closed and moved to one side underneath the jaws.

    Parameters
    ----------
    mu_per_bank : int
        The monitor units to deliver for each bank transmission test.
    mu_per_ref : int
        The monitor units to deliver for the reference open field.
    overreach : float
        The amount to tuck the MLCs under the jaws in mm.
    beam_names : list[str]
        A list containing the names of the beams to use in the following order:
        reference beam, transmission beam bank A, transmission beam bank B
    energy : int
        The energy of the beam.
    fluence_mode : FluenceMode
        The fluence mode of the beam.
    dose_rate : int
        The dose rate of the beam.
    width : float
        The width of the reference field in mm.
    height : float
        The height of the reference field in mm.
    gantry_angle : float
        The gantry angle of the beam in degrees.
    coll_angle : float
        The collimator angle of the beam in degrees.
    couch_vrt : float
        The couch vertical position.
    couch_lat : float
        The couch lateral position.
    couch_lng : float
        The couch longitudinal position.
    couch_rot : float
        The couch rotation in degrees.
    """

    mu_per_bank: int = 100
    mu_per_ref: int = 100
    overreach: float = 10
    beam_names: list[str] = field(
        default_factory=lambda: ["MLC Tx - Ref", "MLC Tx - Bank-A", "MLC Tx - Bank-B"]
    )
    energy: int = 6
    fluence_mode: FluenceMode = FluenceMode.STANDARD
    dose_rate: int = 600
    width: float = 100
    height: float = 100
    gantry_angle: float = 0
    coll_angle: float = 0
    couch_vrt: float = 0
    couch_lat: float = 0
    couch_lng: float = 1000
    couch_rot: float = 0

    # private attributes: common to all beams to facilitate creation
    _x1: float = field(init=False)
    _x2: float = field(init=False)
    _y1: float = field(init=False)
    _y2: float = field(init=False)
    _mlc_is_hd: bool = field(init=False)

    def compute(self, machine: TrueBeamMachine) -> None:
        self._x1 = -self.width / 2
        self._x2 = self.width / 2
        self._y1 = -self.height / 2
        self._y2 = self.height / 2
        self._mlc_is_hd = machine.mlc_is_hd

        keys = ["Ref", "A", "B"]
        names = dict(zip(keys, self.beam_names))
        shaper = Beam.create_shaper(machine)

        # Reference field
        ref = OpenField(
            self._x1,
            self._x2,
            self._y1,
            self._y2,
            self.mu_per_ref,
            defined_by_mlc=False,
            mlc_mode=MLCLeafBoundaryAlignmentMode.OUTWARD,
            energy=self.energy,
            fluence_mode=self.fluence_mode,
            dose_rate=self.dose_rate,
            gantry_angle=self.gantry_angle,
            coll_angle=self.coll_angle,
            couch_vrt=self.couch_vrt,
            couch_lng=self.couch_lng,
            couch_lat=self.couch_lat,
            couch_rot=self.couch_rot,
            padding=20,
            beam_name=names["Ref"],
        )
        ref.compute(machine)
        self.beams.append(ref.beams[0])

        # Transmission field A
        # Bank A is under X2, so the slit should be under X1
        shape = Strip(position=self._x1 - self.overreach, width=1)
        mlc = shaper.get_shape(shape)
        beam = self._beam(names["A"], mlc)
        self.beams.append(beam)

        # Transmission field B
        # Bank B is under X1, so the slit should be under X2
        shape = Strip(position=self._x2 + self.overreach, width=1)
        mlc = shaper.get_shape(shape)
        beam = self._beam(names["B"], mlc)
        self.beams.append(beam)

    def _beam(self, beam_name: str, mlc: list[float]) -> Beam:
        return Beam(
            mlc_is_hd=self._mlc_is_hd,
            beam_name=beam_name,
            energy=self.energy,
            fluence_mode=self.fluence_mode,
            dose_rate=self.dose_rate,
            x1=self._x1,
            x2=self._x2,
            y1=self._y1,
            y2=self._y2,
            gantry_angles=self.gantry_angle,
            coll_angle=self.coll_angle,
            couch_vrt=self.couch_vrt,
            couch_lat=self.couch_lat,
            couch_lng=self.couch_lng,
            couch_rot=self.couch_rot,
            mlc_positions=[mlc],
            metersets=[0, self.mu_per_bank],
        )


@dataclass
class PicketFence(QAProcedure):
    """Add a picket fence beam to the plan.

    Parameters
    ----------
    picket_width : float
        The width of the pickets in mm.
    picket_positions : tuple
        The positions of the pickets in mm relative to the center of the image.
    mu_per_picket : int
        The monitor units for each picket.
    mu_per_transition : int
        The monitor units for MLC transitions between pickets.
    skip_first_picket: bool
        Whether to skip the first picket.
    energy : float
        The energy of the beam.
    fluence_mode : FluenceMode
        The fluence mode of the beam.
    dose_rate : int
        The dose rate of the beam.
    gantry_angle : float
        The gantry angle of the beam.
    coll_angle : float
        The collimator angle of the beam.
    couch_vrt : float
        The couch vertical position.
    couch_lng : float
        The couch longitudinal position.
    couch_lat : float
        The couch lateral position.
    couch_rot : float
        The couch rotation.
    jaw_padding : float
        The padding to add to the X jaws.
    beam_name : str
        The name of the beam.
    """

    picket_width: float = 1
    picket_positions: Sequence[float] = (-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75)
    mu_per_picket: float = 10
    mu_per_transition: float = 2
    skip_first_picket: bool = True
    energy: float = 6
    fluence_mode: FluenceMode = FluenceMode.STANDARD
    dose_rate: int = 600
    gantry_angle: float = 0
    coll_angle: float = 0
    couch_vrt: float = 0
    couch_lng: float = 1000
    couch_lat: float = 0
    couch_rot: float = 0
    jaw_padding: float = 10
    beam_name: str = "Picket fence"

    @classmethod
    def from_varian_reference(cls) -> Self:
        """Add a picket fence that replicates Varian's T0.2_PicketFenceStatic_M120_TB_Rev02.dcm

        .. include:: varian_reference_warning.rst
        """
        # Note #1: jaw x2 is different (cannot be replicated) since the left padding
        # is different from the right padding
        # Note #2: jaws y1,y2 are different (cannot be replicated) since this procedure
        # doesn't hide any leafs, whereas in the Varian plan, the last top/bottom leaves are hidden.
        return cls(
            picket_width=1,
            picket_positions=np.arange(-74.5, 76, 15),
            mu_per_picket=8.125,
            mu_per_transition=1.875,
            skip_first_picket=True,
            jaw_padding=5.5,
        )

    def compute(self, machine: TrueBeamMachine) -> None:
        # check MLC overtravel; machine may prevent delivery if exposing leaf tail
        x1 = min(self.picket_positions) - self.jaw_padding
        x2 = max(self.picket_positions) + self.jaw_padding
        max_dist_to_jaw1 = max(abs(pos - x1) for pos in self.picket_positions)
        max_dist_to_jaw2 = max(abs(pos - x2) for pos in self.picket_positions)
        if max(max_dist_to_jaw1, max_dist_to_jaw2) > machine.specs.max_mlc_overtravel:
            msg = "Picket fence beam exceeds MLC overtravel limits. Lower padding, the number of pickets, or the picket spacing."
            raise ValueError(msg)

        # This is the picket fence sequence delivery motion scheme
        # T is transition, D is Dose, MLC numbers are index of picket positions
        # CP    0 (*) 1 2 3 4 5 6   (*) Add control point if skip_first_picket=False
        # D     0 (D) T D T D T D   (D) Add Dose if skip_first_picket=False
        # MLC   0 (0) 1 1 2 2 3 3   (0) Add MLC if skip_first_picket=False

        mu, pos = [0], [self.picket_positions[0]]
        if not self.skip_first_picket:
            mu += [self.mu_per_picket]
            pos += [self.picket_positions[0]]

        for idx in range(1, len(self.picket_positions)):
            mu += [self.mu_per_transition, self.mu_per_picket]
            pos += 2 * [self.picket_positions[idx]]

        shaper = Beam.create_shaper(machine)
        mlc = [shaper.get_shape(Strip(p, self.picket_width)) for p in pos]
        metersets = np.cumsum(mu)

        beam = Beam(
            beam_name=self.beam_name,
            energy=self.energy,
            dose_rate=self.dose_rate,
            x1=x1,
            x2=x2,
            y1=machine.mlc_boundaries[0],
            y2=machine.mlc_boundaries[-1],
            gantry_angles=self.gantry_angle,
            coll_angle=self.coll_angle,
            couch_vrt=self.couch_vrt,
            couch_lat=self.couch_lat,
            couch_lng=self.couch_lng,
            couch_rot=self.couch_rot,
            mlc_positions=mlc,
            metersets=metersets,
            fluence_mode=self.fluence_mode,
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(beam)


@dataclass
class WinstonLutz(QAProcedure):
    """Add Winston-Lutz beams to the plan. Will create a beam for each set of axes positions.
    Field names are generated automatically based on the axes positions.

    Parameters
    ----------
    x1 : float
        The left edge position.
    x2 : float
        The right edge position.
    y1 : float
        The bottom edge position.
    y2 : float
        The top edge position.
    mu : float
        The monitor units of the beam.
    defined_by_mlc : bool
        Whether the field edges are defined by the MLCs or the jaws.
    mlc_mode : MLCLeafBoundaryAlignmentMode
        Controls how the open field aligns with MLC leaf boundaries along the y-axis.

        * EXACT -- Both ``y1`` and ``y2`` must coincide with an MLC leaf boundary.
          If either edge does not align exactly, an error is raised.
        * ROUND -- If ``y1`` or ``y2`` falls between boundaries, the limits are rounded to the nearest boundary.
        * INWARD -- If ``y1`` or ``y2`` falls between boundaries, the leaf band is treated as "outfield."
          This results in a smaller field in the y-direction.
        * OUTWARD -- If ``y1`` or ``y2`` falls between boundaries, the leaf band is treated as "infield."
          This results in a larger field in the y-direction.
    fields : Iterable[WinstonLutzField]
        The positions of the axes.
    energy : float
        The energy of the beam.
    fluence_mode : FluenceMode
        The fluence mode of the beam.
    dose_rate : int
        The dose rate of the beam.
    couch_vrt : float
        The couch vertical position.
    couch_lng : float
        The couch longitudinal position.
    couch_lat : float
        The couch lateral position.
    padding : float
        The padding to add to the jaws or MLCs.
    """

    x1: float = -10.0
    x2: float = 10.0
    y1: float = -10.0
    y2: float = 10.0
    mu: float = 10.0
    defined_by_mlc: bool = True
    mlc_mode: MLCLeafBoundaryAlignmentMode = MLCLeafBoundaryAlignmentMode.OUTWARD
    fields: Iterable[WinstonLutzField] = (WinstonLutzField(0, 0, 0),)
    energy: float = 6
    fluence_mode: FluenceMode = FluenceMode.STANDARD
    dose_rate: int = 600
    couch_vrt: float = 0
    couch_lng: float = 1000
    couch_lat: float = 0
    padding: float = 5

    def compute(self, machine: TrueBeamMachine) -> None:
        for _field in self.fields:
            g = round(_field.gantry)
            c = round(_field.collimator)
            t = round(_field.couch)
            beam_name = _field.name or f"G{g:03d}C{c:03d}T{t:03d}"
            open_field_procedure = OpenField(
                x1=self.x1,
                x2=self.x2,
                y1=self.y1,
                y2=self.y2,
                mu=self.mu,
                defined_by_mlc=self.defined_by_mlc,
                mlc_mode=self.mlc_mode,
                energy=self.energy,
                dose_rate=self.dose_rate,
                gantry_angle=_field.gantry,  # WL gantry
                coll_angle=_field.collimator,  # WL collimator
                couch_vrt=self.couch_vrt,
                couch_lat=self.couch_lat,
                couch_lng=self.couch_lng,
                couch_rot=_field.couch,  # WL couch
                padding=self.padding,
                beam_name=beam_name,
            )
            open_field_procedure.compute(machine)
            beam = open_field_procedure.beams[0]
            self.beams.append(beam)


@dataclass
class DosimetricLeafGap(QAProcedure):
    """Add beams to measure the Dosimetric Leaf Gap (DLG).

    Parameters
    ----------
    gap_widths : tuple
        The gap widths in mm for the MLC sweeps.
    start_position: int
        The start position of the MLC gap in mm.
    final_position: int
        The final position of the MLC gap in mm
    mu : int
        The monitor units of each beam.
    energy : float
        The energy of the beam.
    fluence_mode : FluenceMode
        The fluence mode of the beam.
    dose_rate : int
        The dose rate of the beam.
    gantry_angle : float
        The gantry angle of the beam.
    coll_angle : float
        The collimator angle of the beam.
    couch_vrt : float
        The couch vertical position.
    couch_lat : float
        The couch lateral position.
    couch_lng : float
        The couch longitudinal position.
    couch_rot : float
        The couch rotation.
    x1 : float
        The left edge position.
    x2 : float
        The right edge position.
    y1 : float
        The bottom edge position.
    y2 : float
        The top edge position.
    """

    gap_widths: Sequence[float] = (2, 4, 6, 10, 14, 16, 20)
    start_position: float = -60
    final_position: float = 60
    mu: int = 100
    energy: float = 6
    fluence_mode: FluenceMode = FluenceMode.STANDARD
    dose_rate: int = 600
    gantry_angle: float = 0
    coll_angle: float = 0
    couch_vrt: float = 0
    couch_lat: float = 0
    couch_lng: float = 1000
    couch_rot: float = 0
    x1: float = -50
    x2: float = 50
    y1: float = -50
    y2: float = 50

    def compute(self, machine: TrueBeamMachine) -> None:
        # validate x_jaw positions
        min_pos = min(self.start_position, self.final_position)
        max_pos = max(self.start_position, self.final_position)
        max_gap = max(self.gap_widths)
        if min_pos + max_gap / 2.0 > self.x1 or max_pos - max_gap / 2.0 < self.x2:
            warnings.warn(
                "Invalid configuration: the MLC must start and end with the gap fully "
                "occluded by the jaws. Failure to do so introduces additional transmission "
                "that can bias the DLG calculation."
            )

        shaper = Beam.create_shaper(machine)
        for gap_width in self.gap_widths:
            mlc_start = shaper.get_shape(Strip(self.start_position, gap_width))
            mlc_final = shaper.get_shape(Strip(self.final_position, gap_width))
            mlc_positions = [mlc_start] + [mlc_final]
            metersets = [0, self.mu]
            beam_name = f"DLG {gap_width:00d}mm"

            beam = Beam(
                mlc_is_hd=machine.mlc_is_hd,
                beam_name=beam_name,
                energy=self.energy,
                fluence_mode=self.fluence_mode,
                dose_rate=self.dose_rate,
                x1=self.x1,
                x2=self.x2,
                y1=self.y1,
                y2=self.y2,
                gantry_angles=self.gantry_angle,
                coll_angle=self.coll_angle,
                couch_vrt=self.couch_vrt,
                couch_lat=self.couch_lat,
                couch_lng=self.couch_lng,
                couch_rot=self.couch_rot,
                mlc_positions=mlc_positions,
                metersets=metersets,
            )
            self.beams.append(beam)


@dataclass
class DoseRate(QAProcedure):
    """Create a single-image dose rate test. Multiple ROIs are generated. A reference beam is also
    created where all ROIs are delivered at the default dose rate for comparison.
    The field names are generated automatically based on the min and max dose rates tested.

    Parameters
    ----------
    dose_rates : tuple
        The dose rates to test in MU/min. Each dose rate will have its own ROI.
    default_dose_rate : int
        The default dose rate. Typically, this is the clinical default. The reference beam
        will be delivered at this dose rate for all ROIs.
    gantry_angle : float
        The gantry angle of the beam.
    desired_mu : int
        The desired monitor units to deliver. It can be that based on the dose rates asked for,
        the MU required might be higher than this value.
    energy : float
        The energy of the beam.
    fluence_mode : FluenceMode
        The fluence mode of the beam.
    coll_angle : float
        The collimator angle of the beam.
    couch_vrt : float
        The couch vertical position.
    couch_lat : float
        The couch lateral position.
    couch_lng : float
        The couch longitudinal position.
    couch_rot : float
        The couch rotation.
    jaw_padding_mm : float
        The padding to add to the X jaws. The X-jaws will close around the ROIs plus this padding.
    roi_size_mm : float
        The width of the ROIs in mm.
    y1 : float
        The bottom jaw position. Usually negative. More negative is lower.
    y2 : float
        The top jaw position. Usually positive. More positive is higher.
    max_sacrificial_move_mm : float
        The maximum distance the sacrificial leaves can move in a given control point.
        Smaller values generate more control points and more back-and-forth movement.
        Too large of values may cause deliverability issues.
    """

    dose_rates: tuple[int, ...] = (100, 300, 500, 600)
    default_dose_rate: int = 600
    gantry_angle: float = 0
    desired_mu: int = 50
    energy: float = 6
    fluence_mode: FluenceMode = FluenceMode.STANDARD
    coll_angle: float = 0
    couch_vrt: float = 0
    couch_lat: float = 0
    couch_lng: float = 1000
    couch_rot: float = 0
    jaw_padding_mm: float = 5
    roi_size_mm: float = 25
    y1: float = -100
    y2: float = 100
    max_sacrificial_move_mm: float = 50

    def compute(self, machine: TrueBeamMachine) -> None:
        if self.roi_size_mm * len(self.dose_rates) > machine.specs.max_mlc_overtravel:
            raise ValueError(
                "The ROI size * number of dose rates must be less than the overall MLC allowable width"
            )
        # calculate MU
        mlc_transition_time = self.roi_size_mm / machine.specs.max_mlc_speed
        min_mu = mlc_transition_time * max(self.dose_rates) * len(self.dose_rates) / 60
        mu = max(self.desired_mu, math.ceil(min_mu))

        # create MLC sacrifices
        times_to_transition = [
            mu * 60 / (dose_rate * len(self.dose_rates))
            for dose_rate in self.dose_rates
        ]
        sacrificial_movements = [
            tt * machine.specs.max_mlc_speed for tt in times_to_transition
        ]

        mlc = Beam.create_modulator(
            machine, sacrifice_max_move_mm=self.max_sacrificial_move_mm
        )
        ref_mlc = Beam.create_modulator(machine)

        roi_centers = np.linspace(
            -self.roi_size_mm * len(self.dose_rates) / 2 + self.roi_size_mm / 2,
            self.roi_size_mm * len(self.dose_rates) / 2 - self.roi_size_mm / 2,
            len(self.dose_rates),
        )
        # we have a starting and ending strip
        ref_mlc.add_strip(
            position_mm=float(roi_centers[0] - self.roi_size_mm / 2),
            strip_width_mm=0,
            meterset_at_target=0,
        )
        mlc.add_strip(
            position_mm=float(roi_centers[0] - self.roi_size_mm / 2),
            strip_width_mm=0,
            meterset_at_target=0,
            initial_sacrificial_gap_mm=5,
        )
        for sacrifice_distance, center in zip(sacrificial_movements, roi_centers):
            ref_mlc.add_rectangle(
                left_position=center - self.roi_size_mm / 2,
                right_position=center + self.roi_size_mm / 2,
                x_outfield_position=-200,
                top_position=max(machine.mlc_boundaries),
                bottom_position=min(machine.mlc_boundaries),
                outer_strip_width=5,
                meterset_at_target=0,
                meterset_transition=0.5 / len(self.dose_rates),
                sacrificial_distance=0,
            )
            ref_mlc.add_strip(
                position_mm=center + self.roi_size_mm / 2,
                strip_width_mm=0,
                meterset_at_target=0,
                meterset_transition=0.5 / len(self.dose_rates),
                sacrificial_distance_mm=0,
            )
            mlc.add_rectangle(
                left_position=center - self.roi_size_mm / 2,
                right_position=center + self.roi_size_mm / 2,
                x_outfield_position=-200,  # not used
                top_position=max(machine.mlc_boundaries),
                bottom_position=min(machine.mlc_boundaries),
                outer_strip_width=5,  # not used
                meterset_at_target=0,
                meterset_transition=0.5 / len(self.dose_rates),
                sacrificial_distance=sacrifice_distance,
            )
            mlc.add_strip(
                position_mm=center + self.roi_size_mm / 2,
                strip_width_mm=0,
                meterset_at_target=0,
                meterset_transition=0.5 / len(self.dose_rates),
                sacrificial_distance_mm=sacrifice_distance,
            )
        ref_beam = Beam(
            beam_name="DR Ref",
            energy=self.energy,
            dose_rate=self.default_dose_rate,
            x1=float(roi_centers[0] - self.roi_size_mm / 2 - self.jaw_padding_mm),
            x2=float(roi_centers[-1] + self.roi_size_mm / 2 + self.jaw_padding_mm),
            y1=self.y1,
            y2=self.y2,
            gantry_angles=self.gantry_angle,
            coll_angle=self.coll_angle,
            couch_vrt=self.couch_vrt,
            couch_lat=self.couch_lat,
            couch_lng=self.couch_lng,
            couch_rot=self.couch_rot,
            fluence_mode=self.fluence_mode,
            mlc_positions=ref_mlc.as_control_points(),
            metersets=[mu * m for m in ref_mlc.as_metersets()],
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(ref_beam)
        beam = Beam(
            beam_name=f"DR{min(self.dose_rates)}-{max(self.dose_rates)}",
            energy=self.energy,
            dose_rate=self.default_dose_rate,
            x1=float(roi_centers[0] - self.roi_size_mm / 2 - self.jaw_padding_mm),
            x2=float(roi_centers[-1] + self.roi_size_mm / 2 + self.jaw_padding_mm),
            y1=self.y1,
            y2=self.y2,
            gantry_angles=self.gantry_angle,
            coll_angle=self.coll_angle,
            couch_vrt=self.couch_vrt,
            couch_lat=self.couch_lat,
            couch_lng=self.couch_lng,
            couch_rot=self.couch_rot,
            fluence_mode=self.fluence_mode,
            mlc_positions=mlc.as_control_points(),
            metersets=[mu * m for m in mlc.as_metersets()],
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(beam)


@dataclass
class MLCSpeed(QAProcedure):
    """Create a single-image MLC speed test. Multiple speeds are generated. A reference beam is also
    generated. The reference beam is delivered at the maximum MLC speed.

    Parameters
    ----------
    speeds : tuple[float]
        The speeds to test in mm/s. Each speed will have its own ROI.
    roi_size_mm : float
        The width of the ROIs in mm.
    mu : int
        The monitor units to deliver.
    default_dose_rate : int
        The dose rate used for the reference beam.
    gantry_angle : float
        The gantry angle of the beam.
    energy : int
        The energy of the beam.
    coll_angle : float
        The collimator angle of the beam.
    couch_vrt : float
        The couch vertical position.
    couch_lat : float
        The couch lateral position.
    couch_lng : float
        The couch longitudinal position.
    couch_rot : float
        The couch rotation.
    fluence_mode : FluenceMode
        The fluence mode of the beam.
    jaw_padding_mm : float
        The padding to add to the X jaws. The X-jaws will close around the ROIs plus this padding.
    y1 : float
        The bottom jaw position. Usually negative. More negative is lower.
    y2 : float
        The top jaw position. Usually positive. More positive is higher.
    beam_name : str
        The name of the beam. The reference beam will be called "MLC Sp Ref".
    max_sacrificial_move_mm : float
        The maximum distance the sacrificial leaves can move in a given control point.
        Smaller values generate more control points and more back-and-forth movement.
        Too large of values may cause deliverability issues.


    Notes
    -----

    The desired speed can be achieved through the following formula:

       speed = roi_size_mm * max dose rate / MU * 60

    We solve for MU with the desired speed. The 60 is for converting the dose rate as MU/min to MU/sec.
    Thus,

        MU = roi_size_mm * max dose rate / speed * 60

    MUs are calculated automatically based on the speed and the ROI size.

    """

    speeds: tuple[float | int, ...] = (5, 10, 15, 20)
    roi_size_mm: float = 20
    mu: int = 50
    default_dose_rate: int = 600
    gantry_angle: float = 0
    energy: float = 6
    coll_angle: float = 0
    couch_vrt: float = 0
    couch_lat: float = 0
    couch_lng: float = 1000
    couch_rot: float = 0
    fluence_mode: FluenceMode = FluenceMode.STANDARD
    jaw_padding_mm: float = 5
    y1: float = -100
    y2: float = 100
    beam_name: str = "MLC Speed"
    max_sacrificial_move_mm: float = 50

    def compute(self, machine: TrueBeamMachine) -> None:
        if max(self.speeds) > machine.specs.max_mlc_speed:
            raise ValueError(
                f"Maximum speed given {max(self.speeds)} is greater than the maximum MLC speed {machine.specs.max_mlc_speed}"
            )
        if min(self.speeds) <= 0:
            raise ValueError("Speeds must be greater than 0")
        if self.roi_size_mm * len(self.speeds) > machine.specs.max_mlc_overtravel:
            raise ValueError(
                "The ROI size * number of speeds must be less than the overall MLC allowable width"
            )
        # create MLC positions
        times_to_transition = [self.roi_size_mm / speed for speed in self.speeds]
        sacrificial_movements = [
            tt * machine.specs.max_mlc_speed for tt in times_to_transition
        ]

        mlc = Beam.create_modulator(
            machine, sacrifice_max_move_mm=self.max_sacrificial_move_mm
        )
        ref_mlc = Beam.create_modulator(machine)

        roi_centers = np.linspace(
            -self.roi_size_mm * len(self.speeds) / 2 + self.roi_size_mm / 2,
            self.roi_size_mm * len(self.speeds) / 2 - self.roi_size_mm / 2,
            len(self.speeds),
        )
        # we have a starting and ending strip
        ref_mlc.add_strip(
            position_mm=float(roi_centers[0] - self.roi_size_mm / 2),
            strip_width_mm=0,
            meterset_at_target=0,
        )
        mlc.add_strip(
            position_mm=float(roi_centers[0] - self.roi_size_mm / 2),
            strip_width_mm=0,
            meterset_at_target=0,
            initial_sacrificial_gap_mm=5,
        )
        for sacrifice_distance, center in zip(sacrificial_movements, roi_centers):
            ref_mlc.add_rectangle(
                left_position=center - self.roi_size_mm / 2,
                right_position=center + self.roi_size_mm / 2,
                x_outfield_position=-200,  # not relevant
                top_position=max(machine.mlc_boundaries),
                bottom_position=min(machine.mlc_boundaries),
                outer_strip_width=5,  # not relevant
                meterset_at_target=0,
                meterset_transition=0.5 / len(self.speeds),
                sacrificial_distance=0,
            )
            ref_mlc.add_strip(
                position_mm=center + self.roi_size_mm / 2,
                strip_width_mm=0,
                meterset_at_target=0,
                meterset_transition=0.5 / len(self.speeds),
                sacrificial_distance_mm=0,
            )
            mlc.add_rectangle(
                left_position=center - self.roi_size_mm / 2,
                right_position=center + self.roi_size_mm / 2,
                x_outfield_position=-200,  # not used
                top_position=max(machine.mlc_boundaries),
                bottom_position=min(machine.mlc_boundaries),
                outer_strip_width=5,  # not used
                meterset_at_target=0,
                meterset_transition=0.5 / len(self.speeds),
                sacrificial_distance=sacrifice_distance,
            )
            mlc.add_strip(
                position_mm=center + self.roi_size_mm / 2,
                strip_width_mm=0,
                meterset_at_target=0,
                meterset_transition=0.5 / len(self.speeds),
                sacrificial_distance_mm=sacrifice_distance,
            )
        ref_beam = Beam(
            beam_name=f"{self.beam_name} Ref",
            energy=self.energy,
            dose_rate=self.default_dose_rate,
            x1=float(roi_centers[0] - self.roi_size_mm / 2 - self.jaw_padding_mm),
            x2=float(roi_centers[-1] + self.roi_size_mm / 2 + self.jaw_padding_mm),
            y1=self.y1,
            y2=self.y2,
            gantry_angles=self.gantry_angle,
            coll_angle=self.coll_angle,
            couch_vrt=self.couch_vrt,
            couch_lat=self.couch_lat,
            couch_lng=self.couch_lng,
            couch_rot=self.couch_rot,
            fluence_mode=self.fluence_mode,
            mlc_positions=ref_mlc.as_control_points(),
            metersets=[self.mu * m for m in ref_mlc.as_metersets()],
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(ref_beam)
        beam = Beam(
            beam_name=self.beam_name,
            energy=self.energy,
            dose_rate=self.default_dose_rate,
            x1=float(roi_centers[0] - self.roi_size_mm / 2 - self.jaw_padding_mm),
            x2=float(roi_centers[-1] + self.roi_size_mm / 2 + self.jaw_padding_mm),
            y1=self.y1,
            y2=self.y2,
            gantry_angles=self.gantry_angle,
            coll_angle=self.coll_angle,
            couch_vrt=self.couch_vrt,
            couch_lat=self.couch_lat,
            couch_lng=self.couch_lng,
            couch_rot=self.couch_rot,
            fluence_mode=self.fluence_mode,
            mlc_positions=mlc.as_control_points(),
            metersets=[self.mu * m for m in mlc.as_metersets()],
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(beam)


@dataclass
class GantrySpeed(QAProcedure):
    """Create a single-image gantry speed test. Multiple speeds are generated. A reference beam is also
    generated. The reference beam is delivered without gantry movement.

    Parameters
    ----------
    speeds : tuple
        The gantry speeds to test. Each speed will have its own ROI.
    max_dose_rate : int
        The max dose rate clinically allowed for the energy.
    start_gantry_angle : float
        The starting gantry angle. The gantry will rotate around this point. It is up to the user
        to know what the machine's limitations are. (i.e. don't go through 180 for Varian machines).
        The ending gantry angle will be the starting angle + the sum of the gantry deltas generated
        by the speed ROIs. Slower speeds require more gantry angle to reach the same MU.
    energy : float
        The energy of the beam.
    fluence_mode : FluenceMode
        The fluence mode of the beam.
    coll_angle : float
        The collimator angle of the beam.
    couch_vrt : float
        The couch vertical position.
    couch_lat : float
        The couch lateral position.
    couch_lng : float
        The couch longitudinal position.
    couch_rot : float
        The couch rotation.
    beam_name : str
        The name of the beam.
    gantry_rot_dir : GantryDirection
        The direction of gantry rotation.
    jaw_padding_mm : float
        The padding to add to the X jaws. The X-jaws will close around the ROIs plus this padding.
    roi_size_mm : float
        The width of the ROIs in mm.
    y1 : float
        The bottom jaw position. Usually negative. More negative is lower.
    y2 : float
        The top jaw position. Usually positive. More positive is higher.
    mu : int
        The monitor units of the beam.

    Notes
    -----

    The gantry angle to cover can be determined via the following:

     gantry speed = gantry_range * max_dose_rate / (MU * 60)

     We can thus solve for the gantry range:

        gantry_range = gantry_speed * MU * 60 / max_dose_rate

    """

    speeds: tuple[float | int, ...] = (2, 3, 4, 4.8)
    max_dose_rate: int = 600
    start_gantry_angle: float = 179
    energy: float = 6
    fluence_mode: FluenceMode = FluenceMode.STANDARD
    coll_angle: float = 0
    couch_vrt: float = 0
    couch_lat: float = 0
    couch_lng: float = 1000
    couch_rot: float = 0
    beam_name: str = "GS"
    gantry_rot_dir: GantryDirection = GantryDirection.CLOCKWISE
    jaw_padding_mm: float = 5
    roi_size_mm: float = 30
    y1: float = -100
    y2: float = 100
    mu: int = 120

    def compute(self, machine: TrueBeamMachine) -> None:
        if max(self.speeds) > machine.specs.max_gantry_speed:
            raise ValueError(
                f"Maximum speed given {max(self.speeds)} is greater than the maximum gantry speed {machine.specs.max_gantry_speed}"
            )
        if self.roi_size_mm * len(self.speeds) > machine.specs.max_mlc_overtravel:
            raise ValueError(
                "The ROI size * number of speeds must be less than the overall MLC allowable width"
            )
        # determine sacrifices and gantry angles
        gantry_deltas = [
            speed * self.mu * 60 / self.max_dose_rate for speed in self.speeds
        ]
        gantry_sign = -1 if self.gantry_rot_dir == GantryDirection.CLOCKWISE else 1
        g_angles_uncorrected = [self.start_gantry_angle] + (
            self.start_gantry_angle + gantry_sign * np.cumsum(gantry_deltas)
        ).tolist()
        gantry_angles = [round(wrap360(angle), 2) for angle in g_angles_uncorrected]

        if sum(gantry_deltas) >= 360:
            raise ValueError(
                "Gantry travel is >360 degrees. Lower the beam MU, use fewer speeds, or decrease the desired gantry speeds"
            )

        mlc = Beam.create_modulator(machine)
        ref_mlc = Beam.create_modulator(machine)

        roi_centers = np.linspace(
            -self.roi_size_mm * len(self.speeds) / 2 + self.roi_size_mm / 2,
            self.roi_size_mm * len(self.speeds) / 2 - self.roi_size_mm / 2,
            len(self.speeds),
        )
        # we have a starting and ending strip
        ref_mlc.add_strip(
            position_mm=float(roi_centers[0]),
            strip_width_mm=self.roi_size_mm,
            meterset_at_target=0,
        )
        mlc.add_strip(
            position_mm=float(roi_centers[0]),
            strip_width_mm=self.roi_size_mm,
            meterset_at_target=0,
        )
        for center, gantry_angle in zip(roi_centers, gantry_angles):
            ref_mlc.add_strip(
                position_mm=center,
                strip_width_mm=self.roi_size_mm,
                meterset_at_target=0,
                meterset_transition=1 / len(self.speeds),
            )
            mlc.add_strip(
                position_mm=center,
                strip_width_mm=self.roi_size_mm,
                meterset_at_target=0,
                meterset_transition=1 / len(self.speeds),
            )

        beam = Beam(
            beam_name=self.beam_name,
            energy=self.energy,
            dose_rate=self.max_dose_rate,
            x1=min(roi_centers) - self.roi_size_mm - self.jaw_padding_mm,
            x2=max(roi_centers) + self.roi_size_mm + self.jaw_padding_mm,
            y1=self.y1,
            y2=self.y2,
            gantry_angles=gantry_angles,
            coll_angle=self.coll_angle,
            couch_vrt=self.couch_vrt,
            couch_lat=self.couch_lat,
            couch_lng=self.couch_lng,
            couch_rot=self.couch_rot,
            fluence_mode=self.fluence_mode,
            mlc_positions=mlc.as_control_points(),
            metersets=[self.mu * m for m in mlc.as_metersets()],
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(beam)
        ref_beam = Beam(
            beam_name=f"{self.beam_name} Ref",
            energy=self.energy,
            dose_rate=self.max_dose_rate,
            x1=min(roi_centers) - self.roi_size_mm - self.jaw_padding_mm,
            x2=max(roi_centers) + self.roi_size_mm + self.jaw_padding_mm,
            y1=self.y1,
            y2=self.y2,
            gantry_angles=gantry_angles[-1],
            coll_angle=self.coll_angle,
            couch_vrt=self.couch_vrt,
            couch_lat=self.couch_lat,
            couch_lng=self.couch_lng,
            couch_rot=self.couch_rot,
            fluence_mode=self.fluence_mode,
            mlc_positions=ref_mlc.as_control_points(),
            metersets=[self.mu * m for m in ref_mlc.as_metersets()],
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(ref_beam)


@dataclass
class VMATDRGS(QAProcedure):
    """Create beams like Clif Ling VMAT DRGS tests. The defaults use an optimized selection for a TrueBeam.

    Parameters
    ----------
    dose_rates : tuple
        The dose rates to test in MU/min. Each dose rate will have its own ROI.
    gantry_speeds : tuple
        The gantry speeds to tests in deg/sec. Each gantry speed will have its own ROI.
    mu_per_segment : float
        The number of MUs to deliver to each ROI
    mu_per_transition : float
        The number of MUs to deliver during while the MLCs move from one ROI to the next.
    correct_fluence : bool
        The original DRGS plans have an incorrect fluence on the initial and final transitions.
        Use False to replicate the original plans, otherwise use True to have a more uniform fluence.
    gantry_motion_per_transition : float
        The number of degrees that the gantry should rotate while the MLCs move from one ROI to the next.
    gantry_rotation_clockwise : bool
        The direction of the gantry rotation. If True, the gantry will rotate clockwise
    initial_gantry_offset : float
        The initial gantry offset in degrees. E.g. If initial_gantry_offset=1 and gantry_rotation_clockwise=True,
        then start angle = 181 IEC. If gantry_rotation_clockwise=False, then start angle = 179 IEC
    mlc_span : float
        The total size of the field in mm. Initial/final MLC position = +/- mlc_span/2
    mlc_motion_reverse : bool
        The direction of MLC motion. If False, the leaves move in positive direction (IEC)
        from -mlc_span/2 to +mlc_span/2. If True, the leaves move in negative direction (IEC)
        from +mlc_span/2 to -mlc_span/2.
    mlc_gap : float
        The MLC gap between ROIs in mm. This creates a darker region to help visualize the ROIs boundaries.
    jaw_padding : float
        The added jaw position in mm with respect to the initial/final MLC positions
    energy : float
        The energy of the beam.
    fluence_mode : FluenceMode
        The fluence mode of the beam.
    max_dose_rate : int
        The max dose rate in MU/min. This is used to compute the control point sequence to achieve the test dose_rates
    reference_beam_mu : float
        The number of MU's to be delivered in the reference beam (static beam)
    reference_beam_add_before : bool
        Whether to add the reference_beam before or after the dynamic beam. If True, the gantry angle is set to
        the initial gantry angle of the dynamic beam. If False, the gantry angle is set to the final gantry angle
        of the dynamic beam.
    dynamic_delivery_at_static_gantry : tuple
        There is one beam created for each static gantry angle. These beams contain the same control point sequence
        as the dynamic beam, but the gantry angle is replaced by a single value. There will be no modulation of
        dose rate and gantry speeds, and can be used as an alternative reference beam.
    """

    dose_rates: tuple[float, ...] = (600, 600, 600, 600, 500, 400, 200)
    gantry_speeds: tuple[float, ...] = (3, 4, 5, 6, 6, 6, 6)
    mu_per_segment: float = 48.0
    mu_per_transition: float = 8.0
    correct_fluence: bool = True
    gantry_motion_per_transition: float = 10.0
    gantry_rotation_clockwise: bool = True
    initial_gantry_offset: float = 1.0
    mlc_span: float = 138.0
    mlc_motion_reverse: bool = True
    mlc_gap: float = 2.0
    jaw_padding: float = 0.0
    energy: float = 6
    fluence_mode: FluenceMode = FluenceMode.STANDARD
    max_dose_rate: int = 600
    reference_beam_mu: float = 100.0
    reference_beam_add_before: bool = False
    dynamic_delivery_at_static_gantry: tuple[float, ...] = ()

    # Prevent using a gantry angle of 180°, which can cause ambiguity in the rotation direction.
    MIN_GANTRY_OFFSET: float = field(init=False, default=0.1)
    # The reference beam may be acquired prior to or following the dynamic beam
    # (as specified by the reference_beam_add_before argument).
    # These attributes record the indices of the respective beams.
    dynamic_beam_idx: int = field(init=False)
    reference_beam_idx: int = field(init=False)

    # private attributes: common to all beams to facilitate creation
    _x1: float = field(init=False)
    _x2: float = field(init=False)
    _y1: float = field(init=False)
    _y2: float = field(init=False)
    machine: TrueBeamMachine = field(init=False)

    @property
    def reference_beam(self) -> Beam:
        return self.beams[self.reference_beam_idx]

    @property
    def dynamic_beam(self) -> Beam:
        return self.beams[self.dynamic_beam_idx]

    @classmethod
    def from_varian_reference(cls) -> Self:
        """Add two beams that replicates Varian's T2_DoseRateGantrySpeed_M120_TB_Rev02.dcm

        .. include:: varian_reference_warning.rst
        """
        return cls(
            dose_rates=(600, 600, 600, 600, 502.691, 335.128, 167.564),
            gantry_speeds=(2.75, 3.056, 3.438, 4.296, 4.8, 4.8, 4.8),
            mu_per_segment=48.0,
            mu_per_transition=8.0,
            correct_fluence=False,
            gantry_motion_per_transition=10.0,
            gantry_rotation_clockwise=False,
            initial_gantry_offset=1.0,
            mlc_span=138.0,
            mlc_motion_reverse=True,
            mlc_gap=2.0,
            jaw_padding=0.0,
            max_dose_rate=600,
            reference_beam_mu=400.0,
        )

    def compute(self, machine: TrueBeamMachine) -> None:
        # store parameters common to all beams
        mlc_boundaries = (
            MLC_BOUNDARIES_TB_HD120 if machine.mlc_is_hd else MLC_BOUNDARIES_TB_MIL120
        )
        self._y1 = mlc_boundaries[0]
        self._y2 = mlc_boundaries[-1]
        self._x1 = -(self.mlc_span / 2 + self.jaw_padding)
        self._x2 = self.mlc_span / 2 + self.jaw_padding
        self.machine = machine

        # convert/cast variables
        gantry_speeds = np.array(self.gantry_speeds, dtype=float)
        dose_rates = np.array(self.dose_rates, dtype=float)
        mu_per_strip = float(self.mu_per_segment)
        mu_per_transition = float(self.mu_per_transition)
        gantry_motion_per_transition = float(self.gantry_motion_per_transition)

        # Verify inputs:
        if len(gantry_speeds) != len(dose_rates):
            raise ValueError("gantry_speeds and dose_rates must have the same length")
        if self.initial_gantry_offset < self.MIN_GANTRY_OFFSET:
            msg = f"The initial gantry offset cannot be smaller than {self.MIN_GANTRY_OFFSET} deg. Using 180 deg can cause ambiguity in the rotation direction."
            raise ValueError(msg)

        gantry_speeds_normalized = gantry_speeds / machine.specs.max_gantry_speed
        dose_rates_normalized = dose_rates / self.max_dose_rate
        # Verify that there are no requested speeds above limit
        if np.any(gantry_speeds_normalized > 1):
            raise ValueError("Requested gantry_speeds cannot exceed max_gantry_speed")
        if np.any(dose_rates_normalized > 1):
            raise ValueError("Requested dose_rates cannot exceed max_dose_rate")
        # Verify that at least one axis is maxed out for all control points
        norm_max = np.max((gantry_speeds_normalized, dose_rates_normalized), axis=0)
        if not np.all(norm_max == 1):
            raise ValueError("At least one axis must be maxed out")

        # calculate unmodulated variables
        num_strip = len(gantry_speeds)
        mu_per_sec = dose_rates / 60.0
        time_to_deliver_per_strip = mu_per_strip / mu_per_sec
        gantry_motions_per_strip = gantry_speeds * time_to_deliver_per_strip
        strip_spacing = float(self.mlc_span + self.mlc_gap) / num_strip
        strip_width = strip_spacing - self.mlc_gap

        # This is the modulation computation
        # delivery motion scheme (T is transition, D is Dose, numbers are index of calculated values (1-based))
        # CP    0 1 2 3 4 5 6 7 8 9
        # G     0 T 1 T 2 T 3 T 4 T
        # D     0 * D T D T D T D *     , * On the 1st and last transition the dose needs to be scaled to the mlc motion to prevent overdosage
        # MLC   S 1 1 2 2 3 3 4 4 E     , S is start positon and E is end position

        shaper = Beam.create_shaper(machine)
        strip_pos = np.linspace(-1, 1, num_strip) * (num_strip - 1) / 2 * strip_spacing
        k = -1 if self.mlc_motion_reverse else 1
        if self.mlc_motion_reverse:
            strip_pos = np.flip(strip_pos)
        # Initial control point (gantry_motion, dose_motion, mlc_position)
        gm, dm, mp = [0.0], [0.0], [shaper.get_shape(Strip(k * -self.mlc_span / 2, 0))]
        # Dynamic sequence
        for s in range(num_strip):
            gm += [gantry_motion_per_transition, gantry_motions_per_strip[s]]
            dm += [mu_per_transition, mu_per_strip]
            mp += 2 * [shaper.get_shape(Strip(strip_pos[s], strip_width))]
        # Final control point
        gm += [gantry_motion_per_transition]
        dm += [mu_per_transition]
        mp += [shaper.get_shape(Strip(k * self.mlc_span / 2, 0))]

        # Finalize values
        dose_motion = np.array(dm)
        if self.correct_fluence:
            correction_factor = 1 - self.mlc_gap / strip_spacing
            dose_motion[[1, -1]] = mu_per_transition * correction_factor
        cumulative_mu = np.cumsum(dose_motion)
        gantry_angles_without_offset = np.cumsum(gm)
        gantry_angles_var = gantry_angles_without_offset + self.initial_gantry_offset
        gantry_angles = (180 - gantry_angles_var) % 360
        if self.gantry_rotation_clockwise:
            gantry_angles = 360 - gantry_angles
        mlc = mp

        # Extra verifications on gantry rotation
        if gantry_angles_without_offset[-1] > 360 - 2 * self.MIN_GANTRY_OFFSET:
            msg = "The selected parameters require the gantry to rotate more than 360 degrees. Please select new parameters."
            raise ValueError(msg)
        if gantry_angles_var[-1] > 360 - self.MIN_GANTRY_OFFSET:
            msg = "The gantry rotation exceeds 360 degrees. Reduce the initial_gantry_offset"
            raise ValueError(msg)

        # Create dynamic beam
        dynamic_beam = self._beam("VMAT-DRGS-Dyn", cumulative_mu, gantry_angles, mlc)

        # Create reference beam
        ref_meterset = [0, self.reference_beam_mu]
        ref_ga = 2 * [float(gantry_angles[0 if self.reference_beam_add_before else -1])]
        ref_mlc = 2 * [shaper.get_shape(Strip(0, self.mlc_span))]
        reference_beam = self._beam("VMAT-DRGS-Ref", ref_meterset, ref_ga, ref_mlc)

        # Append the dynamic and reference beams according to the order defined in init
        beams: list[Beam | None] = 2 * [None]
        self.dynamic_beam_idx = 1 if self.reference_beam_add_before else 0
        self.reference_beam_idx = 0 if self.reference_beam_add_before else 1
        beams[self.dynamic_beam_idx] = dynamic_beam
        beams[self.reference_beam_idx] = reference_beam

        # Add static beams
        for gantry_angle in self.dynamic_delivery_at_static_gantry:
            beam = self._beam(
                f"VMAT-DRGS-G{gantry_angle:03d}", cumulative_mu, [gantry_angle], mlc
            )
            beams.append(beam)

        self.beams = beams

    def _beam(
        self,
        beam_name: str,
        metersets: Sequence[float],
        gantry_angles: Sequence[float],
        mlc: list[list[float]],
    ) -> Beam:
        """Multiple similar beams are created for the VMAT test.
        Common parameters are stored as attributes, whereas the dynamic axes
        are passed as arguments to this method."""

        return Beam(
            mlc_is_hd=self.machine.mlc_is_hd,
            beam_name=beam_name,
            energy=self.energy,
            fluence_mode=self.fluence_mode,
            dose_rate=self.max_dose_rate,
            metersets=metersets,
            gantry_angles=gantry_angles,
            x1=self._x1,
            x2=self._x2,
            y1=self._y1,
            y2=self._y2,
            mlc_positions=mlc,
            coll_angle=0,
            couch_vrt=0,
            couch_lat=0,
            couch_lng=0,
            couch_rot=0,
        )

    def plot_control_points(self, specs: MachineSpecs | None = None) -> None:
        """Plot the control points from dynamic beam

        Parameters
        ----------
        specs : MachineSpecs | None
            The machine specs used to compute speeds.
        """
        specs = specs or self.machine.specs
        beam = self.dynamic_beam
        beam.plot_control_points(specs)

    def plot_fluence(self, imager: Imager, show: bool = True) -> None:
        """Plot the fluence for the reference and dynamic beams

        Parameters
        ----------
        imager : Imager
            The target imager.
        show : bool, optional
            Whether to show the plots. Default is True.
        """
        self.reference_beam.plot_fluence(imager, show)
        self.dynamic_beam.plot_fluence(imager, show)

    def plot_fluence_profile(self, imager: Imager, zoom: float = 10) -> go.Figure:
        """Plot the fluence profile for the dynamic beam

        Parameters
        ----------
        imager : Imager
            The target imager.
        zoom: float
            The zoom factor in % around the max value, i.e. ylim = 1 + [-1, 1] * zoom/100

        Returns
        -------
        go.Figure
            The Plotly figure containing the fluence profile plot.
        """
        beam = self.dynamic_beam
        fluence = beam.generate_fluence(imager)
        profile = fluence[imager.shape[0] // 2, :]
        profile_max = profile.max()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(profile)),
                y=profile,
                mode="lines",
                name="Fluence Profile",
            )
        )
        fig.update_layout(
            yaxis_range=[
                (1 - zoom / 100) * profile_max,
                (1 + zoom / 100) * profile_max,
            ],
            title="Fluence Profile",
            xaxis_title="Position",
            yaxis_title="Fluence",
        )
        fig.show()

        return fig


@dataclass
class VMATDRMLC(QAProcedure):
    """Create beams like Clif Ling VMAT DRMLC tests. The defaults use an optimized
    selection for a TrueBeam.

    Parameters
    ----------
    mlc_speeds : tuple
        The mlc speeds rates to test in mm/sec. Each dose rate will have its own ROI.
    gantry_speeds: tuple | None
        The gantry speeds. When None it will default to max gantry speed for all segments.
    segment_width : tuple
        The width of each exposed segment in mm.
    gantry_rotation_clockwise : bool
        The direction of the gantry rotation. If True, the gantry will rotate clockwise
    initial_gantry_offset : float
        The initial gantry offset in degrees. E.g. If initial_gantry_offset=1 and gantry_rotation_clockwise=True,
        then start angle = 181 IEC. If gantry_rotation_clockwise=False, then start angle = 179 IEC
    mlc_motion_reverse : bool
        The direction of MLC motion. If False, the leaves move in positive direction (IEC)
        from -mlc_span/2 to +mlc_span/2. If True, the leaves move in negative direction (IEC)
        from +mlc_span/2 to -mlc_span/2.
    interpolation_factor : int
        Interpolation factor to create control points with finer resolution.
    jaw_padding : float
        The added jaw position in mm with respect to the initial/final MLC positions
    energy : float
        The energy of the beam.
    fluence_mode : FluenceMode
        The fluence mode of the beam.
    max_dose_rate : int
        The max dose rate in MU/min. This is used to compute the control point sequence to achieve the test dose_rates
    reference_beam_mu : float
        The number of MU's to be delivered in the reference beam (static beam)
    reference_beam_add_before : bool
        Whether to add the reference_beam before or after the dynamic beam. If True, the gantry angle is set to
        the initial gantry angle of the dynamic beam. If False, the gantry angle is set to the final gantry angle
        of the dynamic beam.
    dynamic_delivery_at_static_gantry : tuple
        There is one beam created for each static gantry angle. These beams contain the same control point sequence
        as the dynamic beam, but the gantry angle is replaced by a single value. There will be no modulation of
        dose rate and gantry speeds, and can be used as an alternative reference beam.
    """

    mlc_speeds: tuple[float, ...] = (15.0, 20.0, 10.0, 5.0)
    gantry_speeds: tuple[float, ...] | None = None
    segment_width: float = 30.0
    gantry_rotation_clockwise: bool = False
    initial_gantry_offset: float = 10.0
    mlc_motion_reverse: bool = False
    interpolation_factor: int = 1
    jaw_padding: float = 0.0
    energy: float = 6
    fluence_mode: FluenceMode = FluenceMode.STANDARD
    max_dose_rate: int = 600
    reference_beam_mu: float = 100.0
    reference_beam_add_before: bool = False
    dynamic_delivery_at_static_gantry: tuple[float, ...] = ()

    # Prevent using a gantry angle of 180°, which can cause ambiguity in the rotation direction.
    MIN_GANTRY_OFFSET: float = field(init=False, default=0.1)
    # The reference beam may be acquired prior to or following the dynamic beam
    # (as specified by the reference_beam_add_before argument).
    # These attributes record the indices of the respective beams.
    dynamic_beam_idx: int = field(init=False)
    reference_beam_idx: int = field(init=False)

    # private attributes: common to all beams to facilitate creation
    _x1: float = field(init=False)
    _x2: float = field(init=False)
    _y1: float = field(init=False)
    _y2: float = field(init=False)
    machine: TrueBeamMachine = field(init=False)

    @property
    def reference_beam(self) -> Beam:
        return self.beams[self.reference_beam_idx]

    @property
    def dynamic_beam(self) -> Beam:
        return self.beams[self.dynamic_beam_idx]

    @classmethod
    def from_varian_reference(cls) -> Self:
        """Add two beams that replicates Varian's T3_MLCSpeed_M120_TB_Rev02.dcm

        .. include:: varian_reference_warning.rst
        """
        return cls(
            mlc_speeds=(13.714, 20.0, 8.0, 4.0),
            gantry_speeds=(4.8, 4.0, 4.8, 4.8),
            segment_width=30,
            gantry_rotation_clockwise=False,
            initial_gantry_offset=10.0,
            mlc_motion_reverse=False,
            interpolation_factor=3,
            jaw_padding=0.0,
            max_dose_rate=600,
            reference_beam_mu=120.0,
            reference_beam_add_before=False,
        )

    def compute(self, machine: TrueBeamMachine):
        # Nomenclature
        # rois: the regions to be irradiated (related to mlc_speeds)
        # segment: each roi is made of 2 segments, one for each bank motion (related to control points)
        # Since this is about creating control points, most variables use a (implicit) _per_segment suffix unless specified.

        # convert/cast variables
        num_roi = len(self.mlc_speeds)
        gantry_speed_per_roi = self.gantry_speeds
        if not gantry_speed_per_roi:
            gantry_speed_per_roi = num_roi * [machine.specs.max_gantry_speed]
        gantry_speeds = np.array(np.repeat(gantry_speed_per_roi, 2), dtype=float)
        mlc_speeds = np.array(np.repeat(self.mlc_speeds, 2), dtype=float)

        # Verify inputs:
        if self.initial_gantry_offset < self.MIN_GANTRY_OFFSET:
            msg = f"The initial gantry offset cannot be smaller than {self.MIN_GANTRY_OFFSET} deg. Using 180 deg can cause ambiguity in the rotation direction."
            raise ValueError(msg)
        if any(mlc_speeds > machine.specs.max_mlc_speed):
            raise ValueError("Requested mlc_speeds cannot exceed max_mlc_speed")

        # The selected maximum MLC speed does not need to match the machine’s
        # maximum MLC speed. Regardless, the selected max MLC speed will be paired
        # with the maximum dose rate, and the remaining dose rates will be
        # interpolated accordingly.
        max_mlc_speed = max(mlc_speeds)
        dose_rates = mlc_speeds / max_mlc_speed * float(self.max_dose_rate)

        # Further verifications:
        gantry_speeds_normalized = gantry_speeds / machine.specs.max_gantry_speed
        dose_rates_normalized = dose_rates / self.max_dose_rate
        # Verify that there are no requested speeds above limit
        if np.any(gantry_speeds_normalized > 1):
            raise ValueError("Requested gantry_speeds cannot exceed max_gantry_speed")
        # Verify that at least one axis is maxed out for all control points
        norm_max = np.max((gantry_speeds_normalized, dose_rates_normalized), axis=0)
        if not np.all(norm_max == 1):
            raise ValueError("At least one axis must be maxed out")

        mu_per_sec = dose_rates / 60.0
        time_to_deliver = self.segment_width / mlc_speeds
        mu_per_segment = mu_per_sec * time_to_deliver
        gantry_motion = gantry_speeds * time_to_deliver

        # Extra verifications on gantry rotation
        gantry_angles_without_offset = np.cumsum([0] + gantry_motion.tolist())
        if gantry_angles_without_offset[-1] > 360 - 2 * self.MIN_GANTRY_OFFSET:
            msg = "The selected parameters require the gantry to rotate more than 360 degrees. Please select new parameters."
            raise ValueError(msg)
        gantry_angles_var = gantry_angles_without_offset + self.initial_gantry_offset
        if gantry_angles_var[-1] > 360 - self.MIN_GANTRY_OFFSET:
            msg = "The gantry rotation exceeds 360 degrees. Reduce the initial_gantry_offset"
            raise ValueError(msg)

        # This is the modulation computation
        # D is Dose, A/B are the banks, numbers are index of calculated values
        # CP    0 1 2 3 4 5 6 7 8
        # D     0 D D D D D D D D
        # A     0 1 1 2 2 3 3 4 4
        # B     0 0 1 1 2 2 3 3 4

        roi_limits = np.linspace(-1, 1, num_roi + 1) * 2 * self.segment_width
        mlc_a = roi_limits[np.repeat(range(num_roi + 1), [1] + num_roi * [2])]
        mlc_b = roi_limits[np.repeat(range(num_roi + 1), num_roi * [2] + [1])]
        if self.mlc_motion_reverse:
            mlc_a = np.flip(mlc_a)
            mlc_b = np.flip(mlc_b)
        shaper = Beam.create_shaper(machine)
        z = zip(mlc_a, mlc_b)
        mlc = [shaper.get_shape(Strip.from_minmax(b, a)) for (a, b) in z]
        cumulative_mu = np.cumsum([0] + mu_per_segment.tolist())

        # store parameters common to all beams
        mlc_boundaries = (
            MLC_BOUNDARIES_TB_HD120 if machine.mlc_is_hd else MLC_BOUNDARIES_TB_MIL120
        )
        self._y1 = mlc_boundaries[0]
        self._y2 = mlc_boundaries[-1]
        self._x1 = min(roi_limits) - self.jaw_padding
        self._x2 = max(roi_limits) + self.jaw_padding
        self.machine = machine

        # Finalize values
        gantry_angles = (180 - gantry_angles_var) % 360
        if self.gantry_rotation_clockwise:
            gantry_angles = 360 - gantry_angles
        num_segments = len(cumulative_mu)
        x = range(num_segments)
        k = num_segments - 1
        x_ = np.linspace(0, k, k * self.interpolation_factor + 1)
        cumulative_mu = make_interp_spline(x, cumulative_mu, k=1)(x_)
        gantry_angles = make_interp_spline(x, gantry_angles, k=1)(x_)
        mlc = make_interp_spline(x, np.array(mlc).T, k=1, axis=1)(x_).T.tolist()

        # Create dynamic beam
        dynamic_beam = self._beam("VMAT-DRMLC-Dyn", cumulative_mu, gantry_angles, mlc)

        # Create reference beam
        ref_meterset = [0, self.reference_beam_mu]
        ref_ga = 2 * [float(gantry_angles[0 if self.reference_beam_add_before else -1])]
        ref_mlc = 2 * [shaper.get_shape(Strip.from_minmax(min(mlc_b), max(mlc_a)))]
        ref_beam = self._beam("VMAT-DRMLC-Ref", ref_meterset, ref_ga, ref_mlc)

        # Append the dynamic and reference beams according to the order defined in init
        beams: list[Beam | None] = 2 * [None]
        self.dynamic_beam_idx = 1 if self.reference_beam_add_before else 0
        self.reference_beam_idx = 0 if self.reference_beam_add_before else 1
        beams[self.dynamic_beam_idx] = dynamic_beam
        beams[self.reference_beam_idx] = ref_beam

        # Add static beams
        for gantry_angle in self.dynamic_delivery_at_static_gantry:
            beam = self._beam(
                f"VMAT-DRMLC-G{gantry_angle:03d}", cumulative_mu, [gantry_angle], mlc
            )
            beams.append(beam)

        self.beams = beams

    def _beam(
        self,
        beam_name: str,
        metersets: Sequence[float],
        gantry_angles: Sequence[float],
        mlc: list[list[float]],
    ) -> Beam:
        """Multiple similar beams are created for the VMAT test.
        Common parameters are stored as attributes, whereas the dynamic axes
        are passed as arguments to this method."""
        return Beam(
            mlc_is_hd=self.machine.mlc_is_hd,
            beam_name=beam_name,
            energy=self.energy,
            fluence_mode=self.fluence_mode,
            dose_rate=self.max_dose_rate,
            metersets=metersets,
            gantry_angles=gantry_angles,
            x1=self._x1,
            x2=self._x2,
            y1=self._y1,
            y2=self._y2,
            mlc_positions=mlc,
            coll_angle=0,
            couch_vrt=0,
            couch_lat=0,
            couch_lng=0,
            couch_rot=0,
        )

    def plot_control_points(self, specs: MachineSpecs | None = None) -> None:
        """Plot the control points from dynamic beam

        Parameters
        ----------
        specs : MachineSpecs | None
            The machine specs used to compute speeds.
        """
        specs = specs or self.machine.specs
        beam = self.dynamic_beam
        beam.plot_control_points(specs)

    def plot_fluence(self, imager: Imager, show: bool = True) -> None:
        """Plot the fluence for the reference and dynamic beams

        Parameters
        ----------
        imager : Imager
            The target imager.
        show : bool, optional
            Whether to show the plots. Default is True.
        """
        self.reference_beam.plot_fluence(imager, show)
        self.dynamic_beam.plot_fluence(imager, show)

    def plot_fluence_profile(self, imager: Imager, zoom: float = 10) -> go.Figure:
        """Plot the fluence profile for the dynamic beam

        Parameters
        ----------
        imager : Imager
            The target imager.
        zoom: float
            The zoom factor in % around the max value, i.e. ylim = 1 + [-1, 1] * zoom/100

        Returns
        -------
        go.Figure
            The Plotly figure containing the fluence profile plot.
        """
        beam = self.dynamic_beam
        fluence = beam.generate_fluence(imager)
        profile = fluence[imager.shape[0] // 2, :]
        profile_max = profile.max()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(profile)),
                y=profile,
                mode="lines",
                name="Fluence Profile",
            )
        )
        fig.update_layout(
            yaxis_range=[
                (1 - zoom / 100) * profile_max,
                (1 + zoom / 100) * profile_max,
            ],
            title="Fluence Profile",
            xaxis_title="Position",
            yaxis_title="Fluence",
        )
        fig.show()

        return fig
