import math
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt

from conjuror.images.simulators import Imager
from conjuror.plans.mlc import MLCShaper
from conjuror.plans.plan_generator import QAProcedureBase, TrueBeamMachine, FluenceMode, \
    Beam, OvertravelError, GantryDirection, MLC_BOUNDARIES_TB_HD120, \
    MLC_BOUNDARIES_TB_MIL120, MachineSpecs
from conjuror.utils import wrap360


class QAProcedure(QAProcedureBase):

    @staticmethod
    def _create_mlc(
            machine: TrueBeamMachine, sacrifice_gap_mm: float = None,
            sacrifice_max_move_mm: float = None
    ) -> MLCShaper:
        """Utility to create MLC shaper instances."""
        return MLCShaper(
            leaf_y_positions=machine.mlc_boundaries,
            max_mlc_position=machine.machine_specs.max_mlc_position,
            max_overtravel_mm=machine.machine_specs.max_mlc_overtravel,
            sacrifice_gap_mm=sacrifice_gap_mm,
            sacrifice_max_move_mm=sacrifice_max_move_mm,
        )

class OpenField(QAProcedure):
    """Create an open field beam."""

    def __init__(
        self,
        machine: TrueBeamMachine,
        x1: float,
        x2: float,
        y1: float,
        y2: float,
        defined_by_mlcs: bool = True,
        energy: float = 6,
        fluence_mode: FluenceMode = FluenceMode.STANDARD,
        dose_rate: int = 600,
        gantry_angle: float = 0,
        coll_angle: float = 0,
        couch_vrt: float = 0,
        couch_lng: float = 1000,
        couch_lat: float = 0,
        couch_rot: float = 0,
        mu: int = 200,
        padding_mm: float = 5,
        beam_name: str = "Open",
        outside_strip_width_mm: float = 5,
    ):
        """Create an open field beam.

        Parameters
        ----------
        machine : TrueBeamMachine
            The target machine.
        x1 : float
            The left jaw position.
        x2 : float
            The right jaw position.
        y1 : float
            The bottom jaw position.
        y2 : float
            The top jaw position.
        defined_by_mlcs : bool
            Whether the field edges are defined by the MLCs or the jaws.
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
        mu : int
            The monitor units of the beam.
        padding_mm : float
            The padding to add to the jaws or MLCs.
        beam_name : str
            The name of the beam.
        outside_strip_width_mm : float
            The width of the strip of MLCs outside the field. The MLCs will be placed to the
            left, under the X1 jaw by ~2cm.
        """
        super().__init__()
        if defined_by_mlcs:
            mlc_padding = 0
            jaw_padding = padding_mm
        else:
            mlc_padding = padding_mm
            jaw_padding = 0
        mlc = self._create_mlc(machine)
        mlc.add_rectangle(
            left_position=x1 - mlc_padding,
            right_position=x2 + mlc_padding,
            top_position=y2 + mlc_padding,
            bottom_position=y1 - mlc_padding,
            outer_strip_width=outside_strip_width_mm,
            x_outfield_position=x1 - mlc_padding - jaw_padding - 20,
            meterset_at_target=1.0,
        )
        beam = Beam.for_truebeam(
            beam_name=beam_name,
            energy=energy,
            dose_rate=dose_rate,
            x1=x1 - jaw_padding,
            x2=x2 + jaw_padding,
            y1=y1 - jaw_padding,
            y2=y2 + jaw_padding,
            gantry_angles=gantry_angle,
            coll_angle=coll_angle,
            couch_vrt=couch_vrt,
            couch_lat=couch_lat,
            couch_lng=couch_lng,
            couch_rot=couch_rot,
            mlc_positions=mlc.as_control_points(),
            metersets=[mu * m for m in mlc.as_metersets()],
            fluence_mode=fluence_mode,
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(beam)


class MLCTransmission(QAProcedure):
    def __init__(self,
        machine: TrueBeamMachine,
        bank: Literal["A", "B"],
        mu: int = 50,
        overreach: float = 10,
        beam_name: str = "MLC Tx",
        energy: int = 6,
        dose_rate: int = 600,
        x1: float = -50,
        x2: float = 50,
        y1: float = -100,
        y2: float = 100,
        gantry_angle: float = 0,
        coll_angle: float = 0,
        couch_vrt: float = 0,
        couch_lat: float = 0,
        couch_lng: float = 1000,
        couch_rot: float = 0,
        fluence_mode: FluenceMode = FluenceMode.STANDARD,
    ):
        """Add a single-image MLC transmission beam to the plan.
        The beam is delivered with the MLCs closed and moved to one side underneath the jaws.

        Parameters
        ----------
        machine : TrueBeamMachine
            The target machine.
        bank : str
            The MLC bank to move. Either "A" or "B".
        mu : int
            The monitor units to deliver.
        overreach : float
            The amount to tuck the MLCs under the jaws in mm.
        beam_name : str
            The name of the beam.
        energy : int
            The energy of the beam.
        dose_rate : int
            The dose rate of the beam.
        x1 : float
            The left jaw position. Usually negative. More negative is left.
        x2 : float
            The right jaw position. Usually positive. More positive is right.
        y1 : float
            The bottom jaw position. Usually negative. More negative is lower.
        y2 : float
            The top jaw position. Usually positive. More positive is higher.
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
        fluence_mode : FluenceMode
            The fluence mode of the beam.
        """
        super().__init__()
        mlc = self._create_mlc(machine)
        if bank == "A":
            mlc_tips = x2 + overreach
        elif bank == "B":
            mlc_tips = x1 - overreach
        else:
            raise ValueError("Bank must be 'A' or 'B'")
        # test for overtravel
        if abs(x2 - x1) + overreach > machine.machine_specs.max_mlc_overtravel:
            raise OvertravelError(
                "The MLC overtravel is too large for the given jaw positions and overreach. Reduce the x-jaw opening size and/or overreach value."
            )
        mlc.add_strip(
            position_mm=mlc_tips,
            strip_width_mm=1,
            meterset_at_target=1,
        )
        beam = Beam.for_truebeam(
            beam_name=f"{beam_name} {bank}",
            energy=energy,
            dose_rate=dose_rate,
            x1=x1,
            x2=x2,
            y1=y1,
            y2=y2,
            gantry_angles=gantry_angle,
            coll_angle=coll_angle,
            couch_vrt=couch_vrt,
            couch_lat=couch_lat,
            couch_lng=couch_lng,
            couch_rot=couch_rot,
            mlc_positions=mlc.as_control_points(),
            metersets=[mu * m for m in mlc.as_metersets()],
            fluence_mode=fluence_mode,
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(beam)


class PicketFence(QAProcedure):
    def __init__(self,
        machine: TrueBeamMachine,
        strip_width_mm: float = 3,
        strip_positions_mm: tuple[float | int, ...] = (-45, -30, -15, 0, 15, 30, 45),
        y1: float = -100,
        y2: float = 100,
        fluence_mode: FluenceMode = FluenceMode.STANDARD,
        dose_rate: int = 600,
        energy: float = 6,
        gantry_angle: float = 0,
        coll_angle: float = 0,
        couch_vrt: float = 0,
        couch_lng: float = 1000,
        couch_lat: float = 0,
        couch_rot: float = 0,
        mu: int = 200,
        jaw_padding_mm: float = 10,
        beam_name: str = "PF",
        max_sacrificial_move_mm: float = 50,
    ):
        """Add a picket fence beam to the plan.

        Parameters
        ----------
        machine : TrueBeamMachine
            The target machine.
        strip_width_mm : float
            The width of the strips in mm.
        strip_positions_mm : tuple
            The positions of the strips in mm relative to the center of the image.
        y1 : float
            The bottom jaw position. Usually negative. More negative is lower.
        y2 : float
            The top jaw position. Usually positive. More positive is higher.
        fluence_mode : FluenceMode
            The fluence mode of the beam.
        dose_rate : int
            The dose rate of the beam.
        energy : float
            The energy of the beam.
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
        mu : int
            The monitor units of the beam.
        jaw_padding_mm : float
            The padding to add to the X jaws.
        beam_name : str
            The name of the beam.
        max_sacrificial_move_mm : float
            The maximum distance the sacrificial leaves can move in a given control point.
            Smaller values generate more control points and more back-and-forth movement.
            Too large of values may cause deliverability issues.
        """
        super().__init__()
        # check MLC overtravel; machine may prevent delivery if exposing leaf tail
        x1 = min(strip_positions_mm) - jaw_padding_mm
        x2 = max(strip_positions_mm) + jaw_padding_mm
        max_dist_to_jaw = max(
            max(abs(pos - x1), abs(pos + x2)) for pos in strip_positions_mm
        )
        if max_dist_to_jaw > machine.machine_specs.max_mlc_overtravel:
            raise ValueError(
                "Picket fence beam exceeds MLC overtravel limits. Lower padding, the number of pickets, or the picket spacing."
            )
        mlc = self._create_mlc(machine, sacrifice_max_move_mm=max_sacrificial_move_mm)
        # create initial starting point; start under the jaws
        mlc.add_strip(
            position_mm=strip_positions_mm[0] - 2,
            strip_width_mm=strip_width_mm,
            meterset_at_target=0,
        )

        for strip in strip_positions_mm:
            # starting control point
            mlc.add_strip(
                position_mm=strip,
                strip_width_mm=strip_width_mm,
                meterset_at_target=1 / len(strip_positions_mm),
            )
        beam = Beam.for_truebeam(
            beam_name=beam_name,
            energy=energy,
            dose_rate=dose_rate,
            x1=x1,
            x2=x2,
            y1=y1,
            y2=y2,
            gantry_angles=gantry_angle,
            coll_angle=coll_angle,
            couch_vrt=couch_vrt,
            couch_lat=couch_lat,
            couch_lng=couch_lng,
            couch_rot=couch_rot,
            mlc_positions=mlc.as_control_points(),
            metersets=[mu * m for m in mlc.as_metersets()],
            fluence_mode=fluence_mode,
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(beam)


class WinstonLutz(QAProcedure):
    def __init__(self,
        machine: TrueBeamMachine,
        x1: float = -10,
        x2: float = 10,
        y1: float = -10,
        y2: float = 10,
        defined_by_mlcs: bool = True,
        energy: float = 6,
        fluence_mode: FluenceMode = FluenceMode.STANDARD,
        dose_rate: int = 600,
        axes_positions: Iterable[dict] = ({"gantry": 0, "collimator": 0, "couch": 0},),
        couch_vrt: float = 0,
        couch_lng: float = 1000,
        couch_lat: float = 0,
        mu: int = 10,
        padding_mm: float = 5,
    ):
        """Add Winston-Lutz beams to the plan. Will create a beam for each set of axes positions.
        Field names are generated automatically based on the axes positions.

        Parameters
        ----------
        machine : TrueBeamMachine
            The target machine.
        x1 : float
            The left jaw position.
        x2 : float
            The right jaw position.
        y1 : float
            The bottom jaw position.
        y2 : float
            The top jaw position.
        defined_by_mlcs : bool
            Whether the field edges are defined by the MLCs or the jaws.
        energy : float
            The energy of the beam.
        fluence_mode : FluenceMode
            The fluence mode of the beam.
        dose_rate : int
            The dose rate of the beam.
        axes_positions : Iterable[dict]
            The positions of the axes. Each dict should have keys 'gantry', 'collimator', 'couch', and optionally 'name'.
        couch_vrt : float
            The couch vertical position.
        couch_lng : float
            The couch longitudinal position.
        couch_lat : float
            The couch lateral position.
        mu : int
            The monitor units of the beam.
        padding_mm : float
            The padding to add. If defined by the MLCs, this is the padding of the jaws. If defined by the jaws,
            this is the padding of the MLCs.
        """
        super().__init__()
        for axes in axes_positions:
            if defined_by_mlcs:
                mlc_padding = 0
                jaw_padding = padding_mm
            else:
                mlc_padding = padding_mm
                jaw_padding = 0
            mlc = self._create_mlc(machine)
            mlc.add_rectangle(
                left_position=x1 - mlc_padding,
                right_position=x2 + mlc_padding,
                top_position=y2 + mlc_padding,
                bottom_position=y1 - mlc_padding,
                outer_strip_width=5,
                meterset_at_target=1.0,
                x_outfield_position=x1 - mlc_padding - jaw_padding - 20,
            )
            beam_name = (
                axes.get("name")
                or f"G{axes['gantry']:g}C{axes['collimator']:g}P{axes['couch']:g}"
            )
            beam = Beam.for_truebeam(
                beam_name=beam_name,
                energy=energy,
                dose_rate=dose_rate,
                x1=x1 - jaw_padding,
                x2=x2 + jaw_padding,
                y1=y1 - jaw_padding,
                y2=y2 + jaw_padding,
                gantry_angles=axes["gantry"],
                coll_angle=axes["collimator"],
                couch_vrt=couch_vrt,
                couch_lat=couch_lat,
                couch_lng=couch_lng,
                couch_rot=0,
                mlc_positions=mlc.as_control_points(),
                metersets=[mu * m for m in mlc.as_metersets()],
                fluence_mode=fluence_mode,
                mlc_is_hd=machine.mlc_is_hd,
            )
            self.beams.append(beam)


class DoseRate(QAProcedure):
    def __init__(self,
        machine: TrueBeamMachine,
        dose_rates: tuple[int, ...] = (100, 300, 500, 600),
        default_dose_rate: int = 600,
        gantry_angle: float = 0,
        desired_mu: int = 50,
        energy: float = 6,
        fluence_mode: FluenceMode = FluenceMode.STANDARD,
        coll_angle: float = 0,
        couch_vrt: float = 0,
        couch_lat: float = 0,
        couch_lng: float = 1000,
        couch_rot: float = 0,
        jaw_padding_mm: float = 5,
        roi_size_mm: float = 25,
        y1: float = -100,
        y2: float = 100,
        max_sacrificial_move_mm: float = 50,
    ):
        """Create a single-image dose rate test. Multiple ROIs are generated. A reference beam is also
        created where all ROIs are delivered at the default dose rate for comparison.
        The field names are generated automatically based on the min and max dose rates tested.

        Parameters
        ----------
        machine : TrueBeamMachine
            The target machine.
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
        super().__init__()
        if roi_size_mm * len(dose_rates) > machine.machine_specs.max_mlc_overtravel:
            raise ValueError(
                "The ROI size * number of dose rates must be less than the overall MLC allowable width"
            )
        # calculate MU
        mlc_transition_time = roi_size_mm / machine.machine_specs.max_mlc_speed
        min_mu = mlc_transition_time * max(dose_rates) * len(dose_rates) / 60
        mu = max(desired_mu, math.ceil(min_mu))

        # create MLC sacrifices
        times_to_transition = [
            mu * 60 / (dose_rate * len(dose_rates)) for dose_rate in dose_rates
        ]
        sacrificial_movements = [tt * machine.machine_specs.max_mlc_speed for tt in times_to_transition]

        mlc = self._create_mlc(machine, sacrifice_max_move_mm=max_sacrificial_move_mm)
        ref_mlc = self._create_mlc(machine)

        roi_centers = np.linspace(
            -roi_size_mm * len(dose_rates) / 2 + roi_size_mm / 2,
            roi_size_mm * len(dose_rates) / 2 - roi_size_mm / 2,
            len(dose_rates),
        )
        # we have a starting and ending strip
        ref_mlc.add_strip(
            position_mm=float(roi_centers[0] - roi_size_mm / 2),
            strip_width_mm=0,
            meterset_at_target=0,
        )
        mlc.add_strip(
            position_mm=float(roi_centers[0] - roi_size_mm / 2),
            strip_width_mm=0,
            meterset_at_target=0,
            initial_sacrificial_gap_mm=5,
        )
        for sacrifice_distance, center in zip(sacrificial_movements, roi_centers):
            ref_mlc.add_rectangle(
                left_position=center - roi_size_mm / 2,
                right_position=center + roi_size_mm / 2,
                x_outfield_position=-200,
                top_position=max(machine.mlc_boundaries),
                bottom_position=min(machine.mlc_boundaries),
                outer_strip_width=5,
                meterset_at_target=0,
                meterset_transition=0.5 / len(dose_rates),
                sacrificial_distance=0,
            )
            ref_mlc.add_strip(
                position_mm=center + roi_size_mm / 2,
                strip_width_mm=0,
                meterset_at_target=0,
                meterset_transition=0.5 / len(dose_rates),
                sacrificial_distance_mm=0,
            )
            mlc.add_rectangle(
                left_position=center - roi_size_mm / 2,
                right_position=center + roi_size_mm / 2,
                x_outfield_position=-200,  # not used
                top_position=max(machine.mlc_boundaries),
                bottom_position=min(machine.mlc_boundaries),
                outer_strip_width=5,  # not used
                meterset_at_target=0,
                meterset_transition=0.5 / len(dose_rates),
                sacrificial_distance=sacrifice_distance,
            )
            mlc.add_strip(
                position_mm=center + roi_size_mm / 2,
                strip_width_mm=0,
                meterset_at_target=0,
                meterset_transition=0.5 / len(dose_rates),
                sacrificial_distance_mm=sacrifice_distance,
            )
        ref_beam = Beam.for_truebeam(
            beam_name="DR Ref",
            energy=energy,
            dose_rate=default_dose_rate,
            x1=float(roi_centers[0] - roi_size_mm / 2 - jaw_padding_mm),
            x2=float(roi_centers[-1] + roi_size_mm / 2 + jaw_padding_mm),
            y1=y1,
            y2=y2,
            gantry_angles=gantry_angle,
            coll_angle=coll_angle,
            couch_vrt=couch_vrt,
            couch_lat=couch_lat,
            couch_lng=couch_lng,
            couch_rot=couch_rot,
            fluence_mode=fluence_mode,
            mlc_positions=ref_mlc.as_control_points(),
            metersets=[mu * m for m in ref_mlc.as_metersets()],
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(ref_beam)
        beam = Beam.for_truebeam(
            beam_name=f"DR{min(dose_rates)}-{max(dose_rates)}",
            energy=energy,
            dose_rate=default_dose_rate,
            x1=float(roi_centers[0] - roi_size_mm / 2 - jaw_padding_mm),
            x2=float(roi_centers[-1] + roi_size_mm / 2 + jaw_padding_mm),
            y1=y1,
            y2=y2,
            gantry_angles=gantry_angle,
            coll_angle=coll_angle,
            couch_vrt=couch_vrt,
            couch_lat=couch_lat,
            couch_lng=couch_lng,
            couch_rot=couch_rot,
            fluence_mode=fluence_mode,
            mlc_positions=mlc.as_control_points(),
            metersets=[mu * m for m in mlc.as_metersets()],
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(beam)


class MLCSpeed(QAProcedure):
    def __init__(self,
        machine: TrueBeamMachine,
        speeds: tuple[float | int, ...] = (5, 10, 15, 20),
        roi_size_mm: float = 20,
        mu: int = 50,
        default_dose_rate: int = 600,
        gantry_angle: float = 0,
        energy: float = 6,
        coll_angle: float = 0,
        couch_vrt: float = 0,
        couch_lat: float = 0,
        couch_lng: float = 1000,
        couch_rot: float = 0,
        fluence_mode: FluenceMode = FluenceMode.STANDARD,
        jaw_padding_mm: float = 5,
        y1: float = -100,
        y2: float = 100,
        beam_name: str = "MLC Speed",
        max_sacrificial_move_mm: float = 50,
    ):
        """Create a single-image MLC speed test. Multiple speeds are generated. A reference beam is also
        generated. The reference beam is delivered at the maximum MLC speed.

        Parameters
        ----------
        machine : TrueBeamMachine
            The target machine.
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
        super().__init__()
        if max(speeds) > machine.machine_specs.max_mlc_speed:
            raise ValueError(
                f"Maximum speed given {max(speeds)} is greater than the maximum MLC speed {machine.machine_specs.max_mlc_speed}"
            )
        if min(speeds) <= 0:
            raise ValueError("Speeds must be greater than 0")
        if roi_size_mm * len(speeds) > machine.machine_specs.max_mlc_overtravel:
            raise ValueError(
                "The ROI size * number of speeds must be less than the overall MLC allowable width"
            )
        # create MLC positions
        times_to_transition = [roi_size_mm / speed for speed in speeds]
        sacrificial_movements = [tt * machine.machine_specs.max_mlc_speed for tt in times_to_transition]

        mlc = self._create_mlc(machine, sacrifice_max_move_mm=max_sacrificial_move_mm)
        ref_mlc = self._create_mlc(machine)

        roi_centers = np.linspace(
            -roi_size_mm * len(speeds) / 2 + roi_size_mm / 2,
            roi_size_mm * len(speeds) / 2 - roi_size_mm / 2,
            len(speeds),
        )
        # we have a starting and ending strip
        ref_mlc.add_strip(
            position_mm=float(roi_centers[0] - roi_size_mm / 2),
            strip_width_mm=0,
            meterset_at_target=0,
        )
        mlc.add_strip(
            position_mm=float(roi_centers[0] - roi_size_mm / 2),
            strip_width_mm=0,
            meterset_at_target=0,
            initial_sacrificial_gap_mm=5,
        )
        for sacrifice_distance, center in zip(sacrificial_movements, roi_centers):
            ref_mlc.add_rectangle(
                left_position=center - roi_size_mm / 2,
                right_position=center + roi_size_mm / 2,
                x_outfield_position=-200,  # not relevant
                top_position=max(machine.mlc_boundaries),
                bottom_position=min(machine.mlc_boundaries),
                outer_strip_width=5,  # not relevant
                meterset_at_target=0,
                meterset_transition=0.5 / len(speeds),
                sacrificial_distance=0,
            )
            ref_mlc.add_strip(
                position_mm=center + roi_size_mm / 2,
                strip_width_mm=0,
                meterset_at_target=0,
                meterset_transition=0.5 / len(speeds),
                sacrificial_distance_mm=0,
            )
            mlc.add_rectangle(
                left_position=center - roi_size_mm / 2,
                right_position=center + roi_size_mm / 2,
                x_outfield_position=-200,  # not used
                top_position=max(machine.mlc_boundaries),
                bottom_position=min(machine.mlc_boundaries),
                outer_strip_width=5,  # not used
                meterset_at_target=0,
                meterset_transition=0.5 / len(speeds),
                sacrificial_distance=sacrifice_distance,
            )
            mlc.add_strip(
                position_mm=center + roi_size_mm / 2,
                strip_width_mm=0,
                meterset_at_target=0,
                meterset_transition=0.5 / len(speeds),
                sacrificial_distance_mm=sacrifice_distance,
            )
        ref_beam = Beam.for_truebeam(
            beam_name=f"{beam_name} Ref",
            energy=energy,
            dose_rate=default_dose_rate,
            x1=float(roi_centers[0] - roi_size_mm / 2 - jaw_padding_mm),
            x2=float(roi_centers[-1] + roi_size_mm / 2 + jaw_padding_mm),
            y1=y1,
            y2=y2,
            gantry_angles=gantry_angle,
            coll_angle=coll_angle,
            couch_vrt=couch_vrt,
            couch_lat=couch_lat,
            couch_lng=couch_lng,
            couch_rot=couch_rot,
            fluence_mode=fluence_mode,
            mlc_positions=ref_mlc.as_control_points(),
            metersets=[mu * m for m in ref_mlc.as_metersets()],
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(ref_beam)
        beam = Beam.for_truebeam(
            beam_name=beam_name,
            energy=energy,
            dose_rate=default_dose_rate,
            x1=float(roi_centers[0] - roi_size_mm / 2 - jaw_padding_mm),
            x2=float(roi_centers[-1] + roi_size_mm / 2 + jaw_padding_mm),
            y1=y1,
            y2=y2,
            gantry_angles=gantry_angle,
            coll_angle=coll_angle,
            couch_vrt=couch_vrt,
            couch_lat=couch_lat,
            couch_lng=couch_lng,
            couch_rot=couch_rot,
            fluence_mode=fluence_mode,
            mlc_positions=mlc.as_control_points(),
            metersets=[mu * m for m in mlc.as_metersets()],
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(beam)



class GantrySpeed(QAProcedure):
    def __init__(self,
        machine: TrueBeamMachine,
        speeds: tuple[float | int, ...] = (2, 3, 4, 4.8),
        max_dose_rate: int = 600,
        start_gantry_angle: float = 179,
        energy: float = 6,
        fluence_mode: FluenceMode = FluenceMode.STANDARD,
        coll_angle: float = 0,
        couch_vrt: float = 0,
        couch_lat: float = 0,
        couch_lng: float = 1000,
        couch_rot: float = 0,
        beam_name: str = "GS",
        gantry_rot_dir: GantryDirection = GantryDirection.CLOCKWISE,
        jaw_padding_mm: float = 5,
        roi_size_mm: float = 30,
        y1: float = -100,
        y2: float = 100,
        mu: int = 120,
    ):
        """Create a single-image gantry speed test. Multiple speeds are generated. A reference beam is also
        generated. The reference beam is delivered without gantry movement.

        Parameters
        ----------
        machine : TrueBeamMachine
            The target machine.
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
        super().__init__()
        if max(speeds) > machine.machine_specs.max_gantry_speed:
            raise ValueError(
                f"Maximum speed given {max(speeds)} is greater than the maximum gantry speed {machine.machine_specs.max_gantry_speed}"
            )
        if roi_size_mm * len(speeds) > machine.machine_specs.max_mlc_overtravel:
            raise ValueError(
                "The ROI size * number of speeds must be less than the overall MLC allowable width"
            )
        # determine sacrifices and gantry angles
        gantry_deltas = [speed * mu * 60 / max_dose_rate for speed in speeds]
        gantry_sign = -1 if gantry_rot_dir == GantryDirection.CLOCKWISE else 1
        g_angles_uncorrected = [start_gantry_angle] + (
            start_gantry_angle + gantry_sign * np.cumsum(gantry_deltas)
        ).tolist()
        gantry_angles = [round(wrap360(angle), 2) for angle in g_angles_uncorrected]

        if sum(gantry_deltas) >= 360:
            raise ValueError(
                "Gantry travel is >360 degrees. Lower the beam MU, use fewer speeds, or decrease the desired gantry speeds"
            )

        mlc = self._create_mlc(machine)
        ref_mlc = self._create_mlc(machine)

        roi_centers = np.linspace(
            -roi_size_mm * len(speeds) / 2 + roi_size_mm / 2,
            roi_size_mm * len(speeds) / 2 - roi_size_mm / 2,
            len(speeds),
        )
        # we have a starting and ending strip
        ref_mlc.add_strip(
            position_mm=float(roi_centers[0]),
            strip_width_mm=roi_size_mm,
            meterset_at_target=0,
        )
        mlc.add_strip(
            position_mm=float(roi_centers[0]),
            strip_width_mm=roi_size_mm,
            meterset_at_target=0,
        )
        for center, gantry_angle in zip(roi_centers, gantry_angles):
            ref_mlc.add_strip(
                position_mm=center,
                strip_width_mm=roi_size_mm,
                meterset_at_target=0,
                meterset_transition=1 / len(speeds),
            )
            mlc.add_strip(
                position_mm=center,
                strip_width_mm=roi_size_mm,
                meterset_at_target=0,
                meterset_transition=1 / len(speeds),
            )

        beam = Beam.for_truebeam(
            beam_name=beam_name,
            energy=energy,
            dose_rate=max_dose_rate,
            x1=min(roi_centers) - roi_size_mm - jaw_padding_mm,
            x2=max(roi_centers) + roi_size_mm + jaw_padding_mm,
            y1=y1,
            y2=y2,
            gantry_angles=gantry_angles,
            coll_angle=coll_angle,
            couch_vrt=couch_vrt,
            couch_lat=couch_lat,
            couch_lng=couch_lng,
            couch_rot=couch_rot,
            fluence_mode=fluence_mode,
            mlc_positions=mlc.as_control_points(),
            metersets=[mu * m for m in mlc.as_metersets()],
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(beam)
        ref_beam = Beam.for_truebeam(
            beam_name=f"{beam_name} Ref",
            energy=energy,
            dose_rate=max_dose_rate,
            x1=min(roi_centers) - roi_size_mm - jaw_padding_mm,
            x2=max(roi_centers) + roi_size_mm + jaw_padding_mm,
            y1=y1,
            y2=y2,
            gantry_angles=gantry_angles[-1],
            coll_angle=coll_angle,
            couch_vrt=couch_vrt,
            couch_lat=couch_lat,
            couch_lng=couch_lng,
            couch_rot=couch_rot,
            fluence_mode=fluence_mode,
            mlc_positions=ref_mlc.as_control_points(),
            metersets=[mu * m for m in ref_mlc.as_metersets()],
            mlc_is_hd=machine.mlc_is_hd,
        )
        self.beams.append(ref_beam)



class VMATDRGS(QAProcedure):
    """Create beams like Clif Ling VMAT DRGS test."""

    # Prevent using a gantry angle of 180°, which can cause ambiguity in the rotation direction.
    MIN_GANTRY_OFFSET = 0.1

    # The reference beam may be acquired prior to or following the dynamic beam
    # (as specified by the reference_beam_add_before argument).
    # These attributes record the indices of the respective beams.
    dynamic_beam_idx: int
    reference_beam_idx: int

    def __init__(
        self,
        machine : TrueBeamMachine,
        dose_rates: tuple[float, ...] = (600, 600, 600, 600, 500, 400, 200),
        gantry_speeds: tuple[float, ...] = (3, 4, 5, 6, 6, 6, 6),
        mu_per_segment: float = 48.,
        mu_per_transition: float = 8.,
        correct_fluence: bool = True,
        gantry_motion_per_transition: float = 10.,
        gantry_rotation_clockwise: bool = True,
        initial_gantry_offset: float = 1.,
        mlc_span: float = 138.,
        mlc_motion_reverse: bool = True,
        mlc_gap: float = 2.,
        jaw_padding: float = 0.,
        energy: float = 6,
        fluence_mode: FluenceMode = FluenceMode.STANDARD,
        max_dose_rate: int = 600,
        reference_beam_mu: float = 100.,
        reference_beam_add_before: bool = False,
        dynamic_delivery_at_static_gantry: tuple[float, ...] = (),
    ):
        """Create beams like Clif Ling VMAT DRGS tests. The defaults use an optimized selection for a TrueBeam.

        Parameters
        ----------
        machine : TrueBeamMachine
            The target machine.
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
        super().__init__()
        # store parameters common to all beams
        self._machine = machine
        self._energy = energy
        self._fluence_mode = fluence_mode
        self._max_dose_rate = max_dose_rate
        mlc_boundaries = MLC_BOUNDARIES_TB_HD120 if machine.mlc_is_hd else MLC_BOUNDARIES_TB_MIL120
        self._y1 = mlc_boundaries[0]
        self._y2 = mlc_boundaries[-1]
        self._x1 = -(mlc_span / 2 + jaw_padding)
        self._x2 = mlc_span / 2 + jaw_padding

        # convert/cast variables
        gantry_speeds = np.array(gantry_speeds)
        dose_rates = np.array(dose_rates)
        mu_per_sec = dose_rates / 60

        # Verify inputs:
        if len(gantry_speeds) != len(dose_rates):
            raise ValueError("gantry_speeds and dose_rates must have the same length")
        if initial_gantry_offset < self.MIN_GANTRY_OFFSET:
            raise ValueError(f"The initial gantry offset cannot be smaller than {self.MIN_GANTRY_OFFSET} deg. Using 180 deg can cause ambiguity in the rotation direction.")

        gantry_speeds_normalized = gantry_speeds / machine.machine_specs.max_gantry_speed
        dose_rates_normalized = dose_rates / max_dose_rate
        # Verify that there are no requested speeds above limit
        if np.any(gantry_speeds_normalized > 1):
            raise ValueError("Requested gantry_speeds cannot exceed max_gantry_speed")
        if np.any(dose_rates_normalized > 1):
            raise ValueError("Requested dose_rates cannot exceed max_dose_rate")
        # Verify that at least one axis is maxed out for all control points
        if not np.all(
            np.max((gantry_speeds_normalized, dose_rates_normalized), axis=0) == 1
        ):
            raise ValueError("At least one axis must be maxed out")

        # calculate unmodulated variables
        num_segments = len(gantry_speeds)
        time_to_deliver_segments = mu_per_segment / mu_per_sec
        gantry_motion_per_segment = gantry_speeds * time_to_deliver_segments
        segment_width = (mlc_span + mlc_gap) / num_segments

        # This is the modulation computation
        # delivery motion scheme (T is transition, D is Dose, numbers are index of calculated values (1-based))
        # CP    0 1 2 3 4 5 6 7 8 9
        # G     0 T 1 T 2 T 3 T 4 T
        # D     0 * D T D T D T D *     , * On the 1st and last transition the dose needs to be scaled to the mlc motion to prevent overdosage
        # MLC   0 * 0 T 0 T 0 T 0 0     , * The first transition is smaller by mlc_gap

        gantry_motion = np.insert(
            gantry_motion_per_segment,
            range(num_segments + 1),
            gantry_motion_per_transition,
        )
        gantry_motion = np.append(0, gantry_motion)

        dose_motion = 1.0 * np.tile([mu_per_segment, mu_per_transition], num_segments)
        dose_motion = np.append([0, mu_per_transition], dose_motion)
        if correct_fluence:
            dose_motion[[1, -1]] = mu_per_transition * (1 - mlc_gap / segment_width)

        mlc_motion_ini = [0, segment_width - mlc_gap]
        mlc_motion_mid = np.tile([0, segment_width], num_segments - 1)
        mlc_motion_end = [0, 0]
        mlc_motion = np.concatenate((mlc_motion_ini, mlc_motion_mid, mlc_motion_end))

        # Extra verifications on the computed variables
        gantry_angles_without_offset = np.cumsum(gantry_motion)
        if gantry_angles_without_offset[-1] > 360 - 2 * self.MIN_GANTRY_OFFSET:
            msg = "The selected parameters require the gantry to rotate more than 360 degrees. Please select new parameters."
            raise ValueError(msg)
        gantry_angles_var = gantry_angles_without_offset + initial_gantry_offset
        if gantry_angles_var[-1] > 360 - self.MIN_GANTRY_OFFSET:
            msg = "The gantry rotation exceeds 360 degrees. Reduce the initial_gantry_offset"
            raise ValueError(msg)

        # Finalize values
        cumulative_mu = np.cumsum(dose_motion)
        mlc_positions = np.cumsum(mlc_motion) - mlc_span / 2
        if mlc_motion_reverse:
            mlc_positions = -mlc_positions
        mlc_positions_b = mlc_positions
        mlc_positions_a = -np.flip(mlc_positions)
        gantry_angles = (180 - gantry_angles_var) % 360
        if gantry_rotation_clockwise:
            gantry_angles = 360 - gantry_angles

        # Store final values
        self._gantry_angles = gantry_angles
        self._cumulative_mu = cumulative_mu
        self._mlc_positions_a = mlc_positions_a
        self._mlc_positions_b = mlc_positions_b

        # Create dynamic beam
        dynamic_beam = self._truebeam_beam(
            "VMAT-T2-Dyn",
            cumulative_mu,
            gantry_angles,
            mlc_positions_a,
            mlc_positions_b,
        )

        # Create reference beam
        reference_meterset = [0, reference_beam_mu]
        reference_gantry_angle = [float(
            gantry_angles[0 if reference_beam_add_before else -1]
        )]
        reference_mlc_positions_a = 2 * [float(mlc_positions_a[0])]
        reference_mlc_positions_b = 2 * [float(mlc_positions_b[-1])]
        reference_beam = self._truebeam_beam(
            "VMAT-T2-Ref",
            reference_meterset,
            reference_gantry_angle,
            reference_mlc_positions_a,
            reference_mlc_positions_b,
        )

        # Append the dynamic and reference beams according to the order defined in init
        beams: list[Beam | None] = 2 * [None]
        self.dynamic_beam_idx = 1 if reference_beam_add_before else 0
        self.reference_beam_idx = 0 if reference_beam_add_before else 1
        beams[self.dynamic_beam_idx] = dynamic_beam
        beams[self.reference_beam_idx] = reference_beam

        # Add static beams
        for gantry_angle in dynamic_delivery_at_static_gantry:
            beam = self._truebeam_beam(
                f"VMAT-T2-Sta-{gantry_angle:03d}",
                cumulative_mu,
                [gantry_angle],
                mlc_positions_a,
                mlc_positions_b,
            )
            beams.append(beam)

        self.beams = beams

    def _truebeam_beam(
        self, beam_name: str,
        metersets: Sequence[float],
        gantry_angles: Sequence[float],
        mlc_positions_a: Sequence[float],
        mlc_positions_b: Sequence[float]
    ) -> Beam:
        """Multiple similar beams are created for the VMAT test.
        Common parameters are stored as attributes, whereas the dynamic axes
        are passed as arguments to this method."""

        # Expand mlc positions for all leaves
        beam_mlc_position_a = np.tile(mlc_positions_a, (60, 1))
        beam_mlc_position_b = np.tile(mlc_positions_b, (60, 1))
        beam_mlc_positions = np.vstack((beam_mlc_position_b, beam_mlc_position_a))
        beam_mlc_positions = beam_mlc_positions.transpose().tolist()

        return Beam.for_truebeam(
            mlc_is_hd=self._machine.mlc_is_hd,
            beam_name=beam_name,
            energy=self._energy,
            fluence_mode=self._fluence_mode,
            dose_rate=self._max_dose_rate,
            metersets=metersets,
            gantry_angles=gantry_angles,
            x1=self._x1,
            x2=self._x2,
            y1=self._y1,
            y2=self._y2,
            mlc_positions=beam_mlc_positions,
            coll_angle=0,
            couch_vrt=0,
            couch_lat=0,
            couch_lng=0,
            couch_rot=0,
        )

    def plot_control_points(
        self,
        specs: MachineSpecs | None = None,
        max_dose_rate: float | None = None,
    ) -> None:
        """Plot the control points
        Rows: Absolute position, relative motion, time to deliver, speed
        Cols: MU, Gantry, MLC
        """
        # This is used mostly for visual inspection during development
        # Axis labeling could be improved

        max_dose_rate = (max_dose_rate or self._max_dose_rate) / 60
        specs = specs or self._machine.machine_specs

        cumulative_meterset, gantry_angles, mlc_positions = _get_control_points(
            self.beams[self.dynamic_beam_idx]
        )

        dose_motion = np.append(0, np.abs(np.diff(cumulative_meterset)))
        gantry_angle_var = (180 - gantry_angles) % 360
        gantry_motion = np.append(0, np.abs(np.diff(gantry_angle_var)))
        mlc_zeros = np.zeros((mlc_positions.shape[0], 1))
        mlc_motion = np.hstack((mlc_zeros, np.diff(mlc_positions, axis=1)))

        # ttd = time to deliver
        ttd_dose = dose_motion / max_dose_rate
        ttd_gantry = gantry_motion / specs.max_gantry_speed
        ttd_mlc = mlc_motion / specs.max_mlc_speed
        times_to_deliver = np.vstack((ttd_dose, ttd_gantry, ttd_mlc))
        time_to_deliver = np.max(np.abs(times_to_deliver), axis=0)

        dose_rate = dose_motion / time_to_deliver * 60
        gantry_speed = gantry_motion / time_to_deliver
        mlc_speed = mlc_motion / time_to_deliver

        num_rows, num_cols = 4, 3

        # Positions
        plot_delta = 0
        plt.subplot(num_rows, num_cols, 1 + plot_delta)
        plt.plot(cumulative_meterset, cumulative_meterset)
        plt.title("MU")
        plt.ylabel("Absolute")
        plt.subplot(num_rows, num_cols, 2 + plot_delta)
        plt.plot(cumulative_meterset, gantry_angles)
        plt.title("Gantry")
        plt.subplot(num_rows, num_cols, 3 + plot_delta)
        plt.plot(cumulative_meterset, mlc_positions[0, :])
        plt.plot(cumulative_meterset, mlc_positions[-1, :])
        plt.title("MLC")

        # Motions
        plot_delta += num_cols
        plt.subplot(num_rows, num_cols, 1 + plot_delta)
        plt.step(cumulative_meterset, dose_motion)
        plt.ylabel("Motion")
        plt.subplot(num_rows, num_cols, 2 + plot_delta)
        plt.step(cumulative_meterset, gantry_motion)
        plt.subplot(num_rows, num_cols, 3 + plot_delta)
        plt.step(cumulative_meterset, mlc_motion[0, :])
        plt.step(cumulative_meterset, mlc_motion[-1, :])

        # Time to deliver
        plot_delta += num_cols
        plt.subplot(num_rows, num_cols, 1 + plot_delta)
        plt.step(cumulative_meterset, time_to_deliver)
        plt.ylabel("Delivery time")
        plt.subplot(num_rows, num_cols, 2 + plot_delta)
        plt.step(cumulative_meterset, time_to_deliver)
        plt.subplot(num_rows, num_cols, 3 + plot_delta)
        plt.step(cumulative_meterset, time_to_deliver)

        # Speeds
        plot_delta += num_cols
        plt.subplot(num_rows, num_cols, 1 + plot_delta)
        plt.step(cumulative_meterset, dose_rate)
        plt.ylabel("Speed")
        plt.subplot(num_rows, num_cols, 2 + plot_delta)
        plt.step(cumulative_meterset, gantry_speed)
        plt.subplot(num_rows, num_cols, 3 + plot_delta)
        plt.step(cumulative_meterset, mlc_speed[0, :])
        plt.step(cumulative_meterset, mlc_speed[-1, :])

        plt.show()
        pass

    def plot_fluence(self, imager: Imager):
        beams = {"Reference": self.reference_beam_idx, "Dynamic": self.dynamic_beam_idx}
        for idx, (key, value) in enumerate(beams.items()):
            beam = self.beams[value]
            fluence = beam.generate_fluence(imager)
            plt.subplot(1,2,idx+1)
            plt.imshow(fluence)
            plt.title(key)
        plt.show()

def _get_control_points(beam: Beam) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """This is a helper function to get the control points from a beam."""
    # This is a quick implementation, it should be polished once there are more
    # procedures using this (e.g. method within beam).
    cps = beam.to_dicom().ControlPointSequence
    num_cp = len(cps)
    cumulative_meterset_weight = np.full(num_cp, np.nan)
    gantry_angles = np.full(num_cp, np.nan)
    mlc_positions = np.full((120, num_cp), np.nan)
    for idx_cp in range(num_cp):
        cp = cps[idx_cp]
        cumulative_meterset_weight[idx_cp] = cp.CumulativeMetersetWeight

        if "GantryAngle" in cp:
            gantry_angles[idx_cp] = cp.GantryAngle

        for bld in cp.BeamLimitingDevicePositionSequence:
            if bld.RTBeamLimitingDeviceType == "MLCX":
                mlc_positions[:, idx_cp] = bld.LeafJawPositions

    # if the axis is static all elements except the first will be nan, so replace with first value
    gantry_angles[np.isnan(gantry_angles)] = gantry_angles[0]
    cumulative_meterset = cumulative_meterset_weight * beam.beam_meterset
    return cumulative_meterset, gantry_angles, mlc_positions

