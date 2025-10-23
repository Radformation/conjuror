from enum import Enum

from conjuror.plans.mlc import MLCShaper
from conjuror.plans.plan_generator import QAProcedureBase, HalcyonMachine, Beam


class Stack(Enum):
    DISTAL = "distal"
    PROXIMAL = "proximal"
    BOTH = "both"

class QAProcedure(QAProcedureBase):

    @staticmethod
    def _create_mlc(machine: HalcyonMachine) -> tuple[MLCShaper, MLCShaper]:
        """Create 2 MLC shaper objects, one for each stack."""
        proximal_mlc = MLCShaper(
            leaf_y_positions=machine.mlc_boundaries_prox,
            max_mlc_position=machine.machine_specs.max_mlc_position,
            max_overtravel_mm=machine.machine_specs.max_mlc_overtravel,
            sacrifice_gap_mm=None,
            sacrifice_max_move_mm=None,
        )
        distal_mlc = MLCShaper(
            leaf_y_positions=machine.mlc_boundaries_dist,
            max_mlc_position=machine.machine_specs.max_mlc_position,
            max_overtravel_mm=machine.machine_specs.max_mlc_overtravel,
            sacrifice_gap_mm=None,
            sacrifice_max_move_mm=None,
        )
        return proximal_mlc, distal_mlc


class PicketFence(QAProcedure):
    def __init__(self,
        machine: HalcyonMachine,
        stack: Stack,
        strip_width_mm: float = 3,
        strip_positions_mm: tuple[float, ...] = (-45, -30, -15, 0, 15, 30, 45),
        gantry_angle: float = 0,
        coll_angle: float = 0,
        couch_vrt: float = 0,
        couch_lng: float = 1000,
        couch_lat: float = 0,
        mu: int = 200,
        beam_name: str = "PF",
    ):
        """Add a picket fence beam to the plan. The beam will be delivered with the MLCs stacked on top of each other.

        Parameters
        ----------
        machine : HalcyonMachine
            The target machine.
        stack: Stack
            Which MLC stack to use for the beam. The other stack will be parked.
        strip_width_mm : float
            The width of the strips in mm.
        strip_positions_mm : tuple
            The positions of the strips in mm.
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
        mu : int
            The monitor units of the beam.
        beam_name : str
            The name of the beam.
        """
        super().__init__()
        prox_mlc, dist_mlc = self._create_mlc(machine)

        # we prepend the positions with an initial starting position 2mm from the first strip
        # that way, each picket is the same cadence where the leaves move into position dynamically.
        # If you didn't do this, the first picket might be different as it has the advantage
        # of starting from a static position vs the rest of the pickets being dynamic.
        strip_positions = [strip_positions_mm[0] - 2, *strip_positions_mm]
        metersets = [0, *[1 / len(strip_positions_mm) for _ in strip_positions_mm]]

        for strip, meterset in zip(strip_positions, metersets):
            if stack in (Stack.DISTAL, Stack.BOTH):
                dist_mlc.add_strip(
                    position_mm=strip,
                    strip_width_mm=strip_width_mm,
                    meterset_at_target=meterset,
                )
                if stack == Stack.DISTAL:
                    prox_mlc.park(meterset=meterset)
            if stack in (Stack.PROXIMAL, Stack.BOTH):
                prox_mlc.add_strip(
                    position_mm=strip,
                    strip_width_mm=strip_width_mm,
                    meterset_at_target=meterset,
                )
                if stack == Stack.PROXIMAL:
                    dist_mlc.park(meterset=meterset)

        beam = Beam.for_halcyon(
            beam_name=beam_name,
            gantry_angles=gantry_angle,
            coll_angle=coll_angle,
            couch_vrt=couch_vrt,
            couch_lat=couch_lat,
            couch_lng=couch_lng,
            proximal_mlc_positions=prox_mlc.as_control_points(),
            distal_mlc_positions=dist_mlc.as_control_points(),
            # can use either MLC for metersets
            metersets=[mu * m for m in prox_mlc.as_metersets()],
        )
        self.beams.append(beam)


