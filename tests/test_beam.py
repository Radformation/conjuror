from unittest import TestCase

import pydicom
from matplotlib import pyplot as plt
from parameterized import parameterized

from conjuror.images.simulators import IMAGER_AS1200
from conjuror.plans.beam import Beam as BeamBase
from conjuror.plans.machine import FluenceMode
from conjuror.plans.truebeam import (
    Beam,
    VMATDRGS,
    DEFAULT_SPECS_TB,
    TrueBeamMachine,
    PicketFence,
)
from tests.utils import get_file_from_cloud_test_repo

TB_MIL_PLAN_FILE = get_file_from_cloud_test_repo(["plan_generator", "Murray-plan.dcm"])
DEFAULT_TRUEBEAM_HD120 = TrueBeamMachine(mlc_is_hd=True)


def create_beam(**kwargs) -> Beam:
    return Beam(
        beam_name=kwargs.get("beam_name", "name"),
        energy=kwargs.get("energy", 6),
        dose_rate=kwargs.get("dose_rate", 600),
        x1=kwargs.get("x1", -5),
        x2=kwargs.get("x2", 5),
        y1=kwargs.get("y1", -5),
        y2=kwargs.get("y2", 5),
        gantry_angles=kwargs.get("gantry_angles", 0),
        coll_angle=kwargs.get("coll_angle", 0),
        couch_vrt=kwargs.get("couch_vrt", 0),
        couch_lng=kwargs.get("couch_lng", 0),
        couch_lat=kwargs.get("couch_lat", 0),
        couch_rot=kwargs.get("couch_rot", 0),
        mlc_is_hd=kwargs.get("mlc_is_hd", False),
        mlc_positions=kwargs.get("mlc_positions", [120 * [0], 120 * [0]]),
        metersets=kwargs.get("metersets", [0, 100]),
        fluence_mode=kwargs.get("fluence_mode", FluenceMode.STANDARD),
    )


class TestBeam(TestCase):
    def test_beam_normal(self):
        # shouldn't raise; happy path
        beam = create_beam(gantry_angles=0)
        beam_dcm = beam.to_dicom()
        self.assertEqual(beam_dcm.BeamName, "name")
        self.assertEqual(beam_dcm.BeamType, "STATIC")
        self.assertEqual(beam_dcm.ControlPointSequence[0].GantryAngle, 0)

    def test_from_dicom(self):
        ds = pydicom.dcmread(TB_MIL_PLAN_FILE)
        beam = BeamBase.from_dicom(ds, 0)
        self.assertEqual(2, beam.number_of_control_points)

    def test_from_dicom_without_primary_fluence_mode_sequence(self):
        # E.g. the picket fence RT plan does not have PrimaryFluenceModeSequence
        ds = pydicom.dcmread(TB_MIL_PLAN_FILE)
        ds.BeamSequence[0].pop("PrimaryFluenceModeSequence")
        beam = BeamBase.from_dicom(ds, 0)
        self.assertEqual(2, beam.number_of_control_points)

    def test_from_dicom_error_if_not_rt_plan(self):
        file = get_file_from_cloud_test_repo(["picket_fence", "AS500#2.dcm"])
        ds = pydicom.dcmread(file)
        with self.assertRaises(ValueError):
            Beam.from_dicom(ds, 0)

    def test_from_dicom_error_if_beam_not_in_plan(self):
        ds = pydicom.dcmread(TB_MIL_PLAN_FILE)
        with self.assertRaises(ValueError):
            Beam.from_dicom(ds, 1)

    def test_too_long_beam_name(self):
        with self.assertRaises(ValueError):
            create_beam(beam_name="superlongbeamname")

    def test_1_mlc_position_for_static(self):
        beam = create_beam(mlc_positions=[[0]], metersets=[0])
        self.assertEqual(beam.to_dicom().BeamType, "STATIC")

    @parameterized.expand(
        [
            ([0, 90], "CW", "DYNAMIC"),
            ([90, 0], "CC", "DYNAMIC"),
            ([270, 90], "CW", "DYNAMIC"),
            ([90, 270], "CC", "DYNAMIC"),
            ([170, -170], "CC", "DYNAMIC"),
            ([-170, 170], "CW", "DYNAMIC"),
            ([0, 0], "NONE", "STATIC"),
        ]
    )
    def test_beam_type_and_gantry_rotation_direction(
        self, gantry_angles, rotation_direction, beam_type
    ):
        beam = create_beam(
            gantry_angles=gantry_angles,
        )
        beam_dcm = beam.to_dicom()
        self.assertEqual(beam_dcm.BeamType, beam_type)
        cp_sequence = beam_dcm.ControlPointSequence
        self.assertEqual(cp_sequence[0].GantryRotationDirection, rotation_direction)
        if beam_type == "DYNAMIC":
            self.assertEqual(cp_sequence[1].GantryRotationDirection, "NONE")
        else:
            self.assertNotIn("GantryRotationDirection", cp_sequence[1])

    def test_jaw_positions(self):
        b = create_beam(x1=-5, x2=7, y1=-11, y2=13)
        dcm = b.to_dicom()
        self.assertEqual(
            len(dcm.ControlPointSequence[0].BeamLimitingDevicePositionSequence), 3
        )
        self.assertEqual(
            dcm.ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[0]
            .LeafJawPositions,
            [-5, 7],
        )
        self.assertEqual(
            dcm.ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[1]
            .LeafJawPositions,
            [-11, 13],
        )

    def test_compute_dynamics(self):
        attr = "gantry_speeds"
        procedure = VMATDRGS()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        beam = procedure.dynamic_beam
        self.assertFalse(hasattr(beam, attr))
        beam.compute_dynamics(DEFAULT_SPECS_TB)
        self.assertTrue(hasattr(beam, attr))

    def test_plot_fluence(self):
        # just tests it works
        machine = TrueBeamMachine(mlc_is_hd=True)
        procedure = PicketFence()
        procedure.compute(machine)
        beam = procedure.beams[0]

        # new figure
        beam.plot_fluence(IMAGER_AS1200)

        # existing figure
        fig, (ax1, ax2) = plt.subplots(1, 2)
        beam.plot_fluence(IMAGER_AS1200, ax1)
        beam.plot_fluence(IMAGER_AS1200, ax2)
        plt.show()

    def test_animate_mlc(self):
        procedure = PicketFence()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        beam = procedure.beams[0]
        fig = beam.animate_mlc()
        self.assertEqual(
            120, sum(True for f in fig.data if f["line"]["color"] == "blue")
        )

    def test_plot_control_points(self):
        procedure = VMATDRGS()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        beam = procedure.dynamic_beam
        fig = beam.plot_control_points(DEFAULT_SPECS_TB)
        self.assertEqual(12, len(fig.axes))
