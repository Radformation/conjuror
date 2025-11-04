import tempfile
from unittest import TestCase

import numpy as np
import pydicom
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from parameterized import parameterized

from conjuror.images.simulators import IMAGER_AS1200
from conjuror.plans import plan_generator_halcyon, plan_generator_truebeam
from conjuror.plans.mlc import (
    interpolate_control_points,
    next_sacrifice_shift,
    split_sacrifice_travel,
    MLCShaper,
)
from conjuror.plans.plan_generator_base import BeamBase, FluenceMode, OvertravelError
from conjuror.plans.plan_generator_halcyon import HalcyonPlanGenerator, Stack
from conjuror.plans.plan_generator_truebeam import (
    TrueBeamMachine,
    TrueBeamPlanGenerator,
    OpenField,
    Beam,
    DoseRate,
    WinstonLutz,
    MLCSpeed,
    GantrySpeed,
    VMATDRGS,
    DEFAULT_SPECS_TB,
    MLCTransmission,
)
from tests.utils import get_file_from_cloud_test_repo

RT_PLAN_FILE = get_file_from_cloud_test_repo(["plan_generator", "Murray-plan.dcm"])
RT_PLAN_DS = pydicom.dcmread(RT_PLAN_FILE)
HALCYON_PLAN_FILE = get_file_from_cloud_test_repo(
    ["plan_generator", "Halcyon Prox.dcm"]
)

DEFAULT_TRUEBEAM_HD120 = TrueBeamMachine(mlc_is_hd=True)


class TestPlanGenerator(TestCase):
    def test_from_rt_plan_file(self):
        # shouldn't raise; happy path
        TrueBeamPlanGenerator.from_rt_plan_file(
            RT_PLAN_FILE, plan_label="label", plan_name="my name"
        )

    def test_from_not_rt_plan_file(self):
        file = get_file_from_cloud_test_repo(["picket_fence", "AS500#2.dcm"])
        with self.assertRaises(ValueError):
            TrueBeamPlanGenerator.from_rt_plan_file(
                file, plan_label="label", plan_name="my name"
            )

    def test_to_file(self):
        pg = TrueBeamPlanGenerator.from_rt_plan_file(
            RT_PLAN_FILE, plan_label="label", plan_name="my name"
        )
        pg.add_procedure(MLCSpeed())
        with tempfile.NamedTemporaryFile(delete=False) as t:
            pg.to_file(t.name)
        # shouldn't raise; should be valid DICOM
        ds = pydicom.dcmread(t.name)
        self.assertEqual(ds.RTPlanLabel, "label")
        self.assertEqual(len(ds.BeamSequence), 2)

    def test_from_rt_plan_dataset(self):
        # happy path using a dicom dataset
        dataset = pydicom.dcmread(RT_PLAN_FILE)
        TrueBeamPlanGenerator(dataset, plan_label="label", plan_name="my name")

    def test_no_patient_id(self):
        ds = pydicom.dcmread(RT_PLAN_FILE)
        ds.pop("PatientID")
        with self.assertRaises(ValueError):
            TrueBeamPlanGenerator(ds, plan_label="label", plan_name="my name")

    def test_no_patient_name(self):
        ds = pydicom.dcmread(RT_PLAN_FILE)
        ds.pop("PatientName")
        with self.assertRaises(ValueError):
            TrueBeamPlanGenerator(ds, plan_label="label", plan_name="my name")

    def test_pass_patient_name(self):
        ds = pydicom.dcmread(RT_PLAN_FILE)
        pg = TrueBeamPlanGenerator(
            ds, plan_label="label", plan_name="my name", patient_name="Jimbo Jones"
        )
        pg_dcm = pg.as_dicom()
        self.assertEqual(pg_dcm.PatientName, "Jimbo Jones")

    def test_pass_patient_id(self):
        ds = pydicom.dcmread(RT_PLAN_FILE)
        pg = TrueBeamPlanGenerator(
            ds, plan_label="label", plan_name="my name", patient_id="12345"
        )
        pg_dcm = pg.as_dicom()
        self.assertEqual(pg_dcm.PatientID, "12345")

    def test_no_tolerance_table(self):
        ds = pydicom.dcmread(RT_PLAN_FILE)
        ds.pop("ToleranceTableSequence")
        with self.assertRaises(ValueError):
            TrueBeamPlanGenerator(ds, plan_label="label", plan_name="my name")

    def test_no_beam_sequence(self):
        ds = pydicom.dcmread(RT_PLAN_FILE)
        ds.pop("BeamSequence")
        with self.assertRaises(ValueError):
            TrueBeamPlanGenerator(ds, plan_label="label", plan_name="my name")

    def test_no_mlc_data(self):
        ds = pydicom.dcmread(RT_PLAN_FILE)
        # pop MLC part of the data; at this point it's just an open field
        ds.BeamSequence[0].BeamLimitingDeviceSequence.pop()
        with self.assertRaises(ValueError):
            TrueBeamPlanGenerator(ds, plan_label="label", plan_name="my name")

    def test_machine_name(self):
        pg = TrueBeamPlanGenerator.from_rt_plan_file(
            RT_PLAN_FILE, plan_label="label", plan_name="my name"
        )
        self.assertEqual(pg.machine_name, "TrueBeamSN5837")

    def test_machine_name_set_on_beam(self):
        """Beam machine name is set when added to the plan"""
        pg = TrueBeamPlanGenerator.from_rt_plan_file(
            RT_PLAN_FILE, plan_label="label", plan_name="my name"
        )
        pg.add_beam(create_beam())
        dcm = pg.as_dicom()
        self.assertEqual(dcm.BeamSequence[0].TreatmentMachineName, "TrueBeamSN5837")

    def test_leaf_boundaries(self):
        pg = TrueBeamPlanGenerator.from_rt_plan_file(
            RT_PLAN_FILE, plan_label="label", plan_name="my name"
        )
        self.assertEqual(len(pg.machine.mlc_boundaries), 61)
        self.assertEqual(max(pg.machine.mlc_boundaries), 200)
        self.assertEqual(min(pg.machine.mlc_boundaries), -200)

    def test_instance_uid_changes(self):
        dcm = pydicom.dcmread(RT_PLAN_FILE)
        pg = TrueBeamPlanGenerator.from_rt_plan_file(
            RT_PLAN_FILE, plan_label="label", plan_name="my name"
        )
        pg_dcm = pg.as_dicom()
        self.assertNotEqual(pg_dcm.SOPInstanceUID, dcm.SOPInstanceUID)

    def test_invert_array(self):
        pg = TrueBeamPlanGenerator.from_rt_plan_file(
            RT_PLAN_FILE, plan_label="label", plan_name="my name"
        )
        procedure = OpenField(x1=100, x2=200, y1=100, y2=200, mu=100)
        pg.add_procedure(procedure)
        # test that non-inverted array is 0
        pg_dcm = pg.to_dicom_images(imager=IMAGER_AS1200, invert=False)
        non_inverted_array = pg_dcm[0].pixel_array
        # when inverted, the corner should NOT be 0
        self.assertAlmostEqual(float(non_inverted_array[0, 0]), 0)

        pg_dcm = pg.to_dicom_images(imager=IMAGER_AS1200, invert=True)
        inverted_array = pg_dcm[0].pixel_array
        # when inverted, the corner should NOT be 0
        self.assertAlmostEqual(float(inverted_array[0, 0]), 1000)

    def test_incorrect_machine_type(self):
        plan_file = RT_PLAN_FILE
        with self.assertRaises(ValueError):
            HalcyonPlanGenerator.from_rt_plan_file(
                plan_file, plan_label="label", plan_name="my name"
            )

        plan_file = HALCYON_PLAN_FILE
        with self.assertRaises(ValueError):
            TrueBeamPlanGenerator.from_rt_plan_file(
                plan_file, plan_label="label", plan_name="my name"
            )


def create_beam(**kwargs) -> BeamBase:
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
        mlc_positions=kwargs.get("mlc_positions", [[0], [0]]),
        metersets=kwargs.get("metersets", [0, 100]),
        fluence_mode=kwargs.get("fluence_mode", FluenceMode.STANDARD),
    )


class TestBeam(TestCase):
    def test_beam_normal(self):
        # shouldn't raise; happy path
        beam = create_beam(
            gantry_angles=0,
        )
        beam_dcm = beam.to_dicom()
        self.assertEqual(beam_dcm.BeamName, "name")
        self.assertEqual(beam_dcm.BeamType, "STATIC")
        self.assertEqual(beam_dcm.ControlPointSequence[0].GantryAngle, 0)

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

    def test_plot_fluence(self):
        # just tests it works
        machine = TrueBeamMachine(mlc_is_hd=True)
        procedure = OpenField.from_machine(machine, x1=-5, x2=7, y1=-11, y2=13)
        beam = procedure.beams[0]

        # new figure
        beam.plot_fluence(IMAGER_AS1200)

        # existing figure
        fig, (ax1, ax2) = plt.subplots(1, 2)
        beam.plot_fluence(IMAGER_AS1200, ax1)
        beam.plot_fluence(IMAGER_AS1200, ax2)
        plt.show()


class TestPlanGeneratorBeams(TestCase):
    """Test real workflow where beams are added"""

    def setUp(self) -> None:
        self.pg = TrueBeamPlanGenerator.from_rt_plan_file(
            RT_PLAN_FILE,
            plan_label="label",
            plan_name="my name",
        )

    def test_add_beam_low_level(self):
        self.pg.add_beam(create_beam(plan_dataset=self.pg.as_dicom()))
        dcm = self.pg.as_dicom()
        self.assertEqual(len(dcm.BeamSequence), 1)
        self.assertEqual(dcm.BeamSequence[0].BeamName, "name")
        self.assertEqual(dcm.BeamSequence[0].BeamNumber, 1)
        self.assertEqual(dcm.FractionGroupSequence[0].NumberOfBeams, 1)
        self.assertEqual(
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset, 100
        )
        self.assertEqual(
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedBeamNumber,
            1,
        )
        nominal_boundaries = (
            RT_PLAN_DS.BeamSequence[0]
            .BeamLimitingDeviceSequence[-1]
            .LeafPositionBoundaries
        )
        actual_boundaries = (
            dcm.BeamSequence[0].BeamLimitingDeviceSequence[-1].LeafPositionBoundaries
        )
        self.assertEqual(nominal_boundaries, actual_boundaries)

    def test_add_2_beams(self):
        self.pg.add_beam(create_beam())
        self.pg.add_beam(create_beam(beam_name="beam2"))
        dcm = self.pg.as_dicom()
        self.assertEqual(len(dcm.BeamSequence), 2)
        self.assertEqual(dcm.FractionGroupSequence[0].NumberOfBeams, 2)
        self.assertEqual(
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset, 100
        )
        self.assertEqual(
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[1].BeamMeterset, 100
        )
        self.assertEqual(dcm.BeamSequence[1].BeamName, "beam2")
        self.assertEqual(dcm.BeamSequence[1].BeamNumber, 2)

    def test_plot_fluence(self):
        # just tests it works
        procedure = OpenField(x1=-5, x2=5, y1=-5, y2=5, mu=100)
        self.pg.add_procedure(procedure)
        figs = self.pg.plot_fluences()
        self.assertIsInstance(figs, list)
        self.assertIsInstance(figs[0], Figure)

    def test_list_procedure(self):
        procedures = self.pg.list_procedures()
        self.assertEqual(len(procedures), 8)

        procedures = TrueBeamPlanGenerator.list_procedures()
        self.assertEqual(len(procedures), 8)


class TestPlanPrefabs(TestCase):
    def setUp(self) -> None:
        self.pg = TrueBeamPlanGenerator.from_rt_plan_file(
            RT_PLAN_FILE,
            plan_label="label",
            plan_name="my name",
        )

    def test_create_open_field(self):
        procedure = OpenField(
            x1=-100,
            x2=100,
            y1=-110,
            y2=110,
            mu=123,
            beam_name="Open Field",
            defined_by_mlcs=True,
            padding_mm=0,
        )
        self.pg.add_procedure(procedure)
        dcm = self.pg.as_dicom()
        self.assertEqual(len(dcm.BeamSequence), 1)
        self.assertEqual(dcm.BeamSequence[0].BeamName, "Open Field")
        self.assertEqual(dcm.BeamSequence[0].BeamNumber, 1)
        self.assertEqual(dcm.FractionGroupSequence[0].NumberOfBeams, 1)
        self.assertEqual(
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset, 123
        )
        self.assertEqual(
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[0]
            .LeafJawPositions,
            [-100, 100],
        )
        self.assertEqual(
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[1]
            .LeafJawPositions,
            [-110, 110],
        )
        self.assertEqual(dcm.BeamSequence[0].BeamType, "STATIC")

    def test_open_field_jaws(self):
        procedure = OpenField(
            x1=-100,
            x2=100,
            y1=-110,
            y2=110,
            mu=123,
            beam_name="Open Field",
            defined_by_mlcs=False,
            padding_mm=0,
        )
        self.pg.add_procedure(procedure)
        dcm = self.pg.as_dicom()
        self.assertEqual(len(dcm.BeamSequence), 1)
        self.assertEqual(
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[0]
            .LeafJawPositions,
            [-100, 100],
        )
        self.assertEqual(
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[1]
            .LeafJawPositions,
            [-110, 110],
        )
        self.assertEqual(dcm.BeamSequence[0].BeamType, "STATIC")

    @parameterized.expand(
        [
            ("valid A", "A", 39.5, -30, None),
            ("valid B", "B", -40.5, -30, None),
            ("Invalid Bank", "C", None, None, ValueError),
            ("Overtravel", "A", None, -150, OvertravelError),
        ]
    )
    def test_transmission_beam(self, _, bank, leaf_pos, x1_pos, expected_error):
        if expected_error:
            with self.assertRaises(expected_error):
                procedure = MLCTransmission(
                    bank=bank,
                    x1=x1_pos,
                    x2=30,
                    y1=-110,
                    y2=110,
                    mu=44,
                    beam_name="MLC Txx",
                )
                self.pg.add_procedure(procedure)
        else:
            procedure = MLCTransmission(
                bank=bank,
                x1=-30,
                x2=30,
                y1=-110,
                y2=110,
                mu=44,
                beam_name="MLC Txx",
            )
            self.pg.add_procedure(procedure)
            dcm = self.pg.as_dicom()
            self.assertEqual(len(dcm.BeamSequence), 1)
            self.assertEqual(dcm.BeamSequence[0].BeamName, f"MLC Txx {bank}")
            self.assertEqual(dcm.BeamSequence[0].BeamNumber, 1)
            self.assertEqual(dcm.FractionGroupSequence[0].NumberOfBeams, 1)
            self.assertEqual(
                dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset, 44
            )
            # check X jaws
            self.assertEqual(
                dcm.BeamSequence[0]
                .ControlPointSequence[0]
                .BeamLimitingDevicePositionSequence[0]
                .LeafJawPositions,
                [-30, 30],
            )
            # check Y jaws
            self.assertEqual(
                dcm.BeamSequence[0]
                .ControlPointSequence[0]
                .BeamLimitingDevicePositionSequence[1]
                .LeafJawPositions,
                [-110, 110],
            )
            # check first MLC position is tucked under the jaws
            self.assertEqual(
                dcm.BeamSequence[0]
                .ControlPointSequence[0]
                .BeamLimitingDevicePositionSequence[-1]
                .LeafJawPositions[0],
                leaf_pos,
            )
            self.assertEqual(dcm.BeamSequence[0].BeamType, "STATIC")

    def test_create_picket_fence(self):
        procedure = plan_generator_truebeam.PicketFence(
            y1=-10,
            y2=10,
            mu=123,
            beam_name="Picket Fence",
            strip_positions_mm=(-50, -30, -10, 10, 30, 50),
        )
        self.pg.add_procedure(procedure)
        dcm = self.pg.as_dicom()
        self.assertEqual(len(dcm.BeamSequence), 1)
        self.assertEqual(dcm.BeamSequence[0].BeamName, "Picket Fence")
        self.assertEqual(dcm.BeamSequence[0].BeamNumber, 1)
        self.assertEqual(dcm.FractionGroupSequence[0].NumberOfBeams, 1)
        self.assertEqual(
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset, 123
        )
        # check X jaws
        self.assertEqual(
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[0]
            .LeafJawPositions,
            [-60, 60],
        )
        # check Y jaws
        self.assertEqual(
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[1]
            .LeafJawPositions,
            [-10, 10],
        )
        # check first MLC position is near the first strip
        self.assertEqual(
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[-1]
            .LeafJawPositions[0],
            -53.5,
        )

    def test_picket_fence_too_wide(self):
        with self.assertRaises(ValueError):
            procedure = plan_generator_truebeam.PicketFence(
                y1=-10,
                y2=10,
                mu=123,
                beam_name="Picket Fence",
                strip_positions_mm=(-100, 100),
            )
            self.pg.add_procedure(procedure)

    def test_winston_lutz_beams(self):
        procedure = WinstonLutz(
            axes_positions=(
                {"gantry": 0, "collimator": 0, "couch": 0},
                {"gantry": 90, "collimator": 0, "couch": 0},
                {"gantry": 180, "collimator": 0, "couch": 45},
            ),
            x1=-10,
            x2=10,
            y1=-10,
            y2=10,
            mu=123,
        )
        self.pg.add_procedure(procedure)
        dcm = self.pg.as_dicom()
        self.assertEqual(len(dcm.BeamSequence), 3)
        self.assertEqual(dcm.BeamSequence[0].BeamName, "G0C0P0")
        self.assertEqual(dcm.BeamSequence[2].BeamName, "G180C0P45")
        self.assertEqual(dcm.BeamSequence[0].BeamNumber, 1)
        self.assertEqual(dcm.BeamSequence[1].BeamNumber, 2)
        self.assertEqual(dcm.BeamSequence[2].BeamNumber, 3)
        self.assertEqual(dcm.FractionGroupSequence[0].NumberOfBeams, 3)
        self.assertEqual(
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset, 123
        )
        self.assertEqual(dcm.BeamSequence[0].ControlPointSequence[0].GantryAngle, 0)
        self.assertEqual(dcm.BeamSequence[1].ControlPointSequence[0].GantryAngle, 90)
        self.assertEqual(dcm.BeamSequence[2].ControlPointSequence[0].GantryAngle, 180)

    def test_winston_lutz_jaw_defined(self):
        procedure = WinstonLutz(
            axes_positions=(
                {"gantry": 0, "collimator": 0, "couch": 0},
                {"gantry": 90, "collimator": 0, "couch": 0},
                {"gantry": 180, "collimator": 0, "couch": 45},
            ),
            x1=-10,
            x2=10,
            y1=-10,
            y2=10,
            mu=123,
            defined_by_mlcs=False,
        )
        self.pg.add_procedure(procedure)
        dcm = self.pg.as_dicom()
        self.assertEqual(len(dcm.BeamSequence), 3)
        self.assertEqual(
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[0]
            .LeafJawPositions,
            [-10, 10],
        )
        self.assertEqual(
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[1]
            .LeafJawPositions,
            [-10, 10],
        )

    def test_dose_rate_beams(self):
        procedure = DoseRate(
            dose_rates=(100, 400, 600),
            y1=-10,
            y2=10,
            desired_mu=123,
            default_dose_rate=600,
        )
        self.pg.add_procedure(procedure)
        dcm = self.pg.as_dicom()
        self.assertEqual(len(dcm.BeamSequence), 2)
        self.assertEqual(dcm.BeamSequence[0].BeamName, "DR Ref")
        self.assertEqual(dcm.BeamSequence[1].BeamName, "DR100-600")
        self.assertEqual(dcm.BeamSequence[0].BeamNumber, 1)
        self.assertEqual(dcm.FractionGroupSequence[0].NumberOfBeams, 2)
        self.assertEqual(
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset, 123
        )

    def test_dose_rate_too_wide(self):
        with self.assertRaises(ValueError):
            procedure = DoseRate(
                dose_rates=(100, 150, 200, 250, 300, 350, 400, 600),
                roi_size_mm=30,
                y1=-10,
                y2=10,
                desired_mu=123,
                default_dose_rate=600,
            )
            self.pg.add_procedure(procedure)

    def test_mlc_speed_beams(self):
        procedure = MLCSpeed(
            speeds=(0.5, 1, 1.5, 2),
            y1=-100,
            y2=100,
            mu=123,
        )
        self.pg.add_procedure(procedure)
        dcm = self.pg.as_dicom()
        self.assertEqual(len(dcm.BeamSequence), 2)
        self.assertEqual(dcm.BeamSequence[0].BeamName, "MLC Speed Ref")
        self.assertEqual(dcm.BeamSequence[1].BeamName, "MLC Speed")
        self.assertEqual(dcm.BeamSequence[0].BeamNumber, 1)
        self.assertEqual(dcm.FractionGroupSequence[0].NumberOfBeams, 2)
        self.assertEqual(
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset, 123
        )
        self.assertEqual(dcm.BeamSequence[0].BeamType, "DYNAMIC")
        self.assertEqual(dcm.BeamSequence[1].BeamType, "DYNAMIC")

    def test_mlc_speed_too_fast(self):
        with self.assertRaises(ValueError):
            procedure = MLCSpeed(
                speeds=(10, 20, 30, 40, 50),
                y1=-100,
                y2=100,
            )
            self.pg.add_procedure(procedure)

    def test_mlc_speed_too_wide(self):
        with self.assertRaises(ValueError):
            procedure = MLCSpeed(
                speeds=(0.5, 1, 1.5, 2),
                roi_size_mm=50,
                y1=-100,
                y2=100,
            )
            self.pg.add_procedure(procedure)

    def test_0_mlc_speed(self):
        with self.assertRaises(ValueError):
            procedure = MLCSpeed(
                speeds=(0, 1, 2),
                y1=-100,
                y2=100,
            )
            self.pg.add_procedure(procedure)

    def test_gantry_speed_beams(self):
        # max speed is 2.5 by default
        procedure = GantrySpeed(
            speeds=(1, 2, 3, 4),
            y1=-100,
            y2=100,
            mu=123,
        )
        self.pg.add_procedure(procedure)
        dcm = self.pg.as_dicom()
        self.assertEqual(len(dcm.BeamSequence), 2)
        self.assertEqual(dcm.BeamSequence[0].BeamName, "GS")
        self.assertEqual(dcm.BeamSequence[1].BeamName, "GS Ref")
        self.assertEqual(dcm.BeamSequence[0].BeamNumber, 1)
        self.assertEqual(dcm.FractionGroupSequence[0].NumberOfBeams, 2)
        self.assertEqual(
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset, 123
        )

    def test_gantry_speed_too_fast(self):
        # max speed is 4.8 by default
        with self.assertRaises(ValueError):
            procedure = GantrySpeed(
                speeds=(1, 2, 3, 4, 5),
                y1=-100,
                y2=100,
            )
            self.pg.add_procedure(procedure)

    def test_gantry_speed_too_wide(self):
        with self.assertRaises(ValueError):
            procedure = GantrySpeed(
                speeds=(1, 2, 3, 4),
                roi_size_mm=50,
                y1=-100,
                y2=100,
            )
            self.pg.add_procedure(procedure)

    def test_gantry_range_over_360(self):
        with self.assertRaises(ValueError):
            procedure = GantrySpeed(
                speeds=(4, 4, 4, 4),
                y1=-100,
                y2=100,
                mu=250,
            )
            self.pg.add_procedure(procedure)

    def test_vmat_drgs(self):
        procedure = VMATDRGS()
        self.pg.add_procedure(procedure)
        dcm = self.pg.as_dicom()
        self.assertEqual(len(dcm.BeamSequence), 2)


HALCYON_MLC_INDEX = {
    Stack.DISTAL: -2,
    Stack.PROXIMAL: -1,
}


class TestHalcyonPrefabs(TestCase):
    def setUp(self) -> None:
        self.pg = HalcyonPlanGenerator.from_rt_plan_file(
            HALCYON_PLAN_FILE,
            plan_label="label",
            plan_name="my name",
        )

    def test_create_picket_fence_proximal(self):
        procedure = plan_generator_halcyon.PicketFence(
            stack=Stack.PROXIMAL,
            mu=123,
            beam_name="Picket Fence",
            strip_positions_mm=(-50, -30, -10, 10, 30, 50),
        )
        self.pg.add_procedure(procedure)
        dcm = self.pg.as_dicom()
        self.assertEqual(len(dcm.BeamSequence), 1)
        self.assertEqual(dcm.BeamSequence[0].BeamName, "Picket Fence")
        self.assertEqual(dcm.BeamSequence[0].BeamNumber, 1)
        self.assertEqual(dcm.FractionGroupSequence[0].NumberOfBeams, 1)
        self.assertEqual(
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset, 123
        )
        # check first CP of proximal is at the PF position
        self.assertEqual(
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[HALCYON_MLC_INDEX[Stack.PROXIMAL]]
            .LeafJawPositions[0],
            -53.5,
        )
        # distal should be parked
        self.assertEqual(
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[HALCYON_MLC_INDEX[Stack.DISTAL]]
            .LeafJawPositions[0],
            -140,
        )
        self.assertEqual(dcm.BeamSequence[0].BeamType, "DYNAMIC")

    def test_create_picket_fence_distal(self):
        procedure = plan_generator_halcyon.PicketFence(
            stack=Stack.DISTAL,
            mu=123,
            beam_name="Picket Fence",
            strip_positions_mm=(-50, -30, -10, 10, 30, 50),
        )
        self.pg.add_procedure(procedure)
        dcm = self.pg.as_dicom()
        self.assertEqual(len(dcm.BeamSequence), 1)
        self.assertEqual(dcm.BeamSequence[0].BeamName, "Picket Fence")
        self.assertEqual(dcm.BeamSequence[0].BeamNumber, 1)
        self.assertEqual(dcm.FractionGroupSequence[0].NumberOfBeams, 1)
        self.assertEqual(
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset, 123
        )
        # check first CP of proximal is parked
        self.assertEqual(
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[HALCYON_MLC_INDEX[Stack.PROXIMAL]]
            .LeafJawPositions[0],
            -140,
        )
        # distal should be at picket position
        self.assertEqual(
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[HALCYON_MLC_INDEX[Stack.DISTAL]]
            .LeafJawPositions[0],
            -53.5,
        )
        self.assertEqual(dcm.BeamSequence[0].BeamType, "DYNAMIC")

    def test_create_picket_fence_both(self):
        procedure = plan_generator_halcyon.PicketFence(
            stack=Stack.BOTH,
            mu=123,
            beam_name="Picket Fence",
            strip_positions_mm=(-50, -30, -10, 10, 30, 50),
        )
        self.pg.add_procedure(procedure)
        dcm = self.pg.as_dicom()
        self.assertEqual(len(dcm.BeamSequence), 1)
        self.assertEqual(dcm.BeamSequence[0].BeamName, "Picket Fence")
        self.assertEqual(dcm.BeamSequence[0].BeamNumber, 1)
        self.assertEqual(dcm.FractionGroupSequence[0].NumberOfBeams, 1)
        self.assertEqual(
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset, 123
        )
        # check first CP of proximal is at the PF position
        self.assertEqual(
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[HALCYON_MLC_INDEX[Stack.PROXIMAL]]
            .LeafJawPositions[0],
            -53.5,
        )
        # distal should be at picket position
        self.assertEqual(
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[HALCYON_MLC_INDEX[Stack.DISTAL]]
            .LeafJawPositions[0],
            -53.5,
        )
        self.assertEqual(dcm.BeamSequence[0].BeamType, "DYNAMIC")


class TestMLCShaper(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # simplistic MLC setup
        cls.leaf_boundaries: tuple[float, ...] = tuple(
            np.arange(start=-200, stop=201, step=5).astype(float)
        )

    def test_init(self):
        MLCShaper(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )

    def test_num_leaves(self):
        shaper = MLCShaper(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        self.assertEqual(shaper.num_leaves, 160)

    def test_meterset_over_1(self):
        shaper = MLCShaper(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        with self.assertRaises(ValueError):
            shaper.add_strip(position_mm=-5, strip_width_mm=0, meterset_at_target=2)

    def test_sacrifice_without_transition_dose(self):
        shaper = MLCShaper(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=400,
            max_overtravel_mm=140,
        )
        with self.assertRaises(ValueError):
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=1,
                meterset_transition=0,
                sacrificial_distance_mm=50,
            )

    def test_initial_sacrificial_gap(self):
        shaper = MLCShaper(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        shaper.add_strip(
            position_mm=-5,
            strip_width_mm=0,
            meterset_at_target=1,
            initial_sacrificial_gap_mm=10,
        )
        self.assertEqual(shaper.control_points[0][0], -10)

    def test_cant_add_sacrificial_gap_after_first_point(self):
        shaper = MLCShaper(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        shaper.add_strip(
            position_mm=-5,
            strip_width_mm=0,
            meterset_at_target=0.2,
            initial_sacrificial_gap_mm=5,
        )
        with self.assertRaises(ValueError) as context:
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=0.2,
                initial_sacrificial_gap_mm=10,
            )
        self.assertIn("already control points", str(context.exception))

    def test_cant_have_initial_sacrifice_and_transition_dose(self):
        shaper = MLCShaper(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        with self.assertRaises(ValueError):
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=0,
                meterset_transition=1,
                initial_sacrificial_gap_mm=5,
            )

    def test_cant_have_meterset_transition_for_first_control_point(self):
        shaper = MLCShaper(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        with self.assertRaises(ValueError) as context:
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=0,
                meterset_transition=1,
            )
        self.assertIn("Cannot have a transition", str(context.exception))

    def test_cant_have_initial_sacrificial_gap_and_sacrificial_distance(self):
        shaper = MLCShaper(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        with self.assertRaises(ValueError) as context:
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=0.5,
                meterset_transition=0.1,
                sacrificial_distance_mm=5,
                initial_sacrificial_gap_mm=5,
            )
        self.assertIn("Cannot specify both", str(context.exception))

    def test_cannot_have_sacrifical_gap_on_secondary_control_point(self):
        shaper = MLCShaper(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        shaper.add_strip(position_mm=-5, strip_width_mm=0, meterset_at_target=0.5)
        with self.assertRaises(ValueError) as context:
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=0.5,
                initial_sacrificial_gap_mm=10,
            )
        self.assertIn("already control points", str(context.exception))

    def test_split_sacrifices(self):
        res = split_sacrifice_travel(distance=33, max_travel=20)
        self.assertCountEqual(res, [20, 13])
        res = split_sacrifice_travel(distance=11, max_travel=20)
        self.assertCountEqual(res, [11])
        res = split_sacrifice_travel(distance=66, max_travel=20)
        self.assertCountEqual(res, [20, 20, 20, 6])

    def test_as_control_points(self):
        shaper = MLCShaper(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        shaper.add_strip(position_mm=-5, strip_width_mm=0, meterset_at_target=1)
        cp = shaper.as_control_points()
        self.assertEqual(
            len(cp), 2
        )  # start and end positions given meterset increments
        self.assertEqual(cp[0][0], -5)

    def test_as_metersets(self):
        shaper = MLCShaper(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        shaper.add_strip(position_mm=-5, strip_width_mm=0, meterset_at_target=1)
        metersets = shaper.as_metersets()
        self.assertEqual(metersets, [0, 1])


class TestNextSacrificeShift(TestCase):
    def test_easy(self):
        target = next_sacrifice_shift(
            current_position_mm=0,
            travel_mm=5,
            x_width_mm=400,
            other_mlc_position=0,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, -5)

    def test_toward_target_right(self):
        target = next_sacrifice_shift(
            current_position_mm=-5,
            travel_mm=50,
            x_width_mm=400,
            other_mlc_position=0,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, 50)

    def test_toward_target_left(self):
        target = next_sacrifice_shift(
            current_position_mm=45,
            travel_mm=50,
            x_width_mm=400,
            other_mlc_position=0,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, -50)

    def test_travel_too_large(self):
        with self.assertRaises(ValueError):
            next_sacrifice_shift(
                current_position_mm=0,
                travel_mm=200,
                x_width_mm=400,
                other_mlc_position=0,
                max_overtravel_mm=140,
            )

    def test_travel_can_be_over_max_overtravel_if_on_other_side(self):
        target = next_sacrifice_shift(
            current_position_mm=0,
            travel_mm=200,
            x_width_mm=400,
            other_mlc_position=100,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, 200)

    def test_at_edge_of_width(self):
        target = next_sacrifice_shift(
            current_position_mm=-180,
            travel_mm=30,
            x_width_mm=400,
            other_mlc_position=-190,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, 30)

        target = next_sacrifice_shift(
            current_position_mm=180,
            travel_mm=30,
            x_width_mm=400,
            other_mlc_position=190,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, -30)

    def test_width_vs_overtravel(self):
        with self.assertRaises(ValueError):
            next_sacrifice_shift(
                current_position_mm=0,
                travel_mm=30,
                x_width_mm=100,
                other_mlc_position=-190,
                max_overtravel_mm=140,
            )


class TestInterpolateControlPoints(TestCase):
    """For these tests, we use a simplified version of a 3-pair MLC. The first and last pair are the sacrificial leaves."""

    def test_control_point_lengths_mismatch(self):
        with self.assertRaises(ValueError):
            interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10],
                interpolation_ratios=[0.5],
                sacrifice_chunks=[5],
                max_overtravel=140,
            )

    def test_no_interpolation(self):
        with self.assertRaises(ValueError):
            interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10, 10],
                interpolation_ratios=[],
                sacrifice_chunks=[5],
                max_overtravel=140,
            )

    def test_interpolate_simple(self):
        interp_cp = interpolate_control_points(
            control_point_start=[0, 0, 0, 0, 0, 0],
            control_point_end=[10, 10, 10, 10, 10, 10],
            interpolation_ratios=[0.5],
            sacrifice_chunks=[1],
            max_overtravel=140,
        )
        # the first, middle, and last values should be the sacrifice
        # the middle values should be the interpolated values
        self.assertEqual(interp_cp, [[-1, 5, -1, -1, 5, -1]])

    def test_interpolate_multiple(self):
        interp_cp = interpolate_control_points(
            control_point_start=[0, 0, 0, 0, 0, 0],
            control_point_end=[10, 10, 10, 10, 10, 10],
            interpolation_ratios=[0.25, 0.5, 0.75],
            sacrifice_chunks=[3, 5, 7],
            max_overtravel=140,
        )
        # the sacrifice goes 0 - 3 -> -3 + 5 -> 2 + 7 -> 9
        cp1 = [-3, 2.5, -3, -3, 2.5, -3]
        self.assertEqual(interp_cp[0], cp1)
        cp2 = [2, 5, 2, 2, 5, 2]
        self.assertEqual(interp_cp[1], cp2)
        cp3 = [9, 7.5, 9, 9, 7.5, 9]
        self.assertEqual(interp_cp[2], cp3)

    def test_overtravel(self):
        # 30 is over the max overtravel of 20
        with self.assertRaises(ValueError):
            interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10, 10, 10],
                interpolation_ratios=[0.5],
                sacrifice_chunks=[30],
                max_overtravel=20,
            )

    def test_interpolation_over_1_or_0(self):
        with self.assertRaises(ValueError):
            interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10, 10, 10],
                interpolation_ratios=[1.5],
                sacrifice_chunks=[5],
                max_overtravel=140,
            )
        with self.assertRaises(ValueError):
            interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10, 10, 10],
                interpolation_ratios=[-0.5],
                sacrifice_chunks=[5],
                max_overtravel=140,
            )


class TestVmatT2(TestCase):
    def test_defaults(self):
        VMATDRGS.from_machine(DEFAULT_TRUEBEAM_HD120)

    def test_plot_control_points(self):
        test = VMATDRGS.from_machine(DEFAULT_TRUEBEAM_HD120)
        test.plot_control_points()

    def test_plot_fluence(self):
        test = VMATDRGS.from_machine(DEFAULT_TRUEBEAM_HD120)
        test.plot_fluence(IMAGER_AS1200)

    def test_plot_profile(self):
        test = VMATDRGS.from_machine(DEFAULT_TRUEBEAM_HD120)
        test.plot_fluence_profile(IMAGER_AS1200)

    def test_replicate_original_test(self):
        original_file = get_file_from_cloud_test_repo(
            [
                "plan_generator",
                "VMAT",
                "Millennium",
                "T2_DoseRateGantrySpeed_M120_TB_Rev02.dcm",
            ]
        )
        beam_name_dyn = "T2_DR_GS"
        beam_name_ref = "OpenBeam"
        max_gantry_speed = 4.8
        max_dose_rate = 600

        # Extract data from the original plan
        ds = pydicom.dcmread(original_file)
        beam_sequence = ds.BeamSequence
        beam_idx = [bs.BeamName == beam_name_dyn for bs in beam_sequence].index(True)
        beam_meterset = int(
            ds.FractionGroupSequence[0].ReferencedBeamSequence[beam_idx].BeamMeterset
        )
        beam = beam_sequence[beam_idx]
        control_point_sequence = beam.ControlPointSequence
        plan_gantry_angle = np.array([cp.GantryAngle for cp in control_point_sequence])
        gantry_angle_var = (180 - plan_gantry_angle) % 360
        plan_cumulative_meterset_weight = np.array(
            [cp.CumulativeMetersetWeight for cp in control_point_sequence]
        )
        plan_cumulative_meterset = plan_cumulative_meterset_weight * beam_meterset
        plan_mlc_position = np.array(
            [
                bld.LeafJawPositions
                for cp in control_point_sequence
                for bld in cp.BeamLimitingDevicePositionSequence
                if bld.RTBeamLimitingDeviceType == "MLCX"
            ]
        )

        # Derive the input parameters
        gantry_motion = np.append(0, np.abs(np.diff(gantry_angle_var)))
        dose_motion = np.append(0, np.abs(np.diff(plan_cumulative_meterset)))
        time_to_deliver_gantry = gantry_motion / max_gantry_speed
        time_to_deliver_mu = dose_motion / (max_dose_rate / 60)
        time_to_deliver = np.max((time_to_deliver_gantry, time_to_deliver_mu), axis=0)
        gantry_speed = gantry_motion / time_to_deliver  #
        dose_rate = dose_motion / time_to_deliver * 60
        gantry_speeds = gantry_speed[2:-1:2]
        dose_rates = dose_rate[2:-1:2]
        initial_gantry_offset = float(180 - plan_gantry_angle[0])
        gantry_motion_per_transition = float(gantry_motion[1])
        gantry_rotation_clockwise = bool(
            plan_gantry_angle[1] - plan_gantry_angle[0] > 0
        )
        mu_per_segment = float(dose_motion[2])
        mu_per_transition = float(dose_motion[1])
        correct_fluence = False
        mlc_span = float(2 * plan_mlc_position[0, 0])
        mlc_gap = float(plan_mlc_position[2, 1] - plan_mlc_position[3, -1])
        mlc_motion_reverse = bool(plan_mlc_position[0, 0] > 0)
        jaw_padding = 0

        # Run
        specs = DEFAULT_SPECS_TB.replace(max_gantry_speed=max_gantry_speed)
        test = VMATDRGS.from_machine(
            TrueBeamMachine(False, specs),
            dose_rates=tuple(dose_rates),
            gantry_speeds=tuple(gantry_speeds),
            mu_per_segment=mu_per_segment,
            mu_per_transition=mu_per_transition,
            correct_fluence=correct_fluence,
            gantry_motion_per_transition=gantry_motion_per_transition,
            gantry_rotation_clockwise=gantry_rotation_clockwise,
            initial_gantry_offset=initial_gantry_offset,
            mlc_span=mlc_span,
            mlc_motion_reverse=mlc_motion_reverse,
            mlc_gap=mlc_gap,
            jaw_padding=jaw_padding,
            max_dose_rate=max_dose_rate,
        )

        # Assert dynamic beam
        cps = test.beams[0].to_dicom().ControlPointSequence
        gantry_angle = [cp.GantryAngle for cp in cps]
        cumulative_meterset_weight = [cp.CumulativeMetersetWeight for cp in cps]
        mlc_position = np.array(
            [cp.BeamLimitingDevicePositionSequence[-1].LeafJawPositions for cp in cps]
        )
        self.assertTrue(np.allclose(gantry_angle, plan_gantry_angle))
        self.assertTrue(
            np.allclose(cumulative_meterset_weight, plan_cumulative_meterset_weight)
        )
        self.assertTrue(np.allclose(mlc_position, plan_mlc_position))

        # Assert reference beam
        beam_idx = [bs.BeamName == beam_name_ref for bs in beam_sequence].index(True)
        beam = beam_sequence[beam_idx]
        cps = beam.ControlPointSequence
        plan_gantry_angle = cps[0].GantryAngle
        plan_cumulative_meterset_weight = [cp.CumulativeMetersetWeight for cp in cps]
        plan_mlc_position = (
            cps[0].BeamLimitingDevicePositionSequence[-1].LeafJawPositions
        )
        cps = test.beams[1].to_dicom().ControlPointSequence
        gantry_angle = cps[0].GantryAngle
        cumulative_meterset_weight = [cp.CumulativeMetersetWeight for cp in cps]
        mlc_position = cps[0].BeamLimitingDevicePositionSequence[-1].LeafJawPositions
        self.assertTrue(np.allclose(gantry_angle, plan_gantry_angle))
        self.assertTrue(
            np.allclose(cumulative_meterset_weight, plan_cumulative_meterset_weight)
        )
        self.assertTrue(np.allclose(mlc_position, plan_mlc_position))

    def test_adding_static_beams(self):
        static_angles = (0, 90, 270, 180)
        test = VMATDRGS.from_machine(
            DEFAULT_TRUEBEAM_HD120, dynamic_delivery_at_static_gantry=static_angles
        )
        expected_number_of_beams = 6  # dynamic, reference, 4x static
        actual_number_of_beams = len(test.beams)
        self.assertEqual(actual_number_of_beams, expected_number_of_beams)

        for idx, expected_angle in enumerate(static_angles):
            actual_angle = (
                test.beams[idx + 2].to_dicom().ControlPointSequence[0].GantryAngle
            )
            self.assertEqual(actual_angle, expected_angle)

    def test_error_if_gantry_speeds_and_dose_rates_have_different_sizes(self):
        gantry_speeds = (1, 2, 3)
        dose_rates = (1, 2)
        with self.assertRaises(ValueError):
            VMATDRGS.from_machine(
                DEFAULT_TRUEBEAM_HD120,
                gantry_speeds=gantry_speeds,
                dose_rates=dose_rates,
            )

    def test_error_if_initial_gantry_offset_less_than_min(self):
        initial_gantry_offset = 0
        with self.assertRaises(ValueError):
            VMATDRGS.from_machine(
                DEFAULT_TRUEBEAM_HD120, initial_gantry_offset=initial_gantry_offset
            )

    def test_error_if_gantry_speeds_above_max(self):
        gantry_speeds = (1, 2, 5)
        dose_rates = (1, 2, 3)
        max_gantry_speed = 4
        specs = DEFAULT_SPECS_TB.replace(max_gantry_speed=max_gantry_speed)
        machine = TrueBeamMachine(False, specs)
        with self.assertRaises(ValueError):
            VMATDRGS.from_machine(
                machine, gantry_speeds=gantry_speeds, dose_rates=dose_rates
            )

    def test_error_if_dose_rates_above_max(self):
        gantry_speeds = (1, 2, 5)
        dose_rates = (1, 2, 300)
        max_dose_rate = 100
        with self.assertRaises(ValueError):
            VMATDRGS.from_machine(
                DEFAULT_TRUEBEAM_HD120,
                gantry_speeds=gantry_speeds,
                dose_rates=dose_rates,
                max_dose_rate=max_dose_rate,
            )

    def test_error_if_axis_not_maxed_out(self):
        gantry_speeds = (5, 1, 1)
        dose_rates = (1, 300, 1)
        max_gantry_speed = 5
        max_dose_rate = 300
        specs = DEFAULT_SPECS_TB.replace(max_gantry_speed=max_gantry_speed)
        machine = TrueBeamMachine(False, specs)
        with self.assertRaises(ValueError):
            VMATDRGS.from_machine(
                machine,
                gantry_speeds=gantry_speeds,
                dose_rates=dose_rates,
                max_dose_rate=max_dose_rate,
            )

    def test_error_if_rotation_larger_than_360(self):
        mu_per_segment = 100
        with self.assertRaises(ValueError):
            VMATDRGS.from_machine(DEFAULT_TRUEBEAM_HD120, mu_per_segment=mu_per_segment)
