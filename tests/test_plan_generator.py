import tempfile
from unittest import TestCase

import numpy as np
import pydicom
from plotly.graph_objects import Figure
import matplotlib.pyplot as plt
from parameterized import parameterized

from conjuror.images.simulators import IMAGER_AS1200
from conjuror.plans.plan_generator import (
    BeamBase,
    FluenceMode,
    PlanGenerator,
)
from conjuror.plans.halcyon import HalcyonMachine
from conjuror.plans.truebeam import (
    TrueBeamMachine,
    OpenField,
    Beam,
    MLCSpeed,
    PicketFence,
)
from tests.utils import get_file_from_cloud_test_repo

TB_MIL_PLAN_FILE = get_file_from_cloud_test_repo(["plan_generator", "Murray-plan.dcm"])
_hd_path = ["plan_generator", "VMAT", "HDMLC", "T0.1_Dosimetry_HD120_TB_Rev02.dcm"]
TB_HD_PLAN_FILE = get_file_from_cloud_test_repo(_hd_path)
HAL_PLAN_FILE = get_file_from_cloud_test_repo(["plan_generator", "Halcyon Prox.dcm"])
TB_MIL_PLAN_DS = pydicom.dcmread(TB_MIL_PLAN_FILE)

DEFAULT_TRUEBEAM_HD120 = TrueBeamMachine(mlc_is_hd=True)


def _get_generator_from_dataset(dataset):
    return PlanGenerator(dataset, plan_label="label", plan_name="name")


def _get_generator_from_file(file):
    return PlanGenerator.from_rt_plan_file(file, plan_label="label", plan_name="name")


class TestPlanGeneratorCreation(TestCase):
    GENERATOR_TEST_PARAMS = [
        (TB_MIL_PLAN_FILE, TrueBeamMachine, False),
        (TB_HD_PLAN_FILE, TrueBeamMachine, True),
        (HAL_PLAN_FILE, HalcyonMachine, None),
    ]

    @parameterized.expand(GENERATOR_TEST_PARAMS)
    def test_from_dataset(self, file: str, plan_generator_type: type, mlc_is_hd: bool):
        dataset = pydicom.dcmread(file)
        pg = _get_generator_from_dataset(dataset)
        self.assertIsInstance(pg.machine, plan_generator_type)
        if isinstance(pg.machine, TrueBeamMachine):
            self.assertEqual(pg.machine.mlc_is_hd, mlc_is_hd)

    @parameterized.expand(GENERATOR_TEST_PARAMS)
    def test_from_rt_plan_file(
        self, file: str, plan_generator_type: type, mlc_is_hd: bool
    ):
        pg = _get_generator_from_file(file)
        self.assertIsInstance(pg.machine, plan_generator_type)
        if isinstance(pg.machine, TrueBeamMachine):
            self.assertEqual(pg.machine.mlc_is_hd, mlc_is_hd)

    def test_from_not_rt_plan_file(self):
        file = get_file_from_cloud_test_repo(["picket_fence", "AS500#2.dcm"])
        with self.assertRaises(ValueError):
            _get_generator_from_file(file)

    def test_to_file(self):
        pg = _get_generator_from_file(TB_MIL_PLAN_FILE)
        pg.add_procedure(MLCSpeed())
        with tempfile.NamedTemporaryFile(delete=False) as t:
            pg.to_file(t.name)
        # shouldn't raise; should be valid DICOM
        ds = pydicom.dcmread(t.name)
        self.assertEqual(ds.RTPlanLabel, "label")
        self.assertEqual(len(ds.BeamSequence), 2)


class TestPlanGeneratorParameters(TestCase):
    def test_no_patient_id(self):
        ds = pydicom.dcmread(TB_MIL_PLAN_FILE)
        ds.pop("PatientID")
        with self.assertRaises(ValueError):
            PlanGenerator(ds, plan_label="plan_label", plan_name="plan_name")

    def test_no_patient_name(self):
        ds = pydicom.dcmread(TB_MIL_PLAN_FILE)
        ds.pop("PatientName")
        with self.assertRaises(ValueError):
            _get_generator_from_dataset(ds)

    def test_pass_patient_name(self):
        patient_name = "name"
        ds = pydicom.dcmread(TB_MIL_PLAN_FILE)
        pg = PlanGenerator(ds, plan_label="l", plan_name="n", patient_name=patient_name)
        pg_dcm = pg.as_dicom()
        self.assertEqual(pg_dcm.PatientName, patient_name)

    def test_pass_patient_id(self):
        patient_id = "id"
        ds = pydicom.dcmread(TB_MIL_PLAN_FILE)
        pg = PlanGenerator(ds, plan_label="l", plan_name="n", patient_id=patient_id)
        pg_dcm = pg.as_dicom()
        self.assertEqual(pg_dcm.PatientID, patient_id)

    def test_no_tolerance_table(self):
        ds = pydicom.dcmread(TB_MIL_PLAN_FILE)
        ds.pop("ToleranceTableSequence")
        with self.assertRaises(ValueError):
            _get_generator_from_dataset(ds)

    def test_no_beam_sequence(self):
        ds = pydicom.dcmread(TB_MIL_PLAN_FILE)
        ds.pop("BeamSequence")
        with self.assertRaises(ValueError):
            _get_generator_from_dataset(ds)

    def test_no_mlc_data(self):
        ds = pydicom.dcmread(TB_MIL_PLAN_FILE)
        # pop MLC part of the data; at this point it's just an open field
        ds.BeamSequence[0].BeamLimitingDeviceSequence.pop()
        with self.assertRaises(ValueError):
            _get_generator_from_dataset(ds)

    def test_machine_name(self):
        pg = _get_generator_from_file(TB_MIL_PLAN_FILE)
        self.assertEqual(pg.machine_name, "TrueBeamSN5837")

    def test_machine_name_set_on_beam(self):
        """Beam machine name is set when added to the plan"""
        pg = _get_generator_from_file(TB_MIL_PLAN_FILE)
        pg.add_beam(create_beam())
        dcm = pg.as_dicom()
        self.assertEqual(dcm.BeamSequence[0].TreatmentMachineName, "TrueBeamSN5837")

    def test_leaf_boundaries(self):
        pg = _get_generator_from_file(TB_MIL_PLAN_FILE)
        self.assertEqual(len(pg.machine.mlc_boundaries), 61)
        self.assertEqual(max(pg.machine.mlc_boundaries), 200)
        self.assertEqual(min(pg.machine.mlc_boundaries), -200)

    def test_instance_uid_changes(self):
        dcm = pydicom.dcmread(TB_MIL_PLAN_FILE)
        pg = _get_generator_from_file(TB_MIL_PLAN_FILE)
        pg_dcm = pg.as_dicom()
        self.assertNotEqual(pg_dcm.SOPInstanceUID, dcm.SOPInstanceUID)

    def test_invert_array(self):
        pg = _get_generator_from_file(TB_MIL_PLAN_FILE)
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
        self.assertAlmostEqual(float(inverted_array[0, 0]), 100)


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
        mlc_positions=kwargs.get("mlc_positions", [120 * [0], 120 * [0]]),
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

    def test_from_dicom(self):
        # shouldn't raise; happy path
        ds = pydicom.dcmread(TB_MIL_PLAN_FILE)
        BeamBase.from_dicom(ds, 0)

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
        self.pg = _get_generator_from_file(TB_MIL_PLAN_FILE)

    def test_add_beam_low_level(self):
        self.pg.add_beam(create_beam())
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
            TB_MIL_PLAN_DS.BeamSequence[0]
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

    def test_beam_roundtrip(self):
        # Round trip:
        # 1. create beam (beam1)
        # 2. create generator and append beam1 to plan
        # 3. load beam from generator (beam2)
        # 4. assert(beam1==beam2)
        beam1 = create_beam()
        pg = _get_generator_from_file(TB_MIL_PLAN_FILE)
        pg.add_beam(beam1)
        beam2 = BeamBase.from_dicom(pg.ds, 0)
        self.assertEqual(beam1.beam_name, beam2.beam_name)
        self.assertEqual(beam1.beam_meterset, beam2.beam_meterset)
        np.testing.assert_array_equal(beam1.gantry_angles, beam2.gantry_angles)
        np.testing.assert_array_equal(beam1.metersets, beam2.metersets)
        for bld_type in ["ASYMX", "ASYMY", "MLCX"]:
            np.testing.assert_array_equal(
                beam1.beam_limiting_device_positions[bld_type],
                beam2.beam_limiting_device_positions[bld_type],
            )

    def test_plot_fluence(self):
        # just tests it works
        procedure = OpenField(x1=-5, x2=5, y1=-5, y2=5, mu=100)
        self.pg.add_procedure(procedure)
        figs = self.pg.plot_fluences(IMAGER_AS1200)
        self.assertIsInstance(figs, list)
        self.assertIsInstance(figs[0], Figure)

    def test_animate_mlc(self):
        procedure = PicketFence()
        self.pg.add_procedure(procedure)
        beam = BeamBase.from_dicom(self.pg.as_dicom(), 0)
        beam.animate_mlc()
        pass

    def test_list_procedure(self):
        procedures = self.pg.list_procedures()
        self.assertEqual(len(procedures), 8)
