from unittest import TestCase

import numpy as np
import pydicom

from parameterized import parameterized

from conjuror.images.simulators import IMAGER_AS1200
from conjuror.plans.plan_generator import PlanGenerator, BeamBase
from conjuror.plans.truebeam import (
    OpenField,
    OpenFieldMLCMode,
    MLCTransmission,
    PicketFence,
    WinstonLutz,
    WinstonLutzMLCMode,
    WinstonLutzField,
    DoseRate,
    MLCSpeed,
    GantrySpeed,
    VMATDRGS,
    DEFAULT_SPECS_TB,
    TrueBeamMachine,
)
from tests.utils import get_file_from_cloud_test_repo

TB_MIL_PLAN_FILE = get_file_from_cloud_test_repo(["plan_generator", "Murray-plan.dcm"])
DEFAULT_TRUEBEAM_HD120 = TrueBeamMachine(mlc_is_hd=True)


class TestProcedures(TestCase):
    def setUp(self) -> None:
        self.pg = PlanGenerator.from_rt_plan_file(
            TB_MIL_PLAN_FILE,
            plan_label="label",
            plan_name="my name",
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
        # max speed is 6.0 by default
        procedure = GantrySpeed(
            speeds=(1, 2, 3, 4, 6.5),
            y1=-100,
            y2=100,
        )
        with self.assertRaises(ValueError):
            self.pg.add_procedure(procedure)

    def test_gantry_speed_too_wide(self):
        procedure = GantrySpeed(
            speeds=(1, 2, 3, 4),
            roi_size_mm=100,
            y1=-100,
            y2=100,
        )
        with self.assertRaises(ValueError):
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


class TestOpenField(TestCase):
    # An Open field is mostly a Rectangle, so some of these tests strategy mimics the Rectangle shape tests
    # There is a significant portion of these tests that are copied in WinstonLutz.

    OPENFIELD_MLC_PARAM = [
        (-2.5, 2.5, OpenFieldMLCMode.EXACT),
        (-2.4, 2.6, OpenFieldMLCMode.ROUND),
        (-2.6, 2.6, OpenFieldMLCMode.INWARD),
        (-2.4, 2.4, OpenFieldMLCMode.OUTWARD),
    ]

    @parameterized.expand(OPENFIELD_MLC_PARAM)
    def test_defined_by_mlc(self, y1, y2, mlc_mode):
        x1, x2 = -100, 100
        procedure = OpenField(x1, x2, y1, y2, mlc_mode=mlc_mode)
        procedure.compute(DEFAULT_TRUEBEAM_HD120)

        bld = procedure.beams[0].beam_limiting_device_positions

        mlc = bld["MLCX"]
        self.assertFalse(np.any(mlc[:, 0] - mlc[:, 1]))
        mlc0 = mlc[:, 0]
        self.assertTrue(all(x == -127.5 for x in mlc0[:29]))
        self.assertTrue(all(x == x1 for x in mlc0[29:31]))
        self.assertTrue(all(x == -127.5 for x in mlc0[31:60]))
        self.assertTrue(all(x == -122.5 for x in mlc0[60 : 29 + 60]))
        self.assertTrue(all(x == x2 for x in mlc0[29 + 60 : 31 + 60]))
        self.assertTrue(all(x == -122.5 for x in mlc0[31 + 60 :]))

        jaw_x = bld["ASYMX"]
        self.assertFalse(np.any(jaw_x[:, 0] - jaw_x[:, 1]))
        self.assertEqual(x1 - 5, jaw_x[0, 0])
        self.assertEqual(x2 + 5, jaw_x[1, 0])

        jaw_y = bld["ASYMY"]
        self.assertFalse(np.any(jaw_y[:, 0] - jaw_y[:, 1]))
        self.assertEqual(y1 - 5, jaw_y[0, 0])
        self.assertEqual(y2 + 5, jaw_y[1, 0])

    OPEN_MLC_PARAM = [
        OpenFieldMLCMode.EXACT,
        OpenFieldMLCMode.ROUND,
        OpenFieldMLCMode.INWARD,
        OpenFieldMLCMode.OUTWARD,
    ]

    @parameterized.expand(OPEN_MLC_PARAM)
    def test_open_mlc(self, mlc_mode):
        x1, x2, y1, y2 = -100, 100, -110, 110
        procedure = OpenField(x1, x2, y1, y2, mlc_mode=mlc_mode)
        procedure.compute(DEFAULT_TRUEBEAM_HD120)

        bld = procedure.beams[0].beam_limiting_device_positions

        mlc = bld["MLCX"]
        self.assertFalse(np.any(mlc[:, 0] - mlc[:, 1]))
        mlc0 = mlc[:, 0]
        self.assertTrue(all(x == -100 for x in mlc0[:60]))
        self.assertTrue(all(x == 100 for x in mlc0[60:]))

    def test_defined_by_jaws(self):
        x1, x2, y1, y2 = -100, 100, -110, 110
        procedure = OpenField(x1, x2, y1, y2, defined_by_mlc=False)
        procedure.compute(DEFAULT_TRUEBEAM_HD120)

        bld = procedure.beams[0].beam_limiting_device_positions

        mlc = bld["MLCX"]
        self.assertFalse(np.any(mlc[:, 0] - mlc[:, 1]))
        mlc0 = mlc[:, 0]
        self.assertTrue(all(x == -105 for x in mlc0[:60]))
        self.assertTrue(all(x == 105 for x in mlc0[60:]))

        jaw_x = bld["ASYMX"]
        self.assertFalse(np.any(jaw_x[:, 0] - jaw_x[:, 1]))
        self.assertEqual(x1, jaw_x[0, 0])
        self.assertEqual(x2, jaw_x[1, 0])

        jaw_y = bld["ASYMY"]
        self.assertFalse(np.any(jaw_y[:, 0] - jaw_y[:, 1]))
        self.assertEqual(y1, jaw_y[0, 0])
        self.assertEqual(y2, jaw_y[1, 0])

    @parameterized.expand([(2, 1, 0, 1), (0, 1, 2, 1)])
    def test_error_if_min_larger_than_max(self, x1, x2, y1, y2):
        procedure = OpenField(x1, x2, y1, y2, 100)
        with self.assertRaises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    @parameterized.expand([(-0.1, 0), (0, 0.1)])
    def test_error_if_mlc_exact_and_not_possible(self, y1, y2):
        procedure = OpenField(0, 1, y1, y2, mlc_mode=OpenFieldMLCMode.EXACT)
        with self.assertRaises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    @parameterized.expand([(-0.1, 0), (0, 0.1)])
    def test_exact_is_ignored_if_defined_by_jaws(self, y1, y2):
        procedure = OpenField(
            0, 1, y1, y2, mlc_mode=OpenFieldMLCMode.EXACT, defined_by_mlc=False
        )
        procedure.compute(DEFAULT_TRUEBEAM_HD120)


class TestMLCTransmission(TestCase):
    def test_defaults(self):
        procedure = MLCTransmission()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        self.assertEqual(3, len(procedure.beams))

    def test_banks(self):
        procedure = MLCTransmission()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)

        for beam_idx in [1, 2]:
            mult = -1 if beam_idx == 1 else 1
            slit_pos = mult * (procedure.width / 2 + procedure.overreach)
            mlc_nominal = 60 * [2 * [slit_pos - 0.5]] + 60 * [2 * [slit_pos + 0.5]]

            bld = procedure.beams[beam_idx].beam_limiting_device_positions
            np.testing.assert_array_equal(mlc_nominal, bld["MLCX"])
            np.testing.assert_array_equal([[-50, -50], [50, 50]], bld["ASYMX"])
            np.testing.assert_array_equal([[-50, -50], [50, 50]], bld["ASYMY"])

    def test_beam_names(self):
        beam_names = ["Ref", "A", "B"]
        procedure = MLCTransmission(beam_names=beam_names)
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        actual = [b.beam_name for b in procedure.beams]
        self.assertEqual(beam_names, actual)


class TestPicketFence(TestCase):
    def test_defaults(self):
        procedure = PicketFence()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        self.assertEqual(1, len(procedure.beams))

    def test_replicate_varian_plan(self):
        path = [
            "plan_generator",
            "VMAT",
            "Millennium",
            "T0.2_PicketFenceStatic_M120_TB_Rev02.dcm",
        ]
        picket_fence_file = get_file_from_cloud_test_repo(path)
        ds = pydicom.dcmread(picket_fence_file)
        beam_nominal = BeamBase.from_dicom(ds, 0)

        procedure = PicketFence(
            picket_width=1,
            picket_positions=np.arange(-74.5, 76, 15),
            mu_per_picket=8.125,
            mu_per_transition=1.875,
            skip_first_picket=True,
            jaw_padding=5.5,
        )
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        beam_actual = procedure.beams[0]

        mu_nominal, mu_actual = beam_nominal.metersets, beam_actual.metersets
        np.testing.assert_array_almost_equal(mu_nominal, mu_actual, decimal=2)

        bls_nominal = beam_nominal.beam_limiting_device_positions
        bld_actual = beam_actual.beam_limiting_device_positions
        np.testing.assert_array_equal(bls_nominal["MLCX"], bld_actual["MLCX"])
        np.testing.assert_array_equal(bls_nominal["ASYMX"][0], bld_actual["ASYMX"][0])
        # Note #1: jaw x2 is different (cannot be replicated) since the left padding
        # is different than the right padding
        # Note #2: jaws y1,y2 are different (cannot be replicated) since this procedure
        # doesn't hide any leafs, whereas in the Varian plan, the last top/bottom leaves were hidden

    def test_error_if_too_wide(self):
        procedure = PicketFence(picket_positions=(-100, 100))
        with self.assertRaises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)


class TestWinstonLutz(TestCase):
    # There is a significant portion of these tests that are copied from OpenField
    def test_defaults(self):
        procedure = WinstonLutz()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        self.assertEqual(1, len(procedure.beams))

    def test_non_defaults(self):
        fields = [WinstonLutzField(0, 0, 0, "name1"), WinstonLutzField(90.7, 0.7, 10.7)]
        names_nominal = ["name1", "G091C001T011"]
        procedure = WinstonLutz(fields=fields)
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        self.assertEqual(2, len(procedure.beams))
        for f, b, n in zip(fields, procedure.beams, names_nominal):
            self.assertEqual(f.gantry, b.gantry_angles[0])
            self.assertEqual(f.collimator, b.coll_angle)
            self.assertEqual(f.couch, b.couch_rot)
            self.assertEqual(b.beam_name, n)
            pass
        pass

    WL_MLC_PARAM = [
        (-2.5, 2.5, WinstonLutzMLCMode.EXACT),
        (-2.4, 2.6, WinstonLutzMLCMode.ROUND),
        (-2.6, 2.6, WinstonLutzMLCMode.INWARD),
        (-2.4, 2.4, WinstonLutzMLCMode.OUTWARD),
    ]

    @parameterized.expand(WL_MLC_PARAM)
    def test_defined_by_mlc(self, y1, y2, mlc_mode):
        x1, x2 = -100, 100
        procedure = WinstonLutz(x1, x2, y1, y2, mlc_mode=mlc_mode)
        procedure.compute(DEFAULT_TRUEBEAM_HD120)

        bld = procedure.beams[0].beam_limiting_device_positions

        mlc = bld["MLCX"]
        self.assertFalse(np.any(mlc[:, 0] - mlc[:, 1]))
        mlc0 = mlc[:, 0]
        self.assertTrue(all(x == -127.5 for x in mlc0[:29]))
        self.assertTrue(all(x == x1 for x in mlc0[29:31]))
        self.assertTrue(all(x == -127.5 for x in mlc0[31:60]))
        self.assertTrue(all(x == -122.5 for x in mlc0[60 : 29 + 60]))
        self.assertTrue(all(x == x2 for x in mlc0[29 + 60 : 31 + 60]))
        self.assertTrue(all(x == -122.5 for x in mlc0[31 + 60 :]))

        jaw_x = bld["ASYMX"]
        self.assertFalse(np.any(jaw_x[:, 0] - jaw_x[:, 1]))
        self.assertEqual(x1 - 5, jaw_x[0, 0])
        self.assertEqual(x2 + 5, jaw_x[1, 0])

        jaw_y = bld["ASYMY"]
        self.assertFalse(np.any(jaw_y[:, 0] - jaw_y[:, 1]))
        self.assertEqual(y1 - 5, jaw_y[0, 0])
        self.assertEqual(y2 + 5, jaw_y[1, 0])

    def test_defined_by_jaws(self):
        x1, x2, y1, y2 = -100, 100, -110, 110
        procedure = WinstonLutz(x1, x2, y1, y2, defined_by_mlc=False)
        procedure.compute(DEFAULT_TRUEBEAM_HD120)

        bld = procedure.beams[0].beam_limiting_device_positions

        mlc = bld["MLCX"]
        self.assertFalse(np.any(mlc[:, 0] - mlc[:, 1]))
        mlc0 = mlc[:, 0]
        self.assertTrue(all(x == -105 for x in mlc0[:60]))
        self.assertTrue(all(x == 105 for x in mlc0[60:]))

        jaw_x = bld["ASYMX"]
        self.assertFalse(np.any(jaw_x[:, 0] - jaw_x[:, 1]))
        self.assertEqual(x1, jaw_x[0, 0])
        self.assertEqual(x2, jaw_x[1, 0])

        jaw_y = bld["ASYMY"]
        self.assertFalse(np.any(jaw_y[:, 0] - jaw_y[:, 1]))
        self.assertEqual(y1, jaw_y[0, 0])
        self.assertEqual(y2, jaw_y[1, 0])

    @parameterized.expand([(2, 1, 0, 1), (0, 1, 2, 1)])
    def test_error_if_min_larger_than_max(self, x1, x2, y1, y2):
        procedure = WinstonLutz(x1, x2, y1, y2, 100)
        with self.assertRaises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    @parameterized.expand([(-0.1, 0), (0, 0.1)])
    def test_error_if_mlc_exact_and_not_possible(self, y1, y2):
        procedure = WinstonLutz(0, 1, y1, y2, mlc_mode=WinstonLutzMLCMode.EXACT)
        with self.assertRaises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    @parameterized.expand([(-0.1, 0), (0, 0.1)])
    def test_exact_is_ignored_if_defined_by_jaws(self, y1, y2):
        procedure = WinstonLutz(
            0, 1, y1, y2, mlc_mode=WinstonLutzMLCMode.EXACT, defined_by_mlc=False
        )
        procedure.compute(DEFAULT_TRUEBEAM_HD120)


class TestVmatDRGS(TestCase):
    def test_defaults(self):
        VMATDRGS().compute(DEFAULT_TRUEBEAM_HD120)

    def test_plot_control_points(self):
        procedure = VMATDRGS()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        procedure.plot_control_points()

    def test_plot_fluence(self):
        procedure = VMATDRGS()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        procedure.plot_fluence(IMAGER_AS1200)

    def test_plot_profile(self):
        procedure = VMATDRGS()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        procedure.plot_fluence_profile(IMAGER_AS1200)

    def test_replicate_varian_plan(self):
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
        procedure = VMATDRGS(
            dose_rates=tuple(float(d) for d in dose_rates),
            gantry_speeds=tuple(float(g) for g in gantry_speeds),
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
        procedure.compute(TrueBeamMachine(False, specs))

        # Assert dynamic beam
        cps = procedure.beams[0].to_dicom().ControlPointSequence
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
        cps = procedure.beams[1].to_dicom().ControlPointSequence
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
        procedure = VMATDRGS(dynamic_delivery_at_static_gantry=static_angles)
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        expected_number_of_beams = 6  # dynamic, reference, 4x static
        actual_number_of_beams = len(procedure.beams)
        self.assertEqual(actual_number_of_beams, expected_number_of_beams)

        for idx, expected_angle in enumerate(static_angles):
            dcm = procedure.beams[idx + 2].to_dicom()
            actual_angle = dcm.ControlPointSequence[0].GantryAngle
            self.assertEqual(actual_angle, expected_angle)

    def test_error_if_gantry_speeds_and_dose_rates_have_different_sizes(self):
        gantry_speeds = (1, 2, 3)
        dose_rates = (1, 2)
        procedure = VMATDRGS(gantry_speeds=gantry_speeds, dose_rates=dose_rates)
        with self.assertRaises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    def test_error_if_initial_gantry_offset_less_than_min(self):
        initial_gantry_offset = 0
        procedure = VMATDRGS(initial_gantry_offset=initial_gantry_offset)
        with self.assertRaises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    def test_error_if_gantry_speeds_above_max(self):
        gantry_speeds = (1, 2, 5)
        dose_rates = (1, 2, 3)
        max_gantry_speed = 4
        specs = DEFAULT_SPECS_TB.replace(max_gantry_speed=max_gantry_speed)
        machine = TrueBeamMachine(False, specs)
        procedure = VMATDRGS(gantry_speeds=gantry_speeds, dose_rates=dose_rates)
        with self.assertRaises(ValueError):
            procedure.compute(machine)

    def test_error_if_dose_rates_above_max(self):
        gantry_speeds = (1, 2, 5)
        dose_rates = (1, 2, 300)
        max_dose_rate = 100
        procedure = VMATDRGS(
            gantry_speeds=gantry_speeds,
            dose_rates=dose_rates,
            max_dose_rate=max_dose_rate,
        )
        with self.assertRaises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    def test_error_if_axis_not_maxed_out(self):
        gantry_speeds = (5, 1, 1)
        dose_rates = (1, 300, 1)
        max_gantry_speed = 5
        max_dose_rate = 300
        specs = DEFAULT_SPECS_TB.replace(max_gantry_speed=max_gantry_speed)
        machine = TrueBeamMachine(False, specs)
        procedure = VMATDRGS(
            gantry_speeds=gantry_speeds,
            dose_rates=dose_rates,
            max_dose_rate=max_dose_rate,
        )
        with self.assertRaises(ValueError):
            procedure.compute(machine)

    def test_error_if_rotation_larger_than_360(self):
        mu_per_segment = 100
        procedure = VMATDRGS(mu_per_segment=mu_per_segment)
        with self.assertRaises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)
