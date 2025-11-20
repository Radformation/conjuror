from unittest import TestCase

import numpy as np
import pydicom

from parameterized import parameterized

from conjuror.images.simulators import IMAGER_AS1200
from conjuror.plans.plan_generator import PlanGenerator, OvertravelError
from conjuror.plans.truebeam import (
    OpenField,
    MLCTransmission,
    PicketFence,
    WinstonLutz,
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
        procedure = PicketFence(
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
            procedure = PicketFence(
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
