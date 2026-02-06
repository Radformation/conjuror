import numpy as np
import pydicom
import pytest

from conjuror.images.simulators import IMAGER_AS1200
from conjuror.plans.plan_generator import PlanGenerator
from conjuror.plans.beam import Beam
from conjuror.plans.truebeam import (
    DEFAULT_SPECS_TB,
    TrueBeamMachine,
    OpenField,
    MLCLeafBoundaryAlignmentMode,
    MLCTransmission,
    PicketFence,
    WinstonLutz,
    WinstonLutzField,
    DoseRate,
    MLCSpeed,
    GantrySpeed,
    VMATDRGS,
    VMATDRMLC,
    DosimetricLeafGap,
)
from tests.utils import get_file_from_cloud_test_repo

TB_MIL_PLAN_FILE = get_file_from_cloud_test_repo(["plan_generator", "Murray-plan.dcm"])
DEFAULT_TRUEBEAM_HD120 = TrueBeamMachine(mlc_is_hd=True)

RUN_PLOT_TESTS = True


class TestProcedures:
    def test_adding_procedure(self):
        pg = PlanGenerator.from_rt_plan_file(
            TB_MIL_PLAN_FILE,
            plan_label="label",
            plan_name="my name",
        )
        pg.add_procedure(OpenField(-10, 10, -10, 10))  # 1
        pg.add_procedure(MLCTransmission())  # 3
        pg.add_procedure(PicketFence())  # 1
        pg.add_procedure(WinstonLutz())  # 1
        pg.add_procedure(DoseRate())  # 2
        pg.add_procedure(MLCSpeed())  # 2
        pg.add_procedure(GantrySpeed())  # 2
        pg.add_procedure(VMATDRGS())  # 2
        dcm = pg.as_dicom()
        assert len(dcm.BeamSequence) == 14


class TestOpenField:
    # An Open field is mostly a Rectangle, so some of these tests strategy mimics the Rectangle shape tests
    # There is a significant portion of these tests that are copied in WinstonLutz.

    OPENFIELD_MLC_PARAM = [
        (-2.5, 2.5, MLCLeafBoundaryAlignmentMode.EXACT),
        (-2.4, 2.6, MLCLeafBoundaryAlignmentMode.ROUND),
        (-2.6, 2.6, MLCLeafBoundaryAlignmentMode.INWARD),
        (-2.4, 2.4, MLCLeafBoundaryAlignmentMode.OUTWARD),
    ]

    @pytest.mark.parametrize("y1,y2,mlc_mode", OPENFIELD_MLC_PARAM)
    def test_defined_by_mlc(self, y1, y2, mlc_mode):
        x1, x2 = -100, 100
        procedure = OpenField(x1, x2, y1, y2, mlc_mode=mlc_mode)
        procedure.compute(DEFAULT_TRUEBEAM_HD120)

        bld = procedure.beams[0].beam_limiting_device_positions

        mlc = bld["MLCX"]
        assert not np.any(mlc[:, 0] - mlc[:, 1])
        mlc0 = mlc[:, 0]
        assert all(x == -127.5 for x in mlc0[:29])
        assert all(x == x1 for x in mlc0[29:31])
        assert all(x == -127.5 for x in mlc0[31:60])
        assert all(x == -122.5 for x in mlc0[60 : 29 + 60])
        assert all(x == x2 for x in mlc0[29 + 60 : 31 + 60])
        assert all(x == -122.5 for x in mlc0[31 + 60 :])

        jaw_x = bld["ASYMX"]
        assert not np.any(jaw_x[:, 0] - jaw_x[:, 1])
        assert jaw_x[0, 0] == x1 - 5
        assert jaw_x[1, 0] == x2 + 5

        jaw_y = bld["ASYMY"]
        assert not np.any(jaw_y[:, 0] - jaw_y[:, 1])
        assert jaw_y[0, 0] == y1 - 5
        assert jaw_y[1, 0] == y2 + 5

    OPEN_MLC_PARAM = [
        MLCLeafBoundaryAlignmentMode.EXACT,
        MLCLeafBoundaryAlignmentMode.ROUND,
        MLCLeafBoundaryAlignmentMode.INWARD,
        MLCLeafBoundaryAlignmentMode.OUTWARD,
    ]

    @pytest.mark.parametrize("mlc_mode", OPEN_MLC_PARAM)
    def test_open_mlc(self, mlc_mode):
        x1, x2, y1, y2 = -100, 100, -110, 110
        procedure = OpenField(x1, x2, y1, y2, mlc_mode=mlc_mode)
        procedure.compute(DEFAULT_TRUEBEAM_HD120)

        bld = procedure.beams[0].beam_limiting_device_positions

        mlc = bld["MLCX"]
        assert not np.any(mlc[:, 0] - mlc[:, 1])
        mlc0 = mlc[:, 0]
        assert all(x == -100 for x in mlc0[:60])
        assert all(x == 100 for x in mlc0[60:])

    def test_defined_by_jaws(self):
        x1, x2, y1, y2 = -100, 100, -110, 110
        procedure = OpenField(x1, x2, y1, y2, defined_by_mlc=False)
        procedure.compute(DEFAULT_TRUEBEAM_HD120)

        bld = procedure.beams[0].beam_limiting_device_positions

        mlc = bld["MLCX"]
        assert not np.any(mlc[:, 0] - mlc[:, 1])
        mlc0 = mlc[:, 0]
        assert all(x == -105 for x in mlc0[:60])
        assert all(x == 105 for x in mlc0[60:])

        jaw_x = bld["ASYMX"]
        assert not np.any(jaw_x[:, 0] - jaw_x[:, 1])
        assert jaw_x[0, 0] == x1
        assert jaw_x[1, 0] == x2

        jaw_y = bld["ASYMY"]
        assert not np.any(jaw_y[:, 0] - jaw_y[:, 1])
        assert jaw_y[0, 0] == y1
        assert jaw_y[1, 0] == y2

    @pytest.mark.parametrize("x1,x2,y1,y2", [(2, 1, 0, 1), (0, 1, 2, 1)])
    def test_error_if_min_larger_than_max(self, x1, x2, y1, y2):
        procedure = OpenField(x1, x2, y1, y2, 100)
        with pytest.raises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    @pytest.mark.parametrize("y1,y2", [(-0.1, 0), (0, 0.1)])
    def test_error_if_mlc_exact_and_not_possible(self, y1, y2):
        procedure = OpenField(0, 1, y1, y2, mlc_mode=MLCLeafBoundaryAlignmentMode.EXACT)
        with pytest.raises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    @pytest.mark.parametrize("y1,y2", [(-0.1, 0), (0, 0.1)])
    def test_exact_is_ignored_if_defined_by_jaws(self, y1, y2):
        procedure = OpenField(
            0,
            1,
            y1,
            y2,
            mlc_mode=MLCLeafBoundaryAlignmentMode.EXACT,
            defined_by_mlc=False,
        )
        procedure.compute(DEFAULT_TRUEBEAM_HD120)


class TestMLCTransmission:
    def test_defaults(self):
        procedure = MLCTransmission()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        assert len(procedure.beams) == 3

    def test_banks(self):
        procedure = MLCTransmission()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)

        for beam_idx in [1, 2]:
            mult = -1 if beam_idx == 1 else 1
            slit_pos = mult * (procedure.width / 2 + procedure.overreach)
            mlc_nominal = 60 * [2 * [slit_pos - 0.5]] + 60 * [2 * [slit_pos + 0.5]]

            bld = procedure.beams[beam_idx].beam_limiting_device_positions
            assert np.all(bld["MLCX"] == mlc_nominal)
            assert np.all(bld["ASYMX"] == [[-50, -50], [50, 50]])
            assert np.all(bld["ASYMY"] == [[-50, -50], [50, 50]])

    def test_beam_names(self):
        beam_names = ["Ref", "A", "B"]
        procedure = MLCTransmission(beam_names=beam_names)
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        actual = [b.beam_name for b in procedure.beams]
        assert beam_names == actual


class TestPicketFence:
    def test_defaults(self):
        procedure = PicketFence()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        assert len(procedure.beams) == 1

    def test_replicate_varian_plan(self):
        path = [
            "plan_generator",
            "VMAT",
            "Millennium",
            "T0.2_PicketFenceStatic_M120_TB_Rev02.dcm",
        ]
        picket_fence_file = get_file_from_cloud_test_repo(path)
        ds = pydicom.dcmread(picket_fence_file)
        beam_nominal = Beam.from_dicom(ds, 0)

        procedure = PicketFence.from_varian_reference()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        beam_actual = procedure.beams[0]

        mu_nominal, mu_actual = beam_nominal.metersets, beam_actual.metersets
        assert mu_actual == pytest.approx(mu_nominal, abs=0.01)

        bls_nominal = beam_nominal.beam_limiting_device_positions
        bld_actual = beam_actual.beam_limiting_device_positions
        assert np.all(bld_actual["MLCX"] == bls_nominal["MLCX"])
        assert all(bld_actual["ASYMX"][0] == bls_nominal["ASYMX"][0])
        # Note #1: jaw x2 is different (cannot be replicated) since the left padding
        # is different than the right padding
        # Note #2: jaws y1,y2 are different (cannot be replicated) since this procedure
        # doesn't hide any leafs, whereas in the Varian plan, the last top/bottom leaves were hidden

    def test_error_if_too_wide(self):
        procedure = PicketFence(picket_positions=(-100, 100))
        with pytest.raises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)


class TestWinstonLutz:
    # There is a significant portion of these tests that are copied from OpenField
    def test_defaults(self):
        procedure = WinstonLutz()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        assert len(procedure.beams) == 1

    def test_non_defaults(self):
        fields = [WinstonLutzField(0, 0, 0, "name1"), WinstonLutzField(90.7, 0.7, 10.7)]
        names_nominal = ["name1", "G091C001T011"]
        procedure = WinstonLutz(fields=fields)
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        assert len(procedure.beams) == 2
        for f, b, n in zip(fields, procedure.beams, names_nominal):
            assert f.gantry == b.gantry_angles[0]
            assert f.collimator == b.coll_angle
            assert f.couch == b.couch_rot
            assert b.beam_name == n

    WL_MLC_PARAM = [
        (-2.5, 2.5, MLCLeafBoundaryAlignmentMode.EXACT),
        (-2.4, 2.6, MLCLeafBoundaryAlignmentMode.ROUND),
        (-2.6, 2.6, MLCLeafBoundaryAlignmentMode.INWARD),
        (-2.4, 2.4, MLCLeafBoundaryAlignmentMode.OUTWARD),
    ]

    @pytest.mark.parametrize("y1,y2,mlc_mode", WL_MLC_PARAM)
    def test_defined_by_mlc(self, y1, y2, mlc_mode):
        x1, x2 = -100, 100
        procedure = WinstonLutz(x1, x2, y1, y2, mlc_mode=mlc_mode)
        procedure.compute(DEFAULT_TRUEBEAM_HD120)

        bld = procedure.beams[0].beam_limiting_device_positions

        mlc = bld["MLCX"]
        assert not np.any(mlc[:, 0] - mlc[:, 1])
        mlc0 = mlc[:, 0]
        assert all(x == -127.5 for x in mlc0[:29])
        assert all(x == x1 for x in mlc0[29:31])
        assert all(x == -127.5 for x in mlc0[31:60])
        assert all(x == -122.5 for x in mlc0[60 : 29 + 60])
        assert all(x == x2 for x in mlc0[29 + 60 : 31 + 60])
        assert all(x == -122.5 for x in mlc0[31 + 60 :])

        jaw_x = bld["ASYMX"]
        assert not np.any(jaw_x[:, 0] - jaw_x[:, 1])
        assert jaw_x[0, 0] == x1 - 5
        assert jaw_x[1, 0] == x2 + 5

        jaw_y = bld["ASYMY"]
        assert not np.any(jaw_y[:, 0] - jaw_y[:, 1])
        assert jaw_y[0, 0] == y1 - 5
        assert jaw_y[1, 0] == y2 + 5

    def test_defined_by_jaws(self):
        x1, x2, y1, y2 = -100, 100, -110, 110
        procedure = WinstonLutz(x1, x2, y1, y2, defined_by_mlc=False)
        procedure.compute(DEFAULT_TRUEBEAM_HD120)

        bld = procedure.beams[0].beam_limiting_device_positions

        mlc = bld["MLCX"]
        assert not np.any(mlc[:, 0] - mlc[:, 1])
        mlc0 = mlc[:, 0]
        assert all(x == -105 for x in mlc0[:60])
        assert all(x == 105 for x in mlc0[60:])

        jaw_x = bld["ASYMX"]
        assert not np.any(jaw_x[:, 0] - jaw_x[:, 1])
        assert jaw_x[0, 0] == x1
        assert jaw_x[1, 0] == x2

        jaw_y = bld["ASYMY"]
        assert not np.any(jaw_y[:, 0] - jaw_y[:, 1])
        assert jaw_y[0, 0] == y1
        assert jaw_y[1, 0] == y2

    @pytest.mark.parametrize("x1,x2,y1,y2", [(2, 1, 0, 1), (0, 1, 2, 1)])
    def test_error_if_min_larger_than_max(self, x1, x2, y1, y2):
        procedure = WinstonLutz(x1, x2, y1, y2, 100)
        with pytest.raises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    @pytest.mark.parametrize("y1,y2", [(-0.1, 0), (0, 0.1)])
    def test_error_if_mlc_exact_and_not_possible(self, y1, y2):
        procedure = WinstonLutz(
            0, 1, y1, y2, mlc_mode=MLCLeafBoundaryAlignmentMode.EXACT
        )
        with pytest.raises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    @pytest.mark.parametrize("y1,y2", [(-0.1, 0), (0, 0.1)])
    def test_exact_is_ignored_if_defined_by_jaws(self, y1, y2):
        procedure = WinstonLutz(
            0,
            1,
            y1,
            y2,
            mlc_mode=MLCLeafBoundaryAlignmentMode.EXACT,
            defined_by_mlc=False,
        )
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        jaw_y = procedure.beams[0].beam_limiting_device_positions["ASYMY"]
        assert jaw_y[0, 0] == y1
        assert jaw_y[1, 0] == y2


class TestDLG:
    def test_dlg(self):
        gap_widths = (2, 4, 6)
        start_position = -50
        final_position = 50
        procedure = DosimetricLeafGap(
            gap_widths=gap_widths,
            start_position=start_position,
            final_position=final_position,
        )
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        assert len(procedure.beams) == len(gap_widths)

        for idx, gap_width in enumerate(gap_widths):
            mlc = procedure.beams[idx].beam_limiting_device_positions["MLCX"]
            assert all(mlc[:60, 0] == start_position - gap_width / 2)
            assert all(mlc[60:, 0] == start_position + gap_width / 2)
            assert all(mlc[:60, 1] == final_position - gap_width / 2)
            assert all(mlc[60:, 1] == final_position + gap_width / 2)

    def test_warning_if_bad_config(self):
        gap_widths = (20,)
        start_position = -50
        final_position = 50
        procedure = DosimetricLeafGap(
            gap_widths=gap_widths,
            start_position=start_position,
            final_position=final_position,
        )
        with pytest.warns(UserWarning):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)


class TestDoseRate:
    @pytest.fixture(autouse=True)
    def setup(self):
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
        assert len(dcm.BeamSequence) == 2
        assert dcm.BeamSequence[0].BeamName == "DR Ref"
        assert dcm.BeamSequence[1].BeamName == "DR100-600"
        assert dcm.BeamSequence[0].BeamNumber == 1
        assert dcm.FractionGroupSequence[0].NumberOfBeams == 2
        assert (
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset == 123
        )

    def test_dose_rate_too_wide(self):
        with pytest.raises(ValueError):
            procedure = DoseRate(
                dose_rates=(100, 150, 200, 250, 300, 350, 400, 600),
                roi_size_mm=30,
                y1=-10,
                y2=10,
                desired_mu=123,
                default_dose_rate=600,
            )
            self.pg.add_procedure(procedure)


class TestMlcSpeed:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.pg = PlanGenerator.from_rt_plan_file(
            TB_MIL_PLAN_FILE,
            plan_label="label",
            plan_name="my name",
        )

    def test_mlc_speed_beams(self):
        procedure = MLCSpeed(
            speeds=(0.5, 1, 1.5, 2),
            y1=-100,
            y2=100,
            mu=123,
        )
        self.pg.add_procedure(procedure)
        dcm = self.pg.as_dicom()
        assert len(dcm.BeamSequence) == 2
        assert dcm.BeamSequence[0].BeamName == "MLC Speed Ref"
        assert dcm.BeamSequence[1].BeamName == "MLC Speed"
        assert dcm.BeamSequence[0].BeamNumber == 1
        assert dcm.FractionGroupSequence[0].NumberOfBeams == 2
        assert (
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset == 123
        )
        assert dcm.BeamSequence[0].BeamType == "DYNAMIC"
        assert dcm.BeamSequence[1].BeamType == "DYNAMIC"

    def test_mlc_speed_too_fast(self):
        with pytest.raises(ValueError):
            procedure = MLCSpeed(
                speeds=(10, 20, 30, 40, 50),
                y1=-100,
                y2=100,
            )
            self.pg.add_procedure(procedure)

    def test_mlc_speed_too_wide(self):
        with pytest.raises(ValueError):
            procedure = MLCSpeed(
                speeds=(0.5, 1, 1.5, 2),
                roi_size_mm=50,
                y1=-100,
                y2=100,
            )
            self.pg.add_procedure(procedure)

    def test_0_mlc_speed(self):
        with pytest.raises(ValueError):
            procedure = MLCSpeed(
                speeds=(0, 1, 2),
                y1=-100,
                y2=100,
            )
            self.pg.add_procedure(procedure)


class TestGantrySpeed:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.pg = PlanGenerator.from_rt_plan_file(
            TB_MIL_PLAN_FILE,
            plan_label="label",
            plan_name="my name",
        )

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
        assert len(dcm.BeamSequence) == 2
        assert dcm.BeamSequence[0].BeamName == "GS"
        assert dcm.BeamSequence[1].BeamName == "GS Ref"
        assert dcm.BeamSequence[0].BeamNumber == 1
        assert dcm.FractionGroupSequence[0].NumberOfBeams == 2
        assert (
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset == 123
        )

    def test_gantry_speed_too_fast(self):
        # max speed is 6.0 by default
        procedure = GantrySpeed(
            speeds=(1, 2, 3, 4, 6.5),
            y1=-100,
            y2=100,
        )
        with pytest.raises(ValueError):
            self.pg.add_procedure(procedure)

    def test_gantry_speed_too_wide(self):
        procedure = GantrySpeed(
            speeds=(1, 2, 3, 4),
            roi_size_mm=100,
            y1=-100,
            y2=100,
        )
        with pytest.raises(ValueError):
            self.pg.add_procedure(procedure)

    def test_gantry_range_over_360(self):
        with pytest.raises(ValueError):
            procedure = GantrySpeed(
                speeds=(4, 4, 4, 4),
                y1=-100,
                y2=100,
                mu=250,
            )
            self.pg.add_procedure(procedure)


class TestVmatDRGS:
    def test_defaults(self):
        VMATDRGS().compute(DEFAULT_TRUEBEAM_HD120)

    def test_replicate_varian_plan(self):
        original_file_path = [
            "plan_generator",
            "VMAT",
            "Millennium",
            "T2_DoseRateGantrySpeed_M120_TB_Rev02.dcm",
        ]
        original_file = get_file_from_cloud_test_repo(original_file_path)
        beam_name_dyn = "T2_DR_GS"
        beam_name_ref = "OpenBeam"
        max_gantry_speed = 4.8

        # Run
        specs = DEFAULT_SPECS_TB.replace(max_gantry_speed=max_gantry_speed)
        procedure = VMATDRGS.from_varian_reference()
        procedure.compute(TrueBeamMachine(False, specs))

        # Extract data from the original plan - dynamic beam
        ds = pydicom.dcmread(original_file)
        beam_sequence = ds.BeamSequence
        beam_idx = [bs.BeamName == beam_name_dyn for bs in beam_sequence].index(True)
        plan_beam_dyn = Beam.from_dicom(ds, beam_idx)
        cumulative_meterset_1 = plan_beam_dyn.metersets
        gantry_angle_1 = plan_beam_dyn.gantry_angles
        mlc_position_1 = plan_beam_dyn.beam_limiting_device_positions["MLCX"]

        # Extract data from the conjuror plan - dynamic beam
        beam = procedure.dynamic_beam
        cumulative_meterset_2 = beam.metersets
        gantry_angle_2 = beam.gantry_angles
        mlc_position_2 = beam.beam_limiting_device_positions["MLCX"]

        # Assert dynamic beam
        assert np.allclose(cumulative_meterset_1, cumulative_meterset_2)
        assert np.allclose(gantry_angle_1, gantry_angle_2, atol=1e-2)
        assert np.allclose(mlc_position_1, mlc_position_2)

        # Extract data from the original plan - reference beam
        ds = pydicom.dcmread(original_file)
        beam_sequence = ds.BeamSequence
        beam_idx = [bs.BeamName == beam_name_ref for bs in beam_sequence].index(True)
        plan_beam_ref = Beam.from_dicom(ds, beam_idx)
        cumulative_meterset_1 = plan_beam_ref.metersets
        gantry_angle_1 = plan_beam_ref.gantry_angles
        mlc_position_1 = plan_beam_ref.beam_limiting_device_positions["MLCX"]

        # Extract data from the conjuror plan - reference beam
        beam = procedure.reference_beam
        cumulative_meterset_2 = beam.metersets
        gantry_angle_2 = beam.gantry_angles
        mlc_position_2 = beam.beam_limiting_device_positions["MLCX"]

        # Assert reference beam
        assert np.allclose(cumulative_meterset_1, cumulative_meterset_2)
        assert np.allclose(gantry_angle_1, gantry_angle_2)
        assert np.allclose(mlc_position_1, mlc_position_2)

    def test_adding_static_beams(self):
        static_angles = (0, 90, 270, 180)
        procedure = VMATDRGS(dynamic_delivery_at_static_gantry=static_angles)
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        expected_number_of_beams = 6  # dynamic, reference, 4x static
        actual_number_of_beams = len(procedure.beams)
        assert actual_number_of_beams == expected_number_of_beams

        for idx, expected_angle in enumerate(static_angles):
            dcm = procedure.beams[idx + 2].to_dicom()
            actual_angle = dcm.ControlPointSequence[0].GantryAngle
            assert actual_angle == expected_angle

    def test_error_if_gantry_speeds_and_dose_rates_have_different_sizes(self):
        gantry_speeds = (1, 2, 3)
        dose_rates = (1, 2)
        procedure = VMATDRGS(gantry_speeds=gantry_speeds, dose_rates=dose_rates)
        with pytest.raises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    def test_error_if_initial_gantry_offset_less_than_min(self):
        initial_gantry_offset = 0
        procedure = VMATDRGS(initial_gantry_offset=initial_gantry_offset)
        with pytest.raises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    def test_error_if_gantry_speeds_above_max(self):
        gantry_speeds = (1, 2, 5)
        dose_rates = (1, 2, 3)
        max_gantry_speed = 4
        specs = DEFAULT_SPECS_TB.replace(max_gantry_speed=max_gantry_speed)
        procedure = VMATDRGS(gantry_speeds=gantry_speeds, dose_rates=dose_rates)
        with pytest.raises(ValueError):
            procedure.compute(TrueBeamMachine(False, specs))

    def test_error_if_dose_rates_above_max(self):
        gantry_speeds = (1, 2, 5)
        dose_rates = (1, 2, 300)
        max_dose_rate = 100
        procedure = VMATDRGS(
            gantry_speeds=gantry_speeds,
            dose_rates=dose_rates,
            max_dose_rate=max_dose_rate,
        )
        with pytest.raises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    def test_error_if_axis_not_maxed_out(self):
        gantry_speeds = (5, 1, 1)
        dose_rates = (1, 300, 1)
        max_gantry_speed = 5
        max_dose_rate = 300
        specs = DEFAULT_SPECS_TB.replace(max_gantry_speed=max_gantry_speed)
        procedure = VMATDRGS(
            gantry_speeds=gantry_speeds,
            dose_rates=dose_rates,
            max_dose_rate=max_dose_rate,
        )
        with pytest.raises(ValueError):
            procedure.compute(TrueBeamMachine(False, specs))

    def test_error_if_rotation_larger_than_360(self):
        mu_per_segment = 100
        procedure = VMATDRGS(mu_per_segment=mu_per_segment)
        with pytest.raises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    @pytest.mark.skipif(not RUN_PLOT_TESTS, reason="skip plot test")
    def test_plot_control_points(self):
        procedure = VMATDRGS()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        procedure.plot_control_points()

    @pytest.mark.skipif(not RUN_PLOT_TESTS, reason="skip plot test")
    def test_plot_fluence(self):
        procedure = VMATDRGS()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        procedure.plot_fluence(IMAGER_AS1200)

    @pytest.mark.skipif(not RUN_PLOT_TESTS, reason="skip plot test")
    def test_plot_profile(self):
        procedure = VMATDRGS()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        procedure.plot_fluence_profile(IMAGER_AS1200)

    @pytest.mark.skipif(not RUN_PLOT_TESTS, reason="skip plot test")
    def test_animate_mlc(self):
        procedure = VMATDRGS()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        procedure.dynamic_beam.animate_mlc()


class TestVmatDRMLC:
    def test_defaults(self):
        procedure = VMATDRMLC()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)

    def test_replicate_varian_plan(self):
        original_file_path = [
            "plan_generator",
            "VMAT",
            "Millennium",
            "T3_MLCSpeed_M120_TB_Rev02.dcm",
        ]
        original_file = get_file_from_cloud_test_repo(original_file_path)
        beam_name_dyn = "T3MLCSpeed"
        beam_name_ref = "OpenBeam"
        max_gantry_speed = 4.8

        # Run
        specs = DEFAULT_SPECS_TB.replace(max_gantry_speed=max_gantry_speed)
        procedure = VMATDRMLC.from_varian_reference()
        procedure.compute(TrueBeamMachine(False, specs))

        # Extract data from the original plan - dynamic beam
        ds = pydicom.dcmread(original_file)
        beam_sequence = ds.BeamSequence
        beam_idx = [bs.BeamName == beam_name_dyn for bs in beam_sequence].index(True)
        plan_beam_dyn = Beam.from_dicom(ds, beam_idx)
        cumulative_meterset_1 = plan_beam_dyn.metersets
        gantry_angle_1 = plan_beam_dyn.gantry_angles
        mlc_position_1 = plan_beam_dyn.beam_limiting_device_positions["MLCX"]

        # Extract data from the conjuror plan - dynamic beam
        beam = procedure.dynamic_beam
        cumulative_meterset_2 = beam.metersets
        gantry_angle_2 = beam.gantry_angles
        mlc_position_2 = beam.beam_limiting_device_positions["MLCX"]

        # Assert dynamic beam
        assert np.allclose(cumulative_meterset_1, cumulative_meterset_2)
        assert np.allclose(gantry_angle_1, gantry_angle_2, atol=1e-3)
        assert np.allclose(mlc_position_1, mlc_position_2)

        # Extract data from the original plan - reference beam
        ds = pydicom.dcmread(original_file)
        beam_sequence = ds.BeamSequence
        beam_idx = [bs.BeamName == beam_name_ref for bs in beam_sequence].index(True)
        plan_beam_ref = Beam.from_dicom(ds, beam_idx)
        cumulative_meterset_1 = plan_beam_ref.metersets
        gantry_angle_1 = plan_beam_ref.gantry_angles
        mlc_position_1 = plan_beam_ref.beam_limiting_device_positions["MLCX"]

        # Extract data from the conjuror plan - reference beam
        beam = procedure.reference_beam
        cumulative_meterset_2 = beam.metersets
        gantry_angle_2 = beam.gantry_angles
        mlc_position_2 = beam.beam_limiting_device_positions["MLCX"]

        # Assert reference beam
        assert np.allclose(cumulative_meterset_1, cumulative_meterset_2)
        assert np.allclose(gantry_angle_1, gantry_angle_2, atol=1e-3)
        assert np.allclose(mlc_position_1, mlc_position_2)

    def test_error_if_initial_gantry_offset_less_than_min(self):
        initial_gantry_offset = 0
        procedure = VMATDRMLC(initial_gantry_offset=initial_gantry_offset)
        with pytest.raises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    def test_error_if_mlc_speeds_above_max(self):
        mlc_speeds = (10.0, 60.0)
        max_mlc_speed = 50.0
        specs = DEFAULT_SPECS_TB.replace(max_mlc_speed=max_mlc_speed)
        procedure = VMATDRMLC(mlc_speeds=mlc_speeds)
        with pytest.raises(ValueError):
            procedure.compute(TrueBeamMachine(False, specs))

    def test_error_if_gantry_speeds_above_max(self):
        max_gantry_speed = 10.0
        mlc_speeds = (10.0, 20.0)
        gantry_speeds = (max_gantry_speed, max_gantry_speed + 1)
        specs = DEFAULT_SPECS_TB.replace(max_gantry_speed=max_gantry_speed)
        procedure = VMATDRMLC(mlc_speeds=mlc_speeds, gantry_speeds=gantry_speeds)
        with pytest.raises(ValueError):
            procedure.compute(TrueBeamMachine(False, specs))

    def test_error_if_axis_not_maxed_out(self):
        mlc_speeds = (10.0, 20.0)
        gantry_speeds = (5.0, 6.0)
        procedure = VMATDRMLC(mlc_speeds=mlc_speeds, gantry_speeds=gantry_speeds)
        with pytest.raises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    def test_error_if_rotation_larger_than_360(self):
        mlc_speeds = (1.0, 20.0)
        procedure = VMATDRMLC(mlc_speeds=mlc_speeds)
        with pytest.raises(ValueError):
            procedure.compute(DEFAULT_TRUEBEAM_HD120)

    @pytest.mark.skipif(not RUN_PLOT_TESTS, reason="skip plot test")
    def test_plot_control_points(self):
        procedure = VMATDRMLC()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        procedure.plot_control_points()

    @pytest.mark.skipif(not RUN_PLOT_TESTS, reason="skip plot test")
    def test_plot_fluence(self):
        procedure = VMATDRMLC()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        procedure.plot_fluence(IMAGER_AS1200)

    @pytest.mark.skipif(not RUN_PLOT_TESTS, reason="skip plot test")
    def test_plot_profile(self):
        procedure = VMATDRMLC()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        procedure.plot_fluence_profile(IMAGER_AS1200)

    @pytest.mark.skipif(not RUN_PLOT_TESTS, reason="skip plot test")
    def test_animate_mlc(self):
        procedure = VMATDRMLC()
        procedure.compute(DEFAULT_TRUEBEAM_HD120)
        procedure.dynamic_beam.animate_mlc()
