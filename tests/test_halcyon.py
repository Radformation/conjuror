import pytest

from conjuror.plans.halcyon import Stack, PicketFence
from conjuror.plans.plan_generator import PlanGenerator
from tests.utils import get_file_from_cloud_test_repo

HAL_PLAN_FILE = get_file_from_cloud_test_repo(["plan_generator", "Halcyon Prox.dcm"])

HALCYON_MLC_INDEX = {
    Stack.DISTAL: -2,
    Stack.PROXIMAL: -1,
}


class TestHalcyonPrefabs:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.pg = PlanGenerator.from_rt_plan_file(
            HAL_PLAN_FILE,
            plan_label="label",
            plan_name="my name",
        )

    def test_create_picket_fence_proximal(self):
        procedure = PicketFence(
            stack=Stack.PROXIMAL,
            mu=123,
            beam_name="Picket Fence",
            strip_positions_mm=(-50, -30, -10, 10, 30, 50),
        )
        self.pg.add_procedure(procedure)
        dcm = self.pg.as_dicom()
        assert len(dcm.BeamSequence) == 1
        assert dcm.BeamSequence[0].BeamName == "Picket Fence"
        assert dcm.BeamSequence[0].BeamNumber == 1
        assert dcm.FractionGroupSequence[0].NumberOfBeams == 1
        assert (
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset == 123
        )
        # check first CP of proximal is at the PF position
        assert (
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[HALCYON_MLC_INDEX[Stack.PROXIMAL]]
            .LeafJawPositions[0]
            == -53.5
        )
        # distal should be parked
        assert (
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[HALCYON_MLC_INDEX[Stack.DISTAL]]
            .LeafJawPositions[0]
            == -140
        )
        assert dcm.BeamSequence[0].BeamType == "DYNAMIC"

    def test_create_picket_fence_distal(self):
        procedure = PicketFence(
            stack=Stack.DISTAL,
            mu=123,
            beam_name="Picket Fence",
            strip_positions_mm=(-50, -30, -10, 10, 30, 50),
        )
        self.pg.add_procedure(procedure)
        dcm = self.pg.as_dicom()
        assert len(dcm.BeamSequence) == 1
        assert dcm.BeamSequence[0].BeamName == "Picket Fence"
        assert dcm.BeamSequence[0].BeamNumber == 1
        assert dcm.FractionGroupSequence[0].NumberOfBeams == 1
        assert (
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset == 123
        )
        # check first CP of proximal is parked
        assert (
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[HALCYON_MLC_INDEX[Stack.PROXIMAL]]
            .LeafJawPositions[0]
            == -140
        )
        # distal should be at picket position
        assert (
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[HALCYON_MLC_INDEX[Stack.DISTAL]]
            .LeafJawPositions[0]
            == -53.5
        )
        assert dcm.BeamSequence[0].BeamType == "DYNAMIC"

    def test_create_picket_fence_both(self):
        procedure = PicketFence(
            stack=Stack.BOTH,
            mu=123,
            beam_name="Picket Fence",
            strip_positions_mm=(-50, -30, -10, 10, 30, 50),
        )
        self.pg.add_procedure(procedure)
        dcm = self.pg.as_dicom()
        assert len(dcm.BeamSequence) == 1
        assert dcm.BeamSequence[0].BeamName == "Picket Fence"
        assert dcm.BeamSequence[0].BeamNumber == 1
        assert dcm.FractionGroupSequence[0].NumberOfBeams == 1
        assert (
            dcm.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset == 123
        )
        # check first CP of proximal is at the PF position
        assert (
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[HALCYON_MLC_INDEX[Stack.PROXIMAL]]
            .LeafJawPositions[0]
            == -53.5
        )
        # distal should be at picket position
        assert (
            dcm.BeamSequence[0]
            .ControlPointSequence[0]
            .BeamLimitingDevicePositionSequence[HALCYON_MLC_INDEX[Stack.DISTAL]]
            .LeafJawPositions[0]
            == -53.5
        )
        assert dcm.BeamSequence[0].BeamType == "DYNAMIC"
