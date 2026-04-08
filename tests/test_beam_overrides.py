import inspect
import unittest
from typing import Any, get_args

from parameterized import parameterized

from conjuror.plans import halcyon, truebeam
from conjuror.plans.beam_overrides import (
    BEAM_OVERRIDE_VALIDATORS,
    BeamOverrideTag,
)
from conjuror.plans.plan_generator import PlanGenerator, QAProcedureBase
from tests.utils import get_file_from_cloud_test_repo

TB_MIL_PLAN_FILE = get_file_from_cloud_test_repo(["plan_generator", "Murray-plan.dcm"])
HAL_PLAN_FILE = get_file_from_cloud_test_repo(["plan_generator", "Halcyon Prox.dcm"])

PARAMS_OVERRIDE: dict[type[QAProcedureBase], dict[str, Any]] = {
    truebeam.OpenField: {"x1": -5.0, "x2": 5.0, "y1": -5.0, "y2": 5.0},
    halcyon.PicketFence: {"stack": halcyon.Stack.PROXIMAL},
}


PROCEDURE_CLASSES_TRUEBEAM = [
    cls
    for _name, cls in inspect.getmembers(truebeam, inspect.isclass)
    if issubclass(cls, QAProcedureBase) and not inspect.isabstract(cls)
]

PROCEDURE_CLASSES_HALCYON = [
    cls
    for _name, cls in inspect.getmembers(halcyon, inspect.isclass)
    if issubclass(cls, QAProcedureBase) and not inspect.isabstract(cls)
]

PROCEDURE_CLASSES = PROCEDURE_CLASSES_TRUEBEAM + PROCEDURE_CLASSES_HALCYON


VALID_OVERRIDES = {
    "BeamName": "ValidName",
    "ControlPointSequence[0].PatientSupportAngle": 45.0,
}

INVALID_OVERRIDES = {
    "BeamName": "",
    "ControlPointSequence[0].PatientSupportAngle": 400.0,
}


def test_literal_tags_match_validator_keys():
    literal_tags = set(get_args(BeamOverrideTag))
    validator_tags = set(BEAM_OVERRIDE_VALIDATORS)
    assert literal_tags == validator_tags


class TestBeamOverrideProcedureAllowLists(unittest.TestCase):
    @parameterized.expand(PROCEDURE_CLASSES)
    def test_allow_list_tags_have_validators(self, cls: type[QAProcedureBase]) -> None:
        assert set(cls.BEAM_OVERRIDE_ALLOW_LIST).issubset(set(BEAM_OVERRIDE_VALIDATORS))


class TestBeamNameValidator(unittest.TestCase):
    validator = BEAM_OVERRIDE_VALIDATORS["BeamName"]

    @parameterized.expand(["Ref", "a" * 64])
    def test_valid(self, value: str):
        assert self.validator(value) is None

    @parameterized.expand(
        [
            ("", "is required"),
            (1, "must be str"),
            ("a" * 65, "exceeds max length 64"),
        ]
    )
    def test_invalid(self, value: Any, message_substring: str):
        with self.assertRaises(Exception) as ctx:
            self.validator(value)

        assert message_substring in str(ctx.exception)


class TestPatientSupportAngleValidator(unittest.TestCase):
    validator = BEAM_OVERRIDE_VALIDATORS["ControlPointSequence[0].PatientSupportAngle"]

    @parameterized.expand([0, -180.5, 360.0])
    def test_valid(self, value: float):
        assert self.validator(value) is None

    @parameterized.expand(
        [
            ("90", "must be int or float"),
            (400.0, "must be in [-360.0, 360.0]"),
            (-400.0, "must be in [-360.0, 360.0]"),
        ]
    )
    def test_invalid(self, value: Any, message_substring: str):
        with self.assertRaises(Exception) as ctx:
            self.validator(value)

        assert message_substring in str(ctx.exception)


def parameterized_func_name(func, param_num, param):
    return f"{func.__name__}_{param.args[0].__name__}_{param.args[1]}"


class TestAddProcedureBeamOverride(unittest.TestCase):
    @parameterized.expand(
        [
            (_procedure, _tag)
            for _procedure in PROCEDURE_CLASSES_TRUEBEAM
            for _tag in _procedure.BEAM_OVERRIDE_ALLOW_LIST
        ],
        name_func=parameterized_func_name,
    )
    def test_valid_override_truebeam(
        self, cls: type[QAProcedureBase], tag: BeamOverrideTag
    ) -> None:
        pg = PlanGenerator.from_rt_plan_file(
            TB_MIL_PLAN_FILE, plan_label="label", plan_name="name"
        )
        kwargs = PARAMS_OVERRIDE.get(cls, {})
        proc = cls(**kwargs)
        before = len(pg.ds.BeamSequence)
        pg.add_procedure(proc, beam_overrides={tag: {0: VALID_OVERRIDES[tag]}})

        # Just check that the override was valid and the beams were added
        assert len(pg.ds.BeamSequence) == before + len(proc.beams)

    @parameterized.expand(
        [
            (_procedure, _tag)
            for _procedure in PROCEDURE_CLASSES_TRUEBEAM
            for _tag in _procedure.BEAM_OVERRIDE_ALLOW_LIST
        ],
        name_func=parameterized_func_name,
    )
    def test_invalid_override_truebeam(
        self, cls: type[QAProcedureBase], tag: BeamOverrideTag
    ) -> None:
        pg = PlanGenerator.from_rt_plan_file(
            TB_MIL_PLAN_FILE, plan_label="label", plan_name="name"
        )
        kwargs = PARAMS_OVERRIDE.get(cls, {})
        proc = cls(**kwargs)
        before = len(pg.ds.BeamSequence)
        with self.assertRaises(ValueError):
            pg.add_procedure(proc, beam_overrides={tag: {0: INVALID_OVERRIDES[tag]}})

        # No beams added to plan
        assert len(pg.ds.BeamSequence) == before

    @parameterized.expand(
        [
            (_procedure, _tag)
            for _procedure in PROCEDURE_CLASSES_HALCYON
            for _tag in _procedure.BEAM_OVERRIDE_ALLOW_LIST
        ],
        name_func=parameterized_func_name,
    )
    def test_valid_override_halcyon(
        self, cls: type[QAProcedureBase], tag: BeamOverrideTag
    ) -> None:
        pg = PlanGenerator.from_rt_plan_file(
            HAL_PLAN_FILE, plan_label="label", plan_name="name"
        )
        kwargs = PARAMS_OVERRIDE.get(cls, {})
        proc = cls(**kwargs)
        before = len(pg.ds.BeamSequence)
        pg.add_procedure(proc, beam_overrides={tag: {0: VALID_OVERRIDES[tag]}})

        # Just check that the override was valid and the beams were added
        assert len(pg.ds.BeamSequence) == before + len(proc.beams)

    @parameterized.expand(
        [
            (_procedure, _tag)
            for _procedure in PROCEDURE_CLASSES_HALCYON
            for _tag in _procedure.BEAM_OVERRIDE_ALLOW_LIST
        ],
        name_func=parameterized_func_name,
    )
    def test_invalid_override_halcyon(
        self, cls: type[QAProcedureBase], tag: BeamOverrideTag
    ) -> None:
        pg = PlanGenerator.from_rt_plan_file(
            HAL_PLAN_FILE, plan_label="label", plan_name="name"
        )
        kwargs = PARAMS_OVERRIDE.get(cls, {})
        proc = cls(**kwargs)
        before = len(pg.ds.BeamSequence)
        with self.assertRaises(ValueError):
            pg.add_procedure(proc, beam_overrides={tag: {0: INVALID_OVERRIDES[tag]}})

        # No beams added to plan
        assert len(pg.ds.BeamSequence) == before
