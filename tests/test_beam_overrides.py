import inspect
import unittest
from typing import get_args, Any

from parameterized import parameterized

from conjuror.plans import halcyon, truebeam
from conjuror.plans.beam_overrides import (
    BEAM_OVERRIDE_VALIDATORS,
    BeamOverrideTag,
)
from conjuror.plans.plan_generator import QAProcedureBase


def _concrete_qa_procedure_classes() -> list[type[QAProcedureBase]]:
    classes: list[type[QAProcedureBase]] = []
    for module in (truebeam, halcyon):
        for _name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, QAProcedureBase) and not inspect.isabstract(cls):
                classes.append(cls)
    return classes


def test_literal_tags_match_validator_keys():
    literal_tags = set(get_args(BeamOverrideTag))
    validator_tags = set(BEAM_OVERRIDE_VALIDATORS)
    assert literal_tags == validator_tags


class TestBeamOverrideProcedureAllowLists(unittest.TestCase):
    @parameterized.expand(_concrete_qa_procedure_classes())
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
