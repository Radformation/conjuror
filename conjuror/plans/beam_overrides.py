from collections.abc import Callable
from functools import partial
from typing import Any, Literal, TypeAlias

BeamOverrideTag: TypeAlias = Literal[
    "BeamName",
    "ControlPointSequence[0].PatientSupportAngle",
]


def validate_long_string(
    value: Any, field_label: str = "LO", max_length: int = 64, required=True
) -> None:
    """Validate a value for DICOM Long String (LO) semantics."""
    if not isinstance(value, str):
        raise TypeError(f"{field_label} must be str")
    if len(value) > max_length:
        raise ValueError(f"{field_label} exceeds max length {max_length}")
    if required and len(value) == 0:
        raise ValueError(f"{field_label} is required and cannot be empty")


def validate_angle(
    value: Any, field_label: str, min_deg: float, max_deg: float
) -> None:
    """Validate a numeric angle in degrees."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_label} must be int or float")

    if not (min_deg <= value <= max_deg):
        raise ValueError(f"{field_label} must be in [{min_deg}, {max_deg}] degrees")


BEAM_OVERRIDE_VALIDATORS: dict[BeamOverrideTag, Callable[[Any], None]] = {
    "BeamName": partial(validate_long_string, field_label="BeamName"),
    "ControlPointSequence[0].PatientSupportAngle": partial(
        validate_angle, field_label="PatientSupportAngle", min_deg=-360.0, max_deg=360.0
    ),
}
