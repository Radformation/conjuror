from collections.abc import Callable
from functools import partial
from typing import Any, Literal, TypeAlias

BeamOverrideTag: TypeAlias = Literal[
    "BeamName",
    "ControlPointSequence[0].PatientSupportAngle",
]

# Map from the beam number, to the overridden value. Beam numbers are 0-indexed,
# and not reflective of the `BeamNumber` DICOM tag on the underlying plan
# Example: {0: 0, 1: 90, 2: 180, 3: 270}
BeamOverride: TypeAlias = dict[int, Any]

# Map from the tag name to the overridden value for each beam number.
# Example: {"GantryAngle": {0: 0, 1: 90, 2: 180, 3: 270}}
BeamOverrides: TypeAlias = dict[BeamOverrideTag, BeamOverride]


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


def validate_overrides(
    overrides: BeamOverrides, valid_tags: frozenset[str], max_beams: int
) -> None:
    """Validate that the overrides are in the correct format and that the tags are valid."""
    for tag, beam_override in overrides.items():
        if tag not in valid_tags:
            raise ValueError(
                f"Invalid tag '{tag}' in overrides. Valid tags are: {valid_tags}"
            )
        if not isinstance(beam_override, dict):
            raise ValueError(
                f"Beam override for tag '{tag}' must be a dictionary of beam number to value."
            )
        for beam_num, value in beam_override.items():
            if not isinstance(beam_num, int):
                raise ValueError(
                    f"Beam number '{beam_num}' in overrides for tag '{tag}' must be an integer."
                )
            if beam_num < 0 or beam_num >= max_beams:
                raise ValueError(
                    f"Beam number '{beam_num}' in overrides for tag '{tag}' must be between 0 and {max_beams - 1}."
                )
            # Validate the value using the appropriate validator for the tag
            validator = BEAM_OVERRIDE_VALIDATORS.get(tag)
            if not validator:
                raise ValueError(f"No validator found for tag '{tag}'")

            try:
                validator(value)
            except Exception as e:
                raise ValueError(
                    f"Invalid value for tag '{tag}' on beam number {beam_num}: {e}"
                ) from e
