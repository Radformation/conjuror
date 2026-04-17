import re
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, TypeAlias

from pydicom import config
from pydicom.datadict import dictionary_VR, tag_for_keyword
from pydicom.dataelem import DataElement
from pydicom.sequence import Sequence as DicomSequence

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


def validate_dicom_keyword_value(keyword: str, value: Any) -> None:
    """Validate *value* against the VR of the given DICOM keyword.

    Looks up the tag and VR via pydicom's data dictionary, then constructs
    a DataElement to run pydicom's full VR-specific validation and type
    coercion.
    """
    tag = tag_for_keyword(keyword)
    if tag is None:
        raise ValueError(f"{keyword}: Unknown DICOM keyword")
    vr = dictionary_VR(tag)
    try:
        DataElement(tag, vr, value, validation_mode=config.RAISE)
    except Exception as e:
        raise ValueError(f"{keyword}: {str(e)}") from e


def validate_long_string(
    value: Any, keyword: str = "BeamName", required: bool = True
) -> None:
    """Validate a value for DICOM Long String (LO) semantics."""
    validate_dicom_keyword_value(keyword, value)
    if required and (value is None or len(str(value)) == 0):
        raise ValueError(f"{keyword}: is required and cannot be empty")


def validate_angle(value: Any, keyword: str, min_deg: float, max_deg: float) -> None:
    """Validate a numeric angle in degrees."""
    validate_dicom_keyword_value(keyword, value)
    num = float(value)
    if not (min_deg <= num <= max_deg):
        raise ValueError(f"{keyword}: must be in [{min_deg}, {max_deg}] degrees")


BEAM_OVERRIDE_VALIDATORS: dict[BeamOverrideTag, Callable[[Any], None]] = {
    "BeamName": partial(validate_long_string, keyword="BeamName"),
    "ControlPointSequence[0].PatientSupportAngle": partial(
        validate_angle, keyword="PatientSupportAngle", min_deg=-360.0, max_deg=360.0
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


# DICOM keywords are letters only (PascalCase); optional [n] for SQ indexing.
DICOM_TAG_RE = re.compile(r"^([A-Z][A-Za-z]*)(?:\[(\d+)\])?$")


def parse_dicom_path_segment(segment: str) -> tuple[str, int | None]:
    m = DICOM_TAG_RE.fullmatch(segment.strip())
    if not m:
        raise ValueError(f"Invalid DICOM path segment: {segment!r}")
    name, idx_s = m.group(1), m.group(2)
    return name, int(idx_s) if idx_s is not None else None


def apply_beam_override(
    beam_sequence: DicomSequence,
    beam_index: int,
    dicom_path: str,
    value: Any,
) -> None:
    """Set ``value`` on one beam item using a dotted path (``BeamName`` or ``Seq[0].Leaf``).

    Preconditions:
    - Path and value should already be validated
    - Each keyword along the path must exist on the current dataset
    - Sequence segments must use ``Name[index]`` syntax.
    """
    if beam_index < 0 or beam_index >= len(beam_sequence):
        raise IndexError(f"beam_index {beam_index} out of range for BeamSequence")

    current = beam_sequence[beam_index]
    parts = dicom_path.split(".")
    for i, segment in enumerate(parts):
        name, idx = parse_dicom_path_segment(segment)
        is_last = i == len(parts) - 1
        if is_last:
            # Update Tag if we're at the leaf node
            if idx is not None:
                raise ValueError(
                    f"Final path segment must not use indexing: {segment} in {dicom_path}"
                )
            if name not in current:
                raise KeyError(
                    f"Beam {beam_index}: keyword {name} not present for path {dicom_path}"
                )
            setattr(current, name, value)
            return

        # To iterate into a sequence, we must have an index.
        if idx is None:
            raise ValueError(
                f"Beam {beam_index}: segment {segment} in {dicom_path} must use "
                f"indexing (e.g. {name}[0]) before further components"
            )
        if name not in current:
            raise KeyError(
                f"Beam {beam_index}: keyword {name} not present for path {dicom_path}"
            )
        elem = current[name]
        if elem.VR != "SQ":
            raise ValueError(
                f"Beam {beam_index}: {name} is not a sequence (VR={elem.VR}) in {dicom_path}"
            )
        sq: DicomSequence = elem.value
        if idx < 0 or idx >= len(sq):
            raise IndexError(
                f"Beam {beam_index}: index [{idx}] out of range for {name} "
                f"(len={len(sq)}) in {dicom_path}"
            )
        current = sq[idx]


def apply_beam_overrides(
    beam_sequence: DicomSequence,
    overrides: BeamOverrides,
    beam_start: int,
) -> None:
    """Apply all overrides; ``beam_start`` is the BeamSequence index of procedure-local beam 0."""
    for tag, per_beam in overrides.items():
        for local_idx, val in per_beam.items():
            apply_beam_override(beam_sequence, beam_start + local_idx, tag, val)
