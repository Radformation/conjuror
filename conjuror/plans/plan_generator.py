import datetime
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Self

import numpy as np
import pydicom
from matplotlib.figure import Figure
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import generate_uid

from ..images.layers import ArrayLayer
from ..images.simulators import Simulator, Imager
from ..utils import wrap180
from .fluence import generate_fluences, plot_fluences


# MLC boundaries are immutable by design, so they are stored as tuples.
# However, pydicom expects lists, so these are converted to lists when necessary (Beam).
MLC_BOUNDARIES_TB_MIL120 = (
    tuple(np.arange(-200, -100 + 1, 10).astype(float))
    + tuple(np.arange(-95, 95 + 1, 5).astype(float))
    + tuple(np.arange(100, 200 + 1, 10).astype(float))
)
MLC_BOUNDARIES_TB_HD120 = (
    tuple(np.arange(-110, -40 + 1, 5).astype(float))
    + tuple(np.arange(-37.5, 37.5 + 1, 2.5).astype(float))
    + tuple(np.arange(40, 110 + 1, 5).astype(float))
)
MLC_BOUNDARIES_HAL_DIST = tuple(np.arange(-140, 140 + 1, 10).astype(float))
MLC_BOUNDARIES_HAL_PROX = tuple(np.arange(-145, 145 + 1, 10).astype(float))


@dataclass(frozen=True)
class MachineSpecs:
    """
    This class is a dataclass holding machine specs

    Parameters
    ----------
    max_gantry_speed : float
        The maximum gantry speed in deg/sec
    max_mlc_position : float
        The max mlc position in mm
    max_mlc_overtravel : float
        The maximum distance in mm the MLC leaves can overtravel from each other as well as the jaw size (for tail exposure protection).
    max_mlc_speed : float
        The maximum speed of the MLC leaves in mm/s.
    """

    max_gantry_speed: float
    max_mlc_position: float
    max_mlc_overtravel: float
    max_mlc_speed: float

    def replace(self, **overrides) -> Self:
        return replace(self, **overrides)


DEFAULT_SPECS_TB = MachineSpecs(
    max_gantry_speed=6.0, max_mlc_position=200, max_mlc_overtravel=140, max_mlc_speed=25
)

DEFAULT_SPECS_HAL = MachineSpecs(
    max_gantry_speed=24.0, max_mlc_position=140, max_mlc_overtravel=140, max_mlc_speed=25
)


@dataclass
class TrueBeamMachine:
    mlc_is_hd: bool
    machine_specs: MachineSpecs = DEFAULT_SPECS_TB

    @property
    def mlc_boundaries(self) -> tuple[float,...]:
        return MLC_BOUNDARIES_TB_HD120 if self.mlc_is_hd else MLC_BOUNDARIES_TB_MIL120


@dataclass
class HalcyonMachine:
    machine_specs: MachineSpecs = DEFAULT_SPECS_HAL
    mlc_boundaries_dist = MLC_BOUNDARIES_HAL_DIST
    mlc_boundaries_prox = MLC_BOUNDARIES_HAL_PROX


class GantryDirection(Enum):
    CLOCKWISE = "CW"
    COUNTER_CLOCKWISE = "CC"
    NONE = "NONE"


class GantrySpeedTransition(Enum):
    LEADING = "leading"
    TRAILING = "trailing"


class FluenceMode(Enum):
    STANDARD = "STANDARD"
    FFF = "FFF"
    SRS = "SRS"


class Beam(ABC):
    """Represents a DICOM beam dataset. Has methods for creating the dataset and adding control points.
    Generally not created on its own but rather under the hood as part of a PlanGenerator object.

    It contains enough independent logic steps that it's worth separating out from the PlanGenerator class.
    """

    ROUNDING_DECIMALS = 6

    def __init__(
        self,
        beam_limiting_device_sequence: DicomSequence,
        beam_name: str,
        energy: float,
        fluence_mode: FluenceMode,
        dose_rate: int,
        metersets: Sequence[float],
        gantry_angles: float | Sequence[float],
        coll_angle: float,
        beam_limiting_device_positions: dict[str, list],
        couch_vrt: float,
        couch_lat: float,
        couch_lng: float,
        couch_rot: float,
    ):
        """
        Parameters
        ----------
        beam_limiting_device_sequence : DicomSequence
            The beam_limiting_device_sequence as defined in the template plan.
        beam_name : str
            The name of the beam. Must be less than 16 characters.
        energy : float
            The energy of the beam.
        fluence_mode : FluenceMode
            The fluence mode of the beam.
        dose_rate : int
            The dose rate of the beam.
        metersets : Sequence[float]
            The meter sets for each control point.
        gantry_angles : Union[float, Sequence[float]]
            The gantry angle(s) of the beam. If a single number, it's assumed to be a static beam. If multiple numbers, it's assumed to be a dynamic beam.
        coll_angle : float
            The collimator angle.
        beam_limiting_device_positions : dict[str, list]
            The positions of the beam_limiting_device_positions for each control point,
            where key is the type of beam limiting device (e.g. "MLCX") and the value contains the positions.
        couch_vrt : float
            The couch vertical position.
        couch_lat : float
            The couch lateral position.
        couch_lng : float
            The couch longitudinal position.
        couch_rot : float
            The couch rotation.
        """

        if len(beam_name) > 16:
            raise ValueError("Beam name must be less than or equal to 16 characters")

        # Private attributes used for dicom creation only
        self._beam_name = beam_name
        self._fluence_mode = fluence_mode
        self._beam_limiting_device_sequence = beam_limiting_device_sequence
        self._energy = energy
        self._dose_rate = dose_rate
        self._coll_angle = coll_angle
        self._couch_vrt = couch_vrt
        self._couch_lat = couch_lat
        self._couch_lng = couch_lng
        self._couch_rot = couch_rot

        # Public attributes (used outside dicom scope, e.g. for plotting)
        # For easier manipulation all variable are stored as np.ndarray of size num_cp,
        # if the axis are static they are replicated to fit the array.
        self.beam_meterset = np.round(metersets[-1], self.ROUNDING_DECIMALS)
        self.number_of_control_points = len(metersets)
        if not isinstance(gantry_angles, Iterable):
            gantry_angles = [gantry_angles] * self.number_of_control_points
        self.metersets = np.array(metersets)
        self.gantry_angles = np.array(gantry_angles)
        self.beam_limiting_device_positions = dict()
        for key, positions in beam_limiting_device_positions.items():
            if len(positions) == 1:
                bld = np.array(self.number_of_control_points * positions)
            else:
                bld = np.array(positions)
            self.beam_limiting_device_positions[key] = bld


    def to_dicom(self) -> Dataset:
        """Return the beam as a DICOM dataset that represents a BeamSequence item."""

        # The Meterset at a given Control Point is equal to Beam Meterset (300A,0086)
        # specified in the Referenced Beam Sequence (300C,0004) of the RT Fraction Scheme Module,
        # multiplied by the Cumulative Meterset Weight (300A,0134) for the Control Point,
        # divided by the Final Cumulative Meterset Weight (300A,010E)
        # https://dicom.innolitics.com/ciods/rt-plan/rt-beams/300a00b0/300a0111/300a0134
        metersets_weights = np.array(self.metersets) / self.metersets[-1]

        # Round all possible dynamic elements  to avoid floating point comparisons.
        # E.g. to evaluate is an axis is static, all elements should be equal to the first
        # Note: using np.isclose does not solve the problem since the tolerance should be the same
        # as Eclipse/Machine, and we don't know which tolerance they use.
        # Here we assume that their tolerance is tighter than ROUNDING_DECIMALS
        metersets_weights = np.round(metersets_weights, self.ROUNDING_DECIMALS)
        gantry_angles = np.round(self.gantry_angles, self.ROUNDING_DECIMALS)
        bld_positions = {
            k: np.round(v, self.ROUNDING_DECIMALS)
            for k, v in self.beam_limiting_device_positions.items()
        }

        # Infer gantry rotation from the gantry angles
        # It assumes the gantry cannot rotate over 180, so there is only one possible direction to go from A to B.
        ga_wrap180 = wrap180(np.array(gantry_angles))
        # This dictionary is used for mapping the sign of the difference with the GantryDirection enum.
        gantry_direction_map = {
            0: GantryDirection.NONE,
            1: GantryDirection.CLOCKWISE,
            -1: GantryDirection.COUNTER_CLOCKWISE,
        }
        gantry_direction = [
            gantry_direction_map[s] for s in np.sign(np.diff(ga_wrap180))
        ]
        # The last GantryRotationDirection should always be 'NONE'
        gantry_direction += [GantryDirection.NONE]

        # Infer if a beam is static or dynamic from the control points
        gantry_is_static = len(set(gantry_direction)) == 1
        dict_bld_is_static = {
            k: np.all(pos == pos[0]) for k, pos in bld_positions.items()
        }
        blds_are_static = np.all(list(dict_bld_is_static.values()))
        beam_is_static = gantry_is_static and blds_are_static
        beam_type = "STATIC" if beam_is_static else "DYNAMIC"

        # Create dataset with basic beam info
        dataset = self._create_basic_beam_info(
            self._beam_name,
            beam_type,
            self._fluence_mode,
            beam_limiting_device_sequence=self._beam_limiting_device_sequence,
            number_of_control_points=self.number_of_control_points,
        )

        # Add initial control point
        cp0 = Dataset()
        cp0.ControlPointIndex = 0
        cp0.NominalBeamEnergy = self._energy
        cp0.DoseRateSet = self._dose_rate
        beam_limiting_device_position_sequence = DicomSequence()
        for key, values in bld_positions.items():
            beam_limiting_device_position = Dataset()
            beam_limiting_device_position.RTBeamLimitingDeviceType = key
            beam_limiting_device_position.LeafJawPositions = list(values[0])
            beam_limiting_device_position_sequence.append(beam_limiting_device_position)
        cp0.BeamLimitingDevicePositionSequence = beam_limiting_device_position_sequence
        cp0.GantryAngle = gantry_angles[0]
        cp0.GantryRotationDirection = gantry_direction[0].value
        cp0.BeamLimitingDeviceAngle = self._coll_angle
        cp0.BeamLimitingDeviceRotationDirection = "NONE"
        cp0.PatientSupportAngle = self._couch_rot
        cp0.PatientSupportRotationDirection = "NONE"
        cp0.TableTopEccentricAngle = 0.0
        cp0.TableTopEccentricRotationDirection = "NONE"
        cp0.TableTopVerticalPosition = self._couch_vrt
        cp0.TableTopLongitudinalPosition = self._couch_lng
        cp0.TableTopLateralPosition = self._couch_lat
        cp0.IsocenterPosition = None
        cp0.CumulativeMetersetWeight = 0.0
        dataset.ControlPointSequence.append(cp0)

        # Add rest of the control points
        for cp_idx in range(1, self.number_of_control_points):
            cp = Dataset()
            cp.ControlPointIndex = cp_idx
            cp.CumulativeMetersetWeight = metersets_weights[cp_idx]

            if not gantry_is_static:
                cp.GantryAngle = gantry_angles[cp_idx]
                cp.GantryRotationDirection = gantry_direction[cp_idx].value

            bld_position_sequence = DicomSequence()
            for bld, positions in bld_positions.items():
                if not dict_bld_is_static[bld]:
                    bld_position = Dataset()
                    bld_position.RTBeamLimitingDeviceType = bld
                    bld_position.LeafJawPositions = list(positions[cp_idx])
                    bld_position_sequence.append(bld_position)
            if len(bld_position_sequence) > 0:
                cp.BeamLimitingDevicePositionSequence = bld_position_sequence

            dataset.ControlPointSequence.append(cp)

        return dataset

    @staticmethod
    def _create_basic_beam_info(
        beam_name: str,
        beam_type: str,
        fluence_mode: FluenceMode,
        beam_limiting_device_sequence: DicomSequence,
        number_of_control_points: int,
    ) -> Dataset:
        beam = Dataset()
        beam.Manufacturer = "Radformation"
        beam.ManufacturerModelName = "RadMachine"
        beam.PrimaryDosimeterUnit = "MU"
        beam.SourceAxisDistance = 1000.0

        # Primary Fluence Mode Sequence
        primary_fluence_mode1 = Dataset()
        if fluence_mode == FluenceMode.STANDARD:
            primary_fluence_mode1.FluenceMode = "STANDARD"
        elif fluence_mode == FluenceMode.FFF:
            primary_fluence_mode1.FluenceMode = "NON_STANDARD"
            primary_fluence_mode1.FluenceModeID = "FFF"
        elif fluence_mode == FluenceMode.SRS:
            primary_fluence_mode1.FluenceMode = "NON_STANDARD"
            primary_fluence_mode1.FluenceModeID = "SRS"
        beam.PrimaryFluenceModeSequence = DicomSequence((primary_fluence_mode1,))

        # Beam Limiting Device Sequence
        beam.BeamLimitingDeviceSequence = beam_limiting_device_sequence

        # beam numbers start at 0 and increment from there.
        beam.BeamName = beam_name
        beam.BeamType = beam_type
        beam.RadiationType = "PHOTON"
        beam.TreatmentDeliveryType = "TREATMENT"
        beam.NumberOfWedges = 0
        beam.NumberOfCompensators = 0
        beam.NumberOfBoli = 0
        beam.NumberOfBlocks = 0
        beam.FinalCumulativeMetersetWeight = 1.0
        beam.NumberOfControlPoints = number_of_control_points

        # Control Point Sequence
        beam.ControlPointSequence = DicomSequence()
        return beam


    @classmethod
    def for_truebeam(
        cls,
        mlc_is_hd: bool,
        beam_name: str,
        energy: float,
        fluence_mode: FluenceMode,
        dose_rate: int,
        metersets: Sequence[float],
        gantry_angles: float | Sequence[float],
        x1: float,
        x2: float,
        y1: float,
        y2: float,
        mlc_positions: list[list[float]],
        coll_angle: float,
        couch_vrt: float,
        couch_lat: float,
        couch_lng: float,
        couch_rot: float,
    ) -> Self:
        """
        Parameters
        ----------
        mlc_is_hd : bool
            Whether the MLC type is HD or Millennium
        beam_name : str
            The name of the beam. Must be less than 16 characters.
        energy : float
            The energy of the beam.
        fluence_mode : FluenceMode
            The fluence mode of the beam.
        dose_rate : int
            The dose rate of the beam.
        metersets : Sequence[float]
            The meter sets for each control point. The length must match the number of control points in mlc_positions.
        gantry_angles : Union[float, Sequence[float]]
            The gantry angle(s) of the beam. If a single number, it's assumed to be a static beam. If multiple numbers, it's assumed to be a dynamic beam.
        x1 : float
            The left jaw position.
        x2 : float
            The right jaw position.
        y1 : float
            The bottom jaw position.
        y2 : float
            The top jaw position.
        mlc_positions : list[list[float]]
            The MLC positions for each control point. This is the x-position of each leaf for each control point.
        coll_angle : float
            The collimator angle.
        couch_vrt : float
            The couch vertical position.
        couch_lat : float
            The couch lateral position.
        couch_lng : float
            The couch longitudinal position.
        couch_rot : float
            The couch rotation.
        """
        jaw_x = Dataset()
        jaw_x.RTBeamLimitingDeviceType = "X"
        jaw_x.NumberOfLeafJawPairs = 1
        jaw_y = Dataset()
        jaw_y.RTBeamLimitingDeviceType = "Y"
        jaw_y.NumberOfLeafJawPairs = 1
        jaw_asymx = Dataset()
        jaw_asymx.RTBeamLimitingDeviceType = "ASYMX"
        jaw_asymx.NumberOfLeafJawPairs = 1
        jaw_asymy = Dataset()
        jaw_asymy.RTBeamLimitingDeviceType = "ASYMX"
        jaw_asymy.NumberOfLeafJawPairs = 1
        mlc = Dataset()
        mlc.RTBeamLimitingDeviceType = "MLCX"
        mlc.NumberOfLeafJawPairs = 60
        mlc.LeafPositionBoundaries = list(MLC_BOUNDARIES_TB_HD120 if mlc_is_hd else MLC_BOUNDARIES_TB_MIL120)

        bld_sequence = DicomSequence((jaw_x, jaw_y, jaw_asymx, jaw_asymy, mlc))

        beam_limiting_device_positions = {
            "ASYMX": [[x1, x2]],
            "ASYMY": [[y1, y2]],
            "MLCX": mlc_positions,
        }

        return cls(
            beam_limiting_device_sequence=bld_sequence,
            beam_name=beam_name,
            energy=energy,
            fluence_mode=fluence_mode,
            dose_rate=dose_rate,
            metersets=metersets,
            gantry_angles=gantry_angles,
            beam_limiting_device_positions=beam_limiting_device_positions,
            coll_angle=coll_angle,
            couch_vrt=couch_vrt,
            couch_lat=couch_lat,
            couch_lng=couch_lng,
            couch_rot=couch_rot,
        )


    @classmethod
    def for_halcyon(
        cls,
        beam_name: str,
        metersets: Sequence[float],
        gantry_angles: float | Sequence[float],
        distal_mlc_positions: list[list[float]],
        proximal_mlc_positions: list[list[float]],
        coll_angle: float,
        couch_vrt: float,
        couch_lat: float,
        couch_lng: float,
    ) -> Self:
        """
        Parameters
        ----------
        beam_name : str
            The name of the beam. Must be less than 16 characters.
        metersets : Sequence[float]
            The meter sets for each control point. The length must match the number of control points in mlc_positions.
        gantry_angles : Union[float, Sequence[float]]
            The gantry angle(s) of the beam. If a single number, it's assumed to be a static beam. If multiple numbers, it's assumed to be a dynamic beam.
        distal_mlc_positions : list[list[float]]
            The distal MLC positions for each control point. This is the x-position of each leaf for each control point.
        proximal_mlc_positions : list[list[float]]
            The proximal MLC positions for each control point. This is the x-position of each leaf for each control point.
        coll_angle : float
            The collimator angle.
        couch_vrt : float
            The couch vertical position.
        couch_lat : float
            The couch lateral position.
        couch_lng : float
            The couch longitudinal position.
        """
        jaw_x = Dataset()
        jaw_x.RTBeamLimitingDeviceType = "X"
        jaw_x.NumberOfLeafJawPairs = 1
        jaw_y = Dataset()
        jaw_y.RTBeamLimitingDeviceType = "Y"
        jaw_y.NumberOfLeafJawPairs = 1
        mlc_x1 = Dataset()
        mlc_x1.RTBeamLimitingDeviceType = "MLCX1"
        mlc_x1.NumberOfLeafJawPairs = 28
        mlc_x1.LeafPositionBoundaries = list(MLC_BOUNDARIES_HAL_DIST)
        mlc_x2 = Dataset()
        mlc_x2.RTBeamLimitingDeviceType = "MLCX2"
        mlc_x2.NumberOfLeafJawPairs = 29
        mlc_x2.LeafPositionBoundaries = list(MLC_BOUNDARIES_HAL_PROX)
        bld_sequence = DicomSequence((jaw_x, jaw_y, mlc_x1, mlc_x2))

        beam_limiting_device_positions = {
            "X": [[-140, 140]],
            "Y": [[-140, 140]],
            "MLCX1": distal_mlc_positions,
            "MLCX2": proximal_mlc_positions,
        }

        return cls(
            beam_limiting_device_sequence=bld_sequence,
            beam_name=beam_name,
            energy=6,
            fluence_mode=FluenceMode.FFF,
            dose_rate=600,
            metersets=metersets,
            gantry_angles=gantry_angles,
            beam_limiting_device_positions=beam_limiting_device_positions,
            coll_angle=coll_angle,
            couch_vrt=couch_vrt,
            couch_lat=couch_lat,
            couch_lng=couch_lng,
            couch_rot=0,
        )


class QAProcedureBase(ABC):
    """An abstract base class for generic QA procedures."""

    beams: list[Beam]

    # beams is a class attribute so is shared among the derived classes.
    # It means that each derived class adds the beams to this instead of starting fresh.
    # Therefore, we need to clear it at the beginning.
    def __init__(self):
        self.beams = list()


class PlanGenerator(ABC):
    """A tool for generating new QA RTPlan files based on an initial, somewhat empty RTPlan file.

    Attributes
    ----------
    machine_name : str
        The name of the machine
    """

    machine_name: str
    machine_specs : MachineSpecs

    def __init__(
        self,
        ds: Dataset,
        plan_label: str,
        plan_name: str,
        patient_name: str | None,
        patient_id: str | None,
        machine_specs : MachineSpecs,
    ):
        """A tool for generating new QA RTPlan files based on an initial, somewhat empty RTPlan file.

        Parameters
        ----------
        ds : Dataset
              The RTPLAN dataset to base the new plan off of. The plan must already have MLC positions.
        plan_label : str
            The label of the new plan.
        plan_name : str
            The name of the new plan.
        patient_name : str, optional
            The name of the patient. If not provided, it will be taken from the RTPLAN file.
        patient_id : str, optional
            The ID of the patient. If not provided, it will be taken from the RTPLAN file.
        machine_specs : MachineSpecs
            The specs of the machine
        """
        if ds.Modality != "RTPLAN":
            raise ValueError("File is not an RTPLAN file")
        self.machine_specs = machine_specs
        patient_name = patient_name or getattr(ds, "PatientName", None)
        if not patient_name:
            raise ValueError(
                "RTPLAN file must have PatientName or pass it via `patient_name`"
            )
        patient_id = patient_id or getattr(ds, "PatientID", None)
        if not patient_id:
            raise ValueError(
                "RTPLAN file must have PatientID or pass it via `patient_id`"
            )
        if not hasattr(ds, "ToleranceTableSequence"):
            raise ValueError("RTPLAN file must have ToleranceTableSequence")
        if not hasattr(ds, "BeamSequence"):
            raise ValueError(
                "RTPLAN file must have at least one beam in the beam sequence"
            )
        has_mlc_data: bool = any(
            "MLC" in bld.RTBeamLimitingDeviceType
            for bs in ds.BeamSequence
            for bld in bs.BeamLimitingDeviceSequence
        )
        if not has_mlc_data:
            raise ValueError("RTPLAN file must have MLC data")

        ######  Create a copy of the template plan
        # A shallow copy won’t work because beam data is cleared.
        # The inherited classes require access to the original beam state to determine leaf boundaries
        self.ds = deepcopy(ds)

        ######  Clear/initialize the metadata for the new plan
        self.ds.PatientName = patient_name
        self.ds.PatientID = patient_id
        self.ds.RTPlanLabel = plan_label
        self.ds.RTPlanName = plan_name
        date = datetime.datetime.now().strftime("%Y%m%d")
        time = datetime.datetime.now().strftime("%H%M%S")

        self.ds.InstanceCreationDate = date
        self.ds.InstanceCreationTime = time
        self.ds.SOPInstanceUID = generate_uid()

        # Patient Setup Sequence
        patient_setup = Dataset()
        patient_setup.PatientPosition = "HFS"
        patient_setup.PatientSetupNumber = 0
        self.ds.PatientSetupSequence = DicomSequence((patient_setup,))

        # Dose Reference Sequence
        dose_ref1 = Dataset()
        dose_ref1.DoseReferenceNumber = 1
        dose_ref1.DoseReferenceUID = generate_uid()
        dose_ref1.DoseReferenceStructureType = "SITE"
        dose_ref1.DoseReferenceDescription = "PTV"
        dose_ref1.DoseReferenceType = "TARGET"
        dose_ref1.DeliveryMaximumDose = 20.0
        dose_ref1.TargetPrescriptionDose = 40.0
        dose_ref1.TargetMaximumDose = 20.0
        self.ds.DoseReferenceSequence = DicomSequence((dose_ref1,))

        # Fraction Group Sequence
        frxn_gp1 = Dataset()
        frxn_gp1.FractionGroupNumber = 1
        frxn_gp1.NumberOfFractionsPlanned = 1
        frxn_gp1.NumberOfBeams = 0
        frxn_gp1.NumberOfBrachyApplicationSetups = 0
        frxn_gp1.ReferencedBeamSequence = DicomSequence()
        self.ds.FractionGroupSequence = DicomSequence((frxn_gp1,))

        # Clear beam sequence
        # This will be filled with the custom beams
        self.ds.BeamSequence = DicomSequence()

        # Machine name
        self.machine_name = ds.BeamSequence[0].TreatmentMachineName

        # Validate machine type
        self._validate_machine_type(ds.BeamSequence)

    @classmethod
    def from_rt_plan_file(cls, rt_plan_file: str | Path, **kwargs) -> Self:
        """Load an existing RTPLAN file and create a new plan based on it.

        Parameters
        ----------
        rt_plan_file : str
            The path to the RTPLAN file.
        kwargs
            See the PlanGenerator constructor for details.
        """
        ds = pydicom.dcmread(rt_plan_file)
        return cls(ds, **kwargs)

    @abstractmethod
    def _validate_machine_type(self, beam_sequence: DicomSequence):
        pass

    def add_beam(self, beam: Beam):
        """Add a beam to the plan using the Beam object. Although public,
        this is a low-level method that is used by the higher-level methods like add_open_field_beam.
        This handles the associated metadata like the referenced beam sequence and fraction group sequence.
        """
        beam_dataset = beam.to_dicom()

        # Update the beam
        beam_dataset.BeamNumber = len(self.ds.BeamSequence) + 1
        beam_dataset.TreatmentMachineName = self.machine_name
        patient_setup_nr = self.ds.PatientSetupSequence[0].PatientSetupNumber
        beam_dataset.ReferencedPatientSetupNumber = patient_setup_nr
        tolerance_table_nr = self.ds.ToleranceTableSequence[0].ToleranceTableNumber
        beam_dataset.ReferencedToleranceTableNumber = tolerance_table_nr
        self.ds.BeamSequence.append(beam_dataset)

        # increment number of beams
        fr = self.ds.FractionGroupSequence[0]
        fr.NumberOfBeams += 1

        # Update plan references
        referenced_beam = Dataset()
        referenced_beam.BeamDose = 1.0
        referenced_beam.BeamMeterset = beam.beam_meterset
        referenced_beam.ReferencedBeamNumber = beam_dataset.BeamNumber
        dose_reference_uid = self.ds.DoseReferenceSequence[0].DoseReferenceUID
        referenced_beam.ReferencedDoseReferenceUID = dose_reference_uid
        self.ds.FractionGroupSequence[0].ReferencedBeamSequence.append(referenced_beam)

    def add_procedure(self, procedure: QAProcedureBase) -> None:
        for beam in procedure.beams:
            self.add_beam(beam)

    def to_file(self, filename: str | Path) -> None:
        """Write the DICOM dataset to file"""
        self.ds.save_as(filename, write_like_original=False)

    def as_dicom(self) -> Dataset:
        """Return the new DICOM dataset."""
        return self.ds

    def plot_fluences(
        self,
        width_mm: float = 400,
        resolution_mm: float = 0.5,
        dtype: np.dtype = np.uint16,
    ) -> list[Figure]:
        """Plot the fluences of the beams generated

        See Also
        --------
        :func:`~pydicom_planar.PlanarImage.plot_fluences`
        """
        return plot_fluences(self.as_dicom(), width_mm, resolution_mm, dtype, show=True)

    def to_dicom_images(
        self, imager: Imager, invert: bool = True
    ) -> list[Dataset]:
        """Generate simulated DICOM images of the plan. This provides a way to
        generate an end-to-end simulation of the plan. The images will always be
        at 1000mm SID.

        Parameters
        ----------
        imager : Imager
            The imager to use to generate the images. This provides the
            size of the image and the pixel size
        invert: bool
            Invert the fluence. Setting to True simulates EPID-style images where
            dose->lower pixel value.
        """
        image_ds = []
        fluences = generate_fluences(
            rt_plan=self.as_dicom(),
            width_mm=imager.shape[1] * imager.pixel_size,
            resolution_mm=imager.pixel_size,
        )
        for beam, fluence in zip(self.ds.BeamSequence, fluences):
            beam_info = beam.ControlPointSequence[0]
            sim = Simulator(imager, sid=1000)
            sim.add_layer(ArrayLayer(fluence))
            ds = sim.as_dicom(
                gantry_angle=beam_info.GantryAngle,
                coll_angle=beam_info.BeamLimitingDeviceAngle,
                table_angle=beam_info.PatientSupportAngle,
                invert_array=invert,
            )
            image_ds.append(ds)
        return image_ds


class TrueBeamPlanGenerator(PlanGenerator):

    machine: TrueBeamMachine

    def __init__(
        self,
        ds: Dataset,
        plan_label: str,
        plan_name: str,
        patient_name: str | None = None,
        patient_id: str | None = None,
        machine_specs: MachineSpecs = DEFAULT_SPECS_TB
    ):
        super().__init__(
            ds,
            plan_label,
            plan_name,
            patient_name,
            patient_id,
            machine_specs
        )

        mlc_is_hd = any(
            bld.LeafPositionBoundaries[0] == -110
            for bs in ds.BeamSequence
            for bld in bs.BeamLimitingDeviceSequence
            if bld.RTBeamLimitingDeviceType == "MLCX"
        )
        self.machine = TrueBeamMachine(mlc_is_hd, machine_specs=machine_specs)

    def _validate_machine_type(self, beam_sequence: DicomSequence):
        has_valid_mlc_data: bool = any(
            bld.RTBeamLimitingDeviceType == "MLCX"
            for bs in beam_sequence
            for bld in bs.BeamLimitingDeviceSequence
        )
        if not has_valid_mlc_data:
            raise ValueError(
                "The machine on the template plan does not seem to be a TrueBeam machine."
            )

class HalcyonPlanGenerator(PlanGenerator):
    """A class to generate a plan with two beams stacked on top of each other such as the Halcyon. This
    also assumes no jaws."""

    machine: HalcyonMachine

    def __init__(
        self,
        ds: Dataset,
        plan_label: str,
        plan_name: str,
        patient_name: str | None = None,
        patient_id: str | None = None,
        machine_specs: MachineSpecs = DEFAULT_SPECS_HAL
    ):
        super().__init__(
            ds,
            plan_label,
            plan_name,
            patient_name,
            patient_id,
            machine_specs
        )
        self.machine = HalcyonMachine(machine_specs=machine_specs)

    def _validate_machine_type(self, beam_sequence: DicomSequence):
        has_valid_mlc_data: bool = any(
            bld.RTBeamLimitingDeviceType == "MLCX1"
            for bs in beam_sequence
            for bld in bs.BeamLimitingDeviceSequence
        )
        if not has_valid_mlc_data:
            raise ValueError(
                "The machine on the template plan does not seem to be a Halcyon machine."
            )


class OvertravelError(ValueError):
    pass
