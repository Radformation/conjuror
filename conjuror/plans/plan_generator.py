import datetime
import inspect
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace, field
from enum import Enum
from pathlib import Path
from typing import Self, TypeVar, Generic

import numpy as np
from plotly import graph_objects as go
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import generate_uid

from ..images.layers import ArrayLayer
from ..images.simulators import Simulator, Imager
from ..utils import wrap180
from .fluence import plot_fluences


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


class MachineBase(ABC):
    machine_specs: MachineSpecs


TMachine = TypeVar("TMachine", bound=MachineBase)


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


class BeamBase(Generic[TMachine], ABC):
    """Represents a DICOM beam dataset. Has methods for creating the dataset and adding control points."""

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
        self._fluence_mode = fluence_mode
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
        self.beam_name = beam_name
        self.beam_meterset = np.round(metersets[-1], self.ROUNDING_DECIMALS)
        self.number_of_control_points = len(metersets)
        self.beam_limiting_device_sequence = beam_limiting_device_sequence
        if not isinstance(gantry_angles, Iterable):
            gantry_angles = [gantry_angles] * self.number_of_control_points
        self.metersets = np.array(metersets)
        self.gantry_angles = np.array(gantry_angles)
        self.beam_limiting_device_positions = dict()
        for key, positions in beam_limiting_device_positions.items():
            rep = self.number_of_control_points if len(positions) == 1 else 1
            bld = np.array(rep * positions).T
            self.beam_limiting_device_positions[key] = bld

    @classmethod
    def from_dicom(cls, ds: Dataset, beam_idx: int):
        """Load a beam from an RT plan dataset

        Parameters
        ----------
        ds : Dataset
            The dataset of the RT Plan.
        beam_idx : int
            The index of the beam to be loaded
        """
        mu = ds.FractionGroupSequence[0].ReferencedBeamSequence[beam_idx].BeamMeterset
        beam = ds.BeamSequence[beam_idx]
        bld = beam.BeamLimitingDeviceSequence
        name = beam.BeamName
        fms = beam.PrimaryFluenceModeSequence[0]
        fluence_mode = FluenceMode.STANDARD
        if fms.FluenceMode == "NON_STANDARD":
            match fms.FluenceModeID:
                case "FFF":
                    fluence_mode = FluenceMode.FFF
                case "SRS":
                    fluence_mode = FluenceMode.SRS
                case _:
                    raise ValueError("FluenceModeID must be either FFF or SRS")

        cp0 = beam.ControlPointSequence[0]
        energy = cp0.NominalBeamEnergy
        dose_rate = cp0.DoseRateSet
        coll_angle = cp0.BeamLimitingDeviceAngle
        couch_vrt = cp0.TableTopVerticalPosition
        couch_lat = cp0.TableTopLateralPosition
        couch_lng = cp0.TableTopLongitudinalPosition
        couch_rot = cp0.TableTopEccentricAngle

        # Initial control point
        gantry_angles = [cp0.GantryAngle]
        cmws = [cp0.CumulativeMetersetWeight]
        bldp = {
            bld.RTBeamLimitingDeviceType: [bld.LeafJawPositions]
            for bld in cp0.BeamLimitingDevicePositionSequence
        }

        # for the next control points the concept is: append new if exists,
        # otherwise append a copy of the previous control point
        for idx in range(1, beam.NumberOfControlPoints):
            cp = beam.ControlPointSequence[idx]

            try:
                gantry_angles.append(cp.GantryAngle)
            except AttributeError:
                gantry_angles.append(gantry_angles[-1])

            try:
                cmws.append(cp.CumulativeMetersetWeight)
            except AttributeError:
                cmws.append(cmws[-1])

            bldps = getattr(cp, "BeamLimitingDevicePositionSequence", {})
            for key in bldp.keys():
                bld_types = [x.RTBeamLimitingDeviceType for x in bldps]
                try:
                    idx = bld_types.index(key)
                    bldp[key].append(bldps[idx].LeafJawPositions)
                except ValueError:
                    bldp[key].append(bldp[key][-1])

        beam_limiting_device_positions = bldp
        metersets = mu * np.array(cmws)

        return cls(
            bld,
            name,
            energy,
            fluence_mode,
            dose_rate,
            metersets,
            gantry_angles,
            coll_angle,
            beam_limiting_device_positions,
            couch_vrt,
            couch_lat,
            couch_lng,
            couch_rot,
        )

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
        metersets_weights = np.array(metersets_weights)  # force array for lint
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
            k: np.all(pos == pos[:, 0:1]) for k, pos in bld_positions.items()
        }
        blds_are_static = np.all(list(dict_bld_is_static.values()))
        beam_is_static = gantry_is_static and blds_are_static
        beam_type = "STATIC" if beam_is_static else "DYNAMIC"

        # Create dataset with basic beam info
        dataset = self._create_basic_beam_info(
            self.beam_name,
            beam_type,
            self._fluence_mode,
            beam_limiting_device_sequence=self.beam_limiting_device_sequence,
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
            beam_limiting_device_position.LeafJawPositions = list(values[:, 0])
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
                    bld_position.LeafJawPositions = list(positions[:, cp_idx])
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

    def generate_fluence(self, imager: Imager) -> np.ndarray:
        """Generate the fluence map from the RT Plan.

        Parameters
        ----------
        imager : Imager
            The imager to use to generate the images. This provides the
            size of the image and the pixel size.

        Returns
        -------
        np.ndarray
            The fluence map. Will be the same shape as the imager.
        """
        meterset_per_cp = np.diff(self.metersets, prepend=0)
        x = imager.pixel_size * (np.arange(imager.shape[1]) - (imager.shape[1] - 1) / 2)
        y = imager.pixel_size * (np.arange(imager.shape[0]) - (imager.shape[0] - 1) / 2)

        stack_fluences = list()
        for key, positions in self.beam_limiting_device_positions.items():
            if "MLC" not in key:
                continue
            stack_fluence = np.zeros(imager.shape)
            number_of_leaf_pairs = int(positions.shape[0] / 2)
            leaves_b = positions[0:number_of_leaf_pairs, :]
            leaves_a = positions[number_of_leaf_pairs:, :]

            stack_fluence_compact = np.zeros((number_of_leaf_pairs, imager.shape[1]))
            for cp_idx in range(1, self.number_of_control_points):
                mu = meterset_per_cp[cp_idx]
                mask = (x > leaves_b[:, cp_idx : cp_idx + 1]) & (
                    x <= leaves_a[:, cp_idx : cp_idx + 1]
                )
                stack_fluence_compact[mask] += mu

            boundaries = next(
                bld
                for bld in self.beam_limiting_device_sequence
                if bld.RTBeamLimitingDeviceType == key
            ).LeafPositionBoundaries
            row_to_leaf_map = np.argmax(np.array([boundaries]).T - y > 0, axis=0) - 1
            for row in range(len(y)):
                leaf = row_to_leaf_map[row]
                if leaf < 0:
                    continue
                stack_fluence[row, :] = stack_fluence_compact[leaf, :]
            stack_fluences.append(stack_fluence)

        fluence = np.min(stack_fluences, axis=0)
        return fluence

    def plot_fluence(self, imager: Imager, show: bool = True) -> go.Figure:
        """Plot the fluence map from the RT Beam.

        Parameters
        ----------
        imager : Imager
            The imager to use to generate the images. This provides the
            size of the image and the pixel size.
        show : bool, optional
            Whether to show the plots. Default is True.
        """
        fluence = self.generate_fluence(imager)
        fig = go.Figure()
        fig.add_heatmap(
            z=fluence,
            colorscale="Viridis",
            colorbar=dict(title="Fluence (a.u.)"),
            showscale=True,
        )
        fig.update_layout(
            title=f"Fluence Map - {self.beam_name}",
        )
        if show:
            fig.show()
        return fig

    def animate_mlc(self, show: bool = True) -> go.Figure:
        """Plot the MLC positions as animation.

        Parameters
        ----------
        show : bool, optional
            Whether to show the plot. Default is True.
        """
        _leaf_length = 200
        blds = {
            bld.RTBeamLimitingDeviceType: bld
            for bld in self.beam_limiting_device_sequence
            if "MLC" in bld.RTBeamLimitingDeviceType
        }

        frames = []
        for cp_idx in range(self.number_of_control_points):
            shapes = []
            for key, positions in self.beam_limiting_device_positions.items():
                if key in ["X", "ASYMX"]:
                    x1 = go.Scatter(
                        x=2 * [positions[0, cp_idx]],
                        y=[-1000, 1000],
                        mode="lines",
                        line=dict(width=2, color="orange"),
                    )
                    x2 = go.Scatter(
                        x=2 * [positions[1, cp_idx]],
                        y=[-1000, 1000],
                        mode="lines",
                        line=dict(width=2, color="orange"),
                    )
                    shapes.append(x1)
                    shapes.append(x2)
                if key in ["Y", "ASYMY"]:
                    y1 = go.Scatter(
                        x=[-1000, 1000],
                        y=2 * [positions[0, cp_idx]],
                        mode="lines",
                        line=dict(width=2, color="orange"),
                    )
                    y2 = go.Scatter(
                        x=[-1000, 1000],
                        y=2 * [positions[1, cp_idx]],
                        mode="lines",
                        line=dict(width=2, color="orange"),
                    )
                    shapes.append(y1)
                    shapes.append(y2)
                if "MLC" not in key:
                    continue

                # MLC
                num_leaf_pairs = blds[key].NumberOfLeafJawPairs
                for leaf in range(num_leaf_pairs):
                    y1 = blds[key].LeafPositionBoundaries[leaf]
                    y2 = blds[key].LeafPositionBoundaries[leaf + 1]
                    y = np.array([y1, y1, y2, y2, y1])

                    pos_b = positions[leaf, cp_idx]
                    x_b = pos_b + _leaf_length * np.array([-1, 0, 0, -1, -1])
                    rect_b = go.Scatter(
                        x=x_b, y=y, mode="lines", line=dict(width=2, color="blue")
                    )

                    pos_a = positions[leaf + num_leaf_pairs, cp_idx]
                    x_a = pos_a + _leaf_length * np.array([0, 1, 1, 0, 0])
                    rect_a = go.Scatter(
                        x=x_a, y=y, mode="lines", line=dict(width=2, color="blue")
                    )

                    shapes.append(rect_b)
                    shapes.append(rect_a)

                frame = go.Frame(data=shapes, name=f"cp_{cp_idx}")
                frames.append(frame)
        data = frames[0].data
        layout = go.Layout(
            showlegend=False,
            xaxis=dict(range=[-200, 200]),
            yaxis=dict(range=[-200, 200]),
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "▶ Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    # "frame": {"duration": 50},
                                    # "transition": {"duration": 0},
                                    "fromcurrent": True,
                                },
                            ],
                        }
                    ],
                    "pad": {"r": 10, "t": 50},
                    "x": 0,
                    "y": 0,
                }
            ],
            sliders=[
                {
                    "currentvalue": {"prefix": "Control point: "},
                    "steps": [
                        {
                            "label": f"{i}",
                            "method": "animate",
                            "args": [
                                [f"cp_{i}"],
                                {
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        }
                        for i in range(len(frames))
                    ],
                    "pad": {"b": 10, "t": 50},
                }
            ],
        )
        fig = go.Figure(data=data, frames=frames, layout=layout)

        if show:
            fig.show()

        return fig


@dataclass
class QAProcedureBase(Generic[TMachine], ABC):
    """An abstract base class for generic QA procedures."""

    beams: list[BeamBase[TMachine]] = field(default_factory=list, kw_only=True)
    machine: TMachine = field(init=False)

    @classmethod
    def from_machine(cls, machine: TMachine, **kwargs) -> Self:
        c = cls(**kwargs)
        c.machine = machine
        c.compute()
        return c

    @abstractmethod
    def compute(self):
        pass


class PlanGenerator(Generic[TMachine]):
    """A tool for generating new QA RTPlan files based on an initial, somewhat empty RTPlan file."""

    def __init__(
        self,
        ds: Dataset,
        plan_label: str,
        plan_name: str,
        patient_name: str | None = None,
        patient_id: str | None = None,
        machine_specs: MachineSpecs = None,
    ):
        """
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
        try:
            mlc = next(
                bld
                for bs in ds.BeamSequence
                for bld in bs.BeamLimitingDeviceSequence
                if "MLCX" in bld.RTBeamLimitingDeviceType
            )
        except StopIteration:
            raise ValueError("RTPLAN file must have MLC data")

        self.machine = _get_machine_type_from_mlc(mlc, machine_specs)

        ######  Clear/initialize the metadata for the new plan
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.RTPlanLabel = plan_label
        ds.RTPlanName = plan_name
        date = datetime.datetime.now().strftime("%Y%m%d")
        time = datetime.datetime.now().strftime("%H%M%S")

        ds.InstanceCreationDate = date
        ds.InstanceCreationTime = time
        ds.SOPInstanceUID = generate_uid()

        # Patient Setup Sequence
        patient_setup = Dataset()
        patient_setup.PatientPosition = "HFS"
        patient_setup.PatientSetupNumber = 0
        ds.PatientSetupSequence = DicomSequence((patient_setup,))

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
        ds.DoseReferenceSequence = DicomSequence((dose_ref1,))

        # Fraction Group Sequence
        frxn_gp1 = Dataset()
        frxn_gp1.FractionGroupNumber = 1
        frxn_gp1.NumberOfFractionsPlanned = 1
        frxn_gp1.NumberOfBeams = 0
        frxn_gp1.NumberOfBrachyApplicationSetups = 0
        frxn_gp1.ReferencedBeamSequence = DicomSequence()
        ds.FractionGroupSequence = DicomSequence((frxn_gp1,))

        # Store attributes
        self.ds = ds
        self.machine_name = ds.BeamSequence[0].TreatmentMachineName

        # Clear beam sequence, this will be filled with the custom beams
        ds.BeamSequence = DicomSequence()

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

    def add_beam(self, beam: BeamBase):
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
        procedure.machine = self.machine
        procedure.compute()
        for beam in procedure.beams:
            self.add_beam(beam)

    def to_file(self, filename: str | Path) -> None:
        """Write the DICOM dataset to file"""
        self.ds.save_as(filename, write_like_original=False)

    def as_dicom(self) -> Dataset:
        """Return the new DICOM dataset."""
        return self.ds

    def plot_fluences(self, imager: Imager) -> list[go.Figure]:
        """Plot the fluences of the beams generated

        See Also
        --------
        :func:`~pydicom_planar.PlanarImage.plot_fluences`
        """
        return plot_fluences(self.as_dicom(), imager, show=True)

    def to_dicom_images(self, imager: Imager, invert: bool = True) -> list[Dataset]:
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
        for idx in range(len(self.ds.BeamSequence)):
            beam = BeamBase.from_dicom(self.ds, idx)
            beam_info = beam.to_dicom().ControlPointSequence[0]
            fluence = beam.generate_fluence(imager)
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

    def list_procedures(self) -> list[str]:
        module = sys.modules[self.machine.__module__]
        procedures = [
            name
            for name, _cls in inspect.getmembers(module, inspect.isclass)
            if issubclass(_cls, QAProcedureBase) and not inspect.isabstract(_cls)
        ]
        return procedures


class OvertravelError(ValueError):
    pass


def _get_machine_type_from_mlc(mlc: Dataset, machine_specs: MachineSpecs):
    """This function acts as factory to build the machine from the mlc data set."""
    # Local imports are used to avoid circular dependencies.
    # When a new machine type is added, this factory must be updated accordingly.
    # This is an intentional design choice: although a plugin/registry pattern could
    # automate new additions, new machine types are expected to be added rarely,
    # and maintaining explicit control in this method is preferred.

    bld_type = mlc.RTBeamLimitingDeviceType

    if bld_type == "MLCX":
        mlc_is_hd = mlc.LeafPositionBoundaries[0] == -110
        from .truebeam import TrueBeamMachine

        machine = TrueBeamMachine(mlc_is_hd, machine_specs)

    elif bld_type == "MLCX1" or bld_type == "MLCX2":
        from .halcyon import HalcyonMachine

        machine = HalcyonMachine(machine_specs)

    else:
        raise ValueError("MLC type not supported")

    return machine
