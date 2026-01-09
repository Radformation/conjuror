import datetime
import inspect
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self, Generic

from plotly import graph_objects as go
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import generate_uid

from .beam import Beam
from .machine import MachineSpecs, TMachine, MachineBase
from ..images.layers import ArrayLayer
from ..images.simulators import Simulator, Imager
from .visualization import plot_fluences


class QAProcedureBase(Generic[TMachine], ABC):
    """An abstract base class for generic QA procedures."""

    beams: list[Beam[TMachine]]

    def __post_init__(self):
        self.beams = []

    @abstractmethod
    def compute(self, machine: TMachine):
        pass


class PlanGenerator(Generic[TMachine]):
    """A tool for generating new QA RTPlan files based on an initial base RTPlan file."""

    def __init__(
        self,
        base_plan: Dataset,
        plan_label: str,
        plan_name: str,
        patient_name: str | None = None,
        patient_id: str | None = None,
        machine_specs: MachineSpecs = None,
    ):
        """
        Parameters
        ----------
        base_plan : Dataset
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
        if base_plan.Modality != "RTPLAN":
            raise ValueError("File is not an RTPLAN file")
        patient_name = patient_name or getattr(base_plan, "PatientName", None)
        if not patient_name:
            msg = "RTPLAN file must have PatientName or pass it via `patient_name`"
            raise ValueError(msg)
        patient_id = patient_id or getattr(base_plan, "PatientID", None)
        if not patient_id:
            msg = "RTPLAN file must have PatientID or pass it via `patient_id`"
            raise ValueError(msg)
        if not hasattr(base_plan, "ToleranceTableSequence"):
            msg = "RTPLAN file must have ToleranceTableSequence"
            raise ValueError(msg)
        if not hasattr(base_plan, "BeamSequence"):
            msg = "RTPLAN file must have at least one beam in the beam sequence"
            raise ValueError(msg)
        try:
            mlc = next(
                bld
                for bs in base_plan.BeamSequence
                for bld in bs.BeamLimitingDeviceSequence
                if "MLCX" in bld.RTBeamLimitingDeviceType
            )
        except StopIteration:
            raise ValueError("RTPLAN file must have MLC data")

        self.machine = _get_machine_type_from_mlc(mlc, machine_specs)

        date = datetime.datetime.now().strftime("%Y%m%d")
        time = datetime.datetime.now().strftime("%H%M%S")

        #### Create mandatory tags (empty if not applicable)
        plan = Dataset()

        # SOP Common Module
        plan.SOPClassUID = base_plan.SOPClassUID
        plan.SOPInstanceUID = generate_uid()

        # Patient Module
        plan.PatientBirthDate = ""
        plan.PatientSex = ""

        # General Study Module
        plan.StudyDate = date
        plan.StudyTime = time
        plan.AccessionNumber = ""
        plan.ReferringPhysicianName = ""
        plan.StudyInstanceUID = generate_uid()
        plan.StudyID = ""

        # RT Series Module
        plan.Modality = "RTPLAN"
        plan.OperatorsName = ""
        plan.SeriesInstanceUID = generate_uid()
        plan.SeriesNumber = ""

        # General Equipment Module
        plan.Manufacturer = ""

        # RT General Plan Module
        plan.RTPlanDate = date
        plan.RTPlanTime = time
        plan.RTPlanGeometry = "TREATMENT_DEVICE"
        plan.PlanIntent = "MACHINE_QA"

        #### Input parameters
        plan.PatientName = patient_name
        plan.PatientID = patient_id
        plan.RTPlanLabel = plan_label
        plan.RTPlanName = plan_name

        #### Copy from base plan
        # RT Tolerance Tables Module - use first table from base plan
        plan.ToleranceTableSequence = (base_plan.ToleranceTableSequence[0],)

        #### Modules required for beams
        # Use 1-indexing for sequence number per:
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_7.5.html
        # Patient Setup Sequence
        patient_setup = Dataset()
        patient_setup.PatientPosition = "HFS"
        patient_setup.PatientSetupNumber = 1
        plan.PatientSetupSequence = DicomSequence((patient_setup,))

        # RT Prescription Module - Dose Reference Sequence
        dose_ref = Dataset()
        dose_ref.DoseReferenceNumber = 1
        dose_ref.DoseReferenceUID = generate_uid()
        dose_ref.DoseReferenceStructureType = "SITE"
        dose_ref.DoseReferenceDescription = "PTV"
        dose_ref.DoseReferenceType = "TARGET"
        dose_ref.DeliveryMaximumDose = 20.0
        dose_ref.TargetPrescriptionDose = 40.0
        dose_ref.TargetMaximumDose = 20.0
        plan.DoseReferenceSequence = DicomSequence((dose_ref,))

        # Fraction Group Sequence
        frxn_gp = Dataset()
        frxn_gp.FractionGroupNumber = 1
        frxn_gp.NumberOfFractionsPlanned = 1
        frxn_gp.NumberOfBeams = 0
        frxn_gp.NumberOfBrachyApplicationSetups = 0
        frxn_gp.ReferencedBeamSequence = DicomSequence()
        plan.FractionGroupSequence = DicomSequence((frxn_gp,))

        # Store attributes
        self._base_plan = base_plan
        self.ds = plan
        self.machine_name = base_plan.BeamSequence[0].TreatmentMachineName

        # Clear beam sequence, this will be filled with the custom beams
        plan.BeamSequence = DicomSequence()

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

    @classmethod
    def from_machine(
        cls,
        machine: MachineBase,
        machine_name: str = "RadMachine",
        plan_label: str = "Radformation",
        plan_name: str = "Radformation",
        patient_name: str = "RadMachine",
        patient_id: str = "RadMachine",
    ) -> Self:
        """Create a plan for a target machine type.

        Parameters
        ----------
        machine : MachineBase
            The target machine.
        machine_name : str
            The target machine name.
        plan_label : str
            The label of the new plan.
        plan_name : str
            The name of the new plan.
        patient_name : str, optional
            The name of the patient. If not provided, it will be taken from the RTPLAN file.
        patient_id : str, optional
            The ID of the patient. If not provided, it will be taken from the RTPLAN file."""
        base_plan = Dataset()

        # Transfer syntax UID (Implicit VR Little Endian: Default Transfer Syntax for DICOM)
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part06/chapter_a.html
        base_plan.is_implicit_VR = True
        base_plan.is_little_endian = True

        # General tags required on the base plan
        base_plan.Modality = "RTPLAN"
        base_plan.PatientName = patient_name
        base_plan.PatientID = patient_id

        # Machine type specific tags required on the base plan
        sop, beam, tolerance_table = _set_defaults_from_machine_type(machine)
        beam.TreatmentMachineName = machine_name
        base_plan.SOPClassUID = sop
        base_plan.BeamSequence = (beam,)
        base_plan.ToleranceTableSequence = (tolerance_table,)
        return cls(base_plan, plan_label, plan_name)

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
        procedure.compute(self.machine)
        for beam in procedure.beams:
            self.add_beam(beam)

    def to_file(self, filename: str | Path) -> None:
        """Write the DICOM dataset to file"""
        self.ds.save_as(
            filename,
            implicit_vr=self._base_plan.is_implicit_VR,
            little_endian=self._base_plan.is_little_endian,
            enforce_file_format=True,
        )

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
            beam = Beam.from_dicom(self.ds, idx)
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


def _get_machine_type_from_mlc(mlc: Dataset, machine_specs: MachineSpecs) -> TMachine:
    """This function acts as factory to build the machine from the mlc data set."""
    # Local imports are used to avoid circular dependencies.
    # When a new machine type is added, this factory must be updated accordingly.
    # This is an intentional design choice: although a plugin/registry pattern could
    # automate new additions, new machine types are expected to be added rarely,
    # and maintaining explicit control in this method is preferred.

    bld_type = mlc.RTBeamLimitingDeviceType

    if bld_type == "MLCX":
        from .truebeam import TrueBeamMachine

        mlc_is_hd = mlc.LeafPositionBoundaries[0] == -110
        machine = TrueBeamMachine(mlc_is_hd, machine_specs)

    elif bld_type == "MLCX1" or bld_type == "MLCX2":
        from .halcyon import HalcyonMachine

        machine = HalcyonMachine(machine_specs)

    else:
        raise ValueError("MLC type not supported")

    return machine


def _set_defaults_from_machine_type(
    machine: MachineBase,
) -> tuple[str, Dataset, Dataset]:
    """This function acts as factory to build the required data set from machine type."""
    # Local imports are used to avoid circular dependencies.
    # When a new machine type is added, this factory must be updated accordingly.
    # This is an intentional design choice: although a plugin/registry pattern could
    # automate new additions, new machine types are expected to be added rarely,
    # and maintaining explicit control in this method is preferred.

    # Note: This function does not create realistic datasets. Instead, the datasets
    # contain only the minimum elements required to be processed by
    # _get_machine_type_from_mlc. For example, LeafPositionBoundaries for TrueBeam
    # includes only what is needed to distinguish between HDMLC and Millennium,
    # whereas for Halcyon it does not exist at all.

    tolerance_table = Dataset()
    tolerance_table.ToleranceTableNumber = 1

    from .truebeam import TrueBeamMachine
    from .halcyon import HalcyonMachine

    bld = Dataset()
    if isinstance(machine, TrueBeamMachine):
        sop = "1.2.840.10008.5.1.4.1.1.481.5"
        bld.RTBeamLimitingDeviceType = "MLCX"
        bld.LeafPositionBoundaries = 2 * [-110 if machine.mlc_is_hd else -200]
    elif isinstance(machine, HalcyonMachine):
        sop = "1.2.246.352.70.1.70"
        bld.RTBeamLimitingDeviceType = "MLCX1"
    else:
        raise ValueError("Unknown machine type")

    beam = Dataset()
    beam.BeamLimitingDeviceSequence = (bld,)

    return sop, beam, tolerance_table
