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
from .machine import MachineSpecs, TMachine
from ..images.layers import ArrayLayer
from ..images.simulators import Simulator, Imager
from .fluence import plot_fluences


class QAProcedureBase(Generic[TMachine], ABC):
    """An abstract base class for generic QA procedures."""

    beams: list[Beam[TMachine]]

    def __post_init__(self):
        self.beams = []

    @abstractmethod
    def compute(self, machine: TMachine):
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
