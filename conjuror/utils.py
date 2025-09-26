from datetime import datetime

import numpy as np
from numpy import ndarray
from pydicom import FileMetaDataset, Dataset
from pydicom.uid import generate_uid, RTImageStorage, ExplicitVRLittleEndian, SecondaryCaptureImageStorage


def wrap360(value: float | ndarray) -> float | ndarray:
    """Wrap the input values to the interval [0, 360)"""
    return value % 360


def wrap180(value: float | ndarray) -> float | ndarray:
    """Wrap the input values to the interval [-180, 180)"""
    return wrap360(value + 180) - 180


def geometric_center_idx(array: ndarray) -> float:
    """Returns the center index and value of the profile.

    If the profile has an even number of array the centre lies between the two centre indices and the centre
    value is the average of the two centre array else the centre index and value are returned.
    """
    return (array.shape[0] - 1) / 2.0


def _rt_image_position(array: ndarray, dpmm: float) -> list[float]:
    """Calculate the RT Image Position of the array."""
    rows, cols = array.shape
    pixel_size_mm = 1.0 / dpmm

    # Calculate total physical size
    width_mm = cols * pixel_size_mm
    height_mm = rows * pixel_size_mm

    # Calculate RT Image Position
    # Origin is at center, so upper-left pixel is offset by half width and height
    x_position = -(width_mm / 2) + (pixel_size_mm / 2)
    y_position = -(height_mm / 2) + (pixel_size_mm / 2)
    return [x_position, y_position]

def array_to_dicom(
    array: ndarray,
    sid: float,
    gantry: float,
    coll: float,
    couch: float,
    dpi: float,
    extra_tags: dict | None = None,
) -> Dataset:
    """Converts a numpy array into a **simplistic** DICOM file. Not meant to be a full-featured converter. This
    allows for the creation of DICOM files from numpy arrays usually for internal use or image analysis.

    .. note::

        This will convert the image into an uint16 datatype to match the native EPID datatype.

    Parameters
    ----------
    array
        The numpy array to be converted. Must be 2 dimensions.
    sid
        The Source-to-Image distance in mm.
    dpi
        The dots-per-inch value of the image.
    gantry
        The gantry value that the image was taken at.
    coll
        The collimator value that the image was taken at.
    couch
        The couch value that the image was taken at.
    extra_tags
        Additional arguments to pass to the DICOM constructor. These are tags that should be included in the DICOM file.
        These will override any defaults Pylinac might set.
        E.g. PatientName, etc.
    """
    file_meta = FileMetaDataset()
    # Main data elements
    ds = Dataset()
    ds.SOPClassUID = RTImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.StudyDate = datetime.now().strftime("%Y%m%d")
    ds.ContentDate = datetime.now().strftime("%Y%m%d")
    ds.StudyTime = datetime.now().strftime("%H%M%S")
    ds.ContentTime = datetime.now().strftime("%H%M%S")
    ds.Modality = "RTIMAGE"
    ds.OperatorsName = "Pylinac"
    ds.ConversionType = "WSD"
    ds.PatientName = "Pylinac array"
    ds.PatientID = "123456789"
    ds.PatientSex = "O"
    ds.PatientBirthDate = "20000101"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "OTHER"]
    ds.RTImageLabel = "Pylinac image"
    ds.RTImagePlane = "NORMAL"
    ds.RadiationMachineName = "Pylinac"
    ds.RTImagePosition = _rt_image_position(array, dpi)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"  # 0 is black, max is white
    ds.Rows = array.shape[0]
    ds.Columns = array.shape[1]
    ds.BitsAllocated = array.itemsize * 8
    ds.BitsStored = array.itemsize * 8
    ds.HighBit = array.itemsize * 8 - 1
    ds.ImagePlanePixelSpacing = [25.4 / dpi, 25.4 / dpi]
    ds.RadiationMachineSAD = "1000.0"
    ds.RTImageSID = sid
    ds.PrimaryDosimeterUnit = "MU"
    ds.Manufacturer = "Pylinac"
    ds.GantryAngle = f"{gantry:.2f}"
    ds.BeamLimitingDeviceAngle = f"{coll:.2f}"
    ds.PatientSupportAngle = f"{couch:.2f}"
    # Although rare, loading certain types of images/files
    # may declare endian-ness and be different from the system.
    # We want to ensure it's native to the system.
    # See here for recommendations:
    # https://pydicom.github.io/pydicom/stable/tutorials/pixel_data/creation.html#creating-float-pixel-data-and-double-float-pixel-data
    if not array.dtype.isnative:
        array = array.byteswap().view(array.dtype.newbyteorder("="))
    if np.issubdtype(array.dtype, np.floating):
        ds.FloatPixelData = array.tobytes()
    else:
        ds.PixelData = array.tobytes()
        ds.PixelRepresentation = 0  # unsigned ints; only other option is 2's complement

    ds.file_meta = file_meta
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.ImplementationClassUID = generate_uid()

    extra_tags = extra_tags or {}
    for key, value in extra_tags.items():
        setattr(ds, key, value)
    return ds