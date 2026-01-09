from dataclasses import dataclass

import numpy as np
from plotly import graph_objects as go
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import UID

from conjuror.images.layers import Layer
from conjuror.utils import array_to_dicom


def generate_file_metadata() -> Dataset:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = UID(
        "1.2.840.10008.1.2"
    )  # default DICOM transfer syntax
    return file_meta


@dataclass
class Imager:
    """Data class for an imager"""

    pixel_size: float
    shape: tuple[int, int]


IMAGER_AS500 = Imager(pixel_size=0.78125, shape=(384, 512))
IMAGER_AS1000 = Imager(pixel_size=0.390625, shape=(768, 1024))
IMAGER_AS1200 = Imager(pixel_size=0.336, shape=(1280, 1280))


class Simulator:
    """Class for an image simulator"""

    def __init__(self, imager: Imager, sid: float = 1500):
        """

        Parameters
        ----------
        imager : Imager
            Imager to be simulated
        sid : float
            Source to image distance in mm.
        """
        self.imager = imager
        self.sid = sid
        self.image = np.zeros(imager.shape, np.uint16)
        self.mag_factor = sid / 1000

    def add_layer(self, layer: Layer) -> None:
        """Add a layer to the image"""
        self.image = layer.apply(self.image, self.imager.pixel_size, self.mag_factor)

    def as_dicom(
        self,
        gantry_angle: float = 0.0,
        coll_angle: float = 0.0,
        table_angle: float = 0.0,
        invert_array: bool = False,
        tags: dict | None = None,
    ) -> Dataset:
        """Create and return a pydicom Dataset. I.e. create a pseudo-DICOM image."""
        if invert_array:
            array = -self.image + self.image.max() + self.image.min()
        else:
            array = self.image
        return array_to_dicom(
            array=array,
            sid=self.sid,
            gantry=gantry_angle,
            coll=coll_angle,
            couch=table_angle,
            dpi=25.4 / self.imager.pixel_size,
            extra_tags=tags or {},
        )

    def generate_dicom(self, file_out_name: str, *args, **kwargs) -> None:
        """Save the simulated image to a DICOM file.

        See Also
        --------
        as_dicom
        """
        ds = self.as_dicom(*args, **kwargs)
        ds.save_as(file_out_name, write_like_original=False)

    def plot(self, show: bool = True) -> go.Figure:
        """Plot the simulated image."""
        fig = go.Figure()
        fig.add_heatmap(
            z=self.image,
            colorscale="gray",
            x0=-self.image.shape[1] / 2 * self.imager.pixel_size,
            dx=self.imager.pixel_size,
            y0=-self.image.shape[0] / 2 * self.imager.pixel_size,
            dy=self.imager.pixel_size,
        )
        fig.update_layout(
            yaxis_constrain="domain",
            xaxis_scaleanchor="y",
            xaxis_constrain="domain",
            xaxis_title="Crossplane (mm)",
            yaxis_title="Inplane (mm)",
        )
        fig.update_layout(
            title_text=f"Simulated {self.__class__.__name__} @{self.sid}mm SID",
            title_x=0.5,
        )
        if show:
            fig.show()
        return fig
