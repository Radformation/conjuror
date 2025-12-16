from plotly import graph_objects as go
from pydicom import Dataset

from conjuror.images.simulators import Imager
from conjuror.plans.beam import Beam


def plot_fluences(
    plan: Dataset,
    imager: Imager,
    show: bool = True,
) -> list[go.Figure]:
    """Plot the fluences of the dataset. Generates N figures where N is the number of Beams in the plan BeamSequence.

    Parameters
    ----------
    plan : pydicom.Dataset
        The RT Plan dataset. Must contain BeamSequence.
    imager : Imager
        The imager to use to generate the images. This provides the
        size of the image and the pixel size.
    show : bool, optional
        Whether to show the plots. Default is True.

    Returns
    -------
    list[Figure]
        A list of plotly figures, one for each beam in the plan.
    """
    figs = []

    for idx in range(len(plan.BeamSequence)):
        beam = Beam.from_dicom(plan, idx)
        fig = beam.plot_fluence(imager, show)
        figs.append(fig)
    return figs


def plot_fluence(
    plan: Dataset,
    idx: int,
    imager: Imager,
    show: bool = True,
) -> list[go.Figure]:
    """Plot the fluence for beam ``idx`` of the dataset.

    Parameters
    ----------
    plan : pydicom.Dataset
        The RT Plan dataset. Must contain BeamSequence.
    idx : int
        The index of the beam in the dataset.
    imager : Imager
        The imager to use to generate the images. This provides the
        size of the image and the pixel size.
    show : bool, optional
        Whether to show the plots. Default is True.

    Returns
    -------
    Figure
        A plotly figure with the beam fluence
    """
    beam = Beam.from_dicom(plan, idx)
    fig = beam.plot_fluence(imager, show)
    return fig


def animate_mlc(
    plan: Dataset,
    idx: int,
    show: bool = True,
) -> list[go.Figure]:
    """Plot the MLC animation for beam ``idx`` of the dataset.

    Parameters
    ----------
    plan : pydicom.Dataset
        The RT Plan dataset. Must contain BeamSequence.
    idx : int
        The index of the beam in the dataset.
    show : bool, optional
        Whether to show the plots. Default is True.

    Returns
    -------
    Figure
        A plotly figure with the mlc animation
    """
    beam = Beam.from_dicom(plan, idx)
    fig = beam.animate_mlc(show)
    return fig
