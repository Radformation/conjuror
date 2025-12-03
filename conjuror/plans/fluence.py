from plotly import graph_objects as go
from pydicom import Dataset

from conjuror.images.simulators import Imager


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
    from conjuror.plans.plan_generator import BeamBase

    figs = []

    for idx in range(len(plan.BeamSequence)):
        beam = BeamBase.from_dicom(plan, idx)
        fig = beam.plot_fluence(imager, show)
        figs.append(fig)
    return figs
