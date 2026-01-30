from abc import ABC
from dataclasses import dataclass
from typing import Generic, Sequence, Iterable

import numpy as np
import plotly.io as pio
from plotly import graph_objects as go
from plotly import subplots
from pydicom import Sequence as DicomSequence, Dataset
from scipy.interpolate import make_interp_spline

from conjuror.images.simulators import Imager
from conjuror.plans.machine import TMachine, MachineSpecs, GantryDirection, FluenceMode
from conjuror.utils import wrap180


@dataclass
class _BeamLimitingDevice:
    """Helper class intended to facilitate access to BeamLimitingDevices."""

    number_of_leaf_pairs: int
    leaf_position_boundaries: list[float]
    # This maps an imager rows to a given leaf (e.g. rows 450-470 -> leaf #10)
    row_to_leaf_map: np.ndarray
    leaves_a: np.ndarray
    leaves_b: np.ndarray


class BeamVisualizationMixin:
    """This Mixin class adds functionality to visualize beams"""

    def generate_fluence(
        self: "Beam",
        imager: Imager,
        interpolation_factor: int = 100,
    ) -> np.ndarray:
        """Generate the fluence map from the RT Plan.

        Parameters
        ----------
        imager : Imager
            The imager to use to generate the images. This provides the
            size of the image and the pixel size.
        interpolation_factor : int
            Interpolation factor to increase control points resolution.

        Returns
        -------
        np.ndarray
            The fluence map. Will be the same shape as the imager.
        """
        x = imager.pixel_size * (np.arange(imager.shape[1]) - (imager.shape[1] - 1) / 2)
        y = imager.pixel_size * (np.arange(imager.shape[0]) - (imager.shape[0] - 1) / 2)

        # Store MLC data in a single dictionary
        bldseq = self.beam_limiting_device_sequence
        blds = dict[str, _BeamLimitingDevice]()
        for key, positions in self.beam_limiting_device_positions.items():
            if "MLC" not in key:
                continue
            bld = next(blds for blds in bldseq if blds.RTBeamLimitingDeviceType == key)
            blds[key] = _BeamLimitingDevice(
                bld.NumberOfLeafJawPairs,
                bld.LeafPositionBoundaries,
                np.argmax(np.array([bld.LeafPositionBoundaries]).T - y > 0, axis=0) - 1,
                positions[bld.NumberOfLeafJawPairs :, :],
                positions[: bld.NumberOfLeafJawPairs, :],
            )

        # Interpolate data
        num_cp = self.number_of_control_points  # before interpolation
        num_cp_ = interpolation_factor * (num_cp - 1) + 1  # after interpolation
        t = range(num_cp)  # abscissas for interpolation (used t since x is imager axis)
        t_ = np.linspace(0, num_cp - 1, num_cp_)  # evaluated abscissas
        metersets = make_interp_spline(t, self.metersets, k=1)(t_)
        for bld in blds.values():
            bld.leaves_a = make_interp_spline(t, bld.leaves_a, k=1, axis=1)(t_)
            bld.leaves_b = make_interp_spline(t, bld.leaves_b, k=1, axis=1)(t_)

        meterset_per_cp = np.diff(metersets, prepend=0)
        fluence = np.zeros(imager.shape)
        for cp_idx in range(1, num_cp_):
            stack_fluences = list()
            for bld in blds.values():
                leaves_b = bld.leaves_b[:, cp_idx : cp_idx + 1]
                leaves_a = bld.leaves_a[:, cp_idx : cp_idx + 1]
                mu = meterset_per_cp[cp_idx]
                # The mask contains the fluence boolean values, where the y-axis corresponds
                # to the leaves and the x-axis indicates whether a given pixel is irradiated.
                stack_compact = (x > leaves_b) & (x <= leaves_a)
                # This loop expands stack_compact into the full size image stack_fluence
                stack_fluence = np.zeros(imager.shape)
                for row in range(len(y)):
                    leaf = bld.row_to_leaf_map[row]
                    if leaf < 0:
                        continue
                    stack_fluence[row, stack_compact[leaf, :]] = mu
                stack_fluences.append(stack_fluence)
            cp_fluence = np.min(stack_fluences, axis=0)
            fluence += cp_fluence

        # Jaws
        blds = self.beam_limiting_device_positions
        jaws_x = next(val for key, val in blds.items() if key in ["ASYMX", "X"])
        jaws_y = next(val for key, val in blds.items() if key in ["ASYMY", "Y"])
        if np.any(np.diff(jaws_x, axis=1)) or np.any(np.diff(jaws_y, axis=1)):
            raise ValueError("The jaws must be static")
        fluence[:, (x < jaws_x[0, 0]) | (x > jaws_x[1, 0])] = 0
        fluence[(y < jaws_y[0, 0]) | (y > jaws_y[1, 0]), :] = 0

        return fluence

    def plot_fluence(
        self: "Beam",
        imager: Imager,
        interpolation_factor: int = 100,
        show: bool = True,
    ) -> go.Figure:
        """Plot the fluence map from the RT Beam.

        Parameters
        ----------
        imager : Imager
            The imager to use to generate the images. This provides the
            size of the image and the pixel size.
        interpolation_factor : int
            Interpolation factor to increase control points resolution.
        show : bool, optional
            Whether to show the plots. Default is True.
        """
        fluence = self.generate_fluence(imager, interpolation_factor)
        fig = go.Figure()
        fig.add_heatmap(
            z=fluence,
            colorscale="Viridis",
            colorbar=dict(title="MU"),
            showscale=True,
        )
        fig.update_layout(
            title=f"Fluence Map - {self.beam_name}",
        )
        if show:
            fig.show()
        return fig

    def animate_mlc(self: "Beam", show: bool = True) -> go.Figure:
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
            title=f"Beam: {self.beam_name}",
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

    def plot_control_points(
        self: "Beam", specs: MachineSpecs, show: bool = True
    ) -> go.Figure:
        """Plot the control points from dynamic beam
        Rows: Absolute position, relative motion, time to deliver, speed
        Cols: Dose, Gantry, MLC
        """
        # This is used mostly for visual inspection during development
        # Axis labeling could be improved

        self.compute_dynamics(specs)

        def _plot(
            as_line: bool,
            data: np.ndarray,
            reuse_axis: bool = False,
            title: str = "",
            y_label: str = "",
        ) -> None:
            """helper function for plotting"""
            if not reuse_axis:
                idx[0] += 1

            r, c = np.unravel_index(idx[0], (num_rows, num_cols))
            row, col = int(r) + 1, int(c) + 1  # Plotly subplots are 1-based
            x_data = self.metersets if as_line else self.metersets[:-1]
            shape = "linear" if as_line else "hv"
            color = colorway[1] if reuse_axis else colorway[0]
            trace = go.Scatter(
                x=x_data,
                y=data,
                mode="lines",
                line={"shape": shape, "color": color},
            )
            fig.add_trace(trace, row=row, col=col)

            if not reuse_axis:
                fig.layout.annotations[idx[0]].text = title
                fig.update_yaxes(title_text=y_label, row=row, col=col)

        colorway = pio.templates[pio.templates.default].layout.colorway
        idx = [-1]
        num_rows, num_cols = 4, 3
        spt = [" "] * (num_rows * num_cols)
        fig = subplots.make_subplots(rows=num_rows, cols=num_cols, subplot_titles=spt)
        fig.update_layout(showlegend=False)

        # Positions
        _plot(True, self.metersets, title="MU", y_label="Absolute")
        _plot(True, self.gantry_angles, title="Gantry")
        _plot(True, self.beam_limiting_device_positions["MLCX"][0, :], title="MLC")
        _plot(True, self.beam_limiting_device_positions["MLCX"][-1, :], reuse_axis=True)

        # Motions
        _plot(False, self.dose_motions, y_label="Motion")
        _plot(False, self.gantry_motions)
        _plot(False, self.mlc_motions[0, :])
        _plot(False, self.mlc_motions[-1, :], reuse_axis=True)

        # Time to deliver
        _plot(False, self.time_to_deliver, y_label="Delivery time")
        _plot(False, self.time_to_deliver)
        _plot(False, self.time_to_deliver)

        # Speeds
        _plot(False, self.dose_speeds * 60, y_label="Speed")
        _plot(False, self.gantry_speeds)
        _plot(False, self.mlc_speeds[0, :])
        _plot(False, self.mlc_speeds[-1, :], reuse_axis=True)

        if show:
            fig.show()

        return fig


class BeamDynamicsMixin:
    """This Mixin class adds functionalities to calculate dynamic parameters of the beam.

    Nomenclature:

    _motion: difference between consecutive control points
    _speed: speed = _motion/time
    """

    time_to_deliver: np.ndarray
    dose_motions: np.ndarray
    gantry_motions: np.ndarray
    mlc_motions: np.ndarray
    dose_speeds: np.ndarray
    gantry_speeds: np.ndarray
    mlc_speeds: np.ndarray

    def compute_dynamics(self: "Beam", specs: MachineSpecs) -> None:
        # motions
        # note: this is currently hardcoded for TrueBeam. Changes are necessary for Halcyon
        self.dose_motions = np.abs(np.diff(self.metersets))
        gantry_angle_var = (180 - self.gantry_angles) % 360
        self.gantry_motions = np.abs(np.diff(gantry_angle_var))
        mlc_positions = self.beam_limiting_device_positions["MLCX"]
        self.mlc_motions = np.diff(mlc_positions, axis=1)

        # ttd = time to deliver
        ttd_dose = self.dose_motions / (self.dose_rate / 60)
        ttd_gantry = self.gantry_motions / specs.max_gantry_speed
        ttd_mlc = self.mlc_motions / specs.max_mlc_speed
        times_to_deliver = np.vstack((ttd_dose, ttd_gantry, ttd_mlc))
        self.time_to_deliver = np.max(np.abs(times_to_deliver), axis=0)

        # speeds
        # dose_speed is the same as dose_rate but in MU/sec
        self.dose_speeds = self.dose_motions / self.time_to_deliver
        self.gantry_speeds = self.gantry_motions / self.time_to_deliver
        self.mlc_speeds = self.mlc_motions / self.time_to_deliver


class Beam(Generic[TMachine], BeamDynamicsMixin, BeamVisualizationMixin, ABC):
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
        self._couch_vrt = couch_vrt
        self._couch_lat = couch_lat
        self._couch_lng = couch_lng

        # Public attributes (storing only)
        self.dose_rate = dose_rate
        self.coll_angle = coll_angle
        self.couch_rot = couch_rot

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
            The index of the beam to be loaded (zero indexed, i.e. beam #1 -> ind #0).
        """
        if ds.Modality != "RTPLAN":
            raise ValueError("File is not an RTPLAN file")

        if beam_idx >= ds.FractionGroupSequence[0].NumberOfBeams:
            msg = "beam_idx is largen that the number of beams in the plan (note: use zero indexing)."
            raise ValueError(msg)

        mu = ds.FractionGroupSequence[0].ReferencedBeamSequence[beam_idx].BeamMeterset
        beam = ds.BeamSequence[beam_idx]
        bld = beam.BeamLimitingDeviceSequence
        name = beam.BeamName
        fluence_mode = FluenceMode.STANDARD
        pfms = beam.get("PrimaryFluenceModeSequence")
        if pfms and pfms[0].get("FluenceMode") == "NON_STANDARD":
            match pfms.FluenceModeID:
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
        cp0.DoseRateSet = self.dose_rate
        beam_limiting_device_position_sequence = DicomSequence()
        for key, values in bld_positions.items():
            beam_limiting_device_position = Dataset()
            beam_limiting_device_position.RTBeamLimitingDeviceType = key
            beam_limiting_device_position.LeafJawPositions = list(values[:, 0])
            beam_limiting_device_position_sequence.append(beam_limiting_device_position)
        cp0.BeamLimitingDevicePositionSequence = beam_limiting_device_position_sequence
        cp0.GantryAngle = gantry_angles[0]
        cp0.GantryRotationDirection = gantry_direction[0].value
        cp0.BeamLimitingDeviceAngle = self.coll_angle
        cp0.BeamLimitingDeviceRotationDirection = "NONE"
        cp0.PatientSupportAngle = self.couch_rot
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
