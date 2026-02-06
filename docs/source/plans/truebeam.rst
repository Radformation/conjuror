========================
QA Procedures - TrueBeam
========================

Open Field
----------

The ``OpenField`` procedure creates a simple rectangular open field beam,
commonly used for output calibration, flatness, and symmetry measurements.
The field can be defined either by MLC positions or by jaw positions, with
automatic padding applied to ensure proper field coverage.


Basic Usage
^^^^^^^^^^^

The simplest way to create an open field is to specify the field edges in
millimeters:

.. code-block:: python

    from conjuror.plans.truebeam import OpenField

    # Create a 10x20 cm field centered at isocenter
    procedure = OpenField(x1=-50, x2=50, y1=-100, y2=100, mu=100)
    generator.add_procedure(procedure)

The following visualizations show the MLC animation and fluence map for an open
field:

.. grid:: 2
    :gutter: 2

    .. grid-item::
        :columns: 6

        .. plotly::
            :iframe-width: 100%
            :iframe-height: 500px

            from conjuror.plans.truebeam import OpenField, TrueBeamMachine

            # Create and compute the open field
            procedure = OpenField(x1=-50, x2=50, y1=-100, y2=100, mu=100)
            machine = TrueBeamMachine(mlc_is_hd=False)
            procedure.compute(machine)
            beam = procedure.beams[0]

            # Generate MLC animation
            beam.animate_mlc(show=False)

    .. grid-item::
        :columns: 6

        .. plotly::
            :iframe-width: 100%
            :iframe-height: 500px

            from conjuror.images.simulators import IMAGER_AS1200
            from conjuror.plans.truebeam import OpenField, TrueBeamMachine

            # Create and compute the open field
            procedure = OpenField(x1=-50, x2=50, y1=-100, y2=100, mu=100)
            machine = TrueBeamMachine(mlc_is_hd=False)
            procedure.compute(machine)
            beam = procedure.beams[0]

            # Generate fluence map
            beam.plot_fluence(IMAGER_AS1200, show=False)

Field Definition Modes
^^^^^^^^^^^^^^^^^^^^^^

By default, fields are defined by MLC positions (``defined_by_mlc=True``),
which means the MLCs form the field edges and the jaws are opened with
padding. Alternatively, you can define the field by jaw positions.

In both modes, ``padding`` is applied to the *non-defining* device to avoid
clipping:

- **MLC-defined** (``defined_by_mlc=True``): the jaws open to
  ``(x1, x2, y1, y2)`` plus ``padding`` on each side.
- **Jaw-defined** (``defined_by_mlc=False``): the MLC opens to
  ``(x1, x2, y1, y2)`` plus ``padding`` on each side.

.. code-block:: python

    # Field defined by MLCs (default)
    procedure_mlc = OpenField(x1=-50, x2=50, y1=-100, y2=100, defined_by_mlc=True, padding=20)
    # MLCs form the field edges; jaws open 20 mm beyond on each side.

    # Field defined by jaws
    procedure_jaws = OpenField(x1=-50, x2=50, y1=-100, y2=100, defined_by_mlc=False, padding=20)
    # Jaws form the field edges; MLC opens 20 mm beyond on each side.

The following visualizations show the MLC positions for each field definition mode:

.. grid:: 2
    :gutter: 2

    .. grid-item::
        :columns: 6

        .. plotly::
            :iframe-width: 100%
            :iframe-height: 500px

            from conjuror.plans.truebeam import OpenField, TrueBeamMachine

            # Field defined by MLCs
            procedure = OpenField(x1=-50, x2=50, y1=-100, y2=100, defined_by_mlc=True, padding=20, beam_name="MLC Defined")
            machine = TrueBeamMachine(mlc_is_hd=False)
            procedure.compute(machine)
            beam = procedure.beams[0]

            # Generate MLC animation
            beam.animate_mlc(show=False)

    .. grid-item::
        :columns: 6

        .. plotly::
            :iframe-width: 100%
            :iframe-height: 500px

            from conjuror.plans.truebeam import OpenField, TrueBeamMachine

            # Field defined by jaws
            procedure = OpenField(x1=-50, x2=50, y1=-100, y2=100, defined_by_mlc=False, padding=20, beam_name="Jaw Defined")
            machine = TrueBeamMachine(mlc_is_hd=False)
            procedure.compute(machine)
            beam = procedure.beams[0]

            # Generate MLC animation
            beam.animate_mlc(show=False)

MLC Alignment Modes
^^^^^^^^^^^^^^^^^^^

MLC leaves have discrete leaf boundaries along the y-axis. If ``y1`` or ``y2`` do
not land exactly on a leaf boundary, the requested field edge must be resolved
according to a rule.

- **MLC-defined fields** (``defined_by_mlc=True``): ``mlc_mode`` controls how
  the y-edges are aligned to the nearest leaf boundaries.
- **Jaw-defined fields** (``defined_by_mlc=False``): ``mlc_mode`` is ignored;
  the MLC is always treated as ``OUTWARD`` so it will not clip the jaw-defined
  opening.

The following alignment modes apply to MLC-defined fields:

**OUTWARD** (default)
   If ``y1`` or ``y2`` falls between MLC leaf boundaries, the intermediate
   leaf band is treated as "infield" and included in the field. This results
   in a larger field size in the y-direction. This is the default mode and is
   suitable for most general-purpose applications where slight field size
   variations are acceptable.

**INWARD**
   If ``y1`` or ``y2`` falls between MLC leaf boundaries, the intermediate
   leaf band is treated as "outfield" and excluded from the field. This
   results in a smaller field size in the y-direction. Use this mode when you
   want to ensure the field does not exceed the specified dimensions.

**ROUND**
   If ``y1`` or ``y2`` falls between MLC leaf boundaries, the field edges are
   rounded to the nearest MLC boundary. This provides a balanced approach
   that minimizes field size deviation.

**EXACT**
   Both ``y1`` and ``y2`` must coincide exactly with an MLC leaf boundary.
   If either edge does not align exactly, a ``ValueError`` is raised. This
   mode ensures precise field size matching and is required for applications
   that depend on exact field dimensions, such as output calibration or field
   size verification measurements.

.. warning::

   If you are using this procedure for an application that requires **exact**
   y-dimensions (for example output calibration or field-size verification),
   select ``EXACT`` mode. The other modes (``ROUND``, ``INWARD``, ``OUTWARD``)
   can change the delivered y-size whenever the requested edges do not align
   with MLC leaf boundaries.

.. code-block:: python

    from conjuror.plans.truebeam import OpenField, MLCLeafBoundaryAlignmentMode

    # OUTWARD: Include intermediate boundaries in the field (larger field, default)
    procedure_outward = OpenField(..., mlc_mode=MLCLeafBoundaryAlignmentMode.OUTWARD)

    # INWARD: Exclude intermediate boundaries from the field (smaller field)
    procedure_inward = OpenField(..., mlc_mode=MLCLeafBoundaryAlignmentMode.INWARD)

    # ROUND: Round to nearest MLC boundary
    procedure_round = OpenField(..., mlc_mode=MLCLeafBoundaryAlignmentMode.ROUND)

    # EXACT: Field edges must align exactly with MLC boundaries
    procedure_exact = OpenField(..., mlc_mode=MLCLeafBoundaryAlignmentMode.EXACT)

The following visualizations show the MLC positions for each alignment mode using y2=51:

.. grid:: 2
    :gutter: 2

    .. grid-item::
        :columns: 6

        .. plotly::
            :iframe-width: 100%
            :iframe-height: 500px

            from conjuror.plans.truebeam import OpenField, MLCLeafBoundaryAlignmentMode, TrueBeamMachine

            # OUTWARD: Include intermediate boundaries in the field
            procedure = OpenField(x1=-50, x2=50, y1=-51, y2=51, mlc_mode=MLCLeafBoundaryAlignmentMode.OUTWARD, beam_name="OUTWARD")
            machine = TrueBeamMachine(mlc_is_hd=False)
            procedure.compute(machine)
            beam = procedure.beams[0]

            # Generate MLC animation and zoom in on top edge
            fig = beam.animate_mlc(show=False)
            fig.update_layout(xaxis_range=[-80, 80], yaxis_range=[40, 60])
            fig

    .. grid-item::
        :columns: 6

        .. plotly::
            :iframe-width: 100%
            :iframe-height: 500px

            from conjuror.plans.truebeam import OpenField, MLCLeafBoundaryAlignmentMode, TrueBeamMachine

            # INWARD: Exclude intermediate boundaries from the field
            procedure = OpenField(x1=-50, x2=50, y1=-51, y2=51, mlc_mode=MLCLeafBoundaryAlignmentMode.INWARD, beam_name="INWARD")
            machine = TrueBeamMachine(mlc_is_hd=False)
            procedure.compute(machine)
            beam = procedure.beams[0]

            # Generate MLC animation and zoom in on top edge
            fig = beam.animate_mlc(show=False)
            fig.update_layout(xaxis_range=[-80, 80], yaxis_range=[40, 60])
            fig

    .. grid-item::
        :columns: 6

        .. plotly::
            :iframe-width: 100%
            :iframe-height: 500px

            from conjuror.plans.truebeam import OpenField, MLCLeafBoundaryAlignmentMode, TrueBeamMachine

            # ROUND: Round to nearest MLC boundary
            procedure = OpenField(x1=-50, x2=50, y1=-51, y2=51, mlc_mode=MLCLeafBoundaryAlignmentMode.ROUND, beam_name="ROUND")
            machine = TrueBeamMachine(mlc_is_hd=False)
            procedure.compute(machine)
            beam = procedure.beams[0]

            # Generate MLC animation and zoom in on top edge
            fig = beam.animate_mlc(show=False)
            fig.update_layout(xaxis_range=[-80, 80], yaxis_range=[40, 60])
            fig

    .. grid-item::
        :columns: 6

        .. plotly::
            :iframe-width: 100%
            :iframe-height: 500px

            from conjuror.plans.truebeam import OpenField, MLCLeafBoundaryAlignmentMode, TrueBeamMachine
            from plotly import graph_objects as go

            # EXACT: Field edges must align exactly with MLC boundaries
            # This will raise an error since y1=-51, y2=51 don't align with MLC boundaries
            try:
                procedure = OpenField(x1=-50, x2=50, y1=-51, y2=51, mlc_mode=MLCLeafBoundaryAlignmentMode.EXACT, beam_name="EXACT")
                machine = TrueBeamMachine(mlc_is_hd=False)
                procedure.compute(machine)
            except ValueError as e:
                # Show error message on a plot similar to the others
                import textwrap
                fig = go.Figure()
                error_text = f"Error: {str(e)}"
                # Wrap text to fit within the figure (approximately 50 characters per line)
                wrapped_text = "<br>".join(textwrap.wrap(error_text, width=50))
                fig.add_annotation(
                    text=wrapped_text,
                    showarrow=False,
                    font=dict(size=12, color="red"),
                )
                fig.update_layout(
                    title="Beam: EXACT",
                    plot_bgcolor="white",
                    xaxis=dict(showticklabels=False),
                    yaxis=dict(showticklabels=False),
                )
                fig

Customizing Parameters
^^^^^^^^^^^^^^^^^^^^^^

You can adjust the field size/position (``x1``, ``x2``, ``y1``, ``y2``) along
with beam settings such as monitor units, energy, fluence mode, dose rate,
gantry/collimator angles, couch positions, and beam naming.

.. code-block:: python

    from conjuror.plans.machine import FluenceMode

    # High-energy open field with custom monitor units
    procedure = OpenField(
        x1=-100, x2=100, y1=-100, y2=100,   # 20x20 cm field (all values in mm)
        mu=200,                             # Monitor units
        energy=15,                          # 15 MV
        fluence_mode=FluenceMode.FFF,       # Flattening filter free
        dose_rate=600,                      # MU/min
        gantry_angle=270,                   # Gantry at 270 degrees
        coll_angle=90,                      # Collimator at 90 degrees
        couch_vrt=0,                        # Couch vertical position (mm)
        couch_lng=1000,                     # Couch longitudinal position (mm)
        couch_lat=0,                        # Couch lateral position (mm)
        couch_rot=0,                        # Couch rotation (degrees)
        beam_name="Open 15XFFF"
    )

Complete Example
^^^^^^^^^^^^^^^^

.. code-block:: python

    import pydicom
    from conjuror.plans.plan_generator import PlanGenerator
    from conjuror.plans.truebeam import OpenField, MLCLeafBoundaryAlignmentMode

    # Create generator from base plan
    base_plan = pydicom.dcmread(r"C:\path\to\base_plan.dcm")
    generator = PlanGenerator(base_plan, plan_name="Output Calibration", plan_label="Output")

    # Add multiple open fields for different energies
    for energy in [6, 10, 15]:
        procedure = OpenField(
            x1=-100, x2=100, y1=-100, y2=100,
            energy=energy,
            mu=100,
            mlc_mode=MLCLeafBoundaryAlignmentMode.EXACT,
            beam_name=f"Open {energy}MV"
        )
        generator.add_procedure(procedure)

    # Export plan
    generator.to_file("output_calibration_plan.dcm")

MLC Transmission
----------------

The ``MLCTransmission`` procedure generates a small set of beams intended to
support **MLC transmission** measurements. It adds:

- **Reference**: a jaw-defined open field.
- **Bank A**: a transmission beam configured to isolate Bank A transmission
  under a jaw-defined opening.
- **Bank B**: a transmission beam configured to isolate Bank B transmission
  under a jaw-defined opening.

This construction allows you to measure transmission from each bank separately
by comparing the transmission images to the reference image.

Basic Usage
^^^^^^^^^^^

To include an MLC transmission test in a generated plan, add
``MLCTransmission`` to your ``PlanGenerator``:

.. code-block:: python

    import pydicom
    from conjuror.plans.plan_generator import PlanGenerator
    from conjuror.plans.truebeam import MLCTransmission

    base_plan = pydicom.dcmread(r"C:\path\to\base_plan.dcm")
    generator = PlanGenerator(base_plan, plan_name="MLC Transmission", plan_label="Tx")

    # Default: 10x10 cm reference, 100 MU reference, 1000 MU per bank
    procedure = MLCTransmission()
    generator.add_procedure(procedure)

    generator.to_file("mlc_transmission_plan.dcm")

The following visualizations show the MLC positions for the **reference**, **Bank A**,
and **Bank B** beams:

.. grid:: 3
    :gutter: 2

    .. grid-item::
        :columns: 4

        .. plotly::
            :iframe-width: 100%
            :iframe-height: 400px

            from conjuror.plans.truebeam import MLCTransmission, TrueBeamMachine

            # Create and compute the MLC transmission procedure
            procedure = MLCTransmission()
            machine = TrueBeamMachine(mlc_is_hd=False)
            procedure.compute(machine)

            # Reference beam
            beam = procedure.beams[0]
            fig = beam.animate_mlc(show=False)
            # Zoom around the jaw-defined open field
            pad = 30  # mm margin
            x1, x2 = -procedure.width / 2 - pad, procedure.width / 2 + pad
            y1, y2 = -procedure.height / 2 - pad, procedure.height / 2 + pad
            fig.update_layout(
                xaxis_scaleanchor="y",
                xaxis_range=[x1, x2],
                yaxis_range=[y1, y2],
                margin=dict(l=10, r=10, t=30, b=10),
            )
            fig

    .. grid-item::
        :columns: 4

        .. plotly::
            :iframe-width: 100%
            :iframe-height: 400px

            from conjuror.plans.truebeam import MLCTransmission, TrueBeamMachine

            # Create and compute the MLC transmission procedure
            procedure = MLCTransmission()
            machine = TrueBeamMachine(mlc_is_hd=False)
            procedure.compute(machine)

            # Bank A transmission beam
            beam = procedure.beams[1]
            fig = beam.animate_mlc(show=False)
            # Zoom around the jaw-defined open field
            pad = 30  # mm margin
            x1, x2 = -procedure.width / 2 - pad, procedure.width / 2 + pad
            y1, y2 = -procedure.height / 2 - pad, procedure.height / 2 + pad
            fig.update_layout(
                xaxis_scaleanchor="y",
                xaxis_range=[x1, x2],
                yaxis_range=[y1, y2],
                margin=dict(l=10, r=10, t=30, b=10),
            )
            fig

    .. grid-item::
        :columns: 4

        .. plotly::
            :iframe-width: 100%
            :iframe-height: 400px

            from conjuror.plans.truebeam import MLCTransmission, TrueBeamMachine

            # Create and compute the MLC transmission procedure
            procedure = MLCTransmission()
            machine = TrueBeamMachine(mlc_is_hd=False)
            procedure.compute(machine)

            # Bank B transmission beam
            beam = procedure.beams[2]
            fig = beam.animate_mlc(show=False)
            # Zoom around the jaw-defined open field
            pad = 30  # mm margin
            x1, x2 = -procedure.width / 2 - pad, procedure.width / 2 + pad
            y1, y2 = -procedure.height / 2 - pad, procedure.height / 2 + pad
            fig.update_layout(
                xaxis_scaleanchor="y",
                xaxis_range=[x1, x2],
                yaxis_range=[y1, y2],
                margin=dict(l=10, r=10, t=30, b=10),
            )
            fig

Customizing Parameters
^^^^^^^^^^^^^^^^^^^^^^

You can adjust the reference field size (``width``, ``height``), MU per beam
(``mu_per_ref``, ``mu_per_bank``), bank ``overreach``, and beam names, along
with beam settings such as energy, fluence mode, dose rate, gantry/collimator
angles, and couch positions.

.. code-block:: python

    from conjuror.plans.machine import FluenceMode
    from conjuror.plans.truebeam import MLCTransmission

    procedure = MLCTransmission(
        width=200,          # mm (20 cm)
        height=200,         # mm (20 cm)
        mu_per_ref=200,     # reference open field MU
        mu_per_bank=2000,   # MU for each bank transmission beam
        overreach=10,       # mm; shifts the closed MLC bank further under the jaw
        beam_names=["Tx Ref", "Tx Bank-A", "Tx Bank-B"],
        energy=15,                          # 15 MV
        fluence_mode=FluenceMode.FFF,       # Flattening filter free
        dose_rate=600,                      # MU/min
        gantry_angle=0,
        coll_angle=0,
        couch_vrt=0,
        couch_lng=1000,
        couch_lat=0,
        couch_rot=0,
    )
    generator.add_procedure(procedure)

Dosimetric Leaf Gap
-------------------

The ``DosimetricLeafGap`` procedure creates a set of fields with sliding
MLC gaps beams for measuring the **dosimetric leaf gap (DLG)**. For each gap width in
``gap_widths``, it generates one beam with two control points: the gap starts
at ``start_position`` and ends at ``final_position`` (both in mm).

.. warning::

   For a clean DLG measurement, ensure the sweeping gap is **fully occluded by
   the X jaws** at both the start and end positions for the **largest** gap
   width. The procedure will emit a warning if:

   - ``min(start_position, final_position) + max(gap_widths)/2 > x1`` or
   - ``max(start_position, final_position) - max(gap_widths)/2 < x2``

   If the gap is not fully occluded at the endpoints, extra transmission can
   bias the DLG calculation.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

    import pydicom
    from conjuror.plans.plan_generator import PlanGenerator
    from conjuror.plans.truebeam import DosimetricLeafGap

    base_plan = pydicom.dcmread(r"C:\path\to\base_plan.dcm")
    generator = PlanGenerator(base_plan, plan_name="DLG", plan_label="DLG")

    procedure = DosimetricLeafGap()
    generator.add_procedure(procedure)

    generator.to_file("dlg_plan.dcm")

The following visualizations show example MLC motion for two different gap
widths:

.. grid:: 2
    :gutter: 2

    .. grid-item::
        :columns: 6

        .. plotly::
            :iframe-width: 100%
            :iframe-height: 400px

            from conjuror.plans.truebeam import DosimetricLeafGap, TrueBeamMachine

            procedure = DosimetricLeafGap(gap_widths=(2, 20))
            machine = TrueBeamMachine(mlc_is_hd=False)
            procedure.compute(machine)

            beam = procedure.beams[0]  # 2 mm gap
            fig = beam.animate_mlc(show=False)
            pad = 30
            fig.update_layout(
                xaxis_scaleanchor="y",
                xaxis_range=[procedure.x1 - pad, procedure.x2 + pad],
                yaxis_range=[procedure.y1 - pad, procedure.y2 + pad],
                margin=dict(l=10, r=10, t=30, b=10),
            )
            fig

    .. grid-item::
        :columns: 6

        .. plotly::
            :iframe-width: 100%
            :iframe-height: 400px

            from conjuror.plans.truebeam import DosimetricLeafGap, TrueBeamMachine

            procedure = DosimetricLeafGap(gap_widths=(2, 20))
            machine = TrueBeamMachine(mlc_is_hd=False)
            procedure.compute(machine)

            beam = procedure.beams[1]  # 20 mm gap
            fig = beam.animate_mlc(show=False)
            pad = 30
            fig.update_layout(
                xaxis_scaleanchor="y",
                xaxis_range=[procedure.x1 - pad, procedure.x2 + pad],
                yaxis_range=[procedure.y1 - pad, procedure.y2 + pad],
                margin=dict(l=10, r=10, t=30, b=10),
            )
            fig

Customizing Parameters
^^^^^^^^^^^^^^^^^^^^^^

You can adjust gap widths and sweep extent (``gap_widths``, ``start_position``,
``final_position``), MU (``mu``), jaw size (``x1``, ``x2``, ``y1``, ``y2``),
and beam settings such as energy, fluence mode, dose rate, gantry/collimator
angles, and couch positions.

.. code-block:: python

    from conjuror.plans.machine import FluenceMode
    from conjuror.plans.truebeam import DosimetricLeafGap

    procedure = DosimetricLeafGap(
        gap_widths=(2, 4, 6, 10, 14, 16, 20),
        start_position=-60,
        final_position=60,
        mu=100,
        x1=-50, x2=50, y1=-50, y2=50,
        energy=6,
        fluence_mode=FluenceMode.STANDARD,
        dose_rate=600,
        gantry_angle=0,
        coll_angle=0,
        couch_vrt=0,
        couch_lng=1000,
        couch_lat=0,
        couch_rot=0,
    )

Picket Fence
------------

.. autoclass:: conjuror.plans.truebeam.PicketFence
   :members: from_varian_reference

Winston-Lutz
------------

.. autoclass:: conjuror.plans.truebeam.WinstonLutz

Dose Rate
---------

.. autoclass:: conjuror.plans.truebeam.DoseRate

MLC Speed
---------

.. autoclass:: conjuror.plans.truebeam.MLCSpeed

Gantry Speed
------------

.. autoclass:: conjuror.plans.truebeam.GantrySpeed

VMAT Dose Rate & Gantry Speed
------------------------------

.. autoclass:: conjuror.plans.truebeam.VMATDRGS
   :members: from_varian_reference

VMAT MLC Speed
--------------

.. autoclass:: conjuror.plans.truebeam.VMATDRMLC
   :members: from_varian_reference

API Reference
-------------

.. autoclass:: conjuror.plans.truebeam.OpenField
.. autoclass:: conjuror.plans.truebeam.MLCTransmission
.. autoclass:: conjuror.plans.truebeam.DosimetricLeafGap
