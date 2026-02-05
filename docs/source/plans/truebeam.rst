========================
QA Procedures - TrueBeam
========================

Open Field
----------

The ``OpenField`` procedure creates a simple rectangular open field beam,
commonly used for output calibration, flatness, and symmetry measurements.
The field can be defined either by MLC positions or by jaw positions, with
automatic padding applied to ensure proper field coverage.

**Note:** All position values (``x1``, ``x2``, ``y1``, ``y2``, ``padding``,
etc.) are specified in millimeters.

Basic Usage
^^^^^^^^^^^

The simplest way to create an open field is to specify the field edges in
millimeters:

.. code-block:: python

    from conjuror.plans.truebeam import OpenField

    # Create a 10x20 cm field centered at isocenter
    procedure = OpenField(x1=-50, x2=50, y1=-100, y2=100)
    generator.add_procedure(procedure)

Field Definition Modes
^^^^^^^^^^^^^^^^^^^^^^

By default, fields are defined by MLC positions (``defined_by_mlc=True``),
which means the MLCs form the field edges and the jaws are opened with
padding. Alternatively, you can define the field by jaw positions:

.. code-block:: python

    # Field defined by MLCs (default)
    procedure_mlc = OpenField(x1=-50, x2=50, y1=-100, y2=100, defined_by_mlc=True, padding=5)
    # MLCs form the field edges, jaws are opened 5mm beyond

    # Field defined by jaws
    procedure_jaws = OpenField(x1=-50, x2=50, y1=-100, y2=100, defined_by_mlc=False, padding=5)
    # Jaws form the field edges, MLCs are opened 5mm beyond

MLC Alignment Modes
^^^^^^^^^^^^^^^^^^^

When using MLC-defined fields, the ``mlc_mode`` parameter controls how the
field edges align with MLC leaf boundaries along the y-axis:

.. warning::

   If you are using this procedure for any application that requires exact
   field sizes (such as output calibration or field size verification), you
   should select ``EXACT`` mode. Other modes (``ROUND``, ``INWARD``,
   ``OUTWARD``) may change the field size if the specified field edges do not
   align exactly with MLC leaf boundaries, due to discrete MLC leaf widths.
   This could affect the accuracy of measurements or procedures that depend on
   precise field dimensions.

.. code-block:: python

    from conjuror.plans.truebeam import OpenField, OpenFieldMLCMode

    # OUTWARD: Include intermediate boundaries in the field (larger field, default)
    procedure_outward = OpenField(..., mlc_mode=OpenFieldMLCMode.OUTWARD)

    # INWARD: Exclude intermediate boundaries from the field (smaller field)
    procedure_inward = OpenField(..., mlc_mode=OpenFieldMLCMode.INWARD)

    # ROUND: Round to nearest MLC boundary
    procedure_round = OpenField(..., mlc_mode=OpenFieldMLCMode.ROUND)

    # EXACT: Field edges must align exactly with MLC boundaries
    # If the field edges do not align exactly, an error is raised
    procedure_exact = OpenField(..., mlc_mode=OpenFieldMLCMode.EXACT)

Customizing Beam Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can customize monitor units, energy, fluence mode, dose rate, gantry
angle, collimator angle, couch positions, and beam name:

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
    from conjuror.plans.truebeam import OpenField, OpenFieldMLCMode

    # Create generator from base plan
    base_plan = pydicom.dcmread(r"C:\path\to\base_plan.dcm")
    generator = PlanGenerator(base_plan, plan_name="Output Calibration", plan_label="Output")

    # Add multiple open fields for different energies
    for energy in [6, 10, 15]:
        procedure = OpenField(
            x1=-100, x2=100, y1=-100, y2=100,
            energy=energy,
            mu=100,
            mlc_mode=OpenFieldMLCMode.EXACT,
            beam_name=f"Open {energy}MV"
        )
        generator.add_procedure(procedure)

    # Export plan
    generator.to_file("output_calibration_plan.dcm")

API Reference
^^^^^^^^^^^^^

.. autoclass:: conjuror.plans.truebeam.OpenField

MLC Transmission
----------------

.. autoclass:: conjuror.plans.truebeam.MLCTransmission

Picket Fence
------------

.. autoclass:: conjuror.plans.truebeam.PicketFence
   :members: from_varian_reference

Winston-Lutz
------------

.. autoclass:: conjuror.plans.truebeam.WinstonLutz

Dosimetric Leaf Gap
-------------------

.. autoclass:: conjuror.plans.truebeam.DosimetricLeafGap

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
