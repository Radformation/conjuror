============
User's guide
============

Typical use
-----------

.. code-block:: python

    import pydicom
    from conjuror.plans.plan_generator import PlanGenerator
    from conjuror.plans.truebeam import OpenField

    # create generator
    base_plan = pydicom.dcmread(r"C:\path\to\base_plan_truebeam_millennium_mlc.dcm")
    generator = PlanGenerator(base_plan, plan_name="New QA Plan", plan_label="New QA")

    # add procedures
    procedure = OpenField(x1=-5, x2=5, y1=-10, y2=110, defined_by_mlc=True, padding=10)
    generator.add_procedure(procedure)

    # export to file
    generator.to_file("new_plan.dcm")


Creating a generator
--------------------

There are two ways to create a Plan Generator:

* **Using a base plan file** -- Use this option when planning for a specific
  machine available at the institution. It provides the simplest workflow for
  importing the plan into Eclipse.
* **Selecting a machine type directly** -- Use this option when no specific
  machine is recommended, for example when sharing plans or creating plans
  that apply to multiple machines.

While plans created with the Plan Generator can, in principle, be loaded
directly onto the treatment machine, it is recommended to first import them
into Eclipse. Eclipse performs comprehensive plan validation, ensuring all
tags conform to machine specifications. After validation, the plan can then
be exported from Eclipse for delivery on the machine.

Use case 1: Using a base plan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the Plan Generator with a base plan, a base RT Plan file (or dataset)
is required from the specific machine and institution for which the plans will
be generated (see :ref:`creating-a-base-plan`). In most cases, the resulting
plan will be imported into Eclipse and associated with an existing patient.
Using a base plan as a template ensures that machine and patient identifiers
remain consistent with the clinical database.

.. code-block:: python

    import pydicom
    from conjuror.plans.plan_generator import PlanGenerator

    # create generator from a RT plan dataset
    base_plan_dataset = pydicom.dcmread(r"C:\path\to\base_plan_truebeam_millennium_mlc.dcm")
    generator = PlanGenerator(base_plan_dataset, plan_name="New QA Plan", plan_label="New QA")

    # or

    # create generator from a RT plan file
    base_plan_file = r"C:\path\to\base_plan_truebeam_millennium_mlc.dcm"
    generator = PlanGenerator.from_rt_plan_file(base_plan_file, plan_name="New QA Plan", plan_label="New QA")

.. _creating-a-base-plan:

Creating a base plan
""""""""""""""""""""

This is easy to do in Eclipse (and likely other TPSs) by creating/using a QA
patient and creating a simple plan on the machine of interest. The plan
should have at least 1 field and the field must contain MLCs. The MLCs don't
have to do anything; it doesn't need to be dynamic plan. The point is that a
plan like this, regardless of what the MLCs are doing, simply contains the
MLC setup information. In list form, the plan should:

* Be based on a QA/research patient in your R&V (no real patients)
* Have a field with MLCs (static or dynamic)
* Be set to the machine of interest
* Set the tolerance table to the desired table

Once the plan is created and saved, export it to a DICOM file. This file will
be used as the base plan for the generator.

This entire process can be done in the Plan Parameters of Eclipse
as shown below:

.. image:: images/new_qa_plan.gif
    :width: 600
    :align: center

Use DICOM Import/Export to export the plan to a file.


Required tags from base plan
""""""""""""""""""""""""""""

These are the DICOM tags in the base plan that the generator copies into the
new plan.

* Patient Name (0010, 0010) - Patient Name is used to link the new plan to
  this patient when importing.
* Patient ID (0010, 0020) - Patient ID is used to link the new plan to this
  patient when importing.
* Machine Name (300A, 00B2) - Machine Name is used to link the new plan to
  this machine when importing.
* Tolerance Table Sequence (300A, 0046) - The *first Tolerance Table* in this
  sequence is copied to the new plan.
* BeamSequence (300A, 00B0) - At least on beam in this sequence must contain
  MLC positions. This is required to identify the machine type.

Use case 2: Selecting a machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can create a Plan Generator by directly specifying the machine type. This approach is useful when you don't have a base plan file available or when creating plans that don't need to be associated with a specific machine in a clinical database. The machine type determines the MLC configuration and other machine-specific parameters.

.. code-block:: python

    from conjuror.plans.plan_generator import PlanGenerator
    from conjuror.plans.truebeam import TrueBeamMachine
    from conjuror.plans.halcyon import HalcyonMachine

    # create generator for a TrueBeam machine
    machine = TrueBeamMachine(mlc_is_hd=False)
    generator = PlanGenerator.from_machine(
        machine,
        machine_name="TrueBeam",
        plan_name="New QA Plan",
        plan_label="New QA",
        patient_name="QA Patient",
        patient_id="QA001"
    )

    # or create generator for a Halcyon machine
    halcyon_machine = HalcyonMachine()
    generator = PlanGenerator.from_machine(
        halcyon_machine,
        machine_name="Halcyon",
        plan_name="New QA Plan",
        plan_label="New QA",
        patient_name="QA Patient",
        patient_id="QA001"
    )

Adding procedures
-----------------

Once the plan generator has been created, QA procedures can be added to the
plan. The generator is responsible for adding machine information to each
beam and updating the RT Fraction Scheme Module.

.. code-block:: python

    procedure = OpenField(x1=-5, x2=5, y1=-10, y2=110, defined_by_mlc=True, padding=10)
    generator.add_procedure(procedure)

Pre-defined procedures
----------------------

The plan generator comes with pre-defined procedures for typical QA tests
(Picket-Fence, Open field, etc). For a comprehensive list of available
procedures use the list_procedure method:

.. code-block:: python

    generator.list_procedures()

Advanced features
-----------------

Customize machine parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Plan Generator accounts for specific machine parameters — such as maximum
gantry speed and maximum MLC speed — that define the physical limits of the
target treatment machine. These parameters are immutable properties of the
machine and are used when creating certain procedures, such as MLC speed
tests.

By default, most machines use a set of standard parameter values. However,
when necessary, it is possible to define custom machine specifications to
reflect site-specific configurations or non-standard equipment. There are two
supported methods for creating custom machine specifications:

* Take default machine specs and replace one or more parameters.

.. code-block:: python

    from conjuror.plans.plan_generator import PlanGenerator
    from conjuror.plans.truebeam import DEFAULT_SPECS_TB
    specs = DEFAULT_SPECS_TB.replace(max_gantry_speed=4.8, max_mlc_speed=20)
    generator = PlanGenerator(..., machine_specs=specs)


* Create new machine specs via ``MachineSpecs``

.. code-block:: python

    from conjuror.plans.plan_generator import PlanGenerator, MachineSpecs
    specs = MachineSpecs(
       max_gantry_speed=4.8,
       max_mlc_position=200,
       max_mlc_overtravel=100,
       max_mlc_speed = 20)
    generator = PlanGenerator(..., machine_specs=specs)


Create custom procedures
^^^^^^^^^^^^^^^^^^^^^^^^

Custom procedures can be created by extending the ``QAProcedure`` abstract
class in the appropriate machine module. When computing a custom procedure, a
target machine must be specified — for example, when implementing a procedure
to create a circle, the MLC leaf side boundaries need to be known. To simplify
procedure creation without relying on a base plan, you can instantiate a
``Machine`` class and then pass it to the ``compute`` method as an argument.

.. code-block:: python

    from pydantic import Field
    from conjuror.plans.truebeam import QAProcedure, TrueBeamMachine

    class CircleProcedure(QAProcedure):
        """Create a circular MLC aperture."""

        radius: float = Field(
            title="Radius",
            description="The radius of the circle.",
            json_schema_extra={"units": "mm"},
        )

        def compute(self, machine):
            # business logic
            pass

        def plot(self):
            # business logic
            pass

    def test_circle():
        machine = TrueBeamMachine(mlc_is_hd=False)
        circle = CircleProcedure(radius=5.0)
        circle.compute(machine)  # This step is also done automatically in add_procedure
        circle.plot()


API
---

Core classes
^^^^^^^^^^^^
.. autoclass:: conjuror.plans.plan_generator.PlanGenerator
.. autoclass:: conjuror.plans.machine.MachineSpecs

Base classes
^^^^^^^^^^^^
.. autoclass:: conjuror.plans.machine.MachineBase
.. autopydantic_model:: conjuror.plans.plan_generator.QAProcedureBase
   :show-inheritance:
.. autoclass:: conjuror.plans.beam.Beam

Derived classes - TrueBeam
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: conjuror.plans.truebeam.TrueBeamMachine
.. autoclass:: conjuror.plans.truebeam.QAProcedure
.. autoclass:: conjuror.plans.truebeam.Beam

Derived classes - Halcyon
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: conjuror.plans.halcyon.HalcyonMachine
.. autoclass:: conjuror.plans.halcyon.QAProcedure
.. autoclass:: conjuror.plans.halcyon.Beam
