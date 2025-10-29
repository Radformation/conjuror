============
User's guide
============

Typical use
-----------

.. code-block:: python

    from conjuror.plans.plan_generator_truebeam import TrueBeamPlanGenerator, OpenField
    import pydicom

    # create generator
    base_plan = pydicom.dcmread(r"C:\path\to\base_plan_truebeam_millennium_mlc.dcm")
    generator = TrueBeamPlanGenerator(base_plan, plan_name="New QA Plan", plan_label="New QA")

    # add procedures
    procedure = OpenField(x1=-5, x2=5, y1=-10, y2=110, defined_by_mlcs=True, padding_mm=10)
    generator.add_procedure(procedure)

    # export to file
    generator.to_file("new_plan.dcm")


Creating a generator
--------------------

To use the Plan Generator, a base RT Plan file (or dataset) is required from the specific machine and institution for which the plans will be generated (see :ref:`creating-a-base-plan`). In most cases, the resulting plan will be imported into Eclipse and associated with an existing patient. Using a base plan as a template ensures that machine and patient identifiers remain consistent with the clinical database.

While plans created with the Plan Generator can, in principle, be loaded directly onto the treatment machine, it is recommended to first import them into Eclipse. Eclipse performs comprehensive plan validation, ensuring all tags conform to machine specifications. After validation, the plan can then be exported from Eclipse for delivery on the machine.

.. code-block:: python

    import pydicom

    from conjuror.plans.plan_generator_truebeam import TrueBeamPlanGenerator
    base_plan = pydicom.dcmread(r"C:\path\to\base_plan_truebeam_millennium_mlc.dcm")
    generator = TrueBeamPlanGenerator(base_plan, plan_name="New QA Plan", plan_label="New QA")

    # or

    from conjuror.plans.plan_generator_halcyon import HalcyonPlanGenerator
    base_plan = pydicom.dcmread(r"C:\path\to\base_plan_halcyon.dcm")
    generator = HalcyonPlanGenerator(base_plan, plan_name="New QA Plan", plan_label="New QA")

.. _creating-a-base-plan:

Creating a base plan
####################

This is easy to do in Eclipse (and likely other TPSs) by creating/using a QA patient and creating a simple plan on the machine of interest. The plan should have at least 1 field and the field must contain MLCs. The MLCs don't have to do anything; it doesn't need to be dynamic plan. The point is that a plan like this, regardless of what the MLCs are doing, simply contains the MLC setup information. In list form, the plan should:

* Be based on a QA/research patient in your R&V (no real patients)
* Have a field with MLCs (static or dynamic)
* Be set to the machine of interest
* Set the tolerance table to the desired table

Once the plan is created and saved, export it to a DICOM file. This file will be used as the base plan for the generator.

This entire process can be done in the Plan Parameters of Eclipse
as shown below:

.. image:: images/new_qa_plan.gif
    :width: 600
    :align: center

Use DICOM Import/Export to export the plan to a file.


Required tags from base plan
############################

The required tags in the base plan are:

* Patient Name (0010, 0010) - This isn't changed, just referenced so that the exported plan has the same patient name.
* Patient ID (0010, 0020) - This isn't changed, just referenced so that the exported plan has the same patient ID.
* Machine Name (300A, 00B2) - This isn't changed, just referenced so that the exported plan has the same machine name.
* Tolerance Table Sequence (300A, 0046) - This is required and will be reference by the generated beams. Only
  the first tolerance table is considered. This is not changed by the generator.
* BeamSequence (300A, 00B0) - This is required for Truebeam machines to identify the MLC configuration (Millennium MLC or HD MLC). Specifically, the ``LeafPositionBoundaries`` of the last ``BeamLimitingDeviceSequence`` of the first beam.

  .. note::

      Only the first beam is considered. Extra beams are ignored.

Modified tags from base plan
############################

The metadata of the new plan will be mostly copied from the base plan with a few exceptions:

* RT Plan Label (300A, 0003) - This is changed to reflect the new plan label.
* RT Plan Name (300A, 0002) - This is changed to reflect the new plan name.
* Instance Creation Time (0008, 0013) - This is changed to reflect the new plan creation time (now).
* Instance Creation Date (0008, 0012) - This is changed to reflect the new plan creation date (now).
* SOP Instance UID (0008, 0018) - A new, random UID is generated so it doesn't conflict with the original plan.
* Patient Setup Sequence (300A, 0180) - This is overwritten to a new, single setup.
* Dose Reference Sequence (300A, 0016) - This is overwritten to a new, single dose reference.
* Fraction Group Sequence (300A, 0070) - This is overwritten to a new, single fraction group and
  is dynamically updated based on the fields added by the user.
* Beam Sequence (300A, 00B0) - This is overwritten and is dynamically updated based on the
  procedures added by the user.
* Referenced Beam Sequence (300C, 0006) - This is overwritten and is dynamically updated based on the
  procedures added by the user.


Adding procedures
-----------------

Once the plan generator has been created, QA procedures can be added to the plan. The generator is responsible for adding machine information to each beam and updating the RT Fraction Scheme Module.

.. code-block:: python

    procedure = OpenField(x1=-5, x2=5, y1=-10, y2=110, defined_by_mlcs=True, padding_mm=10)
    generator.add_procedure(procedure)

Pre-defined procedures ** TO BE IMPLEMENTED **
----------------------------------------------

The plan generator comes with pre-defined procedures for typical QA tests (Picket-Fence, Open field, etc). For a comprehensive list of available procedures use the list_procedure method:

.. code-block:: python

    generator.list_procedures()

Advanced features
-----------------

Customize machine parameters
############################

The Plan Generator accounts for specific machine parameters — such as maximum gantry speed and maximum MLC speed — that define the physical limits of the target treatment machine. These parameters are immutable properties of the machine and are used when generating certain procedures, such as MLC speed tests.

By default, most machines use a set of standard parameter values. However, when necessary, it is possible to define custom machine specifications to reflect site-specific configurations or non-standard equipment. There are two supported methods for creating custom machine specifications:

* Take default machine specs and replace one or more parameters.

.. code-block:: python

    from conjuror.plans.plan_generator_truebeam import TrueBeamPlanGenerator, DEFAULT_SPECS_TB
    specs = DEFAULT_SPECS_TB.replace(max_gantry_speed=4.8, max_mlc_speed=20)
    generator = TrueBeamPlanGenerator(..., machine_specs=specs)


* Create new machine specs via ``MachineSpecs``

.. code-block:: python

    from conjuror.plans.plan_generator_base import MachineSpecs
    from conjuror.plans.plan_generator_truebeam import TrueBeamPlanGenerator
    specs = MachineSpecs(
       max_gantry_speed=4.8,
       max_mlc_position=200,
       max_mlc_overtravel=100,
       max_mlc_speed = 20)
    generator = TrueBeamPlanGenerator(..., machine_specs=specs)


Create custom procedures
########################

Custom procedures can be created by extending the ``QAProcedureBase`` abstract class. When defining a custom procedure, a target machine must be specified — for example, when implementing a procedure to create a circle, the MLC leaf side boundaries need to be known. To simplify procedure creation without relying on a base plan, you can instantiate a ``Machine`` class and then generate the procedure using the ``.from_machine`` class method.

.. code-block:: python

    from conjuror.plans.plan_generator_base import QAProcedureBase
    from conjuror.plans.plan_generator_truebeam import TrueBeamPlanGenerator

    @dataclass
    class CircleProcedure(QAProcedureBase):

        # parameters
        radius: float

        def compute(self):
            # business logic
            pass

        def plot(self):
            # business logic
            pass

    def test_circle():
        machine = TrueBeamMachine(mlc_is_hd=False)
        circle = CircleProcedure.from_machine(machine, radius = 5.0)
        circle.plot()
