=========
Changelog
=========

Legend
------

* :bdg-success:`Feature` denotes a new feature or ability.
* :bdg-warning:`Fixed` denotes a bug fix.
* :bdg-primary:`Refactor` denotes a code refactor; usually this means an efficiency
  boost or code cleanup.
* :bdg-danger:`Change` denotes a change that may break existing code.

v 0.3.3
-------

General
^^^^^^^

* :bdg-warning:`Fixed` Fixed a bug in Beam creation to accept BeamNames up to
  64 characters long, as per DICOM standard.

v 0.3.2
-------

General
^^^^^^^

* :bdg-warning:`Fixed` Fixed a bug in Beam.from_dicom which would cause it to
  raise an exception when the Beam had a non-standard Fluence Mode.

v 0.3.1
-------

General
^^^^^^^

* :bdg-warning:`Fixed` Removed the ``(3253,1000)`` ExtendedVAPlanInterface
  private tag to avoid conflicts with the generated beams.

v 0.3.0
-------

General
^^^^^^^

* :bdg-success:`Feature` Added support for overriding DICOM Tags in generated beams. Overrides
  can be provided via ``PlanGenerator.add_procedure``. Initial supported tags are ``BeamName``
  and ``ControlPointSequence[0].PatientSupportAngle``.

* :bdg-warning:`Change` The ``beam_name``/``beam_names`` parameter was removed from all
  ``Procedure`` classes. Beam Names will now be generated automatically, with the option
  to override them via the new beam overrides feature.

v 0.2.1
-------

Truebeam
^^^^^^^^

* :bdg-warning:`Fixed` Removed X and Y from beam limiting devices sequence. Now
  it consists of ASYMX, ASYMY and MLC.

v 0.2.0
-------

General
^^^^^^^

* :bdg-success:`Feature` Added versioning structure and changelog.

Creating a generator
^^^^^^^^^^^^^^^^^^^^

* :bdg-warning:`Fixed` The beam ``Manufacturer`` and ``ManufacturerModelName`` are now
  imported from the base plan.

v 0.1.2
-------

General
^^^^^^^

* :bdg-success:`Feature` Initial release.
