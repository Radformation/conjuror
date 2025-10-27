================
General Overview
================

Conjuror
--------

Conjuror is a Python-based toolkit designed to generate DICOM objects for use in radiation therapy workflows.
It simplifies the creation and manipulation of specialized DICOM files, specifically:

* RT Plan (Radiotherapy Plan) objects

These generated files can be used for testing, QA, and automation pipelines within radiation oncology environments.

Conjuror is particularly useful for developers, medical physicists, and QA engineers who need to simulate or construct DICOM datasets compatible with Varian systems such as TrueBeam and Halcyon.


DICOM
-----

DICOM (Digital Imaging and Communications in Medicine) is the international standard (ISO 12052) for storing, transmitting, and managing medical imaging information.
It defines both a file format and a network communication protocol for interoperability between imaging systems, treatment planning systems, and QA tools.

For detailed reference:

`DICOM Standard <https://www.dicomstandard.org/current>`__

`DICOM Part 3: Information Object Definitions <https://dicom.nema.org/medical/dicom/current/output/html/part03.html>`__

`DICOM Composite Information Object Definitions <https://dicom.nema.org/medical/dicom/current/output/html/part03.html#chapter_A>`__

`DICOM RT Plan IOD <https://dicom.nema.org/medical/dicom/current/output/html/part03.html#sect_A.20>`__


Design and Conventions
----------------------

Conjuror follows established clinical and engineering conventions for radiation therapy systems.

* Millimeters (mm) for spatial coordinates and geometry.

* Degrees (°) for angles such as gantry, collimator, and couch rotation.

* IEC 61217 is adopted as the coordinate system standard.

  *IEC. 2011. Ed 2. Radiotherapy Equipment - Coordinates, Movements and Scales.*

System Compatibility
--------------------

Conjuror is tailored for the Varian platforms:

* `ARIA <https://varian.widen.net/s/jdrkjxxqj8/varian_dicomworklist_v18.1_dicomconformancestatement>`__

* `TrueBeam <https://varian.widen.net/s/kvprxqw7zm/varian_truebeam_v4.1_dicom_conformance_statement>`__

* `Halcyon <https://varian.widen.net/s/bnbzp6vs77/varian_halcyon_imaging_v4.0_dicom_conformance_statement>`__

The generated RT objects adhere to system-specific DICOM tag structures and private data elements known to these platforms.

Dependencies and Integration
----------------------------

Conjuror builds on a robust Python ecosystem for medical imaging and QA automation.

Core Dependencies
^^^^^^^^^^^^^^^^^

* pydicom – for reading, writing, and manipulating DICOM datasets.

* numpy – for matrix operations and numeric calculations.

Integration
^^^^^^^^^^^

* pylinac – integration for image analysis.

* RadMachine – integration for UI interface within Radformation’s machine QA environment.
