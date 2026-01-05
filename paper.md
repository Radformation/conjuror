---
title: 'Conjuror: An RT DICOM QA plan generator'
tags:
  - medical physics
  - radiotherapy
  - DICOM
  - quality assurance
  - python
authors:
  - name: João Silveira
    orcid: 0009-0003-8902-6695
    affiliation: 1
  - name: James R. Kerns
    orcid: 0000-0002-2019-7627
    affiliation: 1
  - name: Hasan Ammar
    orcid: TBD
    affiliation: 1
affiliations:
  - name: Radformation, New York, NY, United States of America
    index: 1
date: 5 January 2026
bibliography: paper.bib
---

# Summary

Quality assurance (QA) of linear accelerators (linacs) is a critical component of safe
and effective external beam radiation therapy (RT). QA procedures often require the
delivery of standardized treatment plans to verify machine performance, including
geometric accuracy, dosimetric consistency, and imaging system alignment. While the
Digital Imaging and Communications in Medicine (DICOM) standard defines how radiotherapy
treatment plans (RT Plans) should be represented [@NEMA_DICOM], constructing valid
RT Plan files for QA purposes is challenging due to the complexity of the standard and
the need for machine-specific configurations.

Conjuror is a Python toolkit that  enables the creation of DICOM RT Plans tailored for
Varian linacs QA. The toolkit provides an accessible and reproducible way to generate
test plans for routine and specialized QA procedures, eliminating the need for manual
authoring or reliance on proprietary planning systems.

# Statement of need

Routine and end-to-end QA of linacs requires standardized test plans that probe key
performance characteristics, such as multileaf collimator (MLC) positioning, imaging
isocenter coincidence, and mechanical accuracy. Guidelines such as AAPM Task
Group 142 [@TG142] prescribe many of these tests.

In current practice, generating these QA plans typically involves either (1) manual construction
within proprietary treatment planning systems (TPSs), or (2) relying on vendor-provided
test plans which may not be adaptable to institution-specific protocols or research needs.
Additionally, QA plans are often distributed across multiple files, making it cumbersome
to assemble all required beams in a single cohesive plan and hindering streamlined, automated workflows.

Conjuror addresses this gap by providing an open-source, Python-based framework for
generating DICOM RT Plans specifically for Varian linacs. The library supports automated
plan construction using customizable QA procedures, provides visualization of expected
fluence and control point sequences, and integrates with the broader Python QA ecosystem
(e.g., Pylinac [@Kerns2023]) via pydicom-based DICOM handling [@Mason2022]).

Conjuror addresses this gap by offering an open-source, Python-based framework for
generating DICOM RT Plans specifically for Varian linacs. The library enables:
* Automated generation of QA plans as part of routine machine QA workflows.
* Customization and reproducibility of commonly used QA plans based on pre-prepared templates.
* Visualization of the expected fluence and the sequence of control points of each beam.
* Integration with Python QA ecosystems (e.g. Pylinac [@Kerns2023]), supporting automated analysis and research into novel QA strategies.
* Built on established Python packages such as pydicom [@Mason2022], ensuring compatibility with the broader medical imaging software ecosystem.

# State of the field

Several open-source projects address aspects of linac QA, but none focus on generating
valid DICOM RT Plan objects for test delivery. For example, pylinac [@Kerns2023] and
pymedphys [@Biggs2022] provide robust tools for analyzing machine QA data but assumes
that test plans have already been generated and delivered. As a foundational dependency,
pydicom [@Mason2022] enables reading and writing of DICOM files, but it does not provide
domain-specific functionality for radiotherapy QA plan creation.

In contrast, Conjuror bridges this gap by directly supporting the programmatic
generation of DICOM RT Plans for Varian linac QA.
This focus on reproducible, automated test plan creation complements existing QA
analysis software and contributes to a more complete open-source ecosystem for medical
physics research and clinical practice.


# Software Design

- **RT Beams** - A `Beam` object represents a DICOM RT Beam and contains all information defining the Control Point Sequence of an external radiation beam, such as monitor units, MLC, jaw, gantry, collimator and couch positions.
- **QA procedures** - A `QAProcedure` represents a logical grouping of `Beam` based on domain-specific use cases. Although a QA Procedure is not a DICOM-standard entity, it facilitates the creation of QA tests. For example, a Winston–Lutz procedure may include multiple Beams, each corresponding to a unique combination of gantry, collimator, and couch positions.
- **Plan synthesis** - A `PlanGenerator` builds DICOM RT Plan datasets that encompasses all Beams across all QA Procedures. Since a QA Procedure is not part of the DICOM hierarchy, Beams are stored as a sequential, ungrouped list.
- **Fluence simulation** - An imager simulator generates the fluence of the control points of an RT Beam. The imager simulator accepts configurable pixel sampling and source to imager distance.
- **Visualization** - Plotting helpers (Plotly) animate MLC motion and chart beam control-point dynamics for inspection.

# Research impact statement
Conjuror enables reproducible and scalable QA plan generation workflows by allowing RT Plans to be created and parameterized directly in code. This makes it possible to define QA tests as version-controlled templates, generate systematic plan variations (e.g., MLC positioning sweeps or imaging geometry perturbations), and integrate plan creation into automated pipelines for end-to-end QA. By reducing dependence on interactive plan authoring and supporting standardization across machines and institutions, the library facilitates reproducible QA studies and accelerates development and evaluation of novel QA strategies.

# AI usage disclosure

No generative AI tools were used in the conceptual design or QA procedure domain logic.
ChatGPT 5.1-Pro was used to help polishing this software (e.g. generating unit-tests) and writing of this manuscript.

# Acknowledgements

We thank Radformation for continuing to support the open-source work of this project.

# References
