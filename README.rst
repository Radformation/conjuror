========
Conjuror
========

This is a Python library for creating synthetic RT DICOM images, usually for testing of machine QA analysis software
such as ``pylinac``. It can also generate RT DICOM Plan files for use in delivering machine QA plans.

Installation
============

Conjuror can be installed via pip:

.. code-block:: console

    pip install conjuror

Developer Installation
======================

To install the development version of Conjuror, clone the repository and install it in editable mode:

.. code-block:: console

     uv sync

Additionally, the developer should install ``pre-commit`` hooks to ensure linting and format consistency:

.. code-block:: console

    uv run pre-commit install

For one-off checks of the repo:

.. code-block:: console

   uv run pre-commit run --all-files

Usage
=====

.. code-block:: python

    from conjuror import PlanGenerator, PicketFence
    conjuror = PlanGenerator()
    conjuror.add_beamset(PicketFence(num_pickets=7))
