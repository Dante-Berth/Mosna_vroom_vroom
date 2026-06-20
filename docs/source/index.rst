MOSNA — Multi-Omics Spatial Network Analysis
============================================

*"vroom vroom" edition* 🏎️

**MOSNA** is a Python package for **spatial omics data analysis**. It extracts
clinically relevant features from single-cell spatial measurements
(transcriptomics, proteomics, multiplexed imaging), reconstructs spatial
networks, computes interaction / neighborhood statistics, and trains predictive
models integrating clinical outcomes.

This documentation describes the **accelerated fork**
`Dante-Berth/Mosna_vroom_vroom <https://github.com/Dante-Berth/Mosna_vroom_vroom>`_,
a cleaned-up, modularised and benchmarked version of the original
`AlexCoul/mosna <https://github.com/AlexCoul/mosna>`_ by Alexis Coullomb.

.. admonition:: The science is unchanged
   :class: important

   This fork changes the code **structure** and the **I/O / compute speed**, not
   the math. Equivalence with the original implementation is verified
   bit-for-bit by the test suite (see :doc:`theoretical_concepts` and
   :doc:`getting_started`). Where a kernel was rewritten, outputs are asserted
   identical to the original before any speedup is claimed.

This documentation follows the `Diátaxis <https://diataxis.fr/>`_ structure:

* :doc:`mosna_at_a_glance` — what the library does and how the pieces fit.
* :doc:`getting_started` — install, verify, and run your first analysis.
* :doc:`theoretical_concepts` — the analysis pipeline and the math behind it.
* :doc:`how_to_guides` — task-oriented recipes (fast I/O, niches, modeling…).
* :doc:`api` — the full API reference, generated from docstrings.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   mosna_at_a_glance
   getting_started
   theoretical_concepts
   how_to_guides
   examples
   api

Resources
---------

* Source code: https://github.com/Dante-Berth/Mosna_vroom_vroom
* Original library: https://github.com/AlexCoul/mosna
* Methodology paper (original MOSNA):
  `PMC / spatial omics network analysis <https://www.ncbi.nlm.nih.gov/pmc/>`_
* Changelog of this fork: ``MODIFICATIONS.md`` in the repository root.

Indices and tables
------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
