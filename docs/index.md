# MOSNA — Multi-Omics Spatial Network Analysis

*"vroom vroom" edition* 🏎️

**MOSNA** is a Python package for **spatial omics data analysis**. It extracts
clinically relevant features from single-cell spatial measurements
(transcriptomics, proteomics, multiplexed imaging), reconstructs spatial
networks, computes interaction / neighborhood statistics, and trains predictive
models integrating clinical outcomes.

This documentation describes the **accelerated fork**
[Dante-Berth/Mosna_vroom_vroom](https://github.com/Dante-Berth/Mosna_vroom_vroom),
a cleaned-up, modularised and benchmarked version of the original
[AlexCoul/mosna](https://github.com/AlexCoul/mosna) by Alexis Coullomb.

!!! important "The science is unchanged"

    This fork changes the code **structure** and the **I/O / compute speed**, not
    the math. Equivalence with the original implementation is verified
    bit-for-bit by the test suite (see [Theoretical concepts](theoretical_concepts.md)
    and [Getting started](getting_started.md)). Where a kernel was rewritten,
    outputs are asserted identical to the original before any speedup is claimed.

This documentation follows the [Diátaxis](https://diataxis.fr/) structure:

* [MOSNA at a glance](mosna_at_a_glance.md) — what the library does and how the pieces fit.
* [Getting started](getting_started.md) — install, verify, and run your first analysis.
* [Theoretical concepts](theoretical_concepts.md) — the analysis pipeline and the math behind it.
* [How-to guides](how_to_guides.md) — task-oriented recipes (fast I/O, niches, modeling…).
* [API reference](api.md) — the full API reference, generated from docstrings.

## Resources

* Source code: <https://github.com/Dante-Berth/Mosna_vroom_vroom>
* Original library: <https://github.com/AlexCoul/mosna>
* Methodology paper (original MOSNA):
  [PMC / spatial omics network analysis](https://www.ncbi.nlm.nih.gov/pmc/)
* Changelog of this fork: `MODIFICATIONS.md` in the repository root.
