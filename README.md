# Oscillatory Incompleteness

**Gödel, Symbol Formation, and High-Dimensional Dynamics**

[![Status](https://img.shields.io/badge/status-submitted-blue)]()
[![Journal](https://img.shields.io/badge/journal-Synthese-orange)]()

> **Submitted to Synthese** (December 2025) - Manuscript ID: SYNT-S-25-03761

## Abstract

We prove a Gödel-style incompleteness theorem for a broad class of dynamical systems that generate symbolic codes from high-dimensional oscillatory dynamics via coarse-grained observation. Using the established result that such systems can encode arithmetic, we show that any effective theory capable of reasoning about their symbolic outputs is necessarily incomplete: there exist true properties of the generated symbol streams that it cannot decide.

The source of undecidable truths is the *dimensional bottleneck*—the mismatch between high-dimensional pre-symbolic dynamics and the low-dimensional symbolic interface.

## Repository Structure

```
├── manuscript.tex          # Main paper
├── manuscript.pdf          # Compiled PDF
├── references.bib          # Bibliography
├── code/
│   ├── bottleneck_diagram.py   # Figure 1: Dimensional bottleneck schematic
│   ├── osc_symbols.py          # Figure 2: Van der Pol → symbol sequences
│   ├── kuramoto_compute.py     # Kuramoto network computation demo
│   └── godel_oscillator.py     # Figure 3: Self-referential oscillator
└── figures/
    ├── fig1_vdp_symbols.png
    ├── fig2_bottleneck.png
    └── fig3_godel_oscillator.png
```

## Key Results

- **Theorem 4 (Oscillatory Incompleteness):** Any consistent, recursively axiomatizable, arithmetically adequate theory of oscillatory systems is necessarily incomplete.

- **Corollary 5 (Categorical Formulation):** There exist subobjects of the final coalgebra Σ^ℕ that no effective theory can classify.

## Citation

```bibtex
@article{todd2025oscillatory,
  author  = {Todd, Ian},
  title   = {Oscillatory Incompleteness: {G}ödel, Symbol Formation,
             and High-Dimensional Dynamics},
  journal = {Synthese},
  year    = {2025},
  note    = {Submitted}
}
```

## Contact

Ian Todd - itod2305@uni.sydney.edu.au
Sydney Medical School, University of Sydney
