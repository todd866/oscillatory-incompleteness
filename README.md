# Oscillatory Incompleteness

**Gödel Incompleteness for Oscillatory Code-Forming Systems: A Dynamical and Categorical Perspective**

> Target journal: *Synthese* (Springer) - covered by USyd Read & Publish

## Core Thesis

Any physically reasonable class of oscillatory systems that (i) generates symbolic codes via finite-resolution measurement and (ii) can encode arithmetic will inherit Gödel-style incompleteness: there are true facts about their code-producing dynamics that no fixed effective theory of those dynamics can decide.

The source of undecidable truths is the mismatch between high-dimensional pre-symbolic dynamics and the low-dimensional symbolic interface—i.e., code formation itself.

## Key Literature

### Dynamical systems + undecidability
- Moore (1990) "Unpredictability and Undecidability in Dynamical Systems" - particle in 3D potential simulates TM
- da Costa & Doria (1991) "Undecidability and incompleteness in classical mechanics"
- Platzer - Differential dynamic logic: Gödel applies to ODEs that can define natural numbers

### Categorical/topos Gödel
- Joyal - arithmetic universes
- Maietti, Hofstra - categorical incompleteness proofs

## Paper Structure

1. **Introduction** - Gödel usually about formal syntax; same phenomenon in physically realised symbol-generating systems
2. **Oscillatory Systems and Symbol Formation** - Define OscSys (state space + flow + observation → symbols)
3. **Arithmetic in Oscillatory Systems** - Formalise encoding of arithmetic/TMs
4. **Oscillatory Incompleteness Theorem** - Main result
5. **Categorical Perspective** - Coalgebras, final coalgebra, topos-internal incompleteness
6. **Application: Neural Oscillations** - Brains as oscillatory code-forming systems
7. **Philosophical Implications** - Symbol formation from high-D dynamics as the mechanism behind Gödel
8. **Conclusion**

## Code Components

1. `osc_symbols.py` - Van der Pol → partition → symbol sequence demo
2. `kuramoto_compute.py` - Kuramoto network + discrete layer (universal computation)
3. `godel_oscillator.py` - System whose dynamics depend on its own measured past (self-reference)

## Repository Structure

```
├── manuscript.tex          # Synthese paper draft
├── references.bib          # Bibliography
├── code/
│   ├── osc_symbols.py      # Oscillator → symbols demo
│   ├── kuramoto_compute.py # Oscillatory computation
│   └── godel_oscillator.py # Self-referential oscillator
├── figures/
└── archive/
```

## Related Projects

- `1_falsifiability/` - Limits of falsifiability (BioSystems 2025)
- `6_clinical_validity_bounds/` - Dimensional validity bounds (IJMI, in prep)
- `15_code_formation/` - Code formation theory

## Citation

```bibtex
@article{todd2025oscillatory,
  author  = {Todd, Ian},
  title   = {Oscillatory Incompleteness: {G}ödel, Symbol Formation,
             and High-Dimensional Dynamics},
  journal = {Synthese},
  year    = {2025},
  note    = {In preparation}
}
```

## Contact

Ian Todd - itod2305@uni.sydney.edu.au
