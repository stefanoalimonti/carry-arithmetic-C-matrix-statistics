# Experiments — Paper C

| Script | Description | Referenced in |
|--------|-------------|---------------|
| C01_goe_gue_unitary_transition.py | GOE/GUE/Unitary ensemble χ² classification by base | §3.1 |
| C02_goe_gue_finite_size.py | Finite-size L² diagnostics and Poisson crossover | §3.1 |
| C03_factorials_goe_gue_transition.py | Factorial-structure ensemble transition | §3 (supporting) |
| C04_gue_correlation.py | Markov vs i.i.d. controlled comparison | §3.1 |
| C05_goe_gue_scaling_limit.py | Large-D spacing ratio scaling | §3.3 |
| C06_analytical_ensemble_structure.py | Eigenvalue density analysis | Supporting |
| C07_goe_spacing_ratio.py | Level spacing ratio ⟨r̃⟩ (Atas et al.) | §3.3 |
| C08_beta_interpolation.py | One-parameter GOE↔GUE β interpolation | §4.1 |
| C09_symmetry_mechanism.py | Eigenvector orthogonality mechanism | §4.2 |
| C10_number_variance.py | Number variance Σ²(L) confirmation | §4.6 |
| C11_beta_bound.py | Computational bound β_eff ≤ 2 − ε | §4.3 |
| C12_analytical_beta_lemma.py | Binary Markov model: β_eff vs ρ | §4.4 |
| C13_analytical_foundations.py | **Propositions 1, 2, 3** verification | §2.1, §2.2, §4.5 |

## Shared Utilities

`../src/carry_utils.py` — common functions for carry matrix construction.

## Requirements

- Python >= 3.8, NumPy, SciPy, SymPy
