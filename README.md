# carry-arithmetic-C-matrix-statistics

**Eigenvalue Statistics of Carry Companion Matrices: A Markov-Driven GOE↔GUE Transition in Sparse Non-Hermitian Ensembles**

*Author: Stefano Alimonti* · [ORCID 0009-0009-1183-1698](https://orcid.org/0009-0009-1183-1698)

## Main Result

Carry companion matrices exhibit random-matrix statistics governed by the Markov structure of carry sequences:

- **Full Markov correlation** → GOE-like ($\beta \approx 1$)
- **i.i.d. digits** → near-GUE ($\beta \approx 1.8$)

Three analytical propositions underpin the transition:

1. **Proposition 1:** The carry chain has provably positive lag-1 autocorrelation, with $\mathrm{Corr}(c_2, c_3) = 1/\sqrt{2}$ exactly.
2. **Proposition 2:** The effective rank of the Markov covariance matrix is $g(\rho) = (1-\rho^2)/(1+\rho^2)$, giving $g(1/2) = 3/5$ at the Diaconis–Fulman spectral gap value.
3. **Proposition 3:** The real eigenvalue fraction $f_{\mathrm{real}}(\rho)$ is monotonically increasing in $\rho$, anti-correlated with $g(\rho)$.

A computational bound establishes $\beta_{\mathrm{eff}} \leq 2 - \varepsilon(D)$ with $\varepsilon > 0$ at 99.9% bootstrap confidence.

## Repository Structure

```
paper/carry_matrix_statistics.md       The paper
experiments/
  C01_goe_gue_unitary_transition.py    GOE/GUE/Unitary transition
  C02_goe_gue_finite_size.py           Finite-size scaling
  C03_factorials_goe_gue_transition.py Factorial ensemble transition
  C04_gue_correlation.py               GUE correlation functions
  C05_goe_gue_scaling_limit.py         Scaling limit analysis
  C06_analytical_ensemble_structure.py  Analytical ensemble structure
  C07_goe_spacing_ratio.py             GOE spacing ratio test
  C08_beta_interpolation.py            Beta interpolation curve
  C09_symmetry_mechanism.py            Symmetry breaking mechanism
  C10_number_variance.py               Number variance Σ²(L)
  C11_beta_bound.py                    Computational bound on β_eff
  C12_analytical_beta_lemma.py         Binary Markov model analysis
  C13_analytical_foundations.py        Propositions 1, 2, 3 verification
src/
  carry_utils.py                       Shared utility functions
```

## Reproduction

```bash
pip install numpy scipy sympy
python experiments/C13_analytical_foundations.py   # Propositions 1-3
python experiments/C07_goe_spacing_ratio.py        # Core spacing ratio test
# ... through C12 for all computational results
```

## Dependencies

- Python >= 3.8, NumPy, SciPy, SymPy

## Companion Papers

| Label | Title | Repository |
|-------|-------|------------|
| [A] | Spectral Theory of Carries | [`carry-arithmetic-A-spectral-theory`](https://github.com/stefanoalimonti/carry-arithmetic-A-spectral-theory) |
| [D] | The Carry-Zero Entropy Bound | [`carry-arithmetic-D-factorization-limits`](https://github.com/stefanoalimonti/carry-arithmetic-D-factorization-limits) |
| [E] | The Trace Anomaly of Binary Multiplication | [`carry-arithmetic-E-trace-anomaly`](https://github.com/stefanoalimonti/carry-arithmetic-E-trace-anomaly) |
| [F] | Exact Covariance Structure | [`carry-arithmetic-F-covariance-structure`](https://github.com/stefanoalimonti/carry-arithmetic-F-covariance-structure) |
| [G] | The Angular Uniqueness of Base 2 | [`carry-arithmetic-G-angular-uniqueness`](https://github.com/stefanoalimonti/carry-arithmetic-G-angular-uniqueness) |
| [H] | Carry Polynomials and the Partial Euler Product | [`carry-arithmetic-H-euler-control`](https://github.com/stefanoalimonti/carry-arithmetic-H-euler-control) |
| [P1] | Pi from Pure Arithmetic | [`carry-arithmetic-P1-pi-spectral`](https://github.com/stefanoalimonti/carry-arithmetic-P1-pi-spectral) |
| [P2] | The Sector Ratio in Binary Multiplication | [`carry-arithmetic-P2-sector-ratio`](https://github.com/stefanoalimonti/carry-arithmetic-P2-sector-ratio) |

### Citation

```bibtex
@article{alimonti2026matrix_statistics,
  author  = {Alimonti, Stefano},
  title   = {Eigenvalue Statistics of Carry Companion Matrices},
  year    = {2026},
  note    = {Preprint},
  url     = {https://github.com/stefanoalimonti/carry-arithmetic-C-matrix-statistics}
}
```

## License

Paper: CC BY 4.0. Code: MIT License.
