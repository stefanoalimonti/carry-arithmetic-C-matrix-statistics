# Reader's Guide: A GOE-GUE Transition in Carry Companion Matrices

**Stefano Alimonti** — March 2026

*This guide introduces the carry companion matrix ensemble for readers familiar with random matrix theory but not with carry arithmetic. It explains where the matrices come from, why their correlations are structured, and what drives the universality-class transition.*

---

## 1. Where the Matrices Come From

When two integers $p$ and $q$ are multiplied in base 2, the carry chain $c_0, c_1, \ldots, c_D$ records the overflow at each column position (see [A] for the spectral theory of carries). The Carry Representation Theorem [B] shows that

$$g(x) h(x) = f(x) + (x - 2) Q(x)$$

where $g, h, f$ are the digit polynomials of $p$, $q$, and $N = pq$, and $Q(x)$ is the carry quotient polynomial with coefficients $q_k = -c_{k+1}$. After the standard monic normalization/indexing convention for companion matrices, the **carry companion matrix** $M$ is a $D \times D$ matrix (where $D = \deg Q$) with ones on the sub-diagonal and the carry values $c_1, \ldots, c_D$ in the last column (since the entries are $-q_k = c_{k+1}$).

For a single multiplication, $M$ is deterministic. Over the ensemble of all $D$-bit semiprimes $N = pq$ (with $p, q$ drawn uniformly from $d$-bit primes), the carry coefficients become random, producing a natural random matrix ensemble.

**Key structural feature:** This is a *sparse* non-Hermitian ensemble — randomness lives only in the last column. The remaining entries are fixed (ones on the sub-diagonal, zeros elsewhere). This places the ensemble outside the standard Wigner, Wishart, and Ginibre frameworks.

---

## 2. The Correlation Structure

What makes this ensemble unusual is that the random entries in the last column are **not independent**. Adjacent carry values share input bits: $c_{k+1} = \lfloor(\text{conv}_k + c_k)/2\rfloor$ depends on $c_k$, which in turn depends on $c_{k-1}$. This creates a Markov correlation structure with:

- **Lag-1 autocorrelation** $\text{Corr}(c_2, c_3) = 1/\sqrt{2} \approx 0.707$ (Proposition 1, proved exactly).
- **Bulk autocorrelation** numerically observed to increase with position $j$ toward 1.
- **Spectral gap** $\rho = 1/2$ (the Diaconis-Fulman eigenvalue [A]), governing the exponential decorrelation rate.

If the last-column entries were i.i.d. (same marginals, no correlation), the ensemble would exhibit GUE-like statistics. The Markov correlations then suppress the complex-like behavior of the i.i.d. model: they increase the real-eigenvalue fraction, reduce the effective perturbation rank, and push the statistics toward the GOE side of the interpolation.

---

## 3. The Effective Rank Mechanism

The transition is quantified by the **normalized effective rank** of the covariance matrix. For a stationary Markov chain with lag-1 correlation $\rho$ and length $D$, the covariance matrix is Kac-Murdock-Szego Toeplitz with entries $\Sigma_{ij} = \sigma^2 \rho^{|i-j|}$.

**Proposition 2:** The normalized effective rank (participation ratio) is

$$g(\rho) = \frac{1 - \rho^2}{1 + \rho^2}$$

in the $D \to \infty$ limit. At the physical value $\rho = 1/2$ (Diaconis-Fulman spectral gap):

$$g(1/2) = \frac{3}{5} = 0.60$$

The correlated random vector effectively has only 60% of the degrees of freedom of an uncorrelated one. This rank reduction is the mechanism driving the GOE shift: fewer effective independent directions means more "real-like" spectral statistics.

---

## 4. The Transition in Four Statistics

The GOE-GUE transition is confirmed independently by four spectral observables:

| Statistic | GUE prediction ($\beta = 2$) | Carry matrices | GOE prediction ($\beta = 1$) |
|---|---|---|---|
| Spacing ratio $\langle\tilde{r}\rangle$ | 0.603 | 0.50-0.58 | 0.536 |
| $\beta_{\text{eff}}$ | 2.0 | $\leq 2 - \varepsilon$ | 1.0 |
| Real eigenvalue fraction | low | increasing with $\rho$ | high |
| Number variance $\Sigma^2(L)$ | GUE curve | intermediate | GOE curve |

The interpolation is continuous: as the correlation parameter $\lambda$ varies from 0 (i.i.d.) to 1 (physical Markov), $\beta_{\text{eff}}$ decreases monotonically from $\approx 1.8$ (near-GUE) toward the GOE side, with a slight sub-GOE endpoint in the largest-$D$ physical data. This is analogous to the $\beta$-ensembles of Dumitriu and Edelman, but driven by a *physical* Markov correlation rather than an artificial interpolation.

---

## 5. Important Caveats

- All spectral statistics are **empirical** (high confidence, no rigorous proof).
- Carry matrices are *sparse companion matrices with integer entries* — they do not belong to any standard RMT ensemble.
- The parallel with Riemann zero statistics (GUE) is suggestive but not formal: carry matrices approximate Euler factors of $\zeta(s)$ [B], but the connection to zero statistics is indirect.
- The GOE-GUE transition is driven by the Markov structure of *carries in multiplication*, which is fundamentally different from the time-reversal symmetry that distinguishes GOE from GUE in physics.

---

## 6. Connection to the Broader Project

The carry companion matrix ensemble connects to several other results:

- **Spectral theory [A]:** The Diaconis-Fulman eigenvalue $\rho = 1/2$ that controls the correlation is the same spectral gap that governs carry mixing.
- **Zeta approximation [B]:** The ensemble-averaged spectral determinant $\langle|\det(I - M/l^s)|\rangle$ approximates $|1 - l^{-s}|^{-1}$, linking each matrix to an Euler factor.
- **Covariance structure [F]:** The exact second-moment structure (Cov, autocorrelation) used in Propositions 1-2 is proved rigorously in the companion paper.

---

## References

1. H.L. Montgomery, "The pair correlation of zeros of the zeta function," *Proc. Symp. Pure Math.* 24, 181–193, 1973.
2. N.M. Katz, P. Sarnak, *Random Matrices, Frobenius Eigenvalues, and Monodromy*, AMS, 1999.
3. [A] S. Alimonti, "Spectral Theory of Carries in Positional Multiplication," this series. doi:[10.5281/zenodo.18895593](https://doi.org/10.5281/zenodo.18895593) — [GitHub](https://github.com/stefanoalimonti/carry-arithmetic-A-spectral-theory)
4. [B] S. Alimonti, "Carry Polynomials and the Euler Product," this series. doi:[10.5281/zenodo.18895597](https://doi.org/10.5281/zenodo.18895597) — [GitHub](https://github.com/stefanoalimonti/carry-arithmetic-B-zeta-approximation)
5. [F] S. Alimonti, "Exact Covariance Structure of Binary Carry Chains," this series. doi:[10.5281/zenodo.18895607](https://doi.org/10.5281/zenodo.18895607) — [GitHub](https://github.com/stefanoalimonti/carry-arithmetic-F-covariance-structure)

---

*CC BY 4.0*
