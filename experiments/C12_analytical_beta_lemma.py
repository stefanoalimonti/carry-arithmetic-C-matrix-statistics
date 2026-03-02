"""
C12_analytical_beta_lemma.py — Binary model: β_eff vs Markov correlation ρ

For D-dimensional companion matrices with binary {0,1} entries drawn from
a symmetric Markov chain with correlation ρ:
  - β_eff is monotonically increasing in ρ
  - At ρ = 1/2 (carry chain value), β_eff ≈ 1.0 (GOE)
  - The i.i.d. (ρ=0) baseline is sub-GOE (β ≈ 0.64) due to entry constraints

This differs from actual carry ensembles where i.i.d. gives near-GUE (β ≈ 2)
and Markov correlation reduces β toward GOE. Both models agree that β_eff ≈ 1
at the carry chain correlation ρ = 1/2.

Method:
  - Exact enumeration for D = 5..8 (all binary configurations)
  - Monte Carlo for D = 5..50
  - Correlation sweep ρ = 0..1 for D = 20
  - Effective rank (participation ratio) analysis
  - Atas Wigner surmise for β_eff mapping
"""

import numpy as np
from scipy import integrate
from itertools import product as iproduct
import time


def atas_unnorm_pdf(r, beta):
    return (r + r**2)**beta / (1.0 + r + r**2)**(1.0 + 1.5 * beta)


def atas_mean_rtilde(beta):
    norm, _ = integrate.quad(atas_unnorm_pdf, 0, np.inf, args=(beta,), limit=200)
    lo, _ = integrate.quad(lambda r: r * atas_unnorm_pdf(r, beta), 0, 1, limit=200)
    hi, _ = integrate.quad(
        lambda r: (1.0 / r) * atas_unnorm_pdf(r, beta), 1, np.inf,
        limit=200
    )
    return (lo + hi) / norm


def invert_atas(r_target, beta_lo=0.01, beta_hi=8.0, tol=1e-8):
    from scipy.optimize import brentq
    try:
        return brentq(lambda b: atas_mean_rtilde(b) - r_target, beta_lo, beta_hi, xtol=tol)
    except ValueError:
        return float('nan')


def carry_markov_joint(n_entries, rho=0.5):
    """
    Joint distribution of (a_0, ..., a_{n-1}) from a carry-like Markov chain.
    States: {0, 1} with stationary π = (1/2, 1/2) and transition:
      P(0→0) = (1+ρ)/2,  P(0→1) = (1-ρ)/2
      P(1→0) = (1-ρ)/2,  P(1→1) = (1+ρ)/2
    Correlation between adjacent: ρ. For carry chain: ρ = 1/2.
    """
    trans = np.array([[(1 + rho) / 2, (1 - rho) / 2],
                      [(1 - rho) / 2, (1 + rho) / 2]])
    stationary = np.array([0.5, 0.5])

    configs = list(iproduct([0, 1], repeat=n_entries))
    probs = np.zeros(len(configs))

    for idx, cfg in enumerate(configs):
        p = stationary[cfg[0]]
        for i in range(1, n_entries):
            p *= trans[cfg[i - 1], cfg[i]]
        probs[idx] = p

    return configs, probs


def iid_joint(n_entries):
    """i.i.d. Bernoulli(1/2) distribution."""
    configs = list(iproduct([0, 1], repeat=n_entries))
    probs = np.ones(len(configs)) / len(configs)
    return configs, probs


def spacing_ratio_from_eigenvalues(eigenvalues):
    """
    Compute spacing ratios from eigenvalues.
    For companion matrices (non-Hermitian), use real parts.
    """
    real_parts = np.sort(np.real(eigenvalues))
    spacings = np.diff(real_parts)
    spacings = spacings[spacings > 1e-12]

    if len(spacings) < 2:
        return []

    ratios = []
    for i in range(len(spacings) - 1):
        s1, s2 = spacings[i], spacings[i + 1]
        ratios.append(min(s1, s2) / max(s1, s2))
    return ratios


def companion_matrix(coeffs):
    """Build companion matrix from polynomial coefficients [a_0, ..., a_{D-1}]."""
    D = len(coeffs)
    M = np.zeros((D, D))
    for i in range(1, D):
        M[i, i - 1] = 1.0
    for i in range(D):
        M[i, D - 1] = -coeffs[i]
    return M


def exact_enumeration(D, rho=0.5):
    """
    Exact enumeration of ⟨r̃⟩ for D-dim companion matrices.
    Entries a_0,...,a_{D-2} are random ({0,1}), a_{D-1} = 1 (ULC).
    """
    n_free = D - 1
    configs_markov, probs_markov = carry_markov_joint(n_free, rho)
    configs_iid, probs_iid = iid_joint(n_free)

    rtilde_markov = 0.0
    rtilde_iid = 0.0
    weight_markov = 0.0
    weight_iid = 0.0

    for idx in range(len(configs_markov)):
        cfg = list(configs_markov[idx]) + [1]
        M = companion_matrix(cfg)
        eigs = np.linalg.eigvals(M)
        ratios = spacing_ratio_from_eigenvalues(eigs)
        if ratios:
            mean_r = np.mean(ratios)
            rtilde_markov += probs_markov[idx] * mean_r
            weight_markov += probs_markov[idx]
            rtilde_iid += probs_iid[idx] * mean_r
            weight_iid += probs_iid[idx]

    r_m = rtilde_markov / weight_markov if weight_markov > 0 else float('nan')
    r_i = rtilde_iid / weight_iid if weight_iid > 0 else float('nan')
    return r_m, r_i, weight_markov, weight_iid


def mc_companion_markov(D, n_samples=50000, rho=0.5, rng=None):
    """Monte Carlo: Markov-correlated companion matrices."""
    if rng is None:
        rng = np.random.default_rng(42)

    trans = np.array([[(1 + rho) / 2, (1 - rho) / 2],
                      [(1 - rho) / 2, (1 + rho) / 2]])

    all_ratios = []
    for _ in range(n_samples):
        coeffs = np.zeros(D, dtype=float)
        coeffs[D - 1] = 1.0
        state = rng.integers(0, 2)
        coeffs[0] = state
        for j in range(1, D - 1):
            if rng.random() < trans[state, 1]:
                state = 1
            else:
                state = 0
            coeffs[j] = state

        M = companion_matrix(coeffs)
        eigs = np.linalg.eigvals(M)
        ratios = spacing_ratio_from_eigenvalues(eigs)
        all_ratios.extend(ratios)

    return np.mean(all_ratios) if all_ratios else float('nan')


def mc_companion_iid(D, n_samples=50000, rng=None):
    """Monte Carlo: i.i.d. Bernoulli(1/2) companion matrices."""
    if rng is None:
        rng = np.random.default_rng(123)

    all_ratios = []
    for _ in range(n_samples):
        coeffs = np.zeros(D, dtype=float)
        coeffs[D - 1] = 1.0
        coeffs[:D - 1] = rng.integers(0, 2, size=D - 1).astype(float)

        M = companion_matrix(coeffs)
        eigs = np.linalg.eigvals(M)
        ratios = spacing_ratio_from_eigenvalues(eigs)
        all_ratios.extend(ratios)

    return np.mean(all_ratios) if all_ratios else float('nan')


def main():
    print("=" * 72)
    print("C12: Analytical β_eff lemma — Markov correlation → β_eff < 2")
    print("=" * 72)

    r_goe = atas_mean_rtilde(1.0)
    r_gue = atas_mean_rtilde(2.0)
    r_poisson = 2 * np.log(2) - 1
    print(f"\nReference values:")
    print(f"  ⟨r̃⟩_Poisson = {r_poisson:.6f}")
    print(f"  ⟨r̃⟩_GOE     = {r_goe:.6f}")
    print(f"  ⟨r̃⟩_GUE     = {r_gue:.6f}")

    print(f"\nCarry Markov chain (base 2):")
    print(f"  Corr(c_j, c_{{j+1}}) = +1/2")
    print(f"  Stationary: π(0) = π(1) = 1/2")

    print(f"\n{'='*72}")
    print("PART 1: Exact enumeration (D = 5..8)")
    print(f"{'='*72}")

    print(f"\n{'D':>3} {'⟨r̃⟩_Markov':>12} {'β_Markov':>10} {'⟨r̃⟩_iid':>12} "
          f"{'β_iid':>10} {'Δ⟨r̃⟩':>10} {'wt_M':>8}")

    for D in range(5, 9):
        t0 = time.time()
        r_m, r_i, wm, wi = exact_enumeration(D)
        beta_m = invert_atas(r_m)
        beta_i = invert_atas(r_i)
        elapsed = time.time() - t0
        delta = r_i - r_m
        print(f"{D:3d} {r_m:12.6f} {beta_m:10.4f} {r_i:12.6f} "
              f"{beta_i:10.4f} {delta:+10.6f} {wm:8.4f}  ({elapsed:.1f}s)")

    print(f"\n{'='*72}")
    print("PART 2: Monte Carlo (D = 5, 10, 20, 30, 50)")
    print(f"{'='*72}")

    n_samples = 100000
    rng_m = np.random.default_rng(42)
    rng_i = np.random.default_rng(123)

    print(f"\n{'D':>3} {'⟨r̃⟩_Markov':>12} {'β_Markov':>10} {'⟨r̃⟩_iid':>12} "
          f"{'β_iid':>10} {'Δ⟨r̃⟩':>10}")

    mc_results = []
    for D in [5, 10, 20, 30, 50]:
        t0 = time.time()
        r_m = mc_companion_markov(D, n_samples, rng=np.random.default_rng(42))
        r_i = mc_companion_iid(D, n_samples, rng=np.random.default_rng(123))
        beta_m = invert_atas(r_m) if not np.isnan(r_m) else float('nan')
        beta_i = invert_atas(r_i) if not np.isnan(r_i) else float('nan')
        delta = r_i - r_m if not (np.isnan(r_i) or np.isnan(r_m)) else float('nan')
        elapsed = time.time() - t0
        mc_results.append({'D': D, 'r_m': r_m, 'r_i': r_i,
                          'beta_m': beta_m, 'beta_i': beta_i})
        print(f"{D:3d} {r_m:12.6f} {beta_m:10.4f} {r_i:12.6f} "
              f"{beta_i:10.4f} {delta:+10.6f}  ({elapsed:.1f}s)")

    print(f"\n{'='*72}")
    print("PART 3: Correlation strength sweep (D = 20)")
    print(f"{'='*72}")

    D_test = 20
    n_sweep = 50000
    print(f"\n{'ρ':>6} {'⟨r̃⟩':>12} {'β_eff':>10} {'ε=2-β':>10}")

    for rho in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        r_val = mc_companion_markov(D_test, n_sweep, rho=rho,
                                    rng=np.random.default_rng(42))
        beta_val = invert_atas(r_val) if not np.isnan(r_val) else float('nan')
        eps = 2.0 - beta_val if not np.isnan(beta_val) else float('nan')
        print(f"{rho:6.2f} {r_val:12.6f} {beta_val:10.4f} {eps:+10.4f}")

    print(f"\n{'='*72}")
    print("PART 4: Effective rank analysis")
    print(f"{'='*72}")

    for D in [10, 20, 50]:
        rho = 0.5
        n_samp = 10000

        markov_vecs = np.zeros((n_samp, D - 1))
        iid_vecs = np.zeros((n_samp, D - 1))

        rng_loc = np.random.default_rng(42)
        trans = np.array([[(1 + rho) / 2, (1 - rho) / 2],
                          [(1 - rho) / 2, (1 + rho) / 2]])

        for s in range(n_samp):
            state = rng_loc.integers(0, 2)
            for j in range(D - 1):
                markov_vecs[s, j] = state
                state = 1 if rng_loc.random() < trans[state, 1] else 0
            iid_vecs[s] = rng_loc.integers(0, 2, size=D - 1)

        cov_m = np.cov(markov_vecs.T)
        cov_i = np.cov(iid_vecs.T)

        eig_m = np.sort(np.linalg.eigvalsh(cov_m))[::-1]
        eig_i = np.sort(np.linalg.eigvalsh(cov_i))[::-1]

        eff_rank_m = (np.sum(eig_m))**2 / np.sum(eig_m**2)
        eff_rank_i = (np.sum(eig_i))**2 / np.sum(eig_i**2)

        print(f"\nD = {D}:")
        print(f"  Markov effective rank: {eff_rank_m:.2f} / {D-1} "
              f"= {eff_rank_m/(D-1):.3f}")
        print(f"  i.i.d.  effective rank: {eff_rank_i:.2f} / {D-1} "
              f"= {eff_rank_i/(D-1):.3f}")
        print(f"  Ratio: {eff_rank_m/eff_rank_i:.4f}")

    print(f"\n{'='*72}")
    print("RESULT SUMMARY")
    print(f"{'='*72}")
    print("""
OBSERVATION (Binary Markov companion matrices).

For D-dimensional companion matrices with last-column entries
(a_0, ..., a_{D-2}) in {0,1} from a symmetric 2-state Markov chain
with correlation rho in [0,1] and a_{D-1} = 1 (ULC):

  (i)   beta_eff(rho) is monotonically increasing in rho.
  (ii)  beta_eff(0) ~ 0.64 (sub-GOE) for the i.i.d. baseline.
  (iii) beta_eff(1/2) ~ 1.0 (GOE) at the carry chain correlation.
  (iv)  beta_eff crosses 2 (GUE) near rho ~ 0.75.

EFFECTIVE RANK: The covariance matrix of the Markov random vector
has effective rank (participation ratio) ~ 0.60D at rho=1/2,
compared to ~ 1.00D for i.i.d.

KEY DISTINCTION from actual carry ensembles:
  - Binary model:  i.i.d. baseline is sub-GOE (beta ~ 0.6), and
    positive correlation INCREASES beta toward GOE.
  - Actual carry:  i.i.d. (shuffled) baseline is near-GUE (beta ~ 2),
    and Markov correlation DECREASES beta toward GOE.
  - Both converge to beta_eff ~ 1 (GOE) at the carry chain rho = 1/2.

The binary model correctly identifies the GOE fixed point at rho=1/2
and the effective rank mechanism, but the direction of the effect
(relative to the i.i.d. baseline) depends on the entry distribution.
""")


if __name__ == "__main__":
    main()
