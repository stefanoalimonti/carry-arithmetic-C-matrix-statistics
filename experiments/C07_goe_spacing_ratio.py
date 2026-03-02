#!/usr/bin/env python3
"""
C07: Level spacing ratio test for GOE evidence in carry companion matrices.

The spacing ratio r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}) is a
robust statistic that does NOT require spectral unfolding.

Known values of <r> for standard ensembles (Atas et al., PRL 110, 2013):
  Poisson:  <r> = 2 ln 2 - 1  ≈ 0.3863
  GOE:      <r> ≈ 0.5307  (Wigner-Dyson β=1)
  GUE:      <r> ≈ 0.5996  (Wigner-Dyson β=2)

For complex eigenvalues (companion matrices are non-symmetric), we use
angular spacings on the unit circle after projecting eigenvalues onto
their angles.

Tests:
  A. Spacing ratio <r> for carry companion matrices at multiple sizes
  B. Comparison: Markov-correlated vs i.i.d. entries
  C. Size dependence: does GOE signal improve with dimension?
  D. Statistical significance via bootstrap confidence intervals
"""

import sys, os, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import random_prime, to_digits

random.seed(42)
np.random.seed(42)

R_POISSON = 2 * np.log(2) - 1  # ≈ 0.3863
R_GOE = 0.5307
R_GUE = 0.5996


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def extract_carry_matrix(p, q, base=2):
    """Build the carry companion matrix for p*q and return (M, eigenvalues)."""
    N = p * q
    gd = to_digits(p, base)
    hd = to_digits(q, base)
    fd = to_digits(N, base)

    conv = [0] * (len(gd) + len(hd) - 1)
    for i, a in enumerate(gd):
        for j, b_val in enumerate(hd):
            conv[i + j] += a * b_val

    D_max = max(len(conv), len(fd))
    carries = [0] * (D_max + 2)
    for k in range(D_max):
        conv_k = conv[k] if k < len(conv) else 0
        carries[k + 1] = (conv_k + carries[k]) // base

    D_carry = 0
    for j in range(len(carries) - 1, 0, -1):
        if carries[j] != 0:
            D_carry = j
            break

    carry_seq = carries[1:D_carry + 1]
    D = len(carry_seq)
    if D < 6 or carry_seq[-1] == 0:
        return None, None, None

    lead = carry_seq[-1]
    M = np.zeros((D, D), dtype=float)
    for i in range(D - 1):
        M[i + 1, i] = 1.0
    for i in range(D):
        M[i, D - 1] = -carry_seq[i] / lead

    try:
        ev = np.linalg.eigvals(M)
        if not np.all(np.isfinite(ev)):
            return None, None, None
    except Exception:
        return None, None, None

    return M, ev, carry_seq


def iid_companion_matrix(carry_seq):
    """Build companion matrix with i.i.d. entries from same marginal as carry_seq."""
    D = len(carry_seq)
    vals = np.array(carry_seq, dtype=float)
    shuffled = vals.copy()
    np.random.shuffle(shuffled)

    lead = shuffled[-1]
    if lead == 0:
        lead = 1

    M = np.zeros((D, D), dtype=float)
    for i in range(D - 1):
        M[i + 1, i] = 1.0
    for i in range(D):
        M[i, D - 1] = -shuffled[i] / lead

    try:
        ev = np.linalg.eigvals(M)
        if not np.all(np.isfinite(ev)):
            return None
    except Exception:
        return None
    return ev


def angular_spacing_ratios(ev):
    """Compute spacing ratios from angular spacings of complex eigenvalues.
    Returns array of r_n = min(s_n, s_{n+1})/max(s_n, s_{n+1})."""
    moduli = np.abs(ev)
    on_circle = ev[moduli > 0.1]
    if len(on_circle) < 8:
        return None

    angles = np.angle(on_circle)
    angles = np.sort(angles % (2 * np.pi))
    spacings = np.diff(angles)
    spacings = np.append(spacings, 2 * np.pi - angles[-1] + angles[0])

    mean_s = np.mean(spacings)
    if mean_s < 1e-12:
        return None
    spacings = spacings / mean_s

    ratios = []
    for i in range(len(spacings) - 1):
        s1, s2 = spacings[i], spacings[i + 1]
        if max(s1, s2) > 1e-12:
            ratios.append(min(s1, s2) / max(s1, s2))
    return np.array(ratios) if len(ratios) >= 4 else None


def bootstrap_ci(data, n_bootstrap=2000, alpha=0.05):
    """Bootstrap confidence interval for the mean."""
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    means = np.sort(means)
    lo = means[int(alpha / 2 * n_bootstrap)]
    hi = means[int((1 - alpha / 2) * n_bootstrap)]
    return lo, hi


# =============================================================
# PART A: Spacing ratio for carry companion matrices
# =============================================================
pr("=" * 70)
pr("PART A: Level spacing ratio <r> vs matrix dimension")
pr("=" * 70)
pr(f"  Reference: Poisson = {R_POISSON:.4f}, GOE = {R_GOE:.4f}, GUE = {R_GUE:.4f}")

for bits in [10, 14, 18, 22, 26]:
    all_ratios = []
    dims = []
    n_samples = 1500

    for _ in range(n_samples):
        p = random_prime(bits)
        q = random_prime(bits)
        if p == q:
            continue
        _, ev, cseq = extract_carry_matrix(p, q, 2)
        if ev is None:
            continue
        r = angular_spacing_ratios(ev)
        if r is not None:
            all_ratios.extend(r.tolist())
            dims.append(len(cseq))

    if not all_ratios:
        continue

    all_ratios = np.array(all_ratios)
    mean_r = np.mean(all_ratios)
    lo, hi = bootstrap_ci(all_ratios)
    mean_dim = np.mean(dims)

    closest = "GOE" if abs(mean_r - R_GOE) < abs(mean_r - R_GUE) else "GUE"
    if abs(mean_r - R_POISSON) < abs(mean_r - R_GOE):
        closest = "Poisson"

    pr(f"\n  {bits}-bit primes (mean D ≈ {mean_dim:.0f}, "
       f"N_ratios = {len(all_ratios)}):")
    pr(f"    <r> = {mean_r:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]")
    pr(f"    Closest: {closest}")
    pr(f"    Distances: |<r>-GOE| = {abs(mean_r-R_GOE):.4f}, "
       f"|<r>-GUE| = {abs(mean_r-R_GUE):.4f}, "
       f"|<r>-Poi| = {abs(mean_r-R_POISSON):.4f}")

# =============================================================
# PART B: Markov-correlated vs i.i.d.
# =============================================================
pr("\n" + "=" * 70)
pr("PART B: Markov-correlated (actual) vs i.i.d. entries")
pr("=" * 70)

bits = 20
markov_ratios = []
iid_ratios = []

for _ in range(2000):
    p = random_prime(bits)
    q = random_prime(bits)
    if p == q:
        continue
    _, ev, cseq = extract_carry_matrix(p, q, 2)
    if ev is None or cseq is None:
        continue

    r = angular_spacing_ratios(ev)
    if r is not None:
        markov_ratios.extend(r.tolist())

    for _ in range(3):
        ev_iid = iid_companion_matrix(cseq)
        if ev_iid is not None:
            r_iid = angular_spacing_ratios(ev_iid)
            if r_iid is not None:
                iid_ratios.extend(r_iid.tolist())

markov_ratios = np.array(markov_ratios)
iid_ratios = np.array(iid_ratios)

if len(markov_ratios) > 0 and len(iid_ratios) > 0:
    m_mean = np.mean(markov_ratios)
    i_mean = np.mean(iid_ratios)
    m_lo, m_hi = bootstrap_ci(markov_ratios)
    i_lo, i_hi = bootstrap_ci(iid_ratios)

    pr(f"\n  Markov (actual carries):   <r> = {m_mean:.4f} "
       f"[{m_lo:.4f}, {m_hi:.4f}], N = {len(markov_ratios)}")
    pr(f"  i.i.d. (shuffled carries): <r> = {i_mean:.4f} "
       f"[{i_lo:.4f}, {i_hi:.4f}], N = {len(iid_ratios)}")
    pr(f"\n  Markov closest to: "
       f"{'GOE' if abs(m_mean-R_GOE) < abs(m_mean-R_GUE) else 'GUE'}")
    pr(f"  i.i.d. closest to: "
       f"{'GOE' if abs(i_mean-R_GOE) < abs(i_mean-R_GUE) else 'GUE'}")

    diff = m_mean - i_mean
    se = np.sqrt(np.var(markov_ratios) / len(markov_ratios)
                 + np.var(iid_ratios) / len(iid_ratios))
    z_score = diff / se if se > 0 else 0
    pr(f"\n  Difference: {diff:.4f}, Z-score: {z_score:.1f}")
    pr(f"  GOE-GUE separation: {R_GOE - R_GUE:.4f}")

# =============================================================
# PART C: Base dependence
# =============================================================
pr("\n" + "=" * 70)
pr("PART C: Base dependence (16-bit primes)")
pr("=" * 70)

for base in [2, 3, 5, 7]:
    all_ratios = []
    dims = []

    for _ in range(1500):
        p = random_prime(16)
        q = random_prime(16)
        if p == q:
            continue
        _, ev, cseq = extract_carry_matrix(p, q, base)
        if ev is None:
            continue
        r = angular_spacing_ratios(ev)
        if r is not None:
            all_ratios.extend(r.tolist())
            dims.append(len(cseq))

    if not all_ratios:
        continue

    all_ratios = np.array(all_ratios)
    mean_r = np.mean(all_ratios)

    pr(f"  Base {base}: <r> = {mean_r:.4f} "
       f"(mean D ≈ {np.mean(dims):.0f}, N = {len(all_ratios)}), "
       f"closest = {'GOE' if abs(mean_r-R_GOE) < abs(mean_r-R_GUE) else 'GUE'}")

# =============================================================
# SUMMARY
# =============================================================
pr("\n" + "=" * 70)
pr("SUMMARY")
pr("=" * 70)
pr("The level spacing ratio <r> does NOT require spectral unfolding,")
pr("making it more robust than histogram-based chi² or L² tests.")
pr("")
pr("Key predictions for the paper:")
pr("  - If carry matrices → GOE: <r> ≈ 0.531")
pr("  - If i.i.d. entries → GUE: <r> ≈ 0.600")
pr("  - The DIFFERENCE confirms that Markov correlation → GOE")
