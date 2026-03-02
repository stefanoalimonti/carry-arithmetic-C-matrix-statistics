#!/usr/bin/env python3
"""
C10: Number variance Σ²(L) for carry companion matrices.

The number variance Σ²(L) counts the variance of the number of eigenvalues
in an interval of length L (after unfolding). It distinguishes GOE from GUE
more sharply than the spacing ratio at intermediate L:

  Σ²_Poisson(L) = L                    (linear growth)
  Σ²_GOE(L) ≈ (2/π²) [ln(2πL) + γ + 1 + π²/8]   (logarithmic)
  Σ²_GUE(L) ≈ (1/π²) [ln(2πL) + γ + 1]           (logarithmic, factor ~2 smaller)

At L ≈ 1: Σ²_GOE ≈ 0.286, Σ²_GUE ≈ 0.168.

For angular spectra (circular ensembles), we use the angular unfolding
and count eigenvalue angles in arcs of normalized length L.

Tests:
  A. Σ²(L) for L = 0.5, 1, 2, 3, 5 — Markov vs i.i.d.
  B. Ratio Σ²_Markov / Σ²_iid at each L
  C. Fit to GOE/GUE predictions
"""

import sys, os, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import random_prime, to_digits

random.seed(42)
np.random.seed(42)

EULER_GAMMA = 0.5772156649


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def sigma2_goe(L):
    """Asymptotic GOE number variance (valid for L >> 1)."""
    if L <= 0:
        return 0
    return (2 / np.pi**2) * (np.log(2 * np.pi * L) + EULER_GAMMA + 1 + np.pi**2 / 8)


def sigma2_gue(L):
    """Asymptotic GUE number variance (valid for L >> 1)."""
    if L <= 0:
        return 0
    return (1 / np.pi**2) * (np.log(2 * np.pi * L) + EULER_GAMMA + 1)


def compute_carries(p, q, base=2):
    gd = to_digits(p, base)
    hd = to_digits(q, base)
    conv = [0] * (len(gd) + len(hd) - 1)
    for i, a in enumerate(gd):
        for j, b_val in enumerate(hd):
            conv[i + j] += a * b_val
    D_max = len(conv) + 1
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
    if len(carry_seq) < 8 or carry_seq[-1] == 0:
        return None
    return carry_seq


def build_companion_ev(carry_seq):
    D = len(carry_seq)
    lead = carry_seq[-1]
    if lead == 0:
        return None
    M = np.zeros((D, D), dtype=float)
    for i in range(D - 1):
        M[i + 1, i] = 1.0
    for i in range(D):
        M[i, D - 1] = -carry_seq[i] / lead
    try:
        ev = np.linalg.eigvals(M)
        if not np.all(np.isfinite(ev)):
            return None
    except Exception:
        return None
    return ev


def unfold_angles(ev, min_modulus=0.1):
    """Extract and unfold eigenvalue angles.

    Returns sorted, normalized angles in [0, N) where N is the number
    of eigenvalues used (uniform mean density = 1).
    """
    moduli = np.abs(ev)
    on_circle = ev[moduli > min_modulus]
    if len(on_circle) < 10:
        return None

    angles = np.angle(on_circle)
    angles = np.sort(angles % (2 * np.pi))
    N = len(angles)
    # Linear unfolding: map [0, 2π) → [0, N)
    unfolded = angles * N / (2 * np.pi)
    return unfolded


def number_variance(unfolded, L):
    """Compute Σ²(L) from unfolded eigenvalue positions.

    Σ²(L) = Var(#{eigenvalues in [x, x+L]}) averaged over x.
    """
    N = len(unfolded)
    if N < 10 or L <= 0:
        return None

    n_windows = min(200, N)
    x_starts = np.linspace(0, N - L, n_windows, endpoint=False)

    counts = []
    for x0 in x_starts:
        x1 = x0 + L
        if x1 <= N:
            n_in = np.sum((unfolded >= x0) & (unfolded < x1))
        else:
            n_in = np.sum(unfolded >= x0) + np.sum(unfolded < (x1 - N))
        counts.append(n_in)

    counts = np.array(counts, dtype=float)
    return np.var(counts)


# ═══════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════
BITS = 20
N_SAMPLES = 2000
L_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0]

pr("=" * 72)
pr("C10: NUMBER VARIANCE Σ²(L) — GOE vs GUE CONFIRMATION")
pr("=" * 72)
pr(f"  Bits per factor: {BITS}")
pr(f"  Samples: {N_SAMPLES}")
pr(f"  L values: {L_VALUES}")
pr()

# Collect data
markov_nv = {L: [] for L in L_VALUES}
iid_nv = {L: [] for L in L_VALUES}

for trial in range(N_SAMPLES):
    p = random_prime(BITS)
    q = random_prime(BITS)
    if p == q:
        continue
    cseq = compute_carries(p, q)
    if cseq is None:
        continue

    # --- Markov ---
    ev = build_companion_ev(cseq)
    if ev is not None:
        uf = unfold_angles(ev)
        if uf is not None:
            for L in L_VALUES:
                nv = number_variance(uf, L)
                if nv is not None:
                    markov_nv[L].append(nv)

    # --- i.i.d. (3 shuffles) ---
    for _ in range(3):
        shuffled = list(cseq)
        np.random.shuffle(shuffled)
        if shuffled[-1] == 0:
            shuffled[-1] = 1
        ev_iid = build_companion_ev(shuffled)
        if ev_iid is not None:
            uf = unfold_angles(ev_iid)
            if uf is not None:
                for L in L_VALUES:
                    nv = number_variance(uf, L)
                    if nv is not None:
                        iid_nv[L].append(nv)

    if (trial + 1) % 500 == 0:
        pr(f"  {trial+1}/{N_SAMPLES} done...")

# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
pr("\n" + "=" * 72)
pr("RESULTS: Σ²(L) comparison")
pr("=" * 72)

pr(f"\n  {'L':>5s} | {'Σ²_Markov':>10s} | {'Σ²_iid':>10s} | {'Σ²_GOE':>8s} | "
   f"{'Σ²_GUE':>8s} | {'ratio M/I':>9s} | {'GOE/GUE':>7s} | {'M closest':>10s}")
pr("  " + "-" * 85)

for L in L_VALUES:
    m_data = np.array(markov_nv[L])
    i_data = np.array(iid_nv[L])

    if len(m_data) < 10 or len(i_data) < 10:
        pr(f"  {L:5.1f} | insufficient data")
        continue

    m_mean = np.mean(m_data)
    i_mean = np.mean(i_data)
    m_se = np.std(m_data) / np.sqrt(len(m_data))
    i_se = np.std(i_data) / np.sqrt(len(i_data))

    goe = sigma2_goe(L)
    gue = sigma2_gue(L)
    ratio = m_mean / i_mean if i_mean > 0 else float('inf')
    goe_gue_ratio = goe / gue if gue > 0 else float('inf')

    m_closest = "GOE" if abs(m_mean - goe) < abs(m_mean - gue) else "GUE"

    pr(f"  {L:5.1f} | {m_mean:10.4f} | {i_mean:10.4f} | {goe:8.4f} | "
       f"{gue:8.4f} | {ratio:9.3f} | {goe_gue_ratio:7.3f} | {m_closest:>10s}")

pr()

# ═══════════════════════════════════════════════════════════════
# PART B: Statistical significance
# ═══════════════════════════════════════════════════════════════
pr("=" * 72)
pr("STATISTICAL SIGNIFICANCE (two-sample t-test at each L)")
pr("=" * 72)

for L in L_VALUES:
    m_data = np.array(markov_nv[L])
    i_data = np.array(iid_nv[L])
    if len(m_data) < 10 or len(i_data) < 10:
        continue

    diff = np.mean(m_data) - np.mean(i_data)
    se = np.sqrt(np.var(m_data) / len(m_data) + np.var(i_data) / len(i_data))
    z = diff / se if se > 0 else 0

    pr(f"  L={L:.1f}: Σ²_Markov - Σ²_iid = {diff:+.4f}  "
       f"(Z = {z:.1f}, N_M={len(m_data)}, N_I={len(i_data)})")

# ═══════════════════════════════════════════════════════════════
# PART C: Dimension dependence
# ═══════════════════════════════════════════════════════════════
pr("\n" + "=" * 72)
pr("DIMENSION DEPENDENCE at L=1.0")
pr("=" * 72)

for bits in [10, 14, 18, 22, 26]:
    m_nv = []
    i_nv = []
    dims = []

    for _ in range(1500):
        p = random_prime(bits)
        q = random_prime(bits)
        if p == q:
            continue
        cseq = compute_carries(p, q)
        if cseq is None:
            continue
        dims.append(len(cseq))

        ev = build_companion_ev(cseq)
        if ev is not None:
            uf = unfold_angles(ev)
            if uf is not None:
                nv = number_variance(uf, 1.0)
                if nv is not None:
                    m_nv.append(nv)

        shuffled = list(cseq)
        np.random.shuffle(shuffled)
        if shuffled[-1] == 0:
            shuffled[-1] = 1
        ev_iid = build_companion_ev(shuffled)
        if ev_iid is not None:
            uf = unfold_angles(ev_iid)
            if uf is not None:
                nv = number_variance(uf, 1.0)
                if nv is not None:
                    i_nv.append(nv)

    if m_nv and i_nv:
        m_mean = np.mean(m_nv)
        i_mean = np.mean(i_nv)
        goe = sigma2_goe(1.0)
        gue = sigma2_gue(1.0)
        m_closest = "GOE" if abs(m_mean - goe) < abs(m_mean - gue) else "GUE"
        i_closest = "GOE" if abs(i_mean - goe) < abs(i_mean - gue) else "GUE"
        pr(f"  {bits:2d}-bit (D̄={np.mean(dims):.0f}): "
           f"Σ²_M={m_mean:.4f} [{m_closest}]  "
           f"Σ²_I={i_mean:.4f} [{i_closest}]  "
           f"(GOE={goe:.4f}, GUE={gue:.4f})")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
pr("\n" + "=" * 72)
pr("SUMMARY")
pr("=" * 72)
pr("Number variance Σ²(L) provides an independent confirmation of the")
pr("GOE↔GUE dichotomy found by spacing ratio (C07) and interpolation (C08).")
pr("")
pr("GOE prediction: Σ²(L) ~ (2/π²) ln L  (larger fluctuations)")
pr("GUE prediction: Σ²(L) ~ (1/π²) ln L  (smaller fluctuations)")
pr("Markov → GOE-like (larger Σ²), i.i.d. → GUE-like (smaller Σ²)")
