#!/usr/bin/env python3
"""
C04: GUE 2-Point Correlation — Quantitative Test

For the carry companion matrix ensemble, compute:
  (A) Nearest-neighbor spacing distribution P(s)
  (B) 2-point correlation function R₂(r)
  (C) Form factor K(τ)

Compare with GUE, GOE, and Poisson predictions.

This is the most sensitive test of the universality class of the
carry matrix ensemble and the first step toward a rigorous GUE theorem.
"""

import sys, os, time, random, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import carry_poly_int, quotient_poly_int, random_prime

random.seed(42)
np.random.seed(42)


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def build_companion(Q):
    D = len(Q)
    if D < 3:
        return None
    lead = float(Q[-1])
    if abs(lead) < 1e-30:
        return None
    n = D - 1
    M = np.zeros((n, n))
    for i in range(n - 1):
        M[i + 1, i] = 1.0
    for i in range(n):
        M[i, n - 1] = -float(Q[i]) / lead
    return M


def get_eigenvalue_angles(p, q, base):
    """Get eigenvalue angles on unit circle for carry companion matrix."""
    C = carry_poly_int(p, q, base)
    Q = quotient_poly_int(C, base)
    if len(Q) < 3:
        return None
    M = build_companion(Q)
    if M is None:
        return None
    try:
        ev = np.linalg.eigvals(M)
    except Exception:
        return None

    r_max = np.max(np.abs(ev))
    if r_max < 1e-10:
        return None

    angles = np.sort(np.angle(ev)) % (2 * np.pi)
    angles = np.sort(angles)
    return angles


def unfold_spectrum(angles, D):
    """Unfold eigenvalue angles to have mean spacing 1.
    Uses the simplest unfolding: divide by mean spacing."""
    if len(angles) < 3:
        return None
    spacings = np.diff(angles)
    spacings = np.append(spacings, 2 * np.pi - angles[-1] + angles[0])
    mean_spacing = 2 * np.pi / len(angles)
    unfolded = angles / mean_spacing
    return unfolded, spacings / mean_spacing


def nearest_neighbor_spacings(angles_list):
    """Collect all nearest-neighbor spacings from multiple spectra."""
    all_spacings = []
    for angles in angles_list:
        if angles is None or len(angles) < 5:
            continue
        n = len(angles)
        mean_sp = 2 * np.pi / n
        spacings = np.diff(np.sort(angles))
        wrap = 2 * np.pi - np.sort(angles)[-1] + np.sort(angles)[0]
        spacings = np.append(spacings, wrap)
        all_spacings.extend(spacings / mean_sp)
    return np.array(all_spacings)


def pair_correlation(angles_list, r_max=3.0, n_bins=60):
    """Compute pair correlation function R₂(r) from eigenvalue angles.
    r is in units of mean spacing."""
    hist = np.zeros(n_bins)
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    total_pairs = 0

    for angles in angles_list:
        if angles is None or len(angles) < 5:
            continue
        n = len(angles)
        mean_sp = 2 * np.pi / n
        sa = np.sort(angles)
        for i in range(n):
            for j in range(i + 1, n):
                d = min(sa[j] - sa[i], 2 * np.pi - (sa[j] - sa[i]))
                r = d / mean_sp
                if r < r_max:
                    idx = int(r / r_max * n_bins)
                    if idx < n_bins:
                        hist[idx] += 1
                        total_pairs += 1

    bin_width = r_max / n_bins
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if total_pairs > 0:
        n_spectra = sum(1 for a in angles_list if a is not None and len(a) >= 5)
        avg_n = np.mean([len(a) for a in angles_list if a is not None and len(a) >= 5])
        expected_per_bin = total_pairs * bin_width / r_max
        hist = hist / max(expected_per_bin, 1)

    return bin_centers, hist


def form_factor(angles_list, tau_max=2.0, n_tau=40):
    """Compute spectral form factor K(τ) = ⟨|Σ exp(2πi n_j τ)|²⟩ / D."""
    taus = np.linspace(0.05, tau_max, n_tau)
    K = np.zeros(n_tau)
    n_valid = 0

    for angles in angles_list:
        if angles is None or len(angles) < 5:
            continue
        n = len(angles)
        unfolded = angles * n / (2 * np.pi)
        n_valid += 1

        for ti, tau in enumerate(taus):
            spectral_sum = np.sum(np.exp(2j * np.pi * unfolded * tau))
            K[ti] += np.abs(spectral_sum) ** 2 / n

    if n_valid > 0:
        K /= n_valid
    return taus, K


def gue_spacing_pdf(s):
    """Wigner surmise for GUE: P(s) = (32/π²) s² exp(-4s²/π)."""
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)


def goe_spacing_pdf(s):
    """Wigner surmise for GOE: P(s) = (π/2) s exp(-πs²/4)."""
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)


def poisson_spacing_pdf(s):
    """Poisson: P(s) = exp(-s)."""
    return np.exp(-s)


def gue_r2(r):
    """GUE pair correlation: R₂(r) = 1 - (sin(πr)/(πr))²."""
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc = np.where(np.abs(r) < 1e-10, 1.0, np.sin(np.pi * r) / (np.pi * r))
    return 1 - sinc**2


def gue_form_factor(tau):
    """GUE form factor: K(τ) = min(|τ|, 1)."""
    return np.minimum(np.abs(tau), 1.0)


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("C04: GUE 2-POINT CORRELATION")
    pr("=" * 72)

    # ═══════════════════════════════════════════════════════════════
    # COLLECT EIGENVALUE SPECTRA
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("COLLECTING EIGENVALUE SPECTRA")
    pr(f"{'═' * 72}\n")

    configs = [
        (2,  32, 5000),
        (2,  64, 3000),
        (2, 128, 1000),
        (3,  32, 3000),
        (5,  32, 3000),
    ]

    results = {}
    for base, bits, n_samples in configs:
        label = f"b={base},d={bits}"
        pr(f"  {label}: sampling {n_samples} semiprimes...", end="")
        t1 = time.time()

        angles_list = []
        for _ in range(n_samples):
            p = random_prime(bits)
            q = random_prime(bits)
            angles = get_eigenvalue_angles(p, q, base)
            if angles is not None and len(angles) >= 5:
                angles_list.append(angles)

        dt = time.time() - t1
        avg_D = np.mean([len(a) for a in angles_list]) if angles_list else 0
        pr(f" {len(angles_list)} valid, <D>={avg_D:.0f}, {dt:.1f}s")
        results[label] = angles_list

    # ═══════════════════════════════════════════════════════════════
    # PART A: NEAREST-NEIGHBOR SPACING
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART A: NEAREST-NEIGHBOR SPACING P(s)")
    pr(f"{'═' * 72}\n")

    s_bins = np.linspace(0, 3.5, 70)
    s_centers = (s_bins[:-1] + s_bins[1:]) / 2
    ds = s_bins[1] - s_bins[0]

    for label, angles_list in results.items():
        spacings = nearest_neighbor_spacings(angles_list)
        if len(spacings) < 100:
            continue

        hist, _ = np.histogram(spacings, bins=s_bins, density=True)

        gue_vals = gue_spacing_pdf(s_centers)
        goe_vals = goe_spacing_pdf(s_centers)
        poi_vals = poisson_spacing_pdf(s_centers)

        L2_gue = np.sqrt(np.sum((hist - gue_vals)**2) * ds)
        L2_goe = np.sqrt(np.sum((hist - goe_vals)**2) * ds)
        L2_poi = np.sqrt(np.sum((hist - poi_vals)**2) * ds)

        best = min([("GUE", L2_gue), ("GOE", L2_goe), ("Poisson", L2_poi)],
                   key=lambda x: x[1])

        pr(f"  {label}: {len(spacings)} spacings")
        pr(f"    L²(GUE)={L2_gue:.4f}  L²(GOE)={L2_goe:.4f}  L²(Poisson)={L2_poi:.4f}")
        pr(f"    Best fit: {best[0]} (L²={best[1]:.4f})")

        pr(f"    P(s) samples: s=0.3→{hist[int(0.3/ds)]:.3f} "
           f"(GUE:{gue_vals[int(0.3/ds)]:.3f}, GOE:{goe_vals[int(0.3/ds)]:.3f})")
        pr(f"                  s=1.0→{hist[int(1.0/ds)]:.3f} "
           f"(GUE:{gue_vals[int(1.0/ds)]:.3f}, GOE:{goe_vals[int(1.0/ds)]:.3f})")
        pr(f"                  s=2.0→{hist[int(2.0/ds)]:.3f} "
           f"(GUE:{gue_vals[int(2.0/ds)]:.3f}, GOE:{goe_vals[int(2.0/ds)]:.3f})")
        pr()

    # ═══════════════════════════════════════════════════════════════
    # PART B: PAIR CORRELATION R₂(r)
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART B: PAIR CORRELATION R₂(r)")
    pr(f"{'═' * 72}\n")

    for label, angles_list in results.items():
        r_centers, R2 = pair_correlation(angles_list, r_max=3.0, n_bins=30)
        if np.sum(R2) < 1e-10:
            continue

        R2_gue = gue_r2(r_centers)

        L2_r2 = np.sqrt(np.mean((R2 - R2_gue)**2))

        pr(f"  {label}:")
        pr(f"    R₂ vs GUE: L²={L2_r2:.4f}")
        pr(f"    R₂ samples: r=0.5→{R2[4]:.3f} (GUE:{R2_gue[4]:.3f})")
        pr(f"                r=1.0→{R2[9]:.3f} (GUE:{R2_gue[9]:.3f})")
        pr(f"                r=2.0→{R2[19]:.3f} (GUE:{R2_gue[19]:.3f})")

        repulsion = R2[0] if len(R2) > 0 else -1
        pr(f"    Level repulsion R₂(0): {repulsion:.4f} "
           f"({'strong' if repulsion < 0.3 else 'weak' if repulsion < 0.7 else 'none'})")
        pr()

    # ═══════════════════════════════════════════════════════════════
    # PART C: FORM FACTOR K(τ)
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART C: FORM FACTOR K(τ)")
    pr(f"{'═' * 72}\n")

    for label, angles_list in results.items():
        taus, K = form_factor(angles_list, tau_max=2.0, n_tau=20)
        if np.sum(K) < 1e-10:
            continue

        K_gue = gue_form_factor(taus)

        L2_K = np.sqrt(np.mean((K - K_gue)**2))

        pr(f"  {label}:")
        pr(f"    K(τ) vs GUE: L²={L2_K:.4f}")

        for ti in [2, 5, 9, 14, 19]:
            if ti < len(taus):
                pr(f"    τ={taus[ti]:.2f}: K={K[ti]:.3f} (GUE={K_gue[ti]:.3f})")
        pr()

    # ═══════════════════════════════════════════════════════════════
    # PART D: SCALING WITH D (base=2, varying bits)
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART D: SCALING — P(s) vs D")
    pr(f"{'═' * 72}\n")
    pr("  How does the spacing distribution evolve with matrix size D?")
    pr("  (D ≈ 2·bits for base 2)\n")

    for bits in [16, 32, 64, 128]:
        n_samp = min(5000, max(500, 50000 // bits))
        angles_list = []
        for _ in range(n_samp):
            p = random_prime(bits)
            q = random_prime(bits)
            angles = get_eigenvalue_angles(p, q, 2)
            if angles is not None and len(angles) >= 5:
                angles_list.append(angles)

        spacings = nearest_neighbor_spacings(angles_list)
        if len(spacings) < 50:
            continue

        hist, _ = np.histogram(spacings, bins=s_bins, density=True)
        L2_gue = np.sqrt(np.sum((hist - gue_spacing_pdf(s_centers))**2) * ds)
        L2_goe = np.sqrt(np.sum((hist - goe_spacing_pdf(s_centers))**2) * ds)
        L2_poi = np.sqrt(np.sum((hist - poisson_spacing_pdf(s_centers))**2) * ds)

        avg_D = np.mean([len(a) for a in angles_list])
        pr(f"  d={bits:3d} (<D>={avg_D:5.0f}, n={len(angles_list)}): "
           f"L²(GUE)={L2_gue:.4f}  L²(GOE)={L2_goe:.4f}  L²(Poi)={L2_poi:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # PART E: CONTROL — TRUE GUE/GOE RANDOM MATRICES
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART E: CONTROL — STANDARD RANDOM MATRIX ENSEMBLES")
    pr(f"{'═' * 72}\n")

    for D_ctrl in [32, 64]:
        pr(f"  D = {D_ctrl}:")

        gue_angles = []
        for _ in range(2000):
            H = np.random.randn(D_ctrl, D_ctrl) + 1j * np.random.randn(D_ctrl, D_ctrl)
            H = (H + H.conj().T) / 2
            ev = np.linalg.eigvalsh(H)
            angles = (ev - ev.min()) / (ev.max() - ev.min()) * 2 * np.pi
            gue_angles.append(np.sort(angles))

        sp_gue = nearest_neighbor_spacings(gue_angles)
        hist_gue, _ = np.histogram(sp_gue, bins=s_bins, density=True)
        L2 = np.sqrt(np.sum((hist_gue - gue_spacing_pdf(s_centers))**2) * ds)
        pr(f"    True GUE: L²(GUE)={L2:.4f} ({len(sp_gue)} spacings)")

        goe_angles = []
        for _ in range(2000):
            H = np.random.randn(D_ctrl, D_ctrl)
            H = (H + H.T) / 2
            ev = np.linalg.eigvalsh(H)
            angles = (ev - ev.min()) / (ev.max() - ev.min()) * 2 * np.pi
            goe_angles.append(np.sort(angles))

        sp_goe = nearest_neighbor_spacings(goe_angles)
        hist_goe, _ = np.histogram(sp_goe, bins=s_bins, density=True)
        L2_goe = np.sqrt(np.sum((hist_goe - goe_spacing_pdf(s_centers))**2) * ds)
        L2_gue_on_goe = np.sqrt(np.sum((hist_goe - gue_spacing_pdf(s_centers))**2) * ds)
        pr(f"    True GOE: L²(GOE)={L2_goe:.4f}, L²(GUE)={L2_gue_on_goe:.4f}")
        pr()

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("SYNTHESIS")
    pr(f"{'═' * 72}")
    pr("""
  Questions answered:
    1. Is P(s) closer to GUE or GOE? → See L² distances above
    2. Does R₂(r) show level repulsion at r→0? → Check R₂(0)
    3. Does K(τ) approach min(τ,1) (GUE plateau)? → Check K(τ) values
    4. Does the fit improve with increasing D? → See Part D scaling

  If GUE is confirmed:
    → The carry companion matrix ensemble is in the GUE universality class
    → A rigorous proof  via Markov structure is justified
    → First publishable theorem from the carry framework

  If GOE instead:
    → Time-reversal symmetry is unbroken in the carry dynamics
    → The GOE→GUE transition requires additional structure
    → Need to identify the symmetry-breaking mechanism
""")

    pr(f"\n  Total runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
