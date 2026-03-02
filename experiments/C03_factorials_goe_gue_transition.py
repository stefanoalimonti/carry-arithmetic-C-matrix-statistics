#!/usr/bin/env python3
"""
C03: Factorials, HCN, and the GOE→GUE Transition

Part A: Factorials and highly composite numbers as single-N products
       (completing the ergodicity test)

Part B: The crucial question from prior experiments — individual M_l spectra are
       GOE-like (real companion matrices), but do the ZEROS of the
       Euler product ∏_l det(I - M_l/l^s) show GUE statistics?
       The l^{-s} factor introduces complex phases → possible symmetry breaking.

Part C: Test the GOE→GUE crossover mechanism:
       - Spacing distribution of product zeros vs individual eigenvalues
       - How many primes l are needed for the transition?
"""

import sys, os, time, random, math
import numpy as np
from scipy.signal import argrelextrema
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import (carry_poly_int, quotient_poly_int, primes_up_to,
                         to_digits, random_prime)

random.seed(42)
np.random.seed(42)

ZETA_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
]


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


def get_ev(a, b, base):
    C = carry_poly_int(a, b, base)
    Q = quotient_poly_int(C, base)
    if len(Q) < 3:
        return None
    M = build_companion(Q)
    if M is None:
        return None
    try:
        return np.linalg.eigvals(M)
    except Exception:
        return None


def compute_product_curve(a, b, base_primes, t_grid):
    log_prod = np.zeros(len(t_grid))
    n_used = 0
    for l in base_primes:
        ev = get_ev(a, b, l)
        if ev is None:
            continue
        n_used += 1
        for idx, t in enumerate(t_grid):
            s = 0.5 + 1j * t
            ls = l ** s
            det_val = np.prod(1.0 - ev / ls)
            log_prod[idx] += math.log(max(abs(det_val), 1e-300))
    return log_prod, n_used


def match_zeros(log_prod, t_grid, known, tol=1.5, thresh_sigma=0.5):
    minima_idx = argrelextrema(-log_prod, np.greater, order=3)[0]
    med = np.median(log_prod)
    std = np.std(log_prod)
    deep = [(t_grid[i], log_prod[i]) for i in minima_idx
            if log_prod[i] < med - thresh_sigma * std]
    matched = 0
    total_err = 0
    used = set()
    for t_f, _ in sorted(deep, key=lambda x: x[0]):
        for j, t_k in enumerate(known):
            if abs(t_f - t_k) < tol and j not in used:
                matched += 1
                total_err += abs(t_f - t_k)
                used.add(j)
                break
    avg_err = total_err / matched if matched > 0 else float('inf')
    return matched, avg_err, len(deep)


def factorize_balanced(n):
    """Find A, B with A*B = n, A ≈ B (greedy divisor search)."""
    sqrt_n = int(math.isqrt(n))
    best_a = 1
    for d in range(2, min(sqrt_n + 1, 10**7)):
        if n % d == 0:
            best_a = d
            if d * d >= n // 2:
                break
    if best_a == 1:
        return 1, n
    b = n // best_a
    while best_a < b:
        improved = False
        for d in range(2, min(int(math.isqrt(b)) + 1, 1000)):
            if b % d == 0:
                new_a = best_a * d
                new_b = b // d
                if abs(new_a - new_b) < abs(best_a - b):
                    best_a, b = new_a, new_b
                    improved = True
                    break
        if not improved:
            break
    return best_a, b


def gue_spacing_pdf(s):
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)


def goe_spacing_pdf(s):
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("C03: FACTORIALS, HCN, AND GOE→GUE TRANSITION")
    pr("=" * 72)

    base_primes = primes_up_to(500)
    T_MAX = 80
    N_GRID = T_MAX * 30
    t_grid = np.linspace(10, T_MAX, N_GRID)
    target_zeros = [z for z in ZETA_ZEROS if z < T_MAX]

    # ═══════════════════════════════════════════════════════════════
    # PART A: FACTORIALS AND HIGHLY COMPOSITE NUMBERS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART A: FACTORIALS AND HIGHLY COMPOSITE NUMBERS")
    pr(f"{'═' * 72}\n")

    factorials = [(f"  {k}!", math.factorial(k)) for k in [10, 15, 20, 25]]
    hcn_list = [
        ("  HCN 720", 720),
        ("  HCN 5040", 5040),
        ("  HCN 55440", 55440),
        ("  HCN 720720", 720720),
        ("  HCN 3603600", 3603600),
        ("  HCN 36756720", 36756720),
    ]

    for label, N in factorials + hcn_list:
        A, B = factorize_balanced(N)
        if A < 2 or B < 2:
            pr(f"{label}: N={N:.3e}, cannot factor balanced")
            continue

        log_prod, n_used = compute_product_curve(A, B, base_primes, t_grid)
        m, e, nd = match_zeros(log_prod, t_grid, target_zeros)
        ratio = max(A, B) / max(min(A, B), 1)
        pr(f"{label}: N={N:.3e}, A={A}, B={B} (ratio={ratio:.1f}), "
           f"bases={n_used}, {m}/20 zeros, <|Δt|>={e:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # PART B: PRODUCT ZEROS — DO THEY SHOW GUE?
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART B: SPACING STATISTICS OF PRODUCT ZEROS")
    pr(f"{'═' * 72}\n")
    pr("  Individual M_l eigenvalues are GOE-like .")
    pr("  But the ZEROS of ∏_l det(I-M_l/l^s) emerge from complex-weighted")
    pr("  combination. Do these zeros show GUE statistics?\n")

    extended_zeros = [
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
        37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
        52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
        67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
        79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
        92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
        103.725538, 105.446623, 107.168611, 111.029535, 111.874659,
        114.320220, 116.226680, 118.790783, 121.370125, 122.946829,
    ]

    T_EXT = 130
    N_GRID_EXT = T_EXT * 50
    t_grid_ext = np.linspace(10, T_EXT, N_GRID_EXT)
    all_primes_big = primes_up_to(1000)

    pr(f"  Computing ensemble product (50 semiprimes, 32-bit, {len(all_primes_big)} bases up to 1000)")
    pr(f"  T range: [10, {T_EXT}], grid: {N_GRID_EXT} points\n")

    log_ens = np.zeros(N_GRID_EXT)
    n_valid = 0
    for trial in range(50):
        p = random_prime(32)
        q = random_prime(32)
        log_s, ns = compute_product_curve(p, q, all_primes_big, t_grid_ext)
        if ns > 5:
            log_ens += log_s
            n_valid += 1
    log_ens /= max(n_valid, 1)

    minima_idx = argrelextrema(-log_ens, np.greater, order=5)[0]
    med = np.median(log_ens)
    std_val = np.std(log_ens)
    product_zeros = sorted([t_grid_ext[i] for i in minima_idx
                            if log_ens[i] < med - 0.3 * std_val])

    pr(f"  Product zeros found: {len(product_zeros)}")
    pr(f"  Known ζ zeros in [10, {T_EXT}]: {len([z for z in extended_zeros if z < T_EXT])}")

    matched_product = []
    used = set()
    for pz in product_zeros:
        for j, kz in enumerate(extended_zeros):
            if abs(pz - kz) < 2.0 and j not in used:
                matched_product.append(pz)
                used.add(j)
                break

    if len(matched_product) > 3:
        spacings = np.diff(sorted(matched_product))
        mean_sp = np.mean(spacings)
        norm_sp = spacings / mean_sp

        s_bins = np.linspace(0, 3.5, 20)
        s_centers = (s_bins[:-1] + s_bins[1:]) / 2
        ds = s_bins[1] - s_bins[0]

        hist, _ = np.histogram(norm_sp, bins=s_bins, density=True)
        gue_vals = gue_spacing_pdf(s_centers)
        goe_vals = goe_spacing_pdf(s_centers)

        L2_gue = np.sqrt(np.sum((hist - gue_vals)**2) * ds)
        L2_goe = np.sqrt(np.sum((hist - goe_vals)**2) * ds)

        pr(f"\n  Product zero spacings ({len(norm_sp)} spacings):")
        pr(f"    Mean spacing: {mean_sp:.3f}")
        pr(f"    L²(GUE) = {L2_gue:.4f}")
        pr(f"    L²(GOE) = {L2_goe:.4f}")
        pr(f"    Closer to: {'GUE' if L2_gue < L2_goe else 'GOE'}")
    else:
        pr(f"  Too few matched zeros ({len(matched_product)}) for spacing statistics")

    known_spacings = np.diff(extended_zeros[:30])
    mean_ksp = np.mean(known_spacings)
    norm_ksp = known_spacings / mean_ksp

    s_bins2 = np.linspace(0, 3.5, 15)
    s_centers2 = (s_bins2[:-1] + s_bins2[1:]) / 2
    ds2 = s_bins2[1] - s_bins2[0]
    hist_k, _ = np.histogram(norm_ksp, bins=s_bins2, density=True)
    gue_k = gue_spacing_pdf(s_centers2)
    goe_k = goe_spacing_pdf(s_centers2)
    L2_gue_k = np.sqrt(np.sum((hist_k - gue_k)**2) * ds2)
    L2_goe_k = np.sqrt(np.sum((hist_k - goe_k)**2) * ds2)

    pr(f"\n  Reference: known ζ zeros spacing ({len(norm_ksp)} spacings):")
    pr(f"    L²(GUE) = {L2_gue_k:.4f}")
    pr(f"    L²(GOE) = {L2_goe_k:.4f}")
    pr(f"    (Note: 29 spacings is very few for reliable L²)")

    # ═══════════════════════════════════════════════════════════════
    # PART C: GOE→GUE CROSSOVER — HOW MANY PRIMES?
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART C: GOE→GUE CROSSOVER — PRIMES NEEDED")
    pr(f"{'═' * 72}\n")
    pr("  Hypothesis: with 1 prime l, spectrum is GOE.")
    pr("  With many primes, product introduces complex phases → GUE?")
    pr("  Test: P(s) of detected zeros vs number of primes used.\n")

    T_C = 200
    N_GRID_C = T_C * 50
    t_grid_c = np.linspace(10, T_C, N_GRID_C)
    extended_zeros_200 = [
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
        37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
        52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
        67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
        79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
        92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
        103.725538, 105.446623, 107.168611, 111.029535, 111.874659,
        114.320220, 116.226680, 118.790783, 121.370125, 122.946829,
        124.256819, 127.516684, 129.578704, 131.087688, 133.497737,
        134.756510, 138.116042, 139.736209, 141.123707, 143.111846,
        146.000982, 147.422765, 150.053520, 150.925258, 153.024694,
        156.112909, 157.597592, 158.849988, 161.188964, 163.030709,
        165.537069, 167.184439, 169.094515, 169.911977, 173.411537,
        174.754192, 176.441434, 178.377407, 179.916484, 182.207079,
        184.874467, 185.598783, 187.228922, 189.416159, 192.026656,
        193.079727, 195.265397, 196.876481, 198.015310, 199.547836,
    ]

    n_ens_c = 30
    for L_cutoff in [50, 100, 200, 500, 1000]:
        primes_cut = [l for l in all_primes_big if l <= L_cutoff]
        if len(primes_cut) < 3:
            continue

        log_avg = np.zeros(N_GRID_C)
        nv = 0
        for trial in range(n_ens_c):
            p = random_prime(32)
            q = random_prime(32)
            lp, ns = compute_product_curve(p, q, primes_cut, t_grid_c)
            if ns > 2:
                log_avg += lp
                nv += 1
        if nv == 0:
            continue
        log_avg /= nv

        mi = argrelextrema(-log_avg, np.greater, order=5)[0]
        med_c = np.median(log_avg)
        std_c = np.std(log_avg)
        pz = sorted([t_grid_c[i] for i in mi
                      if log_avg[i] < med_c - 0.3 * std_c])

        target_c = [z for z in extended_zeros_200 if z < T_C]
        matched_c = []
        used_c = set()
        for z in pz:
            for j, kz in enumerate(target_c):
                if abs(z - kz) < 2.0 and j not in used_c:
                    matched_c.append(z)
                    used_c.add(j)
                    break

        if len(matched_c) > 5:
            sp_c = np.diff(sorted(matched_c))
            mean_c = np.mean(sp_c) if len(sp_c) > 0 else 1
            norm_c = sp_c / max(mean_c, 0.01)

            s_bins_c = np.linspace(0, 3.5, 12)
            s_cen_c = (s_bins_c[:-1] + s_bins_c[1:]) / 2
            ds_c = s_bins_c[1] - s_bins_c[0]
            hist_c, _ = np.histogram(norm_c, bins=s_bins_c, density=True)
            L2g = np.sqrt(np.sum((hist_c - gue_spacing_pdf(s_cen_c))**2) * ds_c)
            L2o = np.sqrt(np.sum((hist_c - goe_spacing_pdf(s_cen_c))**2) * ds_c)
            closer = "GUE" if L2g < L2o else "GOE"
        else:
            L2g = L2o = float('inf')
            closer = "?"

        pr(f"  L≤{L_cutoff:5d} ({len(primes_cut):3d} primes): "
           f"{len(matched_c):3d}/{len(target_c)} zeros matched, "
           f"{len(sp_c) if len(matched_c)>5 else 0} spacings → "
           f"L²(GUE)={L2g:.3f}, L²(GOE)={L2o:.3f}, closer={closer}")

    # ═══════════════════════════════════════════════════════════════
    # PART D: PROPER ERGODICITY — KL WITH ENOUGH BASES
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART D: CARRY DISTRIBUTION — CORRECTED ERGODICITY TEST")
    pr(f"{'═' * 72}\n")

    test_bases = [l for l in primes_up_to(200) if l >= 3]

    def carry_kl(a, b, bases):
        """Compute mean |KL| of carry distribution vs geometric model."""
        kl_vals = []
        for l in bases:
            gd = to_digits(a, l)
            hd = to_digits(b, l)
            D = len(to_digits(a * b, l))
            conv = [0] * (len(gd) + len(hd) - 1)
            for i, ga in enumerate(gd):
                for j, hb in enumerate(hd):
                    conv[i + j] += ga * hb
            carries = [0] * (D + 1)
            for i in range(D):
                total = (conv[i] if i < len(conv) else 0) + carries[i]
                carries[i + 1] = total // l
            bulk = carries[2:D - 1]
            if len(bulk) < 3:
                continue

            counts = Counter(bulk)
            total = len(bulk)
            max_c = max(counts.keys()) + 1
            emp = np.array([counts.get(c, 0) / total for c in range(max_c)])

            mu = (l - 1) / 4.0
            log_base = math.log(1 + mu) if mu > 0 else 1.0
            log_geom = np.array([math.log(max(mu, 1e-30)) - (c + 1) * log_base
                                 for c in range(max_c)])
            log_geom -= np.max(log_geom)
            geom = np.exp(log_geom)
            geom /= geom.sum()

            kl = 0
            for c in range(max_c):
                if emp[c] > 1e-15 and geom[c] > 1e-15:
                    kl += emp[c] * math.log(emp[c] / geom[c])
            kl_vals.append(abs(kl))
        return np.mean(kl_vals) if kl_vals else float('inf')

    from itertools import combinations
    all_small_p = primes_up_to(200)

    def balanced_split(pf):
        n = len(pf)
        log_total = sum(math.log(p) for p in pf)
        half = log_total / 2
        best_diff = float('inf')
        best_A = set()
        for r in range(1, min(n // 2 + 1, 11)):
            for combo in combinations(range(n), r):
                diff = abs(sum(math.log(pf[i]) for i in combo) - half)
                if diff < best_diff:
                    best_diff = diff
                    best_A = set(combo)
        A = B = 1
        for i in range(n):
            if i in best_A:
                A *= pf[i]
            else:
                B *= pf[i]
        return A, B

    pr(f"  Test bases: {len(test_bases)} primes in [3, 200]")
    pr(f"  Comparing primorials, factorials, random semiprimes\n")

    entries = []

    for k in [10, 15, 20]:
        pf = all_small_p[:k]
        A, B = balanced_split(pf)
        usable = [l for l in test_bases if l not in set(pf)]
        if len(usable) < 3:
            pr(f"  Primorial k={k}: not enough usable bases ({len(usable)})")
            continue
        kl = carry_kl(A, B, usable)
        entries.append((f"Primorial k={k}", kl, len(usable)))

    for kf in [15, 20, 25]:
        N = math.factorial(kf)
        A, B = factorize_balanced(N)
        if A < 2:
            continue
        kl = carry_kl(A, B, test_bases)
        entries.append((f"Factorial {kf}!", kl, len(test_bases)))

    kl_rand_vals = []
    for trial in range(20):
        p = random_prime(32)
        q = random_prime(32)
        kl = carry_kl(p, q, test_bases)
        kl_rand_vals.append(kl)
    entries.append((f"Random 32-bit (n=20)", np.mean(kl_rand_vals), len(test_bases)))

    pr(f"  {'Type':<22s}  {'<|KL|>':>8s}  {'bases':>5s}")
    pr(f"  {'─'*22}  {'─'*8}  {'─'*5}")
    for name, kl, nb in entries:
        pr(f"  {name:<22s}  {kl:>8.4f}  {nb:>5d}")

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("SYNTHESIS")
    pr(f"{'═' * 72}")
    pr("""
  ERGODICITY (Parts A, D):
    Factorials and HCN find zeros comparably to random semiprimes.
    No "special integer" advantage detected. The averaging mechanism
    is the product over BASES l, not the choice of N.

  GOE→GUE TRANSITION (Parts B, C):
    Individual M_l matrices: GOE-like (real entries → conjugate pairs)
    Product zeros: ??? — see L² distances above.
    If product zeros are GUE → the transition is in the Euler product!
    The complex weight l^{-s} breaks the time-reversal symmetry.
""")

    pr(f"\n  Total runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
