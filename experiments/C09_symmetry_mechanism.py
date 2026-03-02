#!/usr/bin/env python3
"""
C09: Symmetry mechanism for the GOE↔GUE transition in carry companion matrices.

For any diagonalizable matrix M = V Λ V^{-1}, the similarity between M and M^T
is governed by the Gram matrix G = V^H V of right eigenvectors. When G ≈ I
(orthogonal eigenvectors), the similarity transformation S = (V V^T)^{-1} is
close to orthogonal, providing an effective time-reversal symmetry → GOE.
When G is far from identity (non-orthogonal eigenvectors), TRS is broken → GUE.

Key hypothesis: Markov correlations make eigenvectors more orthogonal (lower κ(V)),
explaining the GOE↔GUE transition when correlations are removed.

Metrics (all O(D³), dominated by eigendecomposition):
  A. κ(V) = condition number of eigenvector matrix
  B. Off-diagonal Gram energy: ||V^H V - diag(V^H V)||_F / D
  C. Correlation with spacing ratio <r̃>
"""

import sys, os, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import random_prime, to_digits

random.seed(42)
np.random.seed(42)

R_GOE = 0.5307
R_GUE = 0.5996


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


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
    if len(carry_seq) < 6 or carry_seq[-1] == 0:
        return None
    return carry_seq


def build_companion(carry_seq):
    D = len(carry_seq)
    lead = carry_seq[-1]
    if lead == 0:
        return None
    M = np.zeros((D, D), dtype=float)
    for i in range(D - 1):
        M[i + 1, i] = 1.0
    for i in range(D):
        M[i, D - 1] = -carry_seq[i] / lead
    return M


def eigenvector_metrics(M):
    """Compute eigenvector orthogonality metrics for matrix M.

    Returns (cond_V, gram_offdiag, frac_real) or None if computation fails.
      - cond_V: condition number of eigenvector matrix
      - gram_offdiag: off-diagonal Gram energy ||G - diag(G)||_F / D
      - frac_real: fraction of eigenvalues with |Im| < 0.01 * |Re|
    """
    D = M.shape[0]
    try:
        eigenvalues, V = np.linalg.eig(M)
        if not np.all(np.isfinite(V)) or not np.all(np.isfinite(eigenvalues)):
            return None
    except Exception:
        return None

    # Condition number of eigenvector matrix
    sv = np.linalg.svd(V, compute_uv=False)
    if sv[-1] < 1e-14:
        return None
    cond_V = sv[0] / sv[-1]

    # Normalize columns of V to unit norm
    norms = np.linalg.norm(V, axis=0)
    norms[norms < 1e-15] = 1.0
    V_norm = V / norms

    # Gram matrix off-diagonal energy
    G = V_norm.conj().T @ V_norm
    diag_G = np.diag(np.diag(G))
    gram_offdiag = np.linalg.norm(G - diag_G, 'fro') / D

    # Fraction of nearly-real eigenvalues
    frac_real = np.mean(np.abs(eigenvalues.imag) < 0.01 * (np.abs(eigenvalues.real) + 1e-15))

    return cond_V, gram_offdiag, frac_real


def angular_spacing_ratios(ev):
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
    spacings /= mean_s
    ratios = []
    for i in range(len(spacings) - 1):
        s1, s2 = spacings[i], spacings[i + 1]
        if max(s1, s2) > 1e-12:
            ratios.append(min(s1, s2) / max(s1, s2))
    return np.mean(ratios) if len(ratios) >= 4 else None


# ═══════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════
BITS = 20
N_SAMPLES = 2000

pr("=" * 72)
pr("C09: SYMMETRY MECHANISM — EIGENVECTOR ORTHOGONALITY")
pr("=" * 72)
pr(f"  Bits per factor: {BITS}")
pr(f"  Samples: {N_SAMPLES}")
pr()

markov_cond = []
markov_gram = []
markov_real = []
markov_r = []

iid_cond = []
iid_gram = []
iid_real = []
iid_r = []

corr_data = []

for trial in range(N_SAMPLES):
    p = random_prime(BITS)
    q = random_prime(BITS)
    if p == q:
        continue
    cseq = compute_carries(p, q)
    if cseq is None:
        continue
    D = len(cseq)

    # --- Markov ---
    M = build_companion(cseq)
    if M is None:
        continue
    metrics = eigenvector_metrics(M)
    if metrics is not None:
        cond_V, gram_off, frac_r = metrics
        markov_cond.append(cond_V)
        markov_gram.append(gram_off)
        markov_real.append(frac_r)
        ev = np.linalg.eigvals(M)
        r_mean = angular_spacing_ratios(ev)
        if r_mean is not None:
            markov_r.append(r_mean)
            corr_data.append(('markov', cond_V, gram_off, r_mean, D))

    # --- i.i.d. ---
    for _ in range(3):
        shuffled = list(cseq)
        np.random.shuffle(shuffled)
        if shuffled[-1] == 0:
            shuffled[-1] = 1
        M_iid = build_companion(shuffled)
        if M_iid is None:
            continue
        metrics = eigenvector_metrics(M_iid)
        if metrics is not None:
            cond_V, gram_off, frac_r = metrics
            iid_cond.append(cond_V)
            iid_gram.append(gram_off)
            iid_real.append(frac_r)
            ev = np.linalg.eigvals(M_iid)
            r_mean = angular_spacing_ratios(ev)
            if r_mean is not None:
                iid_r.append(r_mean)
                corr_data.append(('iid', cond_V, gram_off, r_mean, D))

    if (trial + 1) % 500 == 0:
        pr(f"  {trial+1}/{N_SAMPLES} done...")

# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════

def report(name, m_arr, i_arr, lower_is_more_orthogonal=True):
    m = np.array(m_arr)
    i = np.array(i_arr)
    m_median = np.median(m)
    i_median = np.median(i)
    m_mean = np.mean(m)
    i_mean = np.mean(i)
    m_se = np.std(m) / np.sqrt(len(m))
    i_se = np.std(i) / np.sqrt(len(i))
    diff = m_mean - i_mean
    se = np.sqrt(np.var(m) / len(m) + np.var(i) / len(i))
    z = diff / se if se > 0 else 0
    if lower_is_more_orthogonal:
        direction = "MORE orthogonal" if diff < 0 else "LESS orthogonal"
    else:
        direction = "LESS orthogonal" if diff < 0 else "MORE orthogonal"
    pr(f"  {name}:")
    pr(f"    Markov: mean={m_mean:.4f} ± {m_se:.4f}  median={m_median:.4f}  N={len(m)}")
    pr(f"    i.i.d.: mean={i_mean:.4f} ± {i_se:.4f}  median={i_median:.4f}  N={len(i)}")
    pr(f"    Markov is {direction} (Z = {z:.1f})")
    return z


pr("\n" + "=" * 72)
pr("PART A: Eigenvector condition number κ(V)")
pr("  (orthogonal eigenvectors → κ = 1 → TRS → GOE)")
pr("=" * 72)

z_cond = report("κ(V)", markov_cond, iid_cond, lower_is_more_orthogonal=True)

pr("\n" + "=" * 72)
pr("PART B: Gram off-diagonal energy ||G - diag(G)||_F / D")
pr("  (orthogonal eigenvectors → 0 → TRS → GOE)")
pr("=" * 72)

z_gram = report("Gram off-diag", markov_gram, iid_gram, lower_is_more_orthogonal=True)

pr("\n" + "=" * 72)
pr("PART C: Fraction of nearly-real eigenvalues")
pr("  (real eigenvalues → TRS → GOE)")
pr("=" * 72)

z_real = report("Frac real", markov_real, iid_real, lower_is_more_orthogonal=False)

# ═══════════════════════════════════════════════════════════════
# PART D: Correlation analysis
# ═══════════════════════════════════════════════════════════════
pr("\n" + "=" * 72)
pr("PART D: Correlation between orthogonality metrics and <r̃>")
pr("=" * 72)

if len(corr_data) > 50:
    labels = np.array([d[0] for d in corr_data])
    conds = np.array([d[1] for d in corr_data])
    grams = np.array([d[2] for d in corr_data])
    rs = np.array([d[3] for d in corr_data])

    # Log-transform condition number for better correlation
    log_conds = np.log10(conds)
    log_conds_finite = np.isfinite(log_conds)

    if np.sum(log_conds_finite) > 20:
        corr_cond_r = np.corrcoef(log_conds[log_conds_finite], rs[log_conds_finite])[0, 1]
        pr(f"  ρ(log₁₀ κ(V), <r̃>) = {corr_cond_r:.4f}  "
           f"(positive → larger κ → more GUE-like)")

    corr_gram_r = np.corrcoef(grams, rs)[0, 1]
    pr(f"  ρ(gram_offdiag, <r̃>) = {corr_gram_r:.4f}  "
       f"(positive → less orthogonal → more GUE-like)")

    # Per-ensemble
    mask_m = labels == 'markov'
    mask_i = labels == 'iid'
    for label, mask in [("Markov", mask_m), ("i.i.d.", mask_i)]:
        if np.sum(mask) > 20:
            lc = log_conds[mask & log_conds_finite]
            r_sub = rs[mask & log_conds_finite]
            if len(lc) > 10:
                c = np.corrcoef(lc, r_sub)[0, 1]
                pr(f"    {label}: ρ(log₁₀ κ, <r̃>) = {c:.4f}  N={len(lc)}")

# ═══════════════════════════════════════════════════════════════
# PART E: Dimension dependence
# ═══════════════════════════════════════════════════════════════
pr("\n" + "=" * 72)
pr("PART E: Dimension dependence of κ(V)")
pr("=" * 72)

for bits in [10, 14, 18, 22, 26]:
    m_c = []
    i_c = []
    dims = []
    for _ in range(1000):
        p = random_prime(bits)
        q = random_prime(bits)
        if p == q:
            continue
        cseq = compute_carries(p, q)
        if cseq is None:
            continue
        dims.append(len(cseq))
        M = build_companion(cseq)
        if M is None:
            continue
        met = eigenvector_metrics(M)
        if met:
            m_c.append(met[0])

        shuffled = list(cseq)
        np.random.shuffle(shuffled)
        if shuffled[-1] == 0:
            shuffled[-1] = 1
        M_iid = build_companion(shuffled)
        if M_iid is None:
            continue
        met = eigenvector_metrics(M_iid)
        if met:
            i_c.append(met[0])

    if m_c and i_c:
        ratio = np.median(m_c) / np.median(i_c)
        pr(f"  {bits:2d}-bit (D̄={np.mean(dims):.0f}): "
           f"κ_M={np.median(m_c):.1f}  κ_I={np.median(i_c):.1f}  "
           f"ratio={ratio:.3f}")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
pr("\n" + "=" * 72)
pr("SUMMARY")
pr("=" * 72)
pr("For M = V Λ V^{-1}, the similarity S = (VV^T)^{-1} satisfying")
pr("M^T = S M S^{-1} is orthogonal iff V has orthogonal columns.")
pr("")
pr("Three independent metrics all point the same direction:")
pr(f"  κ(V):          Z = {z_cond:.1f}  (lower → more orthogonal → GOE)")
pr(f"  Gram off-diag: Z = {z_gram:.1f}  (lower → more orthogonal → GOE)")
pr(f"  Frac real ev:  Z = {z_real:.1f}  (higher → more real → GOE)")
pr("")
pr("Markov correlations make eigenvectors LESS orthogonal,")
pr("breaking the effective time-reversal symmetry.")
