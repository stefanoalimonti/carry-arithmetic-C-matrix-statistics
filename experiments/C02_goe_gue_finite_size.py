import numpy as np
import sympy
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

def to_digits(n, b):
    if n == 0: return [0]
    res = []
    while n > 0:
        res.append(n % b)
        n //= b
    return res

def extract_quotient_carries(p, q, base=2):
    gd = to_digits(p, base)
    hd = to_digits(q, base)
    conv = np.convolve(gd, hd)
    carries = [0] * (len(conv) + 2)
    for k in range(len(conv)):
        carries[k+1] = (conv[k] + carries[k]) // base
    while len(carries) > 1 and carries[-1] == 0:
        carries.pop()
    if len(carries) < 3: return []
    return carries[-2:0:-1] 

def build_companion_matrix(c):
    D_minus_1 = len(c)
    M = np.zeros((D_minus_1, D_minus_1))
    for i in range(D_minus_1 - 1):
        M[i+1, i] = 1.0
    for i in range(D_minus_1):
        M[i, D_minus_1 - 1] = -c[i]
    return M

def goe_pdf(s):
    return (np.pi / 2.0) * s * np.exp(-np.pi * s**2 / 4.0)

def gue_pdf(s):
    return (32.0 / np.pi**2) * (s**2) * np.exp(-4.0 * s**2 / np.pi)

def run_experiment(bases=[32, 64, 128], bit_size=64, num_samples=2000):
    print("==============================================================")
    print(f" T-11: Extending GOE -> GUE Transition ({bit_size}-bit semiprimes)")
    print("==============================================================")
    
    s_vals = np.linspace(0.01, 3.0, 100)
    
    for base in bases:
        print(f"\n--- BASE: {base} ---")
        min_val = 1 << (bit_size-1)
        max_val = (1 << bit_size) - 1
        
        spacings = []
        
        for _ in range(num_samples):
            p = sympy.randprime(min_val, max_val)
            q = sympy.randprime(min_val, max_val)
            c = extract_quotient_carries(p, q, base)
            if len(c) < 10: continue
            
            M = build_companion_matrix(c)
            ev = np.linalg.eigvals(M)
            
            ev_circle = ev[np.abs(np.abs(ev) - 1.0) < 0.5]
            if len(ev_circle) < 5: continue
            
            angles = np.sort(np.angle(ev_circle))
            diffs = np.diff(angles)
            
            mean_spacing = np.mean(diffs)
            if mean_spacing > 0:
                s_i = diffs / mean_spacing
                spacings.extend(s_i)
                
        if not spacings:
            print("Not enough roots near the unit circle.")
            continue
            
        spacings = np.array(spacings)
        print(f"Collected {len(spacings)} unfolded eigenvalue spacings.")
        
        hist, bin_edges = np.histogram(spacings, bins=100, range=(0.01, 3.0), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        
        mse_goe = np.mean((hist - goe_pdf(bin_centers))**2)
        mse_gue = np.mean((hist - gue_pdf(bin_centers))**2)
        
        print(f"MSE to GOE: {mse_goe:.6f}")
        print(f"MSE to GUE: {mse_gue:.6f}")
        
        if mse_gue < mse_goe:
            print(">> DOMINANT UNIVERSALITY CLASS: GUE (Unitary Ensemble)")
        else:
            print(">> DOMINANT UNIVERSALITY CLASS: GOE (Orthogonal Ensemble)")

if __name__ == "__main__":
    run_experiment()