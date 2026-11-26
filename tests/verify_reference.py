#!/usr/bin/env python3
"""Differential reference verifier for keel::specfun against SciPy and MATLAB R2025b."""

import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import scipy.special as sp
from scipy.stats import ncx2


def load_f64(path: str) -> np.ndarray:
    return np.fromfile(path, dtype="<f8")


def check_array(name: str, got: np.ndarray, ref: np.ndarray, rtol: float, atol: float, results: list):
    assert got.shape == ref.shape, f"{name}: shape mismatch {got.shape} vs {ref.shape}"
    max_rel = 0.0
    for i in range(len(got)):
        g, r = float(got[i]), float(ref[i])
        if math.isnan(r):
            if not math.isnan(g):
                raise AssertionError(f"{name}[{i}]: expected NaN, got {g}")
            continue
        if math.isinf(r):
            if not (math.isinf(g) and (g > 0) == (r > 0)):
                raise AssertionError(f"{name}[{i}]: expected {r}, got {g}")
            continue
        diff = abs(g - r)
        bound = max(atol, rtol * abs(r))
        if math.isnan(g) or diff > bound:
            raise AssertionError(
                f"{name}[{i}]: got {g:.17g}, ref {r:.17g}, diff={diff:.3e} > bound={bound:.3e}"
            )
        rel = diff / max(abs(r), 1e-300)
        if rel > max_rel and abs(r) > atol:
            max_rel = rel
    results.append((name, f"max_rel={max_rel:.2e} ({len(got)} pts)"))


def run_matlab_check(dump_dir: str, results: list):
    matlab_exe = shutil.which("matlab")
    if not matlab_exe:
        cand = r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
        if os.path.exists(cand):
            matlab_exe = cand
    if not matlab_exe:
        return

    out_mat = os.path.join(dump_dir, "matlab_ref.bin")
    m_code = (
        "vals = ["
        "marcumq(0, 2, 1), marcumq(0, 5, 4), marcumq(2, 0, 1), "
        "marcumq(1, 1, 1), marcumq(2, 3, 1), marcumq(3, 2, 3), "
        "marcumq(10, 10, 1), marcumq(20, 20, 1), marcumq(50, 50, 5), "
        "marcumq(50, 70, 1), marcumq(50, 35, 3), marcumq(100, 90, 10), "
        "marcumq(500, 505, 3), marcumq(1000, 1000, 1), "
        "airy(0, -5), airy(1, -5), airy(2, -5), airy(3, -5), "
        "airy(0, 2.5), airy(1, 2.5), airy(2, 2.5), airy(3, 2.5), "
        "expint(0.5), expint(1.0), expint(10.0), "
        "psi(0.5), psi(12.0), gammaln(12.0), gammaln(145.0)"
        "]; "
        f"fid = fopen('{out_mat.replace(chr(92), '/')}', 'wb'); "
        "fwrite(fid, vals, 'double'); fclose(fid);"
    )
    proc = subprocess.run([matlab_exe, "-batch", m_code], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"MATLAB reference failed:\n{proc.stdout}\n{proc.stderr}")

    mref = load_f64(out_mat)
    mq_got = load_f64(os.path.join(dump_dir, "marcumq.bin"))
    # MATLAB marcumq(500, 505, 3) has ~2e-7 relative error vs exact ncx2.sf (which keel matches to ~1e-13).
    check_array("matlab:marcumq", mq_got, mref[:14], rtol=1e-6, atol=1e-15, results=results)

    airy_got = load_f64(os.path.join(dump_dir, "airy.bin"))
    # a_xs index 1 is -5.0, index 7 is 2.5
    airy_sub = np.concatenate([airy_got[4:8], airy_got[28:32]])
    check_array("matlab:airy", airy_sub, mref[14:22], rtol=1e-11, atol=1e-14, results=results)

    exp_got = load_f64(os.path.join(dump_dir, "expint.bin"))
    # exp_xs indices 3 (0.5), 4 (1.0), 7 (10.0)
    check_array("matlab:expint", np.array([exp_got[3], exp_got[4], exp_got[7]]), mref[22:25], rtol=1e-12, atol=1e-15, results=results)

    psi_got = load_f64(os.path.join(dump_dir, "psi.bin"))
    lg_got = load_f64(os.path.join(dump_dir, "gammaln.bin"))
    # xs indices: 7 is 0.5, 12 is 12.0, 14 is 145.0
    check_array("matlab:psi_gammaln", np.array([psi_got[7], psi_got[12], lg_got[12], lg_got[14]]), mref[25:29], rtol=1e-12, atol=1e-14, results=results)


def main():
    if len(sys.argv) < 2:
        print("Usage: verify_reference.py <path/to/test_specfun.exe>", file=sys.stderr)
        sys.exit(2)
    exe = sys.argv[1]
    results = []

    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run([exe, "--dump-ref", tmp], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"test_specfun --dump-ref failed:\n{proc.stdout}\n{proc.stderr}")

        # 1. Gamma / gammaln / psi
        xs = np.array([-100.5, -10.25, -3.5, -0.5, -0.01, 0.01, 0.25, 0.5, 1.0, 1.5, 2.0, 5.5, 12.0, 50.0, 145.0, 170.5])
        check_array("scipy:gamma", load_f64(os.path.join(tmp, "gamma.bin")), sp.gamma(xs), 1e-12, 1e-15, results)
        check_array("scipy:gammaln", load_f64(os.path.join(tmp, "gammaln.bin")), sp.gammaln(xs), 1e-12, 1e-14, results)
        check_array("scipy:psi", load_f64(os.path.join(tmp, "psi.bin")), sp.digamma(xs), 1e-12, 1e-14, results)

        # 2. Beta / gammainc / betainc & inverses
        b_pairs = [(0.5, 0.5), (1.0, 1.0), (2.5, 4.5), (10.0, 20.0), (100.0, 150.0)]
        check_array("scipy:beta", load_f64(os.path.join(tmp, "beta.bin")),
                    np.array([sp.beta(a, b) for a, b in b_pairs]), 1e-12, 1e-300, results)
        check_array("scipy:betaln", load_f64(os.path.join(tmp, "betaln.bin")),
                    np.array([sp.betaln(a, b) for a, b in b_pairs]), 1e-12, 1e-14, results)

        gi_cases = [(0.01, 0.1), (0.5, 0.5), (1.0, 1.0), (2.5, 3.0), (10.0, 5.0),
                    (100.0, 100.0), (490.0, 500.0), (9950.0, 10000.0), (10050.0, 10000.0)]
        check_array("scipy:gammainc", load_f64(os.path.join(tmp, "gammainc.bin")),
                    np.array([sp.gammainc(a, x) for x, a in gi_cases]), 1e-11, 1e-14, results)
        check_array("scipy:gammainc_upper", load_f64(os.path.join(tmp, "gammainc_upper.bin")),
                    np.array([sp.gammaincc(a, x) for x, a in gi_cases]), 1e-11, 1e-14, results)

        ginv_cases = [(1e-10, 0.05), (0.01, 0.2), (0.5, 0.5), (0.73, 3.5), (0.999999, 5.0),
                      (1.0 - 1e-10, 5.0), (0.25, 100.0), (0.8, 1000.0)]
        check_array("scipy:gammaincinv", load_f64(os.path.join(tmp, "gammaincinv.bin")),
                    np.array([sp.gammaincinv(a, p) for p, a in ginv_cases]), 1e-11, 1e-300, results)

        bi_cases = [(0.1, 0.2, 0.3), (0.3, 2.0, 5.0), (0.7, 5.0, 2.0), (0.5, 10.0, 10.0),
                    (0.99, 0.5, 0.5), (0.505, 5000.0, 5000.0), (0.2, 50.0, 200.0)]
        check_array("scipy:betainc", load_f64(os.path.join(tmp, "betainc.bin")),
                    np.array([sp.betainc(a, b, x) for x, a, b in bi_cases]), 1e-10, 1e-14, results)

        binv_cases = [(1e-10, 0.2, 5.0), (0.05, 0.5, 0.5), (0.27, 2.5, 4.0),
                      (0.9, 10.0, 3.0), (1.0 - 1e-10, 0.5, 3.0), (0.5, 100.0, 200.0)]
        check_array("scipy:betaincinv", load_f64(os.path.join(tmp, "betaincinv.bin")),
                    np.array([sp.betaincinv(a, b, p) for p, a, b in binv_cases]), 1e-11, 1e-300, results)

        # 3. Error functions & expint
        exs = np.array([-20.0, -5.0, -1.5, -0.3, 0.0, 0.25, 0.46875, 1.0, 3.5, 5.0, 10.0, 25.0])
        check_array("scipy:erf", load_f64(os.path.join(tmp, "erf.bin")), sp.erf(exs), 1e-13, 1e-15, results)
        check_array("scipy:erfc", load_f64(os.path.join(tmp, "erfc.bin")), sp.erfc(exs), 1e-13, 1e-300, results)
        erfcx_xs = np.append(exs, [100.0, 1e6])
        check_array("scipy:erfcx", load_f64(os.path.join(tmp, "erfcx.bin")), sp.erfcx(erfcx_xs), 1e-13, 1e-300, results)

        ei_ys = np.array([-0.999999, -0.7, -0.1, 0.0, 0.3, 0.999999])
        check_array("scipy:erfinv", load_f64(os.path.join(tmp, "erfinv.bin")), sp.erfinv(ei_ys), 1e-12, 1e-15, results)
        eci_ys = np.array([1e-12, 1e-6, 0.1, 0.5, 1.0, 1.5, 1.999999])
        check_array("scipy:erfcinv", load_f64(os.path.join(tmp, "erfcinv.bin")), sp.erfcinv(eci_ys), 1e-12, 1e-15, results)

        exp_xs = np.array([1e-12, 1e-3, 0.25, 0.5, 1.0, 1.001, 2.5, 10.0, 50.0, 100.0, 500.0])
        check_array("scipy:expint", load_f64(os.path.join(tmp, "expint.bin")), sp.exp1(exp_xs), 1e-12, 1e-300, results)

        # 4. Bessel J, Y, I, K & Airy
        b_cases = [
            (0.0, 0.1), (0.0, 1.0), (0.0, 10.0), (0.0, 50.0),
            (1.0, 0.5), (1.0, 2.0), (2.0, 5.0), (0.5, 1.5),
            (1.0 / 3.0, 2.5), (2.5, 10.0), (5.0, 15.0), (25.0, 20.0),
            (-0.5, 1.5), (-2.3, 4.0), (-10.5, 3.0), (-100.5, 2.5),
            (100.0, 50.0), (100.0, 700.0), (500.0, 720.0), (0.0, 800.0),
        ]
        check_array("scipy:besselj", load_f64(os.path.join(tmp, "besselj.bin")),
                    np.array([sp.jv(nu, x) for nu, x in b_cases]), 1e-10, 1e-14, results)
        check_array("scipy:bessely", load_f64(os.path.join(tmp, "bessely.bin")),
                    np.array([sp.yv(nu, x) for nu, x in b_cases]), 1e-10, 1e-14, results)
        check_array("scipy:besseli", load_f64(os.path.join(tmp, "besseli.bin")),
                    np.array([sp.iv(nu, x) for nu, x in b_cases]), 1e-10, 1e-300, results)
        check_array("scipy:besselk", load_f64(os.path.join(tmp, "besselk.bin")),
                    np.array([sp.kv(nu, x) for nu, x in b_cases]), 1e-10, 1e-300, results)

        a_xs = [-20.0, -5.0, -2.0, -0.5, 0.0, 0.5, 1.0, 2.5, 10.0, 50.0, 100.0]
        airy_ref = []
        for x in a_xs:
            ai, aip, bi, bip = sp.airy(x)
            airy_ref.extend([ai, aip, bi, bip])
        check_array("scipy:airy", load_f64(os.path.join(tmp, "airy.bin")),
                    np.array(airy_ref), 1e-10, 1e-14, results)

        # 5. Legendre, EllipKE, EllipJ, MarcumQ
        lxs = [-0.9, -0.3, 0.0, 0.5, 1.0]
        leg_ref = []
        for x in lxs:
            for m in range(6):
                leg_ref.append(sp.lpmv(m, 5, x))
        check_array("scipy:legendre", load_f64(os.path.join(tmp, "legendre.bin")),
                    np.array(leg_ref), 1e-12, 1e-14, results)

        ems = [0.0, 1e-6, 0.1, 0.5, 0.9, 0.9999, 1.0 - 1e-10]
        ke_ref = []
        for m in ems:
            ke_ref.extend([sp.ellipk(m), sp.ellipe(m)])
        check_array("scipy:ellipke", load_f64(os.path.join(tmp, "ellipke.bin")),
                    np.array(ke_ref), 1e-12, 1e-14, results)

        ej_cases = [(-2.5, 0.3), (0.0, 0.5), (0.7, 0.4), (2.0, 0.9), (5.0, 0.999), (1.5, 0.0), (1.5, 1.0)]
        ej_ref = []
        for u, m in ej_cases:
            sn, cn, dn, _ = sp.ellipj(u, m)
            ej_ref.extend([sn, cn, dn])
        check_array("scipy:ellipj", load_f64(os.path.join(tmp, "ellipj.bin")),
                    np.array(ej_ref), 1e-12, 1e-14, results)

        mq_cases = [
            (0.0, 2.0, 1), (0.0, 5.0, 4), (2.0, 0.0, 1),
            (1.0, 1.0, 1), (2.0, 3.0, 1), (3.0, 2.0, 3),
            (10.0, 10.0, 1), (20.0, 20.0, 1), (50.0, 50.0, 5),
            (50.0, 70.0, 1), (50.0, 35.0, 3), (100.0, 90.0, 10),
            (500.0, 505.0, 3), (1000.0, 1000.0, 1),
        ]
        mq_ref = np.array([1.0 if b == 0 else ncx2.sf(b * b, 2 * m, a * a) for a, b, m in mq_cases])
        check_array("scipy:marcumq", load_f64(os.path.join(tmp, "marcumq.bin")), mq_ref, 1e-11, 1e-15, results)

        run_matlab_check(tmp, results)

    print(f"ALL SPECFUN REFERENCE CHECKS PASSED ({len(results)}/{len(results)}):")
    for name, detail in results:
        print(f"  [PASS] {name} ({detail})")


if __name__ == "__main__":
    main()
