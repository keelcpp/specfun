#include "keel/specfun.h"

#include "keel/math.h"

namespace keel::specfun {

namespace {

constexpr f64 kEps = 1e-16;
constexpr f64 kFpMin = 1e-300;
constexpr i32 kMaxIter = 10000;
constexpr f64 kXMin = 2.0;
constexpr f64 kLn2 = 0.693147180559945309417232121458176568;
constexpr f64 kMaxOrder = 1.0e7;

// Modular sin(pi * x) and cos(pi * x) avoiding catastrophic precision loss for large |x|.
f64 sinpi_exact(f64 x) noexcept {
    f64 r = keel::fmod(x, 2.0);
    if (r < 0.0) r += 2.0;
    if (r == 0.0 || r == 1.0) return 0.0;
    if (r == 0.5) return 1.0;
    if (r == 1.5) return -1.0;
    if (r > 1.0) return -keel::sin(kPi * (r - 1.0));
    if (r > 0.5) return keel::sin(kPi * (1.0 - r));
    return keel::sin(kPi * r);
}

f64 cospi_exact(f64 x) noexcept {
    f64 r = keel::fmod(keel::abs(x), 2.0);
    if (r == 0.0) return 1.0;
    if (r == 1.0) return -1.0;
    if (r == 0.5 || r == 1.5) return 0.0;
    if (r > 1.0) r = 2.0 - r;
    if (r > 0.5) return -keel::cos(kPi * (1.0 - r));
    return keel::cos(kPi * r);
}

// Chebyshev evaluation for Temme's gamma1 / gamma2 helper on x in [-0.5, 0.5].
f64 chebev(f64 a, f64 b, const f64* c, i32 m, f64 x) noexcept {
    const f64 y = (2.0 * x - a - b) / (b - a);
    const f64 y2 = 2.0 * y;
    f64 d = 0.0;
    f64 dd = 0.0;
    for (i32 j = m - 1; j >= 1; --j) {
        const f64 sv = d;
        d = y2 * d - dd + c[j];
        dd = sv;
    }
    return y * d - dd + 0.5 * c[0];
}

// Computes Temme's functions for |x| <= 0.5:
//   gam1 = (1/Gamma(1+x) - 1/Gamma(1-x)) / (2x)
//   gam2 = (1/Gamma(1+x) + 1/Gamma(1-x)) / 2
//   gampl = 1 / Gamma(1+x), gammi = 1 / Gamma(1-x).
void beschb(f64 x, f64& gam1, f64& gam2, f64& gampl, f64& gammi) noexcept {
    static constexpr f64 kC1[7] = {
        -1.142022680371168e0, 6.5165112670737e-3,   3.087090173086e-4, -3.4706269649e-6,
        6.9437664e-9,         3.67795e-11,          -1.356e-13,
    };
    static constexpr f64 kC2[8] = {
        1.843740587300905e0,  -7.68528408447867e-2, 1.2719271366546e-3, -4.9717367042e-6,
        -3.31261198e-8,       2.423096e-10,         -1.702e-13,         -1.49e-15,
    };
    const f64 xx = 8.0 * x * x - 1.0;
    gam1 = chebev(-1.0, 1.0, kC1, 7, xx);
    gam2 = chebev(-1.0, 1.0, kC2, 8, xx);
    gampl = gam2 - x * gam1;
    gammi = gam2 + x * gam1;
}

// Computes J_xnu(x), Y_xnu(x), J'_xnu(x), Y'_xnu(x) for x > 0 and xnu >= 0
// via Steed's continued fractions + Temme's series (Numerical Recipes bessjy),
// with power-of-two rescaling to prevent overflow/underflow at large orders.
void bessjy(f64 x, f64 xnu, f64& rj, f64& ry, f64& rjp, f64& ryp) noexcept {
    const i32 nl = (x < kXMin) ? as<i32>(xnu + 0.5)
                               : keel::max(i32{0}, as<i32>(xnu - x + 1.5));
    const f64 xmu = xnu - as<f64>(nl);
    const f64 xmu2 = xmu * xmu;
    const f64 xi = 1.0 / x;
    const f64 xi2 = 2.0 * xi;
    const f64 w = xi2 / kPi;

    i32 isign = 1;
    f64 h = xnu * xi;
    if (h < kFpMin) h = kFpMin;
    f64 b = xi2 * xnu;
    f64 d = 0.0;
    f64 c = h;
    for (i32 i = 1; i <= kMaxIter; ++i) {
        b += xi2;
        d = b - d;
        if (keel::abs(d) < kFpMin) d = kFpMin;
        c = b - 1.0 / c;
        if (keel::abs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        const f64 del = c * d;
        h = del * h;
        if (d < 0.0) isign = -isign;
        if (keel::abs(del - 1.0) < kEps) break;
    }

    f64 rjl = as<f64>(isign);
    f64 rjpl = h * rjl;
    const f64 rjl1 = rjl;
    const f64 rjp1 = rjpl;
    i32 rj_shift = 0;
    f64 fact = xnu * xi;
    for (i32 l = nl; l >= 1; --l) {
        const f64 rjtemp = fact * rjl + rjpl;
        fact -= xi;
        rjpl = fact * rjtemp - rjl;
        rjl = rjtemp;
        const f64 mag = keel::max(keel::abs(rjl), keel::abs(rjpl));
        if (mag > 1.0e100) {
            rjl = keel::ldexp(rjl, -300);
            rjpl = keel::ldexp(rjpl, -300);
            rj_shift += 300;
        }
    }
    if (rjl == 0.0) rjl = kEps;
    const f64 f = rjpl / rjl;

    f64 rjmu = 0.0;
    f64 rymu = 0.0;
    f64 rymup = 0.0;
    f64 ry1 = 0.0;

    if (x < kXMin) {
        const f64 x2 = 0.5 * x;
        const f64 pimu = kPi * xmu;
        fact = (keel::abs(pimu) < kEps) ? 1.0 : (pimu / keel::sin(pimu));
        d = -keel::log(x2);
        f64 e = xmu * d;
        const f64 fact2 = (keel::abs(e) < kEps) ? 1.0 : (keel::sinh(e) / e);
        f64 gam1 = 0.0;
        f64 gam2 = 0.0;
        f64 gampl = 0.0;
        f64 gammi = 0.0;
        beschb(xmu, gam1, gam2, gampl, gammi);
        f64 ff = (2.0 / kPi) * fact * (gam1 * keel::cosh(e) + gam2 * fact2 * d);
        e = keel::exp(e);
        f64 p = e / (gampl * kPi);
        f64 q = 1.0 / (e * kPi * gammi);
        const f64 pimu2 = 0.5 * pimu;
        const f64 fact3 = (keel::abs(pimu2) < kEps) ? 1.0 : (keel::sin(pimu2) / pimu2);
        const f64 r = kPi * pimu2 * fact3 * fact3;
        c = 1.0;
        d = -x2 * x2;
        f64 sum = ff + r * q;
        f64 sum1 = p;
        for (i32 i = 1; i <= kMaxIter; ++i) {
            const f64 fi = as<f64>(i);
            ff = (fi * ff + p + q) / (fi * fi - xmu2);
            c *= d / fi;
            p /= (fi - xmu);
            q /= (fi + xmu);
            const f64 del = c * (ff + r * q);
            sum += del;
            const f64 del1 = c * p - fi * del;
            sum1 += del1;
            if (keel::abs(del) < (1.0 + keel::abs(sum)) * kEps) break;
        }
        rymu = -sum;
        ry1 = -sum1 * xi2;
        rymup = xmu * xi * rymu - ry1;
        rjmu = w / (rymup - f * rymu);
    } else {
        f64 a = 0.25 - xmu2;
        f64 pq = -0.5 * xi;
        f64 q = 1.0;
        const f64 br = 2.0 * x;
        f64 bi = 2.0;
        fact = a * xi / (pq * pq + q * q);
        f64 cr = br + q * fact;
        f64 ci = bi + pq * fact;
        f64 den = br * br + bi * bi;
        f64 dr = br / den;
        f64 di = -bi / den;
        f64 dlr = cr * dr - ci * di;
        f64 dli = cr * di + ci * dr;
        f64 temp = pq * dlr - q * dli;
        q = pq * dli + q * dlr;
        pq = temp;
        for (i32 i = 2; i <= kMaxIter; ++i) {
            a += as<f64>(2 * (i - 1));
            bi += 2.0;
            dr = a * dr + br;
            di = a * di + bi;
            if (keel::abs(dr) + keel::abs(di) < kFpMin) dr = kFpMin;
            fact = a / (cr * cr + ci * ci);
            cr = br + cr * fact;
            ci = bi - ci * fact;
            if (keel::abs(cr) + keel::abs(ci) < kFpMin) cr = kFpMin;
            den = dr * dr + di * di;
            dr /= den;
            di /= -den;
            dlr = cr * dr - ci * di;
            dli = cr * di + ci * dr;
            temp = pq * dlr - q * dli;
            q = pq * dli + q * dlr;
            pq = temp;
            if (keel::abs(dlr - 1.0) + keel::abs(dli) < kEps) break;
        }
        const f64 gam = (pq - f) / q;
        rjmu = keel::sqrt(w / ((pq - f) * gam + q));
        rjmu = (rjl >= 0.0) ? keel::abs(rjmu) : -keel::abs(rjmu);
        rymu = rjmu * gam;
        rymup = rymu * (pq + q / gam);
        ry1 = xmu * xi * rymu - rymup;
    }

    fact = rjmu / rjl;
    rj = keel::ldexp(rjl1 * fact, -rj_shift);
    rjp = keel::ldexp(rjp1 * fact, -rj_shift);
    for (i32 i = 1; i <= nl; ++i) {
        const f64 rytemp = (xmu + as<f64>(i)) * xi2 * ry1 - rymu;
        rymu = ry1;
        ry1 = rytemp;
        if (is_inf(rymu)) break;
    }
    ry = rymu;
    ryp = xnu * xi * rymu - ry1;
}

// Computes exponentially scaled modified Bessel functions for x > 0 and xnu >= 0:
//   I_xnu(x) = ri_s * exp(x) * 2^(-ri_shift)
//   K_xnu(x) = rk_s * exp(-x) * 2^(rk_shift)
// Completely avoids exp(-x) underflow for x > 700 and intermediate recurrence overflow for large xnu.
void bessik_scaled(f64 x, f64 xnu, f64& ri_s, i32& ri_shift, f64& rk_s, i32& rk_shift) noexcept {
    const i32 nl = as<i32>(xnu + 0.5);
    const f64 xmu = xnu - as<f64>(nl);
    const f64 xmu2 = xmu * xmu;
    const f64 xi = 1.0 / x;
    const f64 xi2 = 2.0 * xi;

    f64 h = xnu * xi;
    if (h < kFpMin) h = kFpMin;
    f64 b = xi2 * xnu;
    f64 d = 0.0;
    f64 c = h;
    for (i32 i = 1; i <= kMaxIter; ++i) {
        b += xi2;
        d = 1.0 / (b + d);
        c = b + 1.0 / c;
        const f64 del = c * d;
        h = del * h;
        if (keel::abs(del - 1.0) < kEps) break;
    }

    f64 ril = 1.0;
    f64 ripl = h;
    const f64 ril1 = 1.0;
    ri_shift = 0;
    f64 fact = xnu * xi;
    for (i32 l = nl; l >= 1; --l) {
        const f64 ritemp = fact * ril + ripl;
        fact -= xi;
        ripl = fact * ritemp + ril;
        ril = ritemp;
        const f64 mag = keel::max(keel::abs(ril), keel::abs(ripl));
        if (mag > 1.0e100) {
            ril = keel::ldexp(ril, -300);
            ripl = keel::ldexp(ripl, -300);
            ri_shift += 300;
        }
    }
    const f64 f = ripl / ril;

    // rkmu_s = exp(x) * K_xmu(x), rk1_s = exp(x) * K_{xmu+1}(x)
    f64 rkmu_s = 0.0;
    f64 rk1_s = 0.0;
    if (x < kXMin) {
        const f64 x2 = 0.5 * x;
        const f64 pimu = kPi * xmu;
        fact = (keel::abs(pimu) < kEps) ? 1.0 : (pimu / keel::sin(pimu));
        d = -keel::log(x2);
        f64 e = xmu * d;
        const f64 fact2 = (keel::abs(e) < kEps) ? 1.0 : (keel::sinh(e) / e);
        f64 gam1 = 0.0;
        f64 gam2 = 0.0;
        f64 gampl = 0.0;
        f64 gammi = 0.0;
        beschb(xmu, gam1, gam2, gampl, gammi);
        f64 ff = fact * (gam1 * keel::cosh(e) + gam2 * fact2 * d);
        f64 sum = ff;
        e = keel::exp(e);
        f64 p = 0.5 * e / gampl;
        f64 q = 0.5 / (e * gammi);
        c = 1.0;
        d = x2 * x2;
        f64 sum1 = p;
        for (i32 i = 1; i <= kMaxIter; ++i) {
            const f64 fi = as<f64>(i);
            ff = (fi * ff + p + q) / (fi * fi - xmu2);
            c *= d / fi;
            p /= (fi - xmu);
            q /= (fi + xmu);
            const f64 del = c * ff;
            sum += del;
            const f64 del1 = c * (p - fi * ff);
            sum1 += del1;
            if (keel::abs(del) < keel::abs(sum) * kEps) break;
        }
        const f64 ex = keel::exp(x);
        rkmu_s = sum * ex;
        rk1_s = (sum1 * xi2) * ex;
    } else {
        b = 2.0 * (1.0 + x);
        d = 1.0 / b;
        f64 delh = d;
        h = delh;
        f64 q1 = 0.0;
        f64 q2 = 1.0;
        const f64 a1 = 0.25 - xmu2;
        f64 q = a1;
        f64 c_cf = a1;
        f64 a = -a1;
        f64 s = 1.0 + q * delh;
        for (i32 i = 2; i <= kMaxIter; ++i) {
            const f64 fi = as<f64>(i);
            a -= 2.0 * (fi - 1.0);
            c_cf = -a * c_cf / fi;
            const f64 qnew = (q1 - b * q2) / a;
            q1 = q2;
            q2 = qnew;
            q += c_cf * qnew;
            b += 2.0;
            d = 1.0 / (b + a * d);
            delh = (b * d - 1.0) * delh;
            h += delh;
            const f64 dels = q * delh;
            s += dels;
            if (keel::abs(dels / s) < kEps) break;
        }
        h = a1 * h;
        rkmu_s = keel::sqrt(kPi / (2.0 * x)) / s;
        rk1_s = rkmu_s * (xmu + x + 0.5 - h) * xi;
    }

    const f64 rkmup_s = xmu * xi * rkmu_s - rk1_s;
    const f64 rimu_s = xi / (f * rkmu_s - rkmup_s);
    ri_s = (rimu_s * ril1) / ril;

    rk_shift = 0;
    for (i32 i = 1; i <= nl; ++i) {
        const f64 rktemp = (xmu + as<f64>(i)) * xi2 * rk1_s + rkmu_s;
        rkmu_s = rk1_s;
        rk1_s = rktemp;
        if (keel::abs(rk1_s) > 1.0e100) {
            rkmu_s = keel::ldexp(rkmu_s, -300);
            rk1_s = keel::ldexp(rk1_s, -300);
            rk_shift += 300;
        }
    }
    rk_s = rkmu_s;
}

f64 combine_scaled(f64 val_s, i32 shift_pow2, f64 exp_arg) noexcept {
    if (val_s == 0.0) return 0.0;
    if (shift_pow2 == 0 && keel::abs(exp_arg) < 700.0) {
        return val_s * keel::exp(exp_arg);
    }
    const f64 sign = (val_s < 0.0) ? -1.0 : 1.0;
    const f64 log_res = keel::log(keel::abs(val_s)) + as<f64>(shift_pow2) * kLn2 + exp_arg;
    if (log_res > 709.782712893384) return sign * Limits<f64>::infinity();
    if (log_res < -745.1332191019412) return 0.0;
    return sign * keel::exp(log_res);
}

bool is_near_integer(f64 v) noexcept { return is_finite(v) && keel::abs(v - keel::round(v)) < 1e-12; }

} // namespace

// ============================================================================
// 5. Bessel, Modified Bessel & Hankel Functions
// ============================================================================

f64 besselj(f64 nu, f64 x) noexcept {
    if (is_nan(nu) || is_nan(x) || is_inf(nu)) return Limits<f64>::quiet_nan();
    if (x == 0.0) {
        if (nu == 0.0) return 1.0;
        if (nu > 0.0) return 0.0;
        if (is_near_integer(nu)) return 0.0;
        return Limits<f64>::infinity();
    }
    if (is_inf(x)) return 0.0;
    if (keel::abs(nu) > kMaxOrder) return Limits<f64>::quiet_nan();
    if (x < 0.0) {
        if (!is_near_integer(nu)) return Limits<f64>::quiet_nan();
        const i64 n = as<i64>(keel::round(nu));
        const f64 v = besselj(nu, -x);
        return (n & 1) ? -v : v;
    }
    if (nu < 0.0) {
        const f64 anu = -nu;
        if (is_near_integer(anu)) {
            const i64 n = as<i64>(keel::round(anu));
            f64 rj = 0.0, ry = 0.0, rjp = 0.0, ryp = 0.0;
            bessjy(x, anu, rj, ry, rjp, ryp);
            return (n & 1) ? -rj : rj;
        }
        f64 rj = 0.0, ry = 0.0, rjp = 0.0, ryp = 0.0;
        bessjy(x, anu, rj, ry, rjp, ryp);
        return cospi_exact(anu) * rj - sinpi_exact(anu) * ry;
    }
    f64 rj = 0.0, ry = 0.0, rjp = 0.0, ryp = 0.0;
    bessjy(x, nu, rj, ry, rjp, ryp);
    return rj;
}

f64 bessely(f64 nu, f64 x) noexcept {
    if (is_nan(nu) || is_nan(x) || is_inf(nu)) return Limits<f64>::quiet_nan();
    if (x < 0.0) return Limits<f64>::quiet_nan();
    if (x == 0.0) return -Limits<f64>::infinity();
    if (is_inf(x)) return 0.0;
    if (keel::abs(nu) > kMaxOrder) return Limits<f64>::quiet_nan();
    if (nu < 0.0) {
        const f64 anu = -nu;
        if (is_near_integer(anu)) {
            const i64 n = as<i64>(keel::round(anu));
            f64 rj = 0.0, ry = 0.0, rjp = 0.0, ryp = 0.0;
            bessjy(x, anu, rj, ry, rjp, ryp);
            return (n & 1) ? -ry : ry;
        }
        f64 rj = 0.0, ry = 0.0, rjp = 0.0, ryp = 0.0;
        bessjy(x, anu, rj, ry, rjp, ryp);
        return sinpi_exact(anu) * rj + cospi_exact(anu) * ry;
    }
    f64 rj = 0.0, ry = 0.0, rjp = 0.0, ryp = 0.0;
    bessjy(x, nu, rj, ry, rjp, ryp);
    return ry;
}

f64 besseli(f64 nu, f64 x) noexcept {
    if (is_nan(nu) || is_nan(x) || is_inf(nu)) return Limits<f64>::quiet_nan();
    if (x == 0.0) {
        if (nu == 0.0) return 1.0;
        if (nu > 0.0) return 0.0;
        if (is_near_integer(nu)) return 0.0;
        return Limits<f64>::infinity();
    }
    if (x < 0.0) {
        if (!is_near_integer(nu)) return Limits<f64>::quiet_nan();
        const i64 n = as<i64>(keel::round(nu));
        const f64 v = besseli(nu, -x);
        return (n & 1) ? -v : v;
    }
    if (is_inf(x)) return Limits<f64>::infinity();
    if (keel::abs(nu) > kMaxOrder) return Limits<f64>::quiet_nan();
    if (nu < 0.0) {
        const f64 anu = -nu;
        f64 ri_s = 0.0, rk_s = 0.0;
        i32 ri_shift = 0, rk_shift = 0;
        bessik_scaled(x, anu, ri_s, ri_shift, rk_s, rk_shift);
        const f64 ri = combine_scaled(ri_s, -ri_shift, x);
        if (is_near_integer(anu)) return ri;
        const f64 rk = combine_scaled(rk_s, rk_shift, -x);
        return ri + (2.0 / kPi) * sinpi_exact(anu) * rk;
    }
    f64 ri_s = 0.0, rk_s = 0.0;
    i32 ri_shift = 0, rk_shift = 0;
    bessik_scaled(x, nu, ri_s, ri_shift, rk_s, rk_shift);
    return combine_scaled(ri_s, -ri_shift, x);
}

f64 besselk(f64 nu, f64 x) noexcept {
    if (is_nan(nu) || is_nan(x) || is_inf(nu)) return Limits<f64>::quiet_nan();
    if (x < 0.0) return Limits<f64>::quiet_nan();
    if (x == 0.0) return Limits<f64>::infinity();
    if (is_inf(x)) return 0.0;
    const f64 anu = keel::abs(nu); // K_{-nu}(x) = K_nu(x)
    if (anu > kMaxOrder) return Limits<f64>::quiet_nan();
    f64 ri_s = 0.0, rk_s = 0.0;
    i32 ri_shift = 0, rk_shift = 0;
    bessik_scaled(x, anu, ri_s, ri_shift, rk_s, rk_shift);
    return combine_scaled(rk_s, rk_shift, -x);
}

c64 besselh(f64 nu, i32 k, f64 x) noexcept {
    if (k != 1 && k != 2) {
        const f64 nan = Limits<f64>::quiet_nan();
        return c64(nan, nan);
    }
    const f64 sign = (k == 1) ? 1.0 : -1.0;
    return c64(besselj(nu, x), sign * bessely(nu, x));
}

// ============================================================================
// 6. Airy Functions
// ============================================================================

f64 airy(i32 k, f64 x) noexcept {
    if (is_nan(x) || k < 0 || k > 3) return Limits<f64>::quiet_nan();

    if (is_inf(x)) {
        if (x > 0.0) {
            // Ai(+inf) = Ai'(+inf) = 0; Bi(+inf) = Bi'(+inf) = +inf
            return (k <= 1) ? 0.0 : Limits<f64>::infinity();
        }
        // Ai(-inf) = Bi(-inf) = 0; Ai'(-inf), Bi'(-inf) oscillate with unbounded amplitude
        return (k == 0 || k == 2) ? 0.0 : Limits<f64>::quiet_nan();
    }

    if (x == 0.0) {
        constexpr f64 kAi0 = 0.35502805388781723926;
        constexpr f64 kAip0 = -0.25881940379280679841;
        constexpr f64 kBi0 = 0.61492662744600073516;
        constexpr f64 kBip0 = 0.44828835735382635791;
        switch (k) {
            case 0: return kAi0;
            case 1: return kAip0;
            case 2: return kBi0;
            case 3: return kBip0;
            default: return Limits<f64>::quiet_nan();
        }
    }

    constexpr f64 kInvSqrt3 = 0.577350269189625764509148780501957456;

    if (x > 0.0) {
        // For x > 110, zeta = (2/3)*x^(3/2) > 769: Ai, Ai' underflow to 0; Bi, Bi' overflow to +inf.
        if (x > 110.0) {
            return (k <= 1) ? 0.0 : Limits<f64>::infinity();
        }
        const f64 zeta = (2.0 / 3.0) * x * keel::sqrt(x);
        const f64 sx3 = keel::sqrt(x / 3.0);
        if (k == 0) return keel::kInvPi * sx3 * besselk(1.0 / 3.0, zeta);
        if (k == 1) return -(x * kInvSqrt3) * keel::kInvPi * besselk(2.0 / 3.0, zeta);
        if (k == 2) return sx3 * (besseli(1.0 / 3.0, zeta) + besseli(-1.0 / 3.0, zeta));
        return (x * kInvSqrt3) * (besseli(2.0 / 3.0, zeta) + besseli(-2.0 / 3.0, zeta));
    }

    const f64 ax = -x;
    const f64 zeta = (2.0 / 3.0) * ax * keel::sqrt(ax);
    if (is_inf(zeta)) {
        return (k == 0 || k == 2) ? 0.0 : Limits<f64>::quiet_nan();
    }
    const f64 sax = keel::sqrt(ax);
    if (k == 0) return (sax / 3.0) * (besselj(1.0 / 3.0, zeta) + besselj(-1.0 / 3.0, zeta));
    if (k == 1) return (ax / 3.0) * (besselj(2.0 / 3.0, zeta) - besselj(-2.0 / 3.0, zeta));
    if (k == 2) return (sax * kInvSqrt3) * (besselj(-1.0 / 3.0, zeta) - besselj(1.0 / 3.0, zeta));
    return (ax * kInvSqrt3) * (besselj(2.0 / 3.0, zeta) + besselj(-2.0 / 3.0, zeta));
}

} // namespace keel::specfun
