#include "keel/specfun.h"

#include "keel/math.h"

namespace keel {

namespace {

constexpr f64 kInvSqrtPi = 0.564189583547756286948079451560772586;
constexpr f64 kSqrt2Pi = 2.50662827463100050241576528481104525;
constexpr f64 kHalfLn2Pi = 0.918938533204672741780329736405617640;
constexpr f64 kEulerGamma = 0.577215664901532860606512090082402431;

// Lanczos g = 7, n = 9 coefficients (Paul Godfrey, full IEEE-754 double accuracy).
constexpr f64 kLanczosG = 7.0;
constexpr f64 kLanczosC[9] = {
    0.99999999999980993227684700473478,
    676.520368121885098567009190444019,
    -1259.13921672240287047156078755283,
    771.3234287776530788486528258894,
    -176.61502916214059906584551354,
    12.507343278686904814458936853,
    -0.13857109526572011689554707,
    9.984369578019570859563e-6,
    1.50563273514931155834e-7,
};

f64 lanczos_sum(f64 z) noexcept {
    f64 x = kLanczosC[0];
    for (i32 i = 1; i < 9; ++i) {
        x += kLanczosC[i] / (z + static_cast<f64>(i));
    }
    return x;
}

// ============================================================================
// Cody (1969) rational Chebyshev approximation for erf / erfc / erfcx.
// Netlib SPECFUN CALERF (double precision, < 2 ULP over all real x).
// jint == 0: erf(x)
// jint == 1: erfc(x)
// jint == 2: erfcx(x) = exp(x^2) * erfc(x)
// ============================================================================
f64 calerf(f64 arg, i32 jint) noexcept {
    if (is_nan(arg)) return arg;

    static constexpr f64 kA[5] = {
        3.16112374387056560e00,
        1.13864154151050156e02,
        3.77485237685302021e02,
        3.20937758913846947e03,
        1.85777706184603153e-1,
    };
    static constexpr f64 kB[4] = {
        2.36012909523441209e01,
        2.44024637934444173e02,
        1.28261652607737228e03,
        2.84423683343917062e03,
    };
    static constexpr f64 kC[9] = {
        5.64188496988670089e-1,
        8.88314979438837594e00,
        6.61191906371416295e01,
        2.98635138197400131e02,
        8.81952221241769090e02,
        1.71204761263407058e03,
        2.05107837782607147e03,
        1.23033935479799725e03,
        2.15311535474403846e-8,
    };
    static constexpr f64 kD[8] = {
        1.57449261107098347e01,
        1.17693950891312499e02,
        5.37181101862009858e02,
        1.62138957456669019e03,
        3.29079923573345963e03,
        4.36261909014324716e03,
        3.43936767414372164e03,
        1.23033935480374942e03,
    };
    static constexpr f64 kP[6] = {
        3.05326634961232344e-1,
        3.60344899949804439e-1,
        1.25781726111229246e-1,
        1.60837851487422766e-2,
        6.58749161529837803e-4,
        1.63153871373020978e-2,
    };
    static constexpr f64 kQ[5] = {
        2.56852019228982242e00,
        1.87295284992346047e00,
        5.27905102951428412e-1,
        6.05183413124413191e-2,
        2.33520497626869185e-3,
    };

    constexpr f64 kThresh = 0.46875;
    constexpr f64 kXSmall = 1.11e-16;
    constexpr f64 kXBig = 26.543;
    constexpr f64 kXHuge = 6.71e7;
    constexpr f64 kXMax = 2.53e307;
    constexpr f64 kXNeg = -26.628;

    const f64 x = arg;
    const f64 y = keel::abs(x);
    f64 result = 0.0;

    if (y <= kThresh) {
        const f64 ysq = (y > kXSmall) ? (y * y) : 0.0;
        f64 xnum = kA[4] * ysq;
        f64 xden = ysq;
        for (i32 i = 0; i < 3; ++i) {
            xnum = (xnum + kA[i]) * ysq;
            xden = (xden + kB[i]) * ysq;
        }
        result = x * (xnum + kA[3]) / (xden + kB[3]);
        if (jint != 0) result = 1.0 - result;
        if (jint == 2) result = keel::exp(ysq) * result;
        return result;
    }

    if (y <= 4.0) {
        f64 xnum = kC[8] * y;
        f64 xden = y;
        for (i32 i = 0; i < 7; ++i) {
            xnum = (xnum + kC[i]) * y;
            xden = (xden + kD[i]) * y;
        }
        result = (xnum + kC[7]) / (xden + kD[7]);
        if (jint != 2) {
            const f64 ysq = keel::trunc(y * 16.0) / 16.0;
            const f64 del = (y - ysq) * (y + ysq);
            result = keel::exp(-ysq * ysq) * keel::exp(-del) * result;
        }
    } else {
        if (y >= kXBig) {
            if ((jint != 2) || (y >= kXMax)) {
                result = 0.0;
            } else if (y >= kXHuge) {
                result = kInvSqrtPi / y;
            } else {
                const f64 ysq = 1.0 / (y * y);
                f64 xnum = kP[5] * ysq;
                f64 xden = ysq;
                for (i32 i = 0; i < 4; ++i) {
                    xnum = (xnum + kP[i]) * ysq;
                    xden = (xden + kQ[i]) * ysq;
                }
                result = ysq * (xnum + kP[4]) / (xden + kQ[4]);
                result = (kInvSqrtPi - result) / y;
            }
        } else {
            const f64 ysq = 1.0 / (y * y);
            f64 xnum = kP[5] * ysq;
            f64 xden = ysq;
            for (i32 i = 0; i < 4; ++i) {
                xnum = (xnum + kP[i]) * ysq;
                xden = (xden + kQ[i]) * ysq;
            }
            result = ysq * (xnum + kP[4]) / (xden + kQ[4]);
            result = (kInvSqrtPi - result) / y;
            if (jint != 2) {
                const f64 ysq2 = keel::trunc(y * 16.0) / 16.0;
                const f64 del = (y - ysq2) * (y + ysq2);
                result = keel::exp(-ysq2 * ysq2) * keel::exp(-del) * result;
            }
        }
    }

    if (jint == 0) {
        result = (0.5 - result) + 0.5;
        if (x < 0.0) result = -result;
    } else if (jint == 1) {
        if (x < 0.0) result = 2.0 - result;
    } else {
        if (x < 0.0) {
            if (x < kXNeg) {
                result = Limits<f64>::infinity();
            } else {
                const f64 ysq = keel::trunc(x * 16.0) / 16.0;
                const f64 del = (x - ysq) * (x + ysq);
                const f64 yexp = keel::exp(ysq * ysq) * keel::exp(del);
                result = (yexp + yexp) - result;
            }
        }
    }
    return result;
}

f64 gser_lower(f64 a, f64 x) noexcept {
    if (x <= 0.0) return 0.0;
    constexpr i32 kMaxIter = 200;
    constexpr f64 kEps = 1e-15;
    f64 ap = a;
    f64 summ = 1.0 / a;
    f64 del = summ;
    for (i32 n = 0; n < kMaxIter; ++n) {
        ap += 1.0;
        del *= x / ap;
        summ += del;
        if (keel::abs(del) < keel::abs(summ) * kEps) break;
    }
    return summ * keel::exp(-x + a * keel::log(x) - gammaln(a));
}

f64 gcf_upper(f64 a, f64 x) noexcept {
    constexpr i32 kMaxIter = 200;
    constexpr f64 kEps = 1e-15;
    constexpr f64 kFpMin = 1e-300;
    f64 b = x + 1.0 - a;
    f64 c = 1.0 / kFpMin;
    f64 d = 1.0 / b;
    f64 h = d;
    for (i32 i = 1; i <= kMaxIter; ++i) {
        const f64 fi = static_cast<f64>(i);
        const f64 an = -fi * (fi - a);
        b += 2.0;
        d = an * d + b;
        if (keel::abs(d) < kFpMin) d = kFpMin;
        c = b + an / c;
        if (keel::abs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        const f64 del = d * c;
        h *= del;
        if (keel::abs(del - 1.0) < kEps) break;
    }
    return keel::exp(-x + a * keel::log(x) - gammaln(a)) * h;
}

f64 betacf(f64 a, f64 b, f64 x) noexcept {
    constexpr i32 kMaxIter = 200;
    constexpr f64 kEps = 1e-15;
    constexpr f64 kFpMin = 1e-300;
    const f64 qab = a + b;
    const f64 qap = a + 1.0;
    const f64 qam = a - 1.0;
    f64 c = 1.0;
    f64 d = 1.0 - qab * x / qap;
    if (keel::abs(d) < kFpMin) d = kFpMin;
    d = 1.0 / d;
    f64 h = d;
    for (i32 m = 1; m <= kMaxIter; ++m) {
        const f64 fm = static_cast<f64>(m);
        const f64 m2 = 2.0 * fm;
        f64 aa = fm * (b - fm) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (keel::abs(d) < kFpMin) d = kFpMin;
        c = 1.0 + aa / c;
        if (keel::abs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + fm) * (qab + fm) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (keel::abs(d) < kFpMin) d = kFpMin;
        c = 1.0 + aa / c;
        if (keel::abs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        const f64 del = d * c;
        h *= del;
        if (keel::abs(del - 1.0) < kEps) break;
    }
    return h;
}

} // namespace

// ============================================================================
// 1. Gamma, Digamma & Incomplete Gamma
// ============================================================================

f64 gammaln(f64 x) noexcept {
    if (is_nan(x)) return x;
    if (is_inf(x)) return Limits<f64>::infinity();
    if (x <= 0.0 && x == keel::floor(x)) return Limits<f64>::infinity();

    if (x < 0.5) {
        // Reflection: ln|Gamma(x)| = ln(pi / |sin(pi*x)|) - ln|Gamma(1 - x)|
        const f64 s = keel::abs(keel::sin(kPi * x));
        return keel::log(kPi / s) - gammaln(1.0 - x);
    }

    // Exact zeros at x = 1 and x = 2.
    if (x == 1.0 || x == 2.0) return 0.0;

    const f64 z = x - 1.0;
    const f64 t = z + kLanczosG + 0.5;
    return kHalfLn2Pi + (z + 0.5) * keel::log(t) - t + keel::log(lanczos_sum(z));
}

f64 gamma(f64 x) noexcept {
    if (is_nan(x)) return x;
    if (x <= 0.0 && x == keel::floor(x)) return Limits<f64>::infinity();
    if (is_inf(x)) return Limits<f64>::infinity();

    // Small positive integers: exact factorials up to 22! (23! exceeds 53-bit exactness).
    if (x >= 1.0 && x <= 23.0 && x == keel::floor(x)) {
        const i32 n = static_cast<i32>(x);
        f64 res = 1.0;
        for (i32 k = 2; k < n; ++k) res *= static_cast<f64>(k);
        return res;
    }

    if (x > 171.624376956302725) return Limits<f64>::infinity();

    if (x < 0.5) {
        // Reflection: Gamma(x) = pi / (sin(pi * x) * Gamma(1 - x))
        const f64 s = keel::sin(kPi * x);
        const f64 lg = gammaln(1.0 - x);
        if (lg > 709.0) return 0.0;
        return kPi / (s * gamma(1.0 - x));
    }

    const f64 z = x - 1.0;
    const f64 t = z + kLanczosG + 0.5;
    const f64 sum = lanczos_sum(z);
    // Split pow(t, z + 0.5) to avoid intermediate overflow for x in [142, 171].
    if (x > 140.0) {
        const f64 half_pow = keel::pow(t, 0.5 * (z + 0.5));
        return kSqrt2Pi * half_pow * keel::exp(-t) * half_pow * sum;
    }
    return kSqrt2Pi * keel::pow(t, z + 0.5) * keel::exp(-t) * sum;
}

f64 psi(f64 x) noexcept {
    if (is_nan(x)) return x;
    if (x == 0.0) return -Limits<f64>::infinity();
    if (x == keel::floor(x) && x < 0.0) return Limits<f64>::quiet_nan();
    if (is_inf(x)) return x > 0.0 ? Limits<f64>::infinity() : Limits<f64>::quiet_nan();

    f64 r = 0.0;
    f64 y = x;
    if (y < 0.5) {
        r -= kPi / keel::tan(kPi * y);
        y = 1.0 - y;
    }
    while (y < 10.0) {
        r -= 1.0 / y;
        y += 1.0;
    }
    const f64 yi = 1.0 / y;
    const f64 yi2 = yi * yi;
    f64 s = keel::log(y) - 0.5 * yi;
    s -= yi2 * (1.0 / 12.0 -
                yi2 * (1.0 / 120.0 -
                       yi2 * (1.0 / 252.0 -
                              yi2 * (1.0 / 240.0 - yi2 * (5.0 / 660.0 - yi2 * (691.0 / 32760.0))))));
    return r + s;
}

f64 gammainc(f64 x, f64 a) noexcept {
    if (is_nan(x) || is_nan(a)) return Limits<f64>::quiet_nan();
    if (x < 0.0 || a <= 0.0) return Limits<f64>::quiet_nan();
    if (x == 0.0) return 0.0;
    if (is_inf(x)) return 1.0;
    if (x < a + 1.0) return gser_lower(a, x);
    return 1.0 - gcf_upper(a, x);
}

f64 gammainc_upper(f64 x, f64 a) noexcept {
    if (is_nan(x) || is_nan(a)) return Limits<f64>::quiet_nan();
    if (x < 0.0 || a <= 0.0) return Limits<f64>::quiet_nan();
    if (x == 0.0) return 1.0;
    if (is_inf(x)) return 0.0;
    if (x < a + 1.0) return 1.0 - gser_lower(a, x);
    return gcf_upper(a, x);
}

f64 gammaincinv(f64 p, f64 a) noexcept {
    if (is_nan(p) || is_nan(a)) return Limits<f64>::quiet_nan();
    if (a <= 0.0) return Limits<f64>::quiet_nan();
    if (p < 0.0 || p > 1.0) return Limits<f64>::quiet_nan();
    if (p == 0.0) return 0.0;
    if (p == 1.0) return Limits<f64>::infinity();

    // Wilson-Hilferty initial approximation.
    const f64 z = kSqrt2 * erfinv(2.0 * p - 1.0);
    const f64 t = 1.0 - 1.0 / (9.0 * a) + z / keel::sqrt(9.0 * a);
    f64 x = a * t * t * t;
    if (!is_finite(x) || x <= 0.0) x = 0.5 * a;

    const f64 lg_a = gammaln(a);
    for (i32 it = 0; it < 60; ++it) {
        const f64 f = gammainc(x, a) - p;
        if (keel::abs(f) < 1e-15) break;
        const f64 logf = (a - 1.0) * keel::log(x) - x - lg_a;
        const f64 df = keel::exp(logf);
        if (df < 1e-300) break;
        f64 dx = f / df;
        if (dx >= x) dx = 0.5 * x;
        x -= dx;
        if (x <= 0.0) x = 1e-300;
        if (keel::abs(dx) < 1e-13 * (keel::abs(x) + 1.0)) break;
    }
    return x;
}

// ============================================================================
// 2. Beta & Incomplete Beta
// ============================================================================

f64 beta(f64 z, f64 w) noexcept {
    if (z > 0.0 && w > 0.0) {
        return keel::exp(gammaln(z) + gammaln(w) - gammaln(z + w));
    }
    return gamma(z) * gamma(w) / gamma(z + w);
}

f64 betaln(f64 z, f64 w) noexcept { return gammaln(z) + gammaln(w) - gammaln(z + w); }

f64 betainc(f64 x, f64 a, f64 b) noexcept {
    if (is_nan(x) || is_nan(a) || is_nan(b)) return Limits<f64>::quiet_nan();
    if (x < 0.0 || x > 1.0 || a <= 0.0 || b <= 0.0) return Limits<f64>::quiet_nan();
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;
    const f64 bt = keel::exp(gammaln(a + b) - gammaln(a) - gammaln(b) + a * keel::log(x) +
                             b * keel::log1p(-x));
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * betacf(a, b, x) / a;
    }
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

f64 betaincinv(f64 p, f64 a, f64 b) noexcept {
    if (is_nan(p) || is_nan(a) || is_nan(b)) return Limits<f64>::quiet_nan();
    if (a <= 0.0 || b <= 0.0) return Limits<f64>::quiet_nan();
    if (p < 0.0 || p > 1.0) return Limits<f64>::quiet_nan();
    if (p == 0.0) return 0.0;
    if (p == 1.0) return 1.0;

    f64 lo = 0.0;
    f64 hi = 1.0;
    f64 x = a / (a + b);
    const f64 lbeta = gammaln(a) + gammaln(b) - gammaln(a + b);

    for (i32 it = 0; it < 80; ++it) {
        const f64 f = betainc(x, a, b) - p;
        if (keel::abs(f) < 1e-15) break;
        if (f > 0.0) {
            hi = x;
        } else {
            lo = x;
        }
        const f64 logf = (a - 1.0) * keel::log(x) + (b - 1.0) * keel::log1p(-x) - lbeta;
        const f64 df = keel::exp(logf);
        f64 x_new = (df > 1e-300) ? (x - f / df) : (0.5 * (lo + hi));
        if (!(x_new > lo && x_new < hi)) x_new = 0.5 * (lo + hi);
        if (keel::abs(x_new - x) < 1e-14 * (keel::abs(x) + 1.0)) {
            x = x_new;
            break;
        }
        x = x_new;
    }
    return x;
}

// ============================================================================
// 3. Error Functions
// ============================================================================

f64 erf(f64 x) noexcept { return calerf(x, 0); }

f64 erfc(f64 x) noexcept { return calerf(x, 1); }

f64 erfcx(f64 x) noexcept { return calerf(x, 2); }

f64 erfinv(f64 y) noexcept {
    if (is_nan(y)) return y;
    if (y > 1.0 || y < -1.0) return Limits<f64>::quiet_nan();
    if (y == 1.0) return Limits<f64>::infinity();
    if (y == -1.0) return -Limits<f64>::infinity();
    if (y == 0.0) return 0.0;

    constexpr f64 kA = 0.147;
    constexpr f64 k2OverPi = 2.0 / kPi;
    const f64 s = (y < 0.0) ? -1.0 : 1.0;
    const f64 ay = keel::abs(y);
    const f64 l = keel::log1p(-ay * ay);
    const f64 t = k2OverPi / kA + 0.5 * l;
    f64 z = s * keel::sqrt(keel::sqrt(t * t - l / kA) - t);

    for (i32 i = 0; i < 4; ++i) {
        const f64 err = erf(z) - y;
        const f64 dz = err / (2.0 * kInvSqrtPi * keel::exp(-z * z));
        z -= dz;
        if (keel::abs(dz) <= 1e-15 * keel::abs(z)) break;
    }
    return z;
}

f64 erfcinv(f64 y) noexcept {
    if (is_nan(y)) return y;
    if (y < 0.0 || y > 2.0) return Limits<f64>::quiet_nan();
    if (y == 0.0) return Limits<f64>::infinity();
    if (y == 2.0) return -Limits<f64>::infinity();
    if (y == 1.0) return 0.0;

    if (y >= 0.25 && y <= 1.75) {
        return erfinv(1.0 - y);
    }

    // Tail handling without 1 - y cancellation.
    const bool upper_tail = (y < 0.25);
    const f64 q = upper_tail ? y : (2.0 - y);
    // Asymptotic initial guess for erfc(z) = q: z ~ sqrt(-ln(q * sqrt(pi))).
    const f64 l = -keel::log(q);
    f64 z = keel::sqrt(l - 0.5 * keel::log(kPi * l));
    if (!is_finite(z) || z <= 0.0) z = 1.0;
    for (i32 i = 0; i < 6; ++i) {
        const f64 err = erfc(z) - q;
        const f64 df = -2.0 * kInvSqrtPi * keel::exp(-z * z);
        if (df == 0.0) break;
        const f64 dz = err / df;
        z -= dz;
        if (keel::abs(dz) <= 1e-15 * keel::abs(z)) break;
    }
    return upper_tail ? z : -z;
}

// ============================================================================
// 4. Exponential Integral
// ============================================================================

f64 expint(f64 x) noexcept {
    if (is_nan(x)) return x;
    if (x == 0.0) return Limits<f64>::infinity();
    if (x < 0.0) return Limits<f64>::quiet_nan();
    if (x > 740.0) return 0.0;
    if (x <= 1.0) {
        f64 term = 1.0;
        f64 sum = 0.0;
        for (i32 n = 1; n <= 100; ++n) {
            const f64 fn = static_cast<f64>(n);
            term *= -x / fn;
            const f64 add = -term / fn;
            sum += add;
            if (keel::abs(add) < keel::abs(sum) * 1e-16) break;
        }
        return -kEulerGamma - keel::log(x) + sum;
    }
    constexpr f64 kTiny = 1e-300;
    f64 b = x + 1.0;
    f64 c = 1.0 / kTiny;
    f64 d = 1.0 / b;
    f64 h = d;
    for (i32 i = 1; i <= 200; ++i) {
        const f64 fi = static_cast<f64>(i);
        const f64 a = -(fi * fi);
        b += 2.0;
        d = 1.0 / (a * d + b);
        c = b + a / c;
        const f64 del = c * d;
        h *= del;
        if (keel::abs(del - 1.0) < 1e-16) break;
    }
    return h * keel::exp(-x);
}

// ============================================================================
// 7. Associated Legendre Functions
// ============================================================================

void legendre(i32 n, f64 x, MutArrayRef<f64> out) noexcept {
    if (n < 0 || out.size() < static_cast<usize>(n) + 1) return;
    const f64 sx = keel::sqrt(keel::max(0.0, 1.0 - x * x));
    for (i32 m = 0; m <= n; ++m) {
        f64 pmm = 1.0;
        for (i32 i = 1; i <= m; ++i) {
            pmm *= -(2.0 * static_cast<f64>(i) - 1.0) * sx;
        }
        if (n == m) {
            out[static_cast<usize>(m)] = pmm;
            continue;
        }
        f64 pmmp1 = x * (2.0 * static_cast<f64>(m) + 1.0) * pmm;
        if (n == m + 1) {
            out[static_cast<usize>(m)] = pmmp1;
            continue;
        }
        f64 pll = 0.0;
        for (i32 l = m + 2; l <= n; ++l) {
            const f64 fl = static_cast<f64>(l);
            const f64 fm = static_cast<f64>(m);
            pll = ((2.0 * fl - 1.0) * x * pmmp1 - (fl + fm - 1.0) * pmm) / (fl - fm);
            pmm = pmmp1;
            pmmp1 = pll;
        }
        out[static_cast<usize>(m)] = pmmp1;
    }
}

// ============================================================================
// 8. Complete Elliptic Integrals & Jacobi Elliptic Functions
// ============================================================================

EllipKE ellipke(f64 m) noexcept {
    if (is_nan(m)) {
        const f64 nan = Limits<f64>::quiet_nan();
        return EllipKE{nan, nan};
    }
    if (m == 1.0) return EllipKE{Limits<f64>::infinity(), 1.0};
    if (m < 0.0 || m > 1.0) {
        const f64 nan = Limits<f64>::quiet_nan();
        return EllipKE{nan, nan};
    }

    f64 a = 1.0;
    f64 g = keel::sqrt(1.0 - m);
    f64 sum = 0.5 * m;
    f64 scale = 0.25;
    for (i32 i = 0; i < 64; ++i) {
        const f64 diff = a - g;
        if (keel::abs(diff) <= 1e-15 * a) break;
        sum += scale * diff * diff;
        const f64 a_new = 0.5 * (a + g);
        const f64 g_new = keel::sqrt(a * g);
        a = a_new;
        g = g_new;
        scale *= 2.0;
    }
    const f64 k = kPi / (2.0 * a);
    const f64 e = k * (1.0 - sum);
    return EllipKE{k, e};
}

EllipJ ellipj(f64 u, f64 m) noexcept {
    if (is_nan(u) || is_nan(m) || m < 0.0 || m > 1.0) {
        const f64 nan = Limits<f64>::quiet_nan();
        return EllipJ{nan, nan, nan};
    }
    if (m == 0.0) return EllipJ{keel::sin(u), keel::cos(u), 1.0};
    if (m == 1.0) {
        const f64 cn = 1.0 / keel::cosh(u);
        return EllipJ{keel::tanh(u), cn, cn};
    }

    constexpr i32 kMax = 16;
    f64 a[kMax + 1];
    f64 c[kMax + 1];
    a[0] = 1.0;
    f64 b = keel::sqrt(1.0 - m);
    c[0] = keel::sqrt(m);
    i32 n = 0;
    for (i32 i = 1; i <= kMax; ++i) {
        const f64 an = 0.5 * (a[i - 1] + b);
        const f64 bn = keel::sqrt(a[i - 1] * b);
        const f64 cn_ = 0.5 * (a[i - 1] - b);
        a[i] = an;
        c[i] = cn_;
        b = bn;
        n = i;
        if (keel::abs(cn_) < 1e-15 * keel::abs(an)) break;
    }
    f64 phi = keel::ldexp(a[n] * u, n);
    for (i32 i = n; i >= 1; --i) {
        phi = 0.5 * (phi + keel::asin((c[i] / a[i]) * keel::sin(phi)));
    }
    const f64 sn = keel::sin(phi);
    const f64 cn = keel::cos(phi);
    const f64 dn = keel::sqrt(1.0 - m * sn * sn);
    return EllipJ{sn, cn, dn};
}

// ============================================================================
// 9. Marcum Q-Function
// ============================================================================

f64 marcumq(f64 a, f64 b, i32 m) noexcept {
    if (is_nan(a) || is_nan(b)) return Limits<f64>::quiet_nan();
    if (m < 1) m = 1;
    if (b <= 0.0) return 1.0;
    if (a == 0.0) {
        const f64 y = 0.5 * b * b;
        f64 acc = 0.0;
        f64 term = keel::exp(-y);
        for (i32 k = 0; k < m; ++k) {
            acc += term;
            term *= y / static_cast<f64>(k + 1);
        }
        return acc;
    }
    const f64 upper = keel::max(b + 20.0, a + 20.0);
    constexpr i32 kSteps = 4096;
    const f64 dx = (upper - b) / static_cast<f64>(kSteps);
    f64 sum = 0.0;
    const f64 fm1 = static_cast<f64>(m - 1);
    for (i32 k = 0; k <= kSteps; ++k) {
        const f64 xv = b + static_cast<f64>(k) * dx;
        const f64 w = (k == 0 || k == kSteps) ? 1.0 : ((k & 1) ? 4.0 : 2.0);
        const f64 f = xv * keel::pow(xv / a, fm1) * keel::exp(-0.5 * (xv * xv + a * a)) *
                      besseli(fm1, a * xv);
        sum += w * f;
    }
    return sum * (dx / 3.0);
}

} // namespace keel
