#include "keel/specfun.h"

#include "keel/math.h"

namespace keel::specfun {

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
        x += kLanczosC[i] / (z + as<f64>(i));
    }
    return x;
}

// Range-reduced sin(pi * x) and cot(pi * x) for accurate reflection formulas at large negative x.
f64 sinpi_reduced(f64 x) noexcept {
    f64 r = keel::fmod(x, 2.0);
    if (r < 0.0) r += 2.0;
    if (r == 0.0 || r == 1.0) return 0.0;
    if (r == 0.5) return 1.0;
    if (r == 1.5) return -1.0;
    if (r > 1.0) return -keel::sin(kPi * (r - 1.0));
    if (r > 0.5) return keel::sin(kPi * (1.0 - r));
    return keel::sin(kPi * r);
}

f64 cotpi_reduced(f64 x) noexcept {
    f64 r = keel::fmod(x, 1.0);
    if (r < 0.0) r += 1.0;
    if (r == 0.5) return 0.0;
    if (r > 0.5) return -1.0 / keel::tan(kPi * (1.0 - r));
    return 1.0 / keel::tan(kPi * r);
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

f64 clamp01(f64 v) noexcept {
    if (v <= 0.0) return 0.0;
    if (v >= 1.0) return 1.0;
    return v;
}

// Stirling series remainder Delta(a) = gammaln(a) - ((a - 0.5)*log(a) - a + 0.5*log(2*pi)), exact for a >= 10.
f64 stirling_corr(f64 a) noexcept {
    const f64 r = 1.0 / a;
    const f64 r2 = r * r;
    return r * (1.0 / 12.0 +
                r2 * (-1.0 / 360.0 +
                      r2 * (1.0 / 1260.0 +
                            r2 * (-1.0 / 1680.0 + r2 * (1.0 / 1188.0 - r2 * (691.0 / 360360.0))))));
}

// Computes log(1 + u) - u accurately even when |u| is tiny.
f64 log1p_minus(f64 u) noexcept {
    if (keel::abs(u) < 0.05) {
        // -u^2/2 + u^3/3 - u^4/4 + ...
        f64 term = -u * u;
        f64 sum = 0.5 * term;
        for (i32 k = 3; k <= 24; ++k) {
            term *= -u;
            const f64 add = term / as<f64>(k);
            sum += add;
            if (keel::abs(add) <= keel::abs(sum) * 1e-16) break;
        }
        return sum;
    }
    return keel::log1p(u) - u;
}

// Cancellation-free evaluation of -x + a*log(x) - gammaln(a) for a > 0, x > 0.
f64 log_gamma_pref(f64 a, f64 x) noexcept {
    if (a < 10.0) {
        return -x + a * keel::log(x) - gammaln(a);
    }
    // -x + a*log(x) - ((a - 0.5)*log(a) - a + 0.5*log(2*pi) + Delta(a))
    // = 0.5*log(a) - 0.5*log(2*pi) - Delta(a) + a*(log(x/a) - (x/a - 1))
    const f64 u = (x - a) / a;
    return 0.5 * keel::log(a) - kHalfLn2Pi - stirling_corr(a) + a * log1p_minus(u);
}

// Cancellation-free evaluation of log P_Pois(k; mu) = -mu + k*log(mu) - gammaln(k + 1).
f64 log_pois_exact(i64 k, f64 mu) noexcept {
    if (k == 0) return -mu;
    const f64 fk = as<f64>(k);
    if (k < 10) {
        return -mu + fk * keel::log(mu) - gammaln(fk + 1.0);
    }
    // gammaln(k + 1) = (k + 0.5)*log(k) - k + 0.5*log(2*pi) + Delta(k)
    const f64 u = (mu - fk) / fk;
    return -0.5 * keel::log(fk) - kHalfLn2Pi - stirling_corr(fk) + fk * log1p_minus(u);
}

// Cancellation-free evaluation of log(x^a * (1 - x)^b / B(a, b)) for a, b > 0 and x in (0, 1).
f64 log_beta_pref(f64 x, f64 a, f64 b) noexcept {
    if (a < 10.0 || b < 10.0) {
        return gammaln(a + b) - gammaln(a) - gammaln(b) + a * keel::log(x) + b * keel::log1p(-x);
    }
    const f64 s = a + b;
    const f64 p = a / s;
    const f64 q = b / s;
    // Stirling decomposition:
    // gammaln(a+b) - gammaln(a) - gammaln(b) + a*log(x) + b*log(1-x)
    // = 0.5*log(a*b / (2*pi*(a+b))) + Delta(a+b) - Delta(a) - Delta(b)
    //   + a*log(x/p) + b*log((1-x)/q)
    // Note: a*(x/p - 1) + b*((1-x)/q - 1) = s*(x - p) + s*((1-x) - q) = 0 identically!
    const f64 u = (x - p) / p;
    const f64 v = ((1.0 - x) - q) / q;
    return 0.5 * (keel::log(a) + keel::log(b) - keel::log(s)) - kHalfLn2Pi +
           stirling_corr(s) - stirling_corr(a) - stirling_corr(b) +
           a * log1p_minus(u) + b * log1p_minus(v);
}

// Temme (1979) uniform asymptotic expansion for regularized incomplete gamma when a is very large.
// Accurate to ~1e-14 for a >= 1e5 across all x > 0.
f64 gammainc_temme(f64 x, f64 a, bool upper) noexcept {
    const f64 lam = x / a;
    const f64 mu = lam - 1.0;
    f64 eta = 0.0;
    if (keel::abs(mu) < 1e-4) {
        // -2*(lam - 1 - log(lam)) = mu^2 * (1 - 2/3*mu + 1/2*mu^2 - 2/5*mu^3 + 1/3*mu^4)
        const f64 poly = 1.0 + mu * (-2.0 / 3.0 + mu * (0.5 + mu * (-0.4 + mu * (1.0 / 3.0 - mu * (2.0 / 7.0)))));
        eta = mu * keel::sqrt(poly);
    } else {
        const f64 arg = 2.0 * (lam - 1.0 - keel::log(lam));
        eta = (mu >= 0.0) ? keel::sqrt(arg) : -keel::sqrt(arg);
    }
    const f64 z = eta * keel::sqrt(0.5 * a);
    const f64 base = upper ? (0.5 * erfc(z)) : (0.5 * erfc(-z));

    // Leading correction R_a(eta) = -exp(-0.5 * a * eta^2) / sqrt(2 * pi * a) * (c0 + c1/a)
    f64 c0 = 0.0;
    if (keel::abs(eta) < 1e-3) {
        c0 = -1.0 / 3.0 + eta * (1.0 / 12.0 + eta * (-23.0 / 540.0 + eta * (353.0 / 12960.0)));
    } else {
        c0 = (1.0 / mu) - (1.0 / eta);
    }
    const f64 corr = keel::exp(-z * z) / keel::sqrt(2.0 * kPi * a) * c0;
    return clamp01(upper ? (base - corr) : (base + corr));
}

f64 gser_lower(f64 a, f64 x) noexcept {
    if (x <= 0.0) return 0.0;
    constexpr i32 kMaxIter = 20000;
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
    const f64 log_pref = log_gamma_pref(a, x);
    if (log_pref < -745.0) {
        const f64 log_res = log_pref + keel::log(summ);
        if (log_res < -745.1332191019412) return 0.0;
        return clamp01(keel::exp(log_res));
    }
    return clamp01(summ * keel::exp(log_pref));
}

f64 gcf_upper(f64 a, f64 x) noexcept {
    constexpr i32 kMaxIter = 20000;
    constexpr f64 kEps = 1e-15;
    constexpr f64 kFpMin = 1e-300;
    f64 b = x + 1.0 - a;
    f64 c = 1.0 / kFpMin;
    f64 d = 1.0 / b;
    f64 h = d;
    for (i32 i = 1; i <= kMaxIter; ++i) {
        const f64 fi = as<f64>(i);
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
    const f64 log_pref = log_gamma_pref(a, x);
    if (log_pref < -745.0) {
        const f64 log_res = log_pref + keel::log(h);
        if (log_res < -745.1332191019412) return 0.0;
        return clamp01(keel::exp(log_res));
    }
    return clamp01(keel::exp(log_pref) * h);
}

f64 betacf(f64 a, f64 b, f64 x) noexcept {
    constexpr i32 kMaxIter = 4000;
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
        const f64 fm = as<f64>(m);
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
        const f64 s = keel::abs(sinpi_reduced(x));
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
        const i32 n = as<i32>(x);
        f64 res = 1.0;
        for (i32 k = 2; k < n; ++k) res *= as<f64>(k);
        return res;
    }

    if (x > 171.624376956302725) return Limits<f64>::infinity();

    if (x < 0.5) {
        // Reflection: Gamma(x) = pi / (sin(pi * x) * Gamma(1 - x))
        const f64 s = sinpi_reduced(x);
        const f64 lg = gammaln(1.0 - x);
        if (lg > 709.0) return (s >= 0.0) ? 0.0 : -0.0;
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
        r -= kPi * cotpi_reduced(y);
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
    if (a >= 1.0e6) return gammainc_temme(x, a, false);
    if (x < a + 1.0) return gser_lower(a, x);
    return clamp01(1.0 - gcf_upper(a, x));
}

f64 gammainc_upper(f64 x, f64 a) noexcept {
    if (is_nan(x) || is_nan(a)) return Limits<f64>::quiet_nan();
    if (x < 0.0 || a <= 0.0) return Limits<f64>::quiet_nan();
    if (x == 0.0) return 1.0;
    if (is_inf(x)) return 0.0;
    if (a >= 1.0e6) return gammainc_temme(x, a, true);
    if (x < a + 1.0) return clamp01(1.0 - gser_lower(a, x));
    return gcf_upper(a, x);
}

f64 gammaincinv(f64 p, f64 a) noexcept {
    if (is_nan(p) || is_nan(a)) return Limits<f64>::quiet_nan();
    if (a <= 0.0) return Limits<f64>::quiet_nan();
    if (p < 0.0 || p > 1.0) return Limits<f64>::quiet_nan();
    if (p == 0.0) return 0.0;
    if (p == 1.0) return Limits<f64>::infinity();

    // Solve upper tail Q(a, x) = q = 1 - p when p > 0.5 to prevent cancellation near p = 1.
    const bool upper = (p > 0.5);
    const f64 q = 1.0 - p;

    // Initial approximation.
    f64 x = 0.0;
    if (a < 1.0 && p < 0.5) {
        const f64 lp = (keel::log(p) + gammaln(a + 1.0)) / a;
        x = (lp < -700.0) ? 1e-300 : keel::exp(lp);
    } else {
        const f64 z = upper ? (kSqrt2 * erfcinv(2.0 * q)) : (-kSqrt2 * erfcinv(2.0 * p));
        const f64 t = 1.0 - 1.0 / (9.0 * a) + z / keel::sqrt(9.0 * a);
        if (t > 0.05) {
            x = a * t * t * t;
        } else {
            const f64 lp = (keel::log(p) + gammaln(a + 1.0)) / a;
            x = (lp < -700.0) ? 1e-300 : keel::exp(lp);
        }
    }
    if (!is_finite(x) || x <= 0.0) x = 0.5 * a;

    // Bracketed Halley/Newton with bisection fallback.
    f64 lo = 0.0;
    f64 hi = Limits<f64>::infinity();
    for (i32 it = 0; it < 100; ++it) {
        // f(x) = P(a, x) - p = q - Q(a, x), strictly increasing in x.
        const f64 f = upper ? (q - gammainc_upper(x, a)) : (gammainc(x, a) - p);
        if (f == 0.0) break;
        if (f > 0.0) {
            hi = x;
        } else {
            lo = x;
        }
        const f64 target = upper ? q : p;
        if (keel::abs(f) <= 4e-15 * target) break;

        const f64 log_df = log_gamma_pref(a, x) - keel::log(x);
        f64 x_new = -1.0;
        if (log_df > -700.0 && log_df < 700.0) {
            const f64 df = keel::exp(log_df);
            const f64 w = f / df;
            // Halley correction: f''/f' = (a - 1)/x - 1; fall back to bisection if flat tail.
            const f64 r = (a - 1.0) / x - 1.0;
            const f64 wr = w * r;
            if (keel::abs(wr) < 1.0) {
                const f64 step = w / (1.0 - 0.5 * wr);
                if (keel::abs(step) <= 1e-14 * x) break;
                x_new = x - step;
            }
        }

        if (!(x_new > lo && x_new < hi)) {
            if (is_finite(hi)) {
                x_new = (lo > 0.0 && hi / lo > 16.0) ? keel::sqrt(lo * hi) : (0.5 * (lo + hi));
            } else {
                x_new = (x > 0.0) ? (2.0 * x) : 1.0;
            }
        }
        const f64 dx = keel::abs(x_new - x);
        x = x_new;
        if (dx <= 1e-14 * x) break;
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
    const f64 log_bt = log_beta_pref(x, a, b);
    if (x < (a + 1.0) / (a + b + 2.0)) {
        const f64 cf = betacf(a, b, x) / a;
        if (log_bt < -745.0) {
            const f64 log_res = log_bt + keel::log(cf);
            return (log_res < -745.1332191019412) ? 0.0 : clamp01(keel::exp(log_res));
        }
        return clamp01(keel::exp(log_bt) * cf);
    }
    const f64 cf = betacf(b, a, 1.0 - x) / b;
    if (log_bt < -745.0) {
        const f64 log_res = log_bt + keel::log(cf);
        const f64 tail = (log_res < -745.1332191019412) ? 0.0 : clamp01(keel::exp(log_res));
        return clamp01(1.0 - tail);
    }
    return clamp01(1.0 - keel::exp(log_bt) * cf);
}

f64 betaincinv(f64 p, f64 a, f64 b) noexcept {
    if (is_nan(p) || is_nan(a) || is_nan(b)) return Limits<f64>::quiet_nan();
    if (a <= 0.0 || b <= 0.0) return Limits<f64>::quiet_nan();
    if (p < 0.0 || p > 1.0) return Limits<f64>::quiet_nan();
    if (p == 0.0) return 0.0;
    if (p == 1.0) return 1.0;

    // Exploit symmetry I_{1-x}(b, a) = 1 - I_x(a, b) to eliminate upper-tail cancellation.
    if (p > 0.5) {
        return 1.0 - betaincinv(1.0 - p, b, a);
    }

    const f64 lbeta = gammaln(a) + gammaln(b) - gammaln(a + b);
    f64 lo = 0.0;
    f64 hi = 1.0;

    // Initial guess: small-p power approximation or Beta mean.
    f64 x = a / (a + b);
    const f64 log_x0 = (keel::log(p) + keel::log(a) + lbeta) / a;
    if (log_x0 < -1.0 && log_x0 > -700.0) {
        const f64 x_small = keel::exp(log_x0);
        if (x_small > 0.0 && x_small < x) x = x_small;
    }

    for (i32 it = 0; it < 100; ++it) {
        const f64 f = betainc(x, a, b) - p;
        if (f == 0.0) break;
        if (f > 0.0) {
            hi = x;
        } else {
            lo = x;
        }
        if (keel::abs(f) <= 4e-15 * p) break;

        const f64 log_df = log_beta_pref(x, a, b) - keel::log(x) - keel::log1p(-x);
        f64 x_new = -1.0;
        if (log_df > -700.0 && log_df < 700.0) {
            const f64 df = keel::exp(log_df);
            const f64 w = f / df;
            // Halley correction: f''/f' = (a - 1)/x - (b - 1)/(1 - x)
            const f64 r = (a - 1.0) / x - (b - 1.0) / (1.0 - x);
            const f64 wr = w * r;
            if (keel::abs(wr) < 1.0) {
                const f64 step = w / (1.0 - 0.5 * wr);
                if (keel::abs(step) <= 1e-14 * x) break;
                x_new = x - step;
            }
        }
        if (!(x_new > lo && x_new < hi)) {
            x_new = (lo > 0.0 && hi / lo > 16.0) ? keel::sqrt(lo * hi) : (0.5 * (lo + hi));
        }
        const f64 dx = keel::abs(x_new - x);
        x = x_new;
        if (dx <= 1e-14 * x) break;
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
    if (ay > 0.75) {
        return s * erfcinv(1.0 - ay);
    }
    const f64 l = keel::log1p(-ay * ay);
    const f64 t = k2OverPi / kA + 0.5 * l;
    f64 z = s * keel::sqrt(keel::sqrt(t * t - l / kA) - t);

    for (i32 i = 0; i < 6; ++i) {
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
            const f64 fn = as<f64>(n);
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
        const f64 fi = as<f64>(i);
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
    if (n < 0 || out.size() < as<usize>(n) + 1) return;
    if (is_nan(x) || keel::abs(x) > 1.0) {
        const f64 nan = Limits<f64>::quiet_nan();
        for (i32 m = 0; m <= n; ++m) {
            out[as<usize>(m)] = nan;
        }
        return;
    }
    const f64 sx = keel::sqrt(keel::max(0.0, 1.0 - x * x));
    for (i32 m = 0; m <= n; ++m) {
        f64 pmm = 1.0;
        for (i32 i = 1; i <= m; ++i) {
            pmm *= -(2.0 * as<f64>(i) - 1.0) * sx;
        }
        if (n == m) {
            out[as<usize>(m)] = pmm;
            continue;
        }
        f64 pmmp1 = x * (2.0 * as<f64>(m) + 1.0) * pmm;
        if (n == m + 1) {
            out[as<usize>(m)] = pmmp1;
            continue;
        }
        f64 pll = 0.0;
        for (i32 l = m + 2; l <= n; ++l) {
            const f64 fl = as<f64>(l);
            const f64 fm = as<f64>(m);
            pll = ((2.0 * fl - 1.0) * x * pmmp1 - (fl + fm - 1.0) * pmm) / (fl - fm);
            pmm = pmmp1;
            pmmp1 = pll;
        }
        out[as<usize>(m)] = pmmp1;
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
    if (a < 0.0 || b < 0.0) return Limits<f64>::quiet_nan();
    if (is_inf(a) && is_inf(b)) return Limits<f64>::quiet_nan();
    if (b == 0.0) return 1.0;
    if (is_inf(b)) return 0.0;
    if (is_inf(a)) return 1.0;
    if (a == 0.0) {
        return gammainc_upper(0.5 * b * b, as<f64>(m));
    }

    const f64 fm = as<f64>(m);
    const f64 lam = 0.5 * a * a;
    const f64 y = 0.5 * b * b;

    // Saddlepoint k0 = floor(0.5 * (hypot(m, a * b) - m)) maximizes the Poisson-Gamma summand.
    const f64 k0_f = keel::floor(0.5 * (keel::hypot(fm, a * b) - fm));
    if (k0_f > 2.0e6) {
        // Large-parameter normal/Sankaran asymptotic for non-central chi-square tail (a*b > 4e6):
        // df = 2m, nc = a^2, threshold = b^2.
        const f64 df = 2.0 * fm;
        const f64 nc = a * a;
        const f64 mean = df + nc;
        const f64 var = 2.0 * (df + 2.0 * nc);
        const f64 z = (b * b - mean) / keel::sqrt(2.0 * var);
        return clamp01(0.5 * erfc(z));
    }

    const i64 k0 = as<i64>(k0_f);
    const f64 fk0 = as<f64>(k0);

    // Cancellation-free log Poisson weights.
    auto log_pois = [](i64 k, f64 mu) noexcept -> f64 {
        return log_pois_exact(k, mu);
    };

    // Choose upper (Q_m) or lower (1 - Q_m) representation so the summed tail is <= ~0.6
    // and split at k0 using summation by parts so EVERY update is strictly additive (zero cancellation).
    const bool use_upper = (y >= lam + fm);

    if (use_upper) {
        // Q_m(a, b) = sum_{k=k0}^inf P_Pois(k; lam) * Q(m + k, y)
        //           + sum_{j=0}^{m + k0 - 2} P_Pois(j; y) * sum_{k=max(0, j - m + 1)}^{k0 - 1} P_Pois(k; lam)
        const f64 log_p0 = log_pois(k0, lam);
        const f64 log_d0 = log_pois(k0 + m, y);
        if (log_p0 < -750.0) return 0.0;

        f64 p_lam = keel::exp(log_p0);
        f64 g = gammainc_upper(y, fm + fk0);
        f64 d = (log_d0 < -745.0) ? 0.0 : keel::exp(log_d0);

        f64 acc = p_lam * g;
        f64 prev_term = acc;
        // Region 1 (upward from k0): Q(m + k + 1, y) = Q(m + k, y) + P_Pois(m + k; y) (pure addition).
        for (i64 k = k0 + 1;; ++k) {
            const f64 fk = as<f64>(k);
            p_lam *= lam / fk;
            g += d;
            if (g > 1.0) g = 1.0;
            d *= y / (fm + fk);
            const f64 term = p_lam * g;
            acc += term;
            if (p_lam == 0.0 || (term <= prev_term && term <= acc * 1e-16)) break;
            prev_term = term;
        }

        // Region 2 (j downward from m + k0 - 2 to 0): pure addition on both S and acc.
        if (k0 > 0) {
            const i64 j_start = m + k0 - 2;
            const f64 log_py = log_pois(j_start, y);
            const f64 log_pl = log_pois(k0 - 1, lam);
            if (log_py + log_pl > -745.0) {
                // Rescale both factors by their starting exponents to prevent premature underflow.
                const f64 scale = keel::exp(log_py + log_pl);
                f64 py = 1.0;
                f64 pl = 1.0;
                f64 s_lam = 1.0;
                f64 sub = py * s_lam;
                f64 prev_sub = sub;
                for (i64 j = j_start - 1; j >= 0; --j) {
                    py *= as<f64>(j + 1) / y;
                    if (j >= m - 1) {
                        const i64 r = j - m + 1;
                        pl *= as<f64>(r + 1) / lam;
                        s_lam += pl;
                    }
                    const f64 term = py * s_lam;
                    sub += term;
                    if (py == 0.0 || (term <= prev_sub && term <= sub * 1e-16)) break;
                    prev_sub = term;
                }
                acc += scale * sub;
            }
        }
        return clamp01(acc);
    }

    // Lower representation: 1 - Q_m(a, b)
    //   = sum_{k=0}^{k0} P_Pois(k; lam) * P(m + k, y)
    //   + sum_{j=m+k0+1}^inf P_Pois(j; y) * sum_{k=k0+1}^{j - m} P_Pois(k; lam)
    const f64 log_p0 = log_pois(k0, lam);
    if (log_p0 < -750.0) return 1.0;

    f64 p_lam = keel::exp(log_p0);
    f64 g = gammainc(y, fm + fk0);
    const f64 log_dm1 = (k0 + m - 1 >= 0) ? log_pois(k0 + m - 1, y) : -1e9;
    f64 d = (log_dm1 < -745.0) ? 0.0 : keel::exp(log_dm1);

    f64 acc = p_lam * g;
    f64 prev_term = acc;
    // Region A (k downward from k0 - 1 to 0): P(m + k - 1, y) = P(m + k, y) + P_Pois(m + k - 1; y) (pure addition).
    for (i64 k = k0 - 1; k >= 0; --k) {
        g += d;
        if (g > 1.0) g = 1.0;
        d *= as<f64>(k + m) / y;
        p_lam *= as<f64>(k + 1) / lam;
        const f64 term = p_lam * g;
        acc += term;
        if (p_lam == 0.0 || (term <= prev_term && term <= acc * 1e-16)) break;
        prev_term = term;
    }

    // Region B (j upward from m + k0 + 1): pure addition on both S and acc.
    {
        const i64 j_start = m + k0 + 1;
        const f64 log_py = log_pois(j_start, y);
        const f64 log_pl = log_pois(k0 + 1, lam);
        if (log_py + log_pl > -745.0) {
            const f64 scale = keel::exp(log_py + log_pl);
            f64 py = 1.0;
            f64 pl = 1.0;
            f64 s_lam = 1.0;
            f64 sub = py * s_lam;
            f64 prev_sub = sub;
            for (i64 j = j_start + 1;; ++j) {
                py *= y / as<f64>(j);
                pl *= lam / as<f64>(j - m);
                s_lam += pl;
                const f64 term = py * s_lam;
                sub += term;
                if (py == 0.0 || (term <= prev_sub && term <= sub * 1e-16)) break;
                prev_sub = term;
            }
            acc += scale * sub;
        }
    }

    return clamp01(1.0 - acc);
}

} // namespace keel::specfun
