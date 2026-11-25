#include "keel/format.h"
#include "keel/math.h"
#include "keel/specfun.h"

namespace {

keel::i32 g_failed = 0;

void check(bool cond, const char* expr, keel::i32 line) {
    if (!cond) {
        keel::eprintln("FAIL [line %d]: %s", line, expr);
        ++g_failed;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* expr, keel::i32 line) {
    const keel::f64 diff = keel::abs(got - expected);
    if (keel::is_nan(got) || diff > tol) {
        keel::eprintln("FAIL [line %d]: %s (got %.17g, expected %.17g, diff %.3e > tol %.3e)",
                       line, expr, got, expected, diff, tol);
        ++g_failed;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define CHECK_NEAR(got, expected, tol) check_near((got), (expected), (tol), #got " ~= " #expected, __LINE__)

void test_gamma_beta() {
    // Poles & special values.
    CHECK(keel::is_inf(keel::specfun::gamma(0.0)) && keel::specfun::gamma(0.0) > 0.0);
    CHECK(keel::is_inf(keel::specfun::gamma(-3.0)) && keel::specfun::gamma(-3.0) > 0.0);
    CHECK(keel::is_inf(keel::specfun::gamma(-keel::Limits<keel::f64>::infinity())));
    CHECK(keel::specfun::gamma(1.0) == 1.0);
    CHECK(keel::specfun::gamma(6.0) == 120.0);
    CHECK(keel::specfun::gammaln(1.0) == 0.0);
    CHECK(keel::specfun::gammaln(2.0) == 0.0);

    // Cross-check against keel::tgamma / keel::lgamma across positive and negative values.
    const keel::f64 xs[] = {-3.5, -1.25, -0.5, 0.1, 0.5, 1.5, 2.7, 5.2, 12.3, 50.0, 150.0};
    for (keel::f64 x : xs) {
        const keel::f64 ref_g = keel::tgamma(x);
        const keel::f64 ref_lg = keel::lgamma(x);
        CHECK_NEAR(keel::specfun::gamma(x), ref_g, 1e-12 * keel::max(1.0, keel::abs(ref_g)));
        CHECK_NEAR(keel::specfun::gammaln(x), ref_lg, 1e-12 * keel::max(1.0, keel::abs(ref_lg)));
    }

    // Digamma psi.
    CHECK(keel::is_inf(keel::specfun::psi(0.0)) && keel::specfun::psi(0.0) < 0.0);
    CHECK(keel::is_nan(keel::specfun::psi(-2.0)));
    CHECK_NEAR(keel::specfun::psi(1.0), -0.57721566490153286, 1e-13);
    CHECK_NEAR(keel::specfun::psi(0.5), -1.9635100260214235, 1e-13);
    CHECK_NEAR(keel::specfun::psi(10.0), 2.2517525890667211, 1e-13);

    // Beta & betaln.
    CHECK_NEAR(keel::specfun::beta(2.0, 3.0), 1.0 / 12.0, 1e-14);
    CHECK_NEAR(keel::specfun::betaln(10.0, 20.0), keel::lgamma(10.0) + keel::lgamma(20.0) - keel::lgamma(30.0), 1e-12);

    // Incomplete gamma & inverse.
    CHECK_NEAR(keel::specfun::gammainc(1.0, 1.0), 1.0 - keel::exp(-1.0), 1e-14);
    CHECK_NEAR(keel::specfun::gammainc_upper(1.0, 1.0), keel::exp(-1.0), 1e-14);
    CHECK_NEAR(keel::specfun::gammainc(0.0, 2.0), 0.0, 1e-15);
    CHECK_NEAR(keel::specfun::gammainc(50.0, 2.0), 1.0, 1e-12);
    CHECK_NEAR(keel::specfun::gammainc(keel::specfun::gammaincinv(0.73, 3.5), 3.5), 0.73, 1e-12);

    // Incomplete beta & inverse.
    CHECK_NEAR(keel::specfun::betainc(0.3, 2.0, 5.0) + keel::specfun::betainc(0.7, 5.0, 2.0), 1.0, 1e-13);
    CHECK_NEAR(keel::specfun::betainc(0.5, 1.0, 1.0), 0.5, 1e-14);
    CHECK_NEAR(keel::specfun::betainc(keel::specfun::betaincinv(0.27, 2.5, 4.0), 2.5, 4.0), 0.27, 1e-12);
}

void test_erf_and_expint() {
    const keel::f64 xs[] = {-20.0, -5.0, -1.5, -0.3, 0.0, 0.25, 0.46875, 1.0, 3.5, 5.0, 10.0, 25.0};
    for (keel::f64 x : xs) {
        CHECK_NEAR(keel::specfun::erf(x), keel::erf(x), 1e-14);
        CHECK_NEAR(keel::specfun::erfc(x), keel::erfc(x), 1e-14 * keel::max(1.0, keel::erfc(x)));
    }
    CHECK_NEAR(keel::specfun::erfcx(0.0), 1.0, 1e-14);
    CHECK_NEAR(keel::specfun::erfcx(1.0), keel::exp(1.0) * keel::erfc(1.0), 1e-14);
    CHECK_NEAR(keel::specfun::erfcx(50.0), 1.0 / (50.0 * keel::sqrt(keel::kPi)), 1e-5);
    CHECK_NEAR(keel::specfun::erfcx(1e6), 1.0 / (1e6 * keel::sqrt(keel::kPi)), 1e-18);

    // Inverse erf / erfc.
    const keel::f64 ys[] = {-0.99, -0.7, -0.1, 0.0, 0.3, 0.99};
    for (keel::f64 y : ys) {
        CHECK_NEAR(keel::specfun::erf(keel::specfun::erfinv(y)), y, 1e-13);
    }
    CHECK_NEAR(keel::specfun::erfc(keel::specfun::erfcinv(1e-12)), 1e-12, 1e-24);
    CHECK_NEAR(keel::specfun::erfc(keel::specfun::erfcinv(1.999999)), 1.999999, 1e-13);

    // Expint E1.
    CHECK(keel::is_inf(keel::specfun::expint(0.0)));
    CHECK_NEAR(keel::specfun::expint(0.5), 0.5597735947761608, 1e-13);
    CHECK_NEAR(keel::specfun::expint(1.0), 0.21938393439552027, 1e-13);
    CHECK_NEAR(keel::specfun::expint(5.0), 0.0011482955912753257, 1e-14);
}

void test_bessel_and_airy() {
    struct BesselRef {
        keel::f64 nu;
        keel::f64 x;
        keel::f64 j;
        keel::f64 y;
        keel::f64 i;
        keel::f64 k;
    };
    const BesselRef refs[] = {
        {0.0, 1.0, 0.76519768655796661, 0.088256964215676997, 1.2660658777520084, 0.42102443824070834},
        {1.0, 2.0, 0.57672480775687363, -0.1070324315409375, 1.5906368546373291, 0.13986588181652246},
        {2.0, 5.0, 0.04656511627775229, 0.3676628826055246, 17.505614966624236, 0.0053089437122234599},
        {0.5, 1.5, 0.64983807475374822, -0.0460831658930973, 1.3871617204034772, 0.2283350522282655},
        {1.0 / 3.0, 2.5, 0.19832093341860585, 0.46145947419129091, 3.1743242297241974, 0.063542537454733372},
        {2.5, 10.0, 0.19665848358181753, -0.16417847961494103, 2028.5127573919356, 2.3931325864627893e-05},
        {5.0, 15.0, 0.13045613456502958, 0.16717271575940015, 144572.01120063406, 2.1878261369258221e-07},
    };
    for (const BesselRef& r : refs) {
        CHECK_NEAR(keel::specfun::besselj(r.nu, r.x), r.j, 1e-11 * keel::max(1.0, keel::abs(r.j)));
        CHECK_NEAR(keel::specfun::bessely(r.nu, r.x), r.y, 1e-11 * keel::max(1.0, keel::abs(r.y)));
        CHECK_NEAR(keel::specfun::besseli(r.nu, r.x), r.i, 1e-11 * keel::max(1.0, keel::abs(r.i)));
        CHECK_NEAR(keel::specfun::besselk(r.nu, r.x), r.k, 1e-11 * keel::max(1.0, keel::abs(r.k)));
    }
    // Negative orders & negative x parity.
    CHECK_NEAR(keel::specfun::besselj(-1.0, 2.0), -keel::specfun::besselj(1.0, 2.0), 1e-14);
    CHECK_NEAR(keel::specfun::besselj(1.0, -2.0), -keel::specfun::besselj(1.0, 2.0), 1e-14);
    CHECK_NEAR(keel::specfun::besseli(-1.0, 2.0), keel::specfun::besseli(1.0, 2.0), 1e-14);
    CHECK_NEAR(keel::specfun::besselk(-2.5, 1.5), keel::specfun::besselk(2.5, 1.5), 1e-14);

    // Hankel.
    const keel::c64 h1 = keel::specfun::besselh(0.0, 1, 1.0);
    const keel::c64 h2 = keel::specfun::besselh(0.0, 2, 1.0);
    CHECK_NEAR(h1.real(), 0.7651976865579666, 1e-13);
    CHECK_NEAR(h1.imag(), 0.08825696421567696, 1e-13);
    CHECK_NEAR(h2.imag(), -h1.imag(), 1e-14);

    // Airy at x = 0, x = 1, x = -2.
    CHECK_NEAR(keel::specfun::airy(0, 0.0), 0.35502805388781724, 1e-14);
    CHECK_NEAR(keel::specfun::airy(1, 0.0), -0.25881940379280680, 1e-14);
    CHECK_NEAR(keel::specfun::airy(2, 0.0), 0.61492662744600074, 1e-14);
    CHECK_NEAR(keel::specfun::airy(3, 0.0), 0.44828835735382636, 1e-14);
    CHECK_NEAR(keel::specfun::airy(0, 1.0), 0.13529241631288141, 1e-12);
    CHECK_NEAR(keel::specfun::airy(1, 1.0), -0.15914744129679321, 1e-12);
    CHECK_NEAR(keel::specfun::airy(2, 1.0), 1.2074235949528713, 1e-12);
    CHECK_NEAR(keel::specfun::airy(0, -2.0), 0.22740742820168557, 1e-12);
    CHECK_NEAR(keel::specfun::airy(1, -2.0), 0.6182590207416910, 1e-12);
    // Wronskian Ai(x)*Bi'(x) - Ai'(x)*Bi(x) = 1/pi.
    const keel::f64 w_pos = keel::specfun::airy(0, 2.5) * keel::specfun::airy(3, 2.5) - keel::specfun::airy(1, 2.5) * keel::specfun::airy(2, 2.5);
    const keel::f64 w_neg = keel::specfun::airy(0, -3.5) * keel::specfun::airy(3, -3.5) - keel::specfun::airy(1, -3.5) * keel::specfun::airy(2, -3.5);
    CHECK_NEAR(w_pos, 1.0 / keel::kPi, 1e-11);
    CHECK_NEAR(w_neg, 1.0 / keel::kPi, 1e-11);
}

void test_legendre_ellip_marcum_comb() {
    // Associated Legendre P_2^m(0.5): P_2^0 = -0.125, P_2^1 = -3*0.5*sqrt(0.75), P_2^2 = 3*0.75 = 2.25.
    keel::f64 p[3] = {0.0, 0.0, 0.0};
    keel::specfun::legendre(2, 0.5, keel::MutArrayRef<keel::f64>(p, 3));
    CHECK_NEAR(p[0], -0.125, 1e-14);
    CHECK_NEAR(p[1], -1.5 * keel::sqrt(0.75), 1e-14);
    CHECK_NEAR(p[2], 2.25, 1e-14);

    // EllipKE.
    const keel::specfun::EllipKE ke0 = keel::specfun::ellipke(0.0);
    CHECK_NEAR(ke0.k, 0.5 * keel::kPi, 1e-14);
    CHECK_NEAR(ke0.e, 0.5 * keel::kPi, 1e-14);
    const keel::specfun::EllipKE ke5 = keel::specfun::ellipke(0.5);
    CHECK_NEAR(ke5.k, 1.8540746773013719, 1e-14);
    CHECK_NEAR(ke5.e, 1.3506438810476755, 1e-14);

    // EllipJ identities: sn^2 + cn^2 == 1, dn^2 + m*sn^2 == 1.
    const keel::specfun::EllipJ ej = keel::specfun::ellipj(0.7, 0.4);
    CHECK_NEAR(ej.sn * ej.sn + ej.cn * ej.cn, 1.0, 1e-14);
    CHECK_NEAR(ej.dn * ej.dn + 0.4 * ej.sn * ej.sn, 1.0, 1e-14);

    // Marcum Q: Q_1(0, b) = exp(-b^2/2), Q_1(a, 0) = 1, Q_1(1, 1) = 0.5*(1 + exp(-1)*I_0(1)).
    CHECK_NEAR(keel::specfun::marcumq(0.0, 2.0, 1), keel::exp(-2.0), 1e-14);
    CHECK_NEAR(keel::specfun::marcumq(2.0, 0.0, 1), 1.0, 1e-14);
    CHECK_NEAR(keel::specfun::marcumq(1.0, 1.0, 1), 0.73287980379682027, 1e-12);

    // Combinatorics & number theory.
    CHECK(keel::specfun::factorial(0) == 1.0);
    CHECK(keel::specfun::factorial(10) == 3628800.0);
    CHECK(keel::is_inf(keel::specfun::factorial(171)));
    CHECK(keel::specfun::nchoosek(10, 3) == 120.0);
    CHECK(keel::specfun::nchoosek(5, 6) == 0.0);
    CHECK(keel::specfun::is_prime_u64(2));
    CHECK(!keel::specfun::is_prime_u64(1));
    CHECK(keel::specfun::is_prime_u64(104729));
    CHECK(!keel::specfun::is_prime_u64(104727));
    auto ps = keel::specfun::primes_up_to(30);
    CHECK(ps.size() == 10 && ps[0] == 2 && ps[9] == 29);
    auto fs = keel::specfun::factor_u64(360);
    CHECK(fs.size() == 6 && fs[0] == 2 && fs[3] == 3 && fs[5] == 5);
}

} // namespace

int main() {
    test_gamma_beta();
    test_erf_and_expint();
    test_bessel_and_airy();
    test_legendre_ellip_marcum_comb();

    if (g_failed != 0) {
        keel::eprintln("test_specfun: %d failure(s)", g_failed);
        return 1;
    }
    keel::println("test_specfun: all tests passed");
    return 0;
}
