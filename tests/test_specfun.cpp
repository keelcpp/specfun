#include "keel/format.h"
#include "keel/fs.h"
#include "keel/math.h"
#include "keel/specfun.h"
#include "keel/string.h"

namespace {

keel::i32 g_failed = 0;

void check(bool cond, const char* expr, keel::i32 line) {
    if (!cond) {
        keel::eprintln("FAIL [line {}]: {}", line, expr);
        ++g_failed;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* expr, keel::i32 line) {
    const keel::f64 diff = keel::abs(got - expected);
    if (keel::is_nan(got) || diff > tol) {
        keel::eprintln("FAIL [line {}]: {} (got {}, expected {}, diff {} > tol {})",
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
    const keel::f64 xs[] = {-100.5, -10.25, -3.5, -1.25, -0.5, 0.1, 0.5, 1.5, 2.7, 5.2, 12.3, 50.0, 150.0};
    for (keel::f64 x : xs) {
        const keel::f64 ref_g = keel::tgamma(x);
        const keel::f64 ref_lg = keel::lgamma(x);
        CHECK_NEAR(keel::specfun::gamma(x), ref_g, 1e-12 * keel::max(1.0, keel::abs(ref_g)));
        CHECK_NEAR(keel::specfun::gammaln(x), ref_lg, 1e-12 * keel::max(1.0, keel::abs(ref_lg)));
    }

    // Digamma psi (including large negative non-integers).
    CHECK(keel::is_inf(keel::specfun::psi(0.0)) && keel::specfun::psi(0.0) < 0.0);
    CHECK(keel::is_nan(keel::specfun::psi(-2.0)));
    CHECK_NEAR(keel::specfun::psi(1.0), -0.57721566490153286, 1e-13);
    CHECK_NEAR(keel::specfun::psi(0.5), -1.9635100260214235, 1e-13);
    CHECK_NEAR(keel::specfun::psi(10.0), 2.2517525890667211, 1e-13);
    CHECK_NEAR(keel::specfun::psi(-100.5), 4.615124601338064, 1e-12);

    // Beta & betaln.
    CHECK_NEAR(keel::specfun::beta(2.0, 3.0), 1.0 / 12.0, 1e-14);
    CHECK_NEAR(keel::specfun::betaln(10.0, 20.0), keel::lgamma(10.0) + keel::lgamma(20.0) - keel::lgamma(30.0), 1e-12);

    // Incomplete gamma & inverse (including large a and extreme tails).
    CHECK_NEAR(keel::specfun::gammainc(1.0, 1.0), 1.0 - keel::exp(-1.0), 1e-14);
    CHECK_NEAR(keel::specfun::gammainc_upper(1.0, 1.0), keel::exp(-1.0), 1e-14);
    CHECK_NEAR(keel::specfun::gammainc(0.0, 2.0), 0.0, 1e-15);
    CHECK_NEAR(keel::specfun::gammainc(50.0, 2.0), 1.0, 1e-12);
    CHECK_NEAR(keel::specfun::gammainc(9950.0, 10000.0), 0.30941788486118255, 1e-11);
    CHECK_NEAR(keel::specfun::gammainc_upper(9950.0, 10000.0), 0.6905821151388174, 1e-11);
    CHECK_NEAR(keel::specfun::gammainc(keel::specfun::gammaincinv(0.73, 3.5), 3.5), 0.73, 1e-12);
    CHECK_NEAR(keel::specfun::gammainc(keel::specfun::gammaincinv(0.01, 0.05), 0.05), 0.01, 1e-12);
    const keel::f64 p_hi = 1.0 - 1e-10;
    CHECK_NEAR(keel::specfun::gammainc_upper(keel::specfun::gammaincinv(p_hi, 5.0), 5.0), 1.0 - p_hi, 1e-22);

    // Incomplete beta & inverse (including large a, b and extreme tails).
    CHECK_NEAR(keel::specfun::betainc(0.3, 2.0, 5.0) + keel::specfun::betainc(0.7, 5.0, 2.0), 1.0, 1e-13);
    CHECK_NEAR(keel::specfun::betainc(0.5, 1.0, 1.0), 0.5, 1e-14);
    CHECK_NEAR(keel::specfun::betainc(0.505, 5000.0, 5000.0), 0.8413447463710382, 1e-12);
    CHECK_NEAR(keel::specfun::betainc(keel::specfun::betaincinv(0.27, 2.5, 4.0), 2.5, 4.0), 0.27, 1e-12);
    CHECK_NEAR(keel::specfun::betainc(keel::specfun::betaincinv(1e-10, 0.2, 5.0), 0.2, 5.0), 1e-10, 1e-22);
    CHECK_NEAR(keel::specfun::betainc(keel::specfun::betaincinv(1.0 - 1e-6, 3.0, 0.5), 3.0, 0.5), 1.0 - 1e-6, 1e-10);
    CHECK_NEAR(keel::specfun::betainc(keel::specfun::betaincinv(p_hi, 0.5, 3.0), 0.5, 3.0), p_hi, 1e-14);
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
    // Negative orders & negative x parity + large-x / large-order stability.
    CHECK_NEAR(keel::specfun::besselj(-1.0, 2.0), -keel::specfun::besselj(1.0, 2.0), 1e-14);
    CHECK_NEAR(keel::specfun::besselj(1.0, -2.0), -keel::specfun::besselj(1.0, 2.0), 1e-14);
    CHECK_NEAR(keel::specfun::besseli(-1.0, 2.0), keel::specfun::besseli(1.0, 2.0), 1e-14);
    CHECK_NEAR(keel::specfun::besselk(-2.5, 1.5), keel::specfun::besselk(2.5, 1.5), 1e-14);
    CHECK(keel::is_inf(keel::specfun::besseli(0.0, 800.0)) && keel::specfun::besseli(0.0, 800.0) > 0.0);
    CHECK(keel::specfun::besselk(0.0, 800.0) == 0.0);
    CHECK_NEAR(keel::specfun::besselk(500.0, 720.0), 4.635255517368995e-242, 1e-252);
    CHECK_NEAR(keel::specfun::besseli(100.0, 700.0), 1.2176413667079262e299, 1e289);
    CHECK(keel::specfun::besselj(200.0, 1.0) == 0.0);
    CHECK_NEAR(keel::specfun::besselj(-100.5, 2.5), 5.491299152659355e146, 1e136);

    // Hankel.
    const keel::c64 h1 = keel::specfun::besselh(0.0, 1, 1.0);
    const keel::c64 h2 = keel::specfun::besselh(0.0, 2, 1.0);
    CHECK_NEAR(h1.real(), 0.7651976865579666, 1e-13);
    CHECK_NEAR(h1.imag(), 0.08825696421567696, 1e-13);
    CHECK_NEAR(h2.imag(), -h1.imag(), 1e-14);

    // Airy at x = 0, x = 1, x = -2, large positive x, and +/-Inf.
    CHECK_NEAR(keel::specfun::airy(0, 0.0), 0.35502805388781724, 1e-14);
    CHECK_NEAR(keel::specfun::airy(1, 0.0), -0.25881940379280680, 1e-14);
    CHECK_NEAR(keel::specfun::airy(2, 0.0), 0.61492662744600074, 1e-14);
    CHECK_NEAR(keel::specfun::airy(3, 0.0), 0.44828835735382636, 1e-14);
    CHECK_NEAR(keel::specfun::airy(0, 1.0), 0.13529241631288141, 1e-12);
    CHECK_NEAR(keel::specfun::airy(1, 1.0), -0.15914744129679321, 1e-12);
    CHECK_NEAR(keel::specfun::airy(2, 1.0), 1.2074235949528713, 1e-12);
    CHECK_NEAR(keel::specfun::airy(0, -2.0), 0.22740742820168557, 1e-12);
    CHECK_NEAR(keel::specfun::airy(1, -2.0), 0.6182590207416910, 1e-12);
    CHECK(keel::specfun::airy(0, 105.0) >= 0.0 && keel::specfun::airy(0, 105.0) < 1e-300);
    CHECK(keel::specfun::airy(1, 105.0) <= 0.0 && keel::specfun::airy(1, 105.0) > -1e-300);
    CHECK(keel::specfun::airy(0, 110.0) == 0.0);
    CHECK(keel::specfun::airy(1, 110.0) == 0.0);
    CHECK(keel::is_inf(keel::specfun::airy(2, 105.0)) && keel::specfun::airy(2, 105.0) > 0.0);
    CHECK(keel::is_inf(keel::specfun::airy(3, 105.0)) && keel::specfun::airy(3, 105.0) > 0.0);
    const keel::f64 inf = keel::Limits<keel::f64>::infinity();
    CHECK(keel::specfun::airy(0, inf) == 0.0);
    CHECK(keel::specfun::airy(1, inf) == 0.0);
    CHECK(keel::is_inf(keel::specfun::airy(2, inf)));
    CHECK(keel::is_inf(keel::specfun::airy(3, inf)));
    CHECK(keel::specfun::airy(0, -inf) == 0.0);
    CHECK(keel::is_nan(keel::specfun::airy(1, -inf)));
    CHECK(keel::specfun::airy(2, -inf) == 0.0);
    CHECK(keel::is_nan(keel::specfun::airy(3, -inf)));

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
    // Out-of-domain |x| > 1 produces NaN.
    keel::specfun::legendre(2, 1.5, keel::MutArrayRef<keel::f64>(p, 3));
    CHECK(keel::is_nan(p[0]) && keel::is_nan(p[1]) && keel::is_nan(p[2]));

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

    // Marcum Q across small, moderate, large, and deep-tail arguments.
    CHECK_NEAR(keel::specfun::marcumq(0.0, 2.0, 1), keel::exp(-2.0), 1e-14);
    CHECK_NEAR(keel::specfun::marcumq(2.0, 0.0, 1), 1.0, 1e-14);
    CHECK_NEAR(keel::specfun::marcumq(1.0, 1.0, 1), 0.73287980379682027, 1e-13);
    CHECK_NEAR(keel::specfun::marcumq(20.0, 20.0, 1), 0.50997667814096959, 1e-12);
    CHECK_NEAR(keel::specfun::marcumq(50.0, 50.0, 5), 0.53585877230456702, 1e-12);
    CHECK_NEAR(keel::specfun::marcumq(50.0, 70.0, 1), 3.2594057967498268e-89, 1e-99);
    CHECK_NEAR(keel::specfun::marcumq(500.0, 505.0, 3), 2.9414098812390249e-07, 1e-17);
    CHECK_NEAR(keel::specfun::marcumq(1000.0, 1000.0, 1), 0.50019947116514463, 1e-11);
    CHECK_NEAR(keel::specfun::marcumq(100.0, 90.0, 10), 1.0, 1e-14);

    // Combinatorics & number theory (including 64-bit Miller-Rabin primality).
    CHECK(keel::specfun::factorial(0) == 1.0);
    CHECK(keel::specfun::factorial(10) == 3628800.0);
    CHECK(keel::is_inf(keel::specfun::factorial(171)));
    CHECK(keel::specfun::nchoosek(10, 3) == 120.0);
    CHECK(keel::specfun::nchoosek(5, 6) == 0.0);
    CHECK(keel::is_inf(keel::specfun::nchoosek(2000, 1000)));
    CHECK(keel::specfun::is_prime_u64(2));
    CHECK(!keel::specfun::is_prime_u64(1));
    CHECK(keel::specfun::is_prime_u64(104729));
    CHECK(!keel::specfun::is_prime_u64(104727));
    CHECK(keel::specfun::is_prime_u64(18446744073709551557ULL));  // largest 64-bit prime
    CHECK(!keel::specfun::is_prime_u64(18446744073709551555ULL)); // composite near 2^64
    CHECK(!keel::specfun::is_prime_u64(341550071728321ULL));      // strong pseudoprime to first 7 primes
    auto ps = keel::specfun::primes_up_to(30);
    CHECK(ps.size() == 10 && ps[0] == 2 && ps[9] == 29);
    auto fs = keel::specfun::factor_u64(360);
    CHECK(fs.size() == 6 && fs[0] == 2 && fs[3] == 3 && fs[5] == 5);
}

bool write_f64_file(keel::StringRef dir, keel::StringRef name, keel::ArrayRef<keel::f64> data) {
    keel::String path = keel::format("{}/{}", dir, name);
    keel::ArrayRef<keel::u8> bv(reinterpret_cast<const keel::u8*>(data.data()), data.size() * sizeof(keel::f64));
    return keel::fs_write_file(path, bv).is_ok();
}

keel::i32 dump_reference(keel::StringRef dir) {
    // 1. Gamma / gammaln / psi grid
    {
        const keel::f64 xs[] = {-100.5, -10.25, -3.5, -0.5, -0.01, 0.01, 0.25, 0.5, 1.0, 1.5, 2.0, 5.5, 12.0, 50.0, 145.0, 170.5};
        keel::Array<keel::f64> g_out, lg_out, psi_out;
        for (keel::f64 x : xs) {
            g_out.push_back(keel::specfun::gamma(x));
            lg_out.push_back(keel::specfun::gammaln(x));
            psi_out.push_back(keel::specfun::psi(x));
        }
        if (!write_f64_file(dir, "gamma.bin", g_out)) return 2;
        if (!write_f64_file(dir, "gammaln.bin", lg_out)) return 2;
        if (!write_f64_file(dir, "psi.bin", psi_out)) return 2;
    }

    // 2. Beta / betaln / gammainc / gammainc_upper / gammaincinv / betainc / betaincinv
    {
        const keel::f64 b_pairs[][2] = {{0.5, 0.5}, {1.0, 1.0}, {2.5, 4.5}, {10.0, 20.0}, {100.0, 150.0}};
        keel::Array<keel::f64> b_out, bl_out;
        for (const auto& p : b_pairs) {
            b_out.push_back(keel::specfun::beta(p[0], p[1]));
            bl_out.push_back(keel::specfun::betaln(p[0], p[1]));
        }
        if (!write_f64_file(dir, "beta.bin", b_out)) return 2;
        if (!write_f64_file(dir, "betaln.bin", bl_out)) return 2;

        const keel::f64 gi_cases[][2] = {
            {0.01, 0.1}, {0.5, 0.5}, {1.0, 1.0}, {2.5, 3.0}, {10.0, 5.0},
            {100.0, 100.0}, {490.0, 500.0}, {9950.0, 10000.0}, {10050.0, 10000.0},
        };
        keel::Array<keel::f64> gil_out, giu_out;
        for (const auto& c : gi_cases) {
            gil_out.push_back(keel::specfun::gammainc(c[0], c[1]));
            giu_out.push_back(keel::specfun::gammainc_upper(c[0], c[1]));
        }
        if (!write_f64_file(dir, "gammainc.bin", gil_out)) return 2;
        if (!write_f64_file(dir, "gammainc_upper.bin", giu_out)) return 2;

        const keel::f64 ginv_cases[][2] = {
            {1e-10, 0.05}, {0.01, 0.2}, {0.5, 0.5}, {0.73, 3.5}, {0.999999, 5.0},
            {1.0 - 1e-10, 5.0}, {0.25, 100.0}, {0.8, 1000.0},
        };
        keel::Array<keel::f64> ginv_out;
        for (const auto& c : ginv_cases) ginv_out.push_back(keel::specfun::gammaincinv(c[0], c[1]));
        if (!write_f64_file(dir, "gammaincinv.bin", ginv_out)) return 2;

        const keel::f64 bi_cases[][3] = {
            {0.1, 0.2, 0.3}, {0.3, 2.0, 5.0}, {0.7, 5.0, 2.0}, {0.5, 10.0, 10.0},
            {0.99, 0.5, 0.5}, {0.505, 5000.0, 5000.0}, {0.2, 50.0, 200.0},
        };
        keel::Array<keel::f64> bi_out;
        for (const auto& c : bi_cases) bi_out.push_back(keel::specfun::betainc(c[0], c[1], c[2]));
        if (!write_f64_file(dir, "betainc.bin", bi_out)) return 2;

        const keel::f64 binv_cases[][3] = {
            {1e-10, 0.2, 5.0}, {0.05, 0.5, 0.5}, {0.27, 2.5, 4.0},
            {0.9, 10.0, 3.0}, {1.0 - 1e-10, 0.5, 3.0}, {0.5, 100.0, 200.0},
        };
        keel::Array<keel::f64> binv_out;
        for (const auto& c : binv_cases) binv_out.push_back(keel::specfun::betaincinv(c[0], c[1], c[2]));
        if (!write_f64_file(dir, "betaincinv.bin", binv_out)) return 2;
    }

    // 3. Error functions & expint
    {
        const keel::f64 exs[] = {-20.0, -5.0, -1.5, -0.3, 0.0, 0.25, 0.46875, 1.0, 3.5, 5.0, 10.0, 25.0};
        keel::Array<keel::f64> erf_o, erfc_o, erfcx_o;
        for (keel::f64 x : exs) {
            erf_o.push_back(keel::specfun::erf(x));
            erfc_o.push_back(keel::specfun::erfc(x));
            erfcx_o.push_back(keel::specfun::erfcx(x));
        }
        erfcx_o.push_back(keel::specfun::erfcx(100.0));
        erfcx_o.push_back(keel::specfun::erfcx(1e6));
        if (!write_f64_file(dir, "erf.bin", erf_o)) return 2;
        if (!write_f64_file(dir, "erfc.bin", erfc_o)) return 2;
        if (!write_f64_file(dir, "erfcx.bin", erfcx_o)) return 2;

        const keel::f64 ei_ys[] = {-0.999999, -0.7, -0.1, 0.0, 0.3, 0.999999};
        keel::Array<keel::f64> erfinv_o;
        for (keel::f64 y : ei_ys) erfinv_o.push_back(keel::specfun::erfinv(y));
        if (!write_f64_file(dir, "erfinv.bin", erfinv_o)) return 2;

        const keel::f64 eci_ys[] = {1e-12, 1e-6, 0.1, 0.5, 1.0, 1.5, 1.999999};
        keel::Array<keel::f64> erfcinv_o;
        for (keel::f64 y : eci_ys) erfcinv_o.push_back(keel::specfun::erfcinv(y));
        if (!write_f64_file(dir, "erfcinv.bin", erfcinv_o)) return 2;

        const keel::f64 exp_xs[] = {1e-12, 1e-3, 0.25, 0.5, 1.0, 1.001, 2.5, 10.0, 50.0, 100.0, 500.0};
        keel::Array<keel::f64> expint_o;
        for (keel::f64 x : exp_xs) expint_o.push_back(keel::specfun::expint(x));
        if (!write_f64_file(dir, "expint.bin", expint_o)) return 2;
    }

    // 4. Bessel J, Y, I, K, H & Airy
    {
        const keel::f64 b_cases[][2] = {
            {0.0, 0.1}, {0.0, 1.0}, {0.0, 10.0}, {0.0, 50.0},
            {1.0, 0.5}, {1.0, 2.0}, {2.0, 5.0}, {0.5, 1.5},
            {1.0 / 3.0, 2.5}, {2.5, 10.0}, {5.0, 15.0}, {25.0, 20.0},
            {-0.5, 1.5}, {-2.3, 4.0}, {-10.5, 3.0}, {-100.5, 2.5},
            {100.0, 50.0}, {100.0, 700.0}, {500.0, 720.0}, {0.0, 800.0},
        };
        keel::Array<keel::f64> bj_o, by_o, bi_o, bk_o;
        for (const auto& c : b_cases) {
            bj_o.push_back(keel::specfun::besselj(c[0], c[1]));
            by_o.push_back(keel::specfun::bessely(c[0], c[1]));
            bi_o.push_back(keel::specfun::besseli(c[0], c[1]));
            bk_o.push_back(keel::specfun::besselk(c[0], c[1]));
        }
        if (!write_f64_file(dir, "besselj.bin", bj_o)) return 2;
        if (!write_f64_file(dir, "bessely.bin", by_o)) return 2;
        if (!write_f64_file(dir, "besseli.bin", bi_o)) return 2;
        if (!write_f64_file(dir, "besselk.bin", bk_o)) return 2;

        const keel::f64 a_xs[] = {-20.0, -5.0, -2.0, -0.5, 0.0, 0.5, 1.0, 2.5, 10.0, 50.0, 100.0};
        keel::Array<keel::f64> airy_o;
        for (keel::f64 x : a_xs) {
            for (keel::i32 k = 0; k < 4; ++k) {
                airy_o.push_back(keel::specfun::airy(k, x));
            }
        }
        if (!write_f64_file(dir, "airy.bin", airy_o)) return 2;
    }

    // 5. Legendre, EllipKE, EllipJ, MarcumQ
    {
        keel::Array<keel::f64> leg_o;
        keel::f64 buf[8];
        const keel::f64 lxs[] = {-0.9, -0.3, 0.0, 0.5, 1.0};
        for (keel::f64 x : lxs) {
            keel::specfun::legendre(5, x, keel::MutArrayRef<keel::f64>(buf, 6));
            for (keel::i32 m = 0; m <= 5; ++m) leg_o.push_back(buf[m]);
        }
        if (!write_f64_file(dir, "legendre.bin", leg_o)) return 2;

        const keel::f64 ems[] = {0.0, 1e-6, 0.1, 0.5, 0.9, 0.9999, 1.0 - 1e-10};
        keel::Array<keel::f64> ke_o;
        for (keel::f64 m : ems) {
            const auto r = keel::specfun::ellipke(m);
            ke_o.push_back(r.k);
            ke_o.push_back(r.e);
        }
        if (!write_f64_file(dir, "ellipke.bin", ke_o)) return 2;

        const keel::f64 ej_cases[][2] = {{-2.5, 0.3}, {0.0, 0.5}, {0.7, 0.4}, {2.0, 0.9}, {5.0, 0.999}, {1.5, 0.0}, {1.5, 1.0}};
        keel::Array<keel::f64> ej_o;
        for (const auto& c : ej_cases) {
            const auto r = keel::specfun::ellipj(c[0], c[1]);
            ej_o.push_back(r.sn);
            ej_o.push_back(r.cn);
            ej_o.push_back(r.dn);
        }
        if (!write_f64_file(dir, "ellipj.bin", ej_o)) return 2;

        const keel::f64 mq_cases[][3] = {
            {0.0, 2.0, 1.0}, {0.0, 5.0, 4.0}, {2.0, 0.0, 1.0},
            {1.0, 1.0, 1.0}, {2.0, 3.0, 1.0}, {3.0, 2.0, 3.0},
            {10.0, 10.0, 1.0}, {20.0, 20.0, 1.0}, {50.0, 50.0, 5.0},
            {50.0, 70.0, 1.0}, {50.0, 35.0, 3.0}, {100.0, 90.0, 10.0},
            {500.0, 505.0, 3.0}, {1000.0, 1000.0, 1.0},
        };
        keel::Array<keel::f64> mq_o;
        for (const auto& c : mq_cases) {
            mq_o.push_back(keel::specfun::marcumq(c[0], c[1], keel::as<keel::i32>(c[2])));
        }
        if (!write_f64_file(dir, "marcumq.bin", mq_o)) return 2;
    }

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && keel::StringRef(argv[1]) == "--dump-ref") {
        return dump_reference(keel::StringRef(argv[2]));
    }

    test_gamma_beta();
    test_erf_and_expint();
    test_bessel_and_airy();
    test_legendre_ellip_marcum_comb();

    if (g_failed != 0) {
        keel::eprintln("test_specfun: {} failure(s)", g_failed);
        return 1;
    }
    keel::println("test_specfun: all tests passed");
    return 0;
}
