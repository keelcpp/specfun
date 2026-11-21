#pragma once

#include "keel/array.h"
#include "keel/complex.h"
#include "keel/core.h"
#include "keel/memory.h"
#include "keel/ref.h"

namespace keel {

// ============================================================================
// 1. Gamma, Digamma & Incomplete Gamma Functions
// ============================================================================

/// Complete gamma function Gamma(x). Non-positive integer poles return +Inf.
[[nodiscard]] f64 gamma(f64 x) noexcept;

/// Natural logarithm of |Gamma(x)|. Non-positive integer poles return +Inf.
[[nodiscard]] f64 gammaln(f64 x) noexcept;

/// Digamma function psi(x) = Gamma'(x) / Gamma(x).
[[nodiscard]] f64 psi(f64 x) noexcept;

/// Regularized lower incomplete gamma P(a, x) = (1 / Gamma(a)) * integral_0^x t^(a-1) e^-t dt.
[[nodiscard]] f64 gammainc(f64 x, f64 a) noexcept;

/// Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x).
[[nodiscard]] f64 gammainc_upper(f64 x, f64 a) noexcept;

/// Inverse regularized lower incomplete gamma: solves gammainc(x, a) = p for x >= 0.
[[nodiscard]] f64 gammaincinv(f64 p, f64 a) noexcept;

// ============================================================================
// 2. Beta & Incomplete Beta Functions
// ============================================================================

/// Complete beta function B(z, w) = Gamma(z) * Gamma(w) / Gamma(z + w).
[[nodiscard]] f64 beta(f64 z, f64 w) noexcept;

/// Natural logarithm of |B(z, w)|.
[[nodiscard]] f64 betaln(f64 z, f64 w) noexcept;

/// Regularized incomplete beta I_x(a, b) = B(x; a, b) / B(a, b) for x in [0, 1], a > 0, b > 0.
[[nodiscard]] f64 betainc(f64 x, f64 a, f64 b) noexcept;

/// Inverse regularized incomplete beta: solves betainc(x, a, b) = p for x in [0, 1].
[[nodiscard]] f64 betaincinv(f64 p, f64 a, f64 b) noexcept;

// ============================================================================
// 3. Error Functions
// ============================================================================

/// Error function erf(x) = (2 / sqrt(pi)) * integral_0^x e^(-t^2) dt.
[[nodiscard]] f64 erf(f64 x) noexcept;

/// Complementary error function erfc(x) = 1 - erf(x).
[[nodiscard]] f64 erfc(f64 x) noexcept;

/// Scaled complementary error function erfcx(x) = exp(x^2) * erfc(x).
[[nodiscard]] f64 erfcx(f64 x) noexcept;

/// Inverse error function erfinv(y) for y in [-1, 1].
[[nodiscard]] f64 erfinv(f64 y) noexcept;

/// Inverse complementary error function erfcinv(y) for y in [0, 2].
[[nodiscard]] f64 erfcinv(f64 y) noexcept;

// ============================================================================
// 4. Exponential Integral
// ============================================================================

/// Exponential integral E_1(x) = integral_x^inf (e^-t / t) dt for x > 0.
[[nodiscard]] f64 expint(f64 x) noexcept;

// ============================================================================
// 5. Bessel, Modified Bessel & Hankel Functions
// ============================================================================

/// Bessel function of the first kind J_nu(x).
[[nodiscard]] f64 besselj(f64 nu, f64 x) noexcept;

/// Bessel function of the second kind (Neumann) Y_nu(x).
[[nodiscard]] f64 bessely(f64 nu, f64 x) noexcept;

/// Modified Bessel function of the first kind I_nu(x).
[[nodiscard]] f64 besseli(f64 nu, f64 x) noexcept;

/// Modified Bessel function of the second kind K_nu(x).
[[nodiscard]] f64 besselk(f64 nu, f64 x) noexcept;

/// Hankel function H_nu^(k)(x) = J_nu(x) +/- i * Y_nu(x) for k in {1, 2}.
/// Returns NaN + i*NaN for invalid k.
[[nodiscard]] c64 besselh(f64 nu, i32 k, f64 x) noexcept;

// ============================================================================
// 6. Airy Functions
// ============================================================================

/// Airy functions for kind k:
///   k = 0: Ai(x),  k = 1: Ai'(x),  k = 2: Bi(x),  k = 3: Bi'(x).
/// Returns NaN for k outside 0..3.
[[nodiscard]] f64 airy(i32 k, f64 x) noexcept;

// ============================================================================
// 7. Associated Legendre Functions
// ============================================================================

/// Computes associated Legendre functions P_n^m(x) for m = 0..n (with Condon-Shortley phase)
/// into `out` (which must have size >= n + 1). Does nothing if n < 0 or out.size() < n + 1.
void legendre(i32 n, f64 x, MutArrayRef<f64> out) noexcept;

// ============================================================================
// 8. Complete Elliptic Integrals & Jacobi Elliptic Functions
// ============================================================================

struct EllipKE {
    f64 k = 0.0;
    f64 e = 0.0;
};

/// Complete elliptic integrals of the first and second kinds K(m), E(m) for m in [0, 1].
[[nodiscard]] EllipKE ellipke(f64 m) noexcept;

struct EllipJ {
    f64 sn = 0.0;
    f64 cn = 0.0;
    f64 dn = 0.0;
};

/// Jacobi elliptic functions sn(u | m), cn(u | m), dn(u | m) for m in [0, 1].
[[nodiscard]] EllipJ ellipj(f64 u, f64 m) noexcept;

// ============================================================================
// 9. Marcum Q-Function
// ============================================================================

/// Generalized Marcum Q-function Q_m(a, b) for integer order m >= 1.
[[nodiscard]] f64 marcumq(f64 a, f64 b, i32 m = 1) noexcept;

// ============================================================================
// 10. Combinatorics & Number Theory
// ============================================================================

/// Exact double factorial n! (returns +Inf for n > 170).
[[nodiscard]] f64 factorial(u64 n) noexcept;

/// Binomial coefficient C(n, k) as IEEE-754 double (returns 0 if k > n).
[[nodiscard]] f64 nchoosek(u64 n, u64 k) noexcept;

/// Deterministic trial-division / Miller-Rabin primality check for 64-bit unsigned integer.
[[nodiscard]] bool is_prime_u64(u64 n) noexcept;

/// Sieve of Eratosthenes returning all primes <= n in ascending order.
[[nodiscard]] Array<u64> primes_up_to(u64 n, Allocator* alloc = nullptr);

/// Prime factorization of n in ascending order with multiplicity (returns {n} for n <= 1).
[[nodiscard]] Array<u64> factor_u64(u64 n, Allocator* alloc = nullptr);

} // namespace keel
