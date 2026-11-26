#include "keel/specfun.h"

#include "keel/arena.h"
#include "keel/math.h"

namespace keel::specfun {

namespace {

// Exact 64-bit modular multiplication (a * b) % m without overflow or 128-bit compiler extensions.
u64 mul_mod_u64(u64 a, u64 b, u64 m) noexcept {
    u64 res = 0;
    a %= m;
    while (b > 0) {
        if ((b & 1ULL) != 0ULL) {
            res = (res >= m - a) ? (res - (m - a)) : (res + a);
        }
        a = (a >= m - a) ? (a - (m - a)) : (a + a);
        b >>= 1;
    }
    return res;
}

u64 pow_mod_u64(u64 base, u64 exp, u64 mod) noexcept {
    u64 res = 1;
    base %= mod;
    while (exp > 0) {
        if ((exp & 1ULL) != 0ULL) {
            res = mul_mod_u64(res, base, mod);
        }
        base = mul_mod_u64(base, base, mod);
        exp >>= 1;
    }
    return res;
}

} // namespace

f64 factorial(u64 n) noexcept {
    if (n > 170) return Limits<f64>::infinity();
    f64 r = 1.0;
    for (u64 i = 2; i <= n; ++i) r *= as<f64>(i);
    return r;
}

f64 nchoosek(u64 n, u64 k) noexcept {
    if (k > n) return 0.0;
    const u64 kk = (k > n - k) ? (n - k) : k;
    if (kk == 0) return 1.0;
    f64 r = 1.0;
    for (u64 i = 0; i < kk; ++i) {
        r = r * as<f64>(n - i) / as<f64>(i + 1);
        if (is_inf(r)) return Limits<f64>::infinity();
    }
    return keel::round(r);
}

bool is_prime_u64(u64 n) noexcept {
    if (n < 2) return false;
    static constexpr u64 kSmallPrimes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (u64 p : kSmallPrimes) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    if (n < 41ULL * 41ULL) return true;

    // Deterministic 7-base Sinclair Miller-Rabin witnesses (proven exact for all u64 < 2^64).
    u64 d = n - 1;
    u32 s = 0;
    while ((d & 1ULL) == 0ULL) {
        d >>= 1;
        ++s;
    }
    static constexpr u64 kWitnesses[] = {2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL, 9780504ULL, 1795265022ULL};
    for (u64 a : kWitnesses) {
        if (a % n == 0) continue;
        u64 x = pow_mod_u64(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (u32 r = 1; r < s; ++r) {
            x = mul_mod_u64(x, x, n);
            if (x == n - 1) {
                composite = false;
                break;
            }
        }
        if (composite) return false;
    }
    return true;
}

Array<u64> primes_up_to(u64 n, Allocator* alloc) {
    Array<u64> out(alloc);
    if (n < 2) return out;
    ArenaTemp scratch = get_scratch(alloc);
    Array<u8> composite(as<usize>(n + 1), u8{0}, scratch.allocator());
    for (u64 i = 2; i <= n / i; ++i) {
        if (!composite[as<usize>(i)]) {
            for (u64 j = i * i; j <= n; j += i) {
                composite[as<usize>(j)] = 1;
            }
        }
    }
    const usize est = as<usize>(as<f64>(n) / keel::log(as<f64>(n) + 1.0)) + 2;
    out.reserve(est);
    for (u64 i = 2; i <= n; ++i) {
        if (!composite[as<usize>(i)]) out.push_back(i);
    }
    return out;
}

Array<u64> factor_u64(u64 n, Allocator* alloc) {
    Array<u64> out(alloc);
    if (n <= 1) {
        out.push_back(n);
        return out;
    }
    u64 m = n;
    while ((m & 1ULL) == 0) {
        out.push_back(2);
        m >>= 1;
    }
    for (u64 p = 3; p <= m / p; p += 2) {
        while (m % p == 0) {
            out.push_back(p);
            m /= p;
        }
    }
    if (m > 1) out.push_back(m);
    return out;
}

} // namespace keel::specfun
