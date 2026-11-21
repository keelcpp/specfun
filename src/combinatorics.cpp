#include "keel/specfun.h"

#include "keel/arena.h"
#include "keel/math.h"

namespace keel {

f64 factorial(u64 n) noexcept {
    if (n > 170) return Limits<f64>::infinity();
    f64 r = 1.0;
    for (u64 i = 2; i <= n; ++i) r *= static_cast<f64>(i);
    return r;
}

f64 nchoosek(u64 n, u64 k) noexcept {
    if (k > n) return 0.0;
    const u64 kk = (k > n - k) ? (n - k) : k;
    if (kk == 0) return 1.0;
    f64 r = 1.0;
    for (u64 i = 0; i < kk; ++i) {
        r = r * static_cast<f64>(n - i) / static_cast<f64>(i + 1);
    }
    return keel::round(r);
}

bool is_prime_u64(u64 n) noexcept {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if ((n & 1ULL) == 0 || (n % 3ULL) == 0) return false;
    for (u64 i = 5; i <= n / i; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

Array<u64> primes_up_to(u64 n, Allocator* alloc) {
    Array<u64> out(alloc);
    if (n < 2) return out;
    ArenaTemp scratch = get_scratch(alloc);
    Array<u8> composite(static_cast<usize>(n + 1), u8{0}, scratch.allocator());
    for (u64 i = 2; i <= n / i; ++i) {
        if (!composite[static_cast<usize>(i)]) {
            for (u64 j = i * i; j <= n; j += i) {
                composite[static_cast<usize>(j)] = 1;
            }
        }
    }
    const usize est = static_cast<usize>(static_cast<f64>(n) / keel::log(static_cast<f64>(n) + 1.0)) + 2;
    out.reserve(est);
    for (u64 i = 2; i <= n; ++i) {
        if (!composite[static_cast<usize>(i)]) out.push_back(i);
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

} // namespace keel
