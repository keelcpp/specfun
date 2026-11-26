# keel::specfun

Portable special mathematical functions and combinatorics built on [`keel::keel`](https://github.com/keelcpp/keel), with zero STL dependencies and identical behavior across Linux, macOS, Windows, and WebAssembly.

## Features

- **Gamma, Digamma & Incomplete Gamma**: `gamma`, `gammaln`, `psi`, `gammainc`, `gammainc_upper`, `gammaincinv`
- **Beta & Incomplete Beta**: `beta`, `betaln`, `betainc`, `betaincinv`
- **Error Functions**: `erf`, `erfc`, `erfcx`, `erfinv`, `erfcinv` (Cody rational Chebyshev approximations + Halley/Newton inversion)
- **Exponential Integral**: `expint` ($E_1(x)$)
- **Bessel, Modified Bessel & Hankel**: `besselj`, `bessely`, `besseli`, `besselk`, `besselh` (Temme/Steed continued fractions for all real orders $\nu$ and all $x$)
- **Airy Functions**: `airy` ($\operatorname{Ai}, \operatorname{Ai}', \operatorname{Bi}, \operatorname{Bi}'$)
- **Associated Legendre Functions**: `legendre` ($P_n^m(x)$ for $m = 0 \dots n$)
- **Elliptic Integrals & Jacobi Elliptic Functions**: `ellipke` (AGM), `ellipj` (descending Landen)
- **Marcum Q-Function**: `marcumq`
- **Combinatorics & Number Theory**: `factorial`, `nchoosek`, `is_prime_u64` (deterministic 7-base Sinclair Miller-Rabin), `primes_up_to`, `factor_u64`
- Zero STL in public headers, compiles cleanly under `-fno-exceptions -fno-rtti`.

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
python tests/verify_reference.py build/Release/test_specfun.exe
```

`tests/verify_reference.py` validates every special function across small, large, negative, and deep-tail arguments against SciPy (`scipy.special`, `scipy.stats.ncx2`) and MATLAB R2025b (`marcumq`, `airy`, `expint`, `psi`, `gammaln`).
