// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef THIRD_PARTY_XNNPACK_YNNPACK_BASE_SIMD_GENERIC_INC_
#define THIRD_PARTY_XNNPACK_YNNPACK_BASE_SIMD_GENERIC_INC_

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bit_cast.h"
#include "ynnpack/base/simd/emulate_fma.h"
#include "ynnpack/base/simd/vec.h"

// This header provides generic implementations of vec<> operations that work by
// splitting the vec into two smaller vectors and executing the operation on
// those. It should be included after implementing specializations of vec<>.

namespace ynn {

namespace simd {

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr,
                                 std::integral_constant<size_t, N> n,
                                 vec<T, N>) {
  std::integral_constant<size_t, N / 2> n2 = {};
  return {load(ptr, n2), load(ptr + n2, n2)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr,
                                 std::integral_constant<size_t, N> n,
                                 zeros<N>) {
  return load(ptr, n, vec<T, N>{});
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr,
                                 std::integral_constant<size_t, N> n,
                                 undef<N>) {
  return load(ptr, n, vec<T, N>{});
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load_aligned(const T* ptr,
                                         std::integral_constant<size_t, N> n,
                                         vec<T, N>) {
  std::integral_constant<size_t, N / 2> n2 = {};
  return {load_aligned(ptr, n2), load_aligned(ptr + n2, n2)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr, size_t n, vec<T, N> src) {
  std::integral_constant<size_t, N / 2> n2 = {};
  if (n < n2) {
    return {load(ptr, n, lo(src)), hi(src)};
  } else {
    return {load(ptr, n2), load(ptr + n2, n - n2, hi(src))};
  }
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr, size_t n, zeros<N>) {
  std::integral_constant<size_t, N / 2> n2 = {};
  if (n < n2) {
    return {load(ptr, n, zeros<N / 2>{}), vec<T, N / 2>{T{0}}};
  } else {
    return {load(ptr, n2), load(ptr + n2, n - n2, zeros<N / 2>{})};
  }
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr, size_t n, undef<N>) {
  std::integral_constant<size_t, N / 2> n2 = {};
  if (n < n2) {
    return {load(ptr, n, undef<N / 2>{}), vec<T, N / 2>{}};
  } else {
    return {load(ptr, n2), load(ptr + n2, n - n2, undef<N / 2>{})};
  }
}

// Store `N` elements of `T` to `ptr`.
template <typename T, size_t N>
YNN_ALWAYS_INLINE void store(T* ptr, vec<T, N> value,
                             std::integral_constant<size_t, N> n) {
  std::integral_constant<size_t, N / 2> n2 = {};
  store(ptr, lo(value), n2);
  store(ptr + n2, hi(value), n2);
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE void store_aligned(T* ptr, vec<T, N> value,
                                     std::integral_constant<size_t, N> n) {
  std::integral_constant<size_t, N / 2> n2 = {};
  store_aligned(ptr, lo(value), n2);
  store_aligned(ptr + n2, hi(value), n2);
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE void store(T* ptr, vec<T, N> value, size_t n) {
  std::integral_constant<size_t, N / 2> n2 = {};
  if (n < n2) {
    store(ptr, lo(value), n);
  } else {
    store(ptr, lo(value), n2);
    store(ptr + n2, hi(value), n - n2);
  }
}

// Arithmetic operators.
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator+(vec<T, N> a, vec<T, N> b) {
  return {lo(a) + lo(b), hi(a) + hi(b)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator-(vec<T, N> a, vec<T, N> b) {
  return {lo(a) - lo(b), hi(a) - hi(b)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator*(vec<T, N> a, vec<T, N> b) {
  return {lo(a) * lo(b), hi(a) * hi(b)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator/(vec<T, N> a, vec<T, N> b) {
  return {lo(a) / lo(b), hi(a) / hi(b)};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator+=(vec<T, N>& a, vec<T, N> b) {
  a = a + b;
  return a;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator-=(vec<T, N>& a, vec<T, N> b) {
  a = a - b;
  return a;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator*=(vec<T, N>& a, vec<T, N> b) {
  a = a * b;
  return a;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator/=(vec<T, N>& a, vec<T, N> b) {
  a = a / b;
  return a;
}

// Boolean operators.
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator&(vec<T, N> a, vec<T, N> b) {
  return {lo(a) & lo(b), hi(a) & hi(b)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator|(vec<T, N> a, vec<T, N> b) {
  return {lo(a) | lo(b), hi(a) | hi(b)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator^(vec<T, N> a, vec<T, N> b) {
  return {lo(a) ^ lo(b), hi(a) ^ hi(b)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator~(vec<T, N> a) {
  return {~lo(a), ~hi(a)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator<<(vec<T, N> a, int b) {
  return {lo(a) << b, hi(a) << b};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator&=(vec<T, N>& a, vec<T, N> b) {
  a = a & b;
  return a;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator|=(vec<T, N>& a, vec<T, N> b) {
  a = a | b;
  return a;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator^=(vec<T, N>& a, vec<T, N> b) {
  a = a ^ b;
  return a;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> min(vec<T, N> a, vec<T, N> b) {
  return {min(lo(a), lo(b)), min(hi(a), hi(b))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> max(vec<T, N> a, vec<T, N> b) {
  return {max(lo(a), lo(b)), max(hi(a), hi(b))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> copysign(vec<T, N> mag, vec<T, N> sgn) {
  return {copysign(lo(mag), lo(sgn)), copysign(hi(mag), hi(sgn))};
};
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> abs(vec<T, N> a) {
  return {abs(lo(a)), abs(hi(a))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> add_sat(vec<T, N> a, vec<T, N> b) {
  return {add_sat(lo(a), lo(b)), add_sat(hi(a), hi(b))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> sub_sat(vec<T, N> a, vec<T, N> b) {
  return {sub_sat(lo(a), lo(b)), sub_sat(hi(a), hi(b))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> floor(vec<T, N> a) {
  return {floor(lo(a)), floor(hi(a))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> floor_log2(vec<T, N> a) {
  return {floor_log2(lo(a)), floor_log2(hi(a))};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> exp2_round(vec<T, N> a) {
  return {exp2_round(lo(a)), exp2_round(hi(a))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> ceil(vec<T, N> a) {
  return {ceil(lo(a)), ceil(hi(a))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> round(vec<T, N> a) {
  return {round(lo(a)), round(hi(a))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> sqrt(vec<T, N> a) {
  return {sqrt(lo(a)), sqrt(hi(a))};
}

#ifdef YNN_HAVE_FMA
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> fma(vec<T, N> a, vec<T, N> b, vec<T, N> acc) {
  return {fma(lo(a), lo(b), lo(acc)), fma(hi(a), hi(b), hi(acc))};
}
#elif defined(YNN_ARCH_EMULATE_FMA)
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> fma(vec<T, N> a, vec<T, N> b, vec<T, N> acc) {
  return emulate_fma(a, b, acc);
}
#define YNN_HAVE_FMA
#endif

#ifdef YNN_HAVE_FMA
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> multiply_add(vec<T, N> a, vec<T, N> b,
                                         vec<T, N> acc) {
  return fma(a, b, acc);
}
#else
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> multiply_add(vec<T, N> a, vec<T, N> b,
                                         vec<T, N> acc) {
  return a * b + acc;
}
#endif

template <typename T, size_t N>
YNN_ALWAYS_INLINE auto operator==(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) == lo(b), hi(a) == hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto operator!=(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) != lo(b), hi(a) != hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto operator<(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) < lo(b), hi(a) < hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto operator<=(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) <= lo(b), hi(a) <= hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto operator>(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) > lo(b), hi(a) > hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto operator>=(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) >= lo(b), hi(a) >= hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto isinf(vec<T, N> a) {
  return concat(isinf(lo(a)), isinf(hi(a)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto isnan(vec<T, N> a) {
  return concat(isnan(lo(a)), isnan(hi(a)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto isfinite(vec<T, N> a) {
  return concat(isfinite(lo(a)), isfinite(hi(a)));
}

template <typename T, size_t N, typename M>
YNN_ALWAYS_INLINE vec<T, N> select(M cond, vec<T, N> a, vec<T, N> b) {
  return {select(lo(cond), lo(a), lo(b)), select(hi(cond), hi(a), hi(b))};
}

template <int Index, typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> extract(vec<T, N> x,
                                    std::integral_constant<size_t, N>) {
  static_assert(Index == 0, "");
  return x;
}
template <int Index, typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N / 2> extract(vec<T, N> x,
                                        std::integral_constant<size_t, N / 2>) {
  static_assert(Index == 0 || Index == 1, "");
  return Index == 0 ? lo(x) : hi(x);
}
template <int Index, typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N / 4> extract(vec<T, N> x,
                                        std::integral_constant<size_t, N / 4>) {
  constexpr std::integral_constant<size_t, N / 2> n2 = {};
  constexpr std::integral_constant<size_t, N / 4> n4 = {};
  return extract<Index % 2>(extract<Index / 2>(x, n2), n4);
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N*2> concat(vec<T, N> a, vec<T, N> b) {
  return {a, b};
}

template <typename To, typename From, size_t N>
YNN_ALWAYS_INLINE vec<To, N> cast(vec<From, N> from, To) {
  return {cast(lo(from), To()), cast(hi(from), To())};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE T horizontal_sum(vec<T, N> x) {
  return horizontal_sum(lo(x) + hi(x));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE T horizontal_min(vec<T, N> x) {
  return horizontal_min(min(lo(x), hi(x)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE T horizontal_max(vec<T, N> x) {
  return horizontal_max(max(lo(x), hi(x)));
}

namespace internal {

// Evaluate a polynomial with coefficients `p` at `x`, where the terms are
// ordered highest to lowest.
template <typename T, size_t N, size_t Degree>
YNN_ALWAYS_INLINE vec<T, N> eval_polynomial(const std::array<T, Degree>& p,
                                            vec<T, N> x) {
  auto y = broadcast<N>(p[0]);
  for (size_t i = 1; i < p.size(); ++i) {
    y = multiply_add(x, y, broadcast<N>(p[i]));
  }
  return y;
}

// Round floating point values assuming the integer value can fit in the
// mantissa of the result.
template <size_t N>
vec<float, N> round_small(vec<float, N> a) {
  const auto magic = broadcast<N>(1.5f * (1 << 23));
  return (magic + a) - magic;
}
template <size_t N>
vec<double, N> round_small(vec<double, N> a) {
  const auto magic = broadcast<N>(1.5 * (1ull << 52));
  return (magic + a) - magic;
}

template <int m, typename T, size_t N, typename P, typename Q>
YNN_ALWAYS_INLINE vec<T, N> exp_impl(vec<T, N> x, const P& p, const Q& q,
                                     T min_x, T max_x, T log2e,
                                     T ln2_hi, T ln2_lo) {
  auto x_clamped = min(max(x, broadcast<N>(min_x)), broadcast<N>(max_x));
  auto b = round_small(x_clamped * broadcast<N>(log2e));
  // Compute b * ln2 with extra precision using both ln2 and the residual
  // T(ln(2)) - ln2_hi
  auto x_prime = multiply_add(b, broadcast<N>(-ln2_hi), x_clamped);
  x_prime = multiply_add(b, broadcast<N>(-ln2_lo), x_prime);

  // Compute 2^b.
  auto exp2b = exp2_round(b);
  exp2b = select(isnan(x), x, exp2b);

  auto p_a_prime = eval_polynomial(p, x_prime);
  auto q_a_prime = eval_polynomial(q, x_prime);

  // Divide the numerator by the denominator, obtaining exp(r).
  auto exp_a_prime = p_a_prime / q_a_prime;

  if (m == 1) {
    // Compute 2^b * (exp(a_prime) + 1) - 1. We avoid catastrophic cancellation
    // by distributing 2^b, and then reassociating the subtraction, so when
    // 2^b = 1 we keep the low bits of the result.
    //
    auto expm1_a =
        multiply_add(exp2b, exp_a_prime, exp2b - broadcast<N>(static_cast<T>(1)));
    return select(isinf(exp2b), exp2b, expm1_a);
  } else {
    // Compute 2^b * exp(a_prime).
    return exp2b * exp_a_prime;
  }
}

}  // namespace internal

template <size_t N>
static vec<float, N> exp(vec<float, N> x) {
  constexpr std::array<float, 4> p = {
      1.0244545527e-02f,
      1.1163228750e-01f,
      5.2334904671e-01f,
      1.0000000000e+00f
  };
  constexpr std::array<float, 4> q = {
      -6.3802185468e-03f,
      +8.8284827769e-02f,
      -4.7665113211e-01f,
      +1.0000000000e+00f
  };
  constexpr float log2e = 1.4426950408889634f;
  constexpr float ln2_hi = 0.693145751953125f;
  constexpr float ln2_lo = 1.4286067653e-06f;
  constexpr float min_x = -88.02969360351562f;
  constexpr float max_x = 88.72283935546875f;

  return internal::exp_impl<0>(x, p, q, min_x, max_x, log2e, ln2_hi, ln2_lo);
}

template <size_t N>
static vec<float, N> expm1(vec<float, N> x) {
  constexpr std::array<float, 4> p = {
      1.6622614115e-02f,
      3.3290067222e-04f,
      1.0000000000e+00f,
      0.0000000000e+00f
  };
  constexpr std::array<float, 4> q = {
      -8.2832705230e-03f,
      +9.9789284170e-02f,
      -4.9966722727e-01f,
      +1.0000000000e+00f
  };
  constexpr float log2e = 1.4426950408889634f;
  constexpr float ln2_hi = 6.9314575195e-01f;
  constexpr float ln2_lo = 1.4286067653e-06f;
  constexpr float min_x = -88.02969360351562f;
  constexpr float max_x = 88.72283935546875f;

  return internal::exp_impl<1>(x, p, q, min_x, max_x, log2e, ln2_hi, ln2_lo);
}

template <size_t N>
static vec<double, N> exp(vec<double, N> x) {
  constexpr std::array<double, 6> p = {
      3.239974270508748009e-05,
      9.791151627310745124e-04,
      1.377501781822836231e-02,
      1.105887196698455144e-01,
      4.989688485947128549e-01,
      9.999999999999998890e-01,
  };
  constexpr std::array<double, 6> q = {
      -3.362442950087436857e-05,
      +1.003649080773944512e-03,
      -1.399594422100019359e-02,
      +1.116198710751181572e-01,
      -5.010311514052862014e-01,
      +1.000000000000000000e+00,
  };
  constexpr double log2e = 1.4426950408889634;
  constexpr double ln2_hi = 6.93147180369123816490e-01;
  constexpr double ln2_lo = 1.90821492927058770002e-10;
  constexpr double min_x = -709.0895657128241;
  constexpr double max_x = 709.782712893384;

  return internal::exp_impl<0>(x, p, q, min_x, max_x, log2e, ln2_hi, ln2_lo);
}

template <size_t N>
static vec<double, N> expm1(vec<double, N> x) {
  constexpr std::array<double, 7> p = {
      -4.030667112776999258e-05,
      +7.475144792880859031e-05,
      -1.694975752712946787e-02,
      +2.813794733261987951e-02,
      -6.103171905131150599e-01,
      +9.999999999999998890e-01,
      +0.000000000000000000e+00,
  };
  constexpr std::array<double, 7> q = {
      +2.037091193345233258e-05,
      -6.424994246691768203e-04,
      +9.505570268034290587e-03,
      -8.187849706955278861e-02,
      +4.166298759224764670e-01,
      -1.110317190513113950e+00,
      +1.000000000000000000e+00,
  };
  constexpr double log2e = 1.4426950408889634;
  constexpr double ln2_hi = 6.93147180369123816490e-01;
  constexpr double ln2_lo = 1.90821492927058770002e-10;
  constexpr double min_x = -709.0895657128241;
  constexpr double max_x = 709.782712893384;

  return internal::exp_impl<1>(x, p, q, min_x, max_x, log2e, ln2_hi, ln2_lo);
}

namespace internal {

constexpr int32_t mantissa_mask(float) { return 0x007FFFFF; }
constexpr int64_t mantissa_mask(double) { return 0x000FFFFFFFFFFFFFll; }

// log(x) ~= x + x^2*P(x)/Q(x)
template <int p, typename Int, typename T, size_t N, typename P, typename Q>
YNN_ALWAYS_INLINE vec<T, N> log_impl(vec<T, N> x, const P& poly_p,
                                     const Q& poly_q) {
  vec<T, N> zero(0);
  vec<T, N> half(0.5);
  vec<T, N> one(1);
  vec<T, N> y = x;
  if (p == 1) {
    y = x + one;
  }
  // Decompose y into 2^exp*r. This is basically the exponent and mantissa of y.
  auto exp = floor_log2(y);

  // We want the mantissa with the exponent of 1.0.
  auto y_bits = bit_cast<vec<Int, N>>(y);
  auto y_norm_bits =
      (y_bits & broadcast<N>(mantissa_mask(T{}))) | bit_cast<vec<Int, N>>(one);
  auto y_norm = bit_cast<vec<T, N>>(y_norm_bits);

  // Normalize such that r is in [sqrt(2)/2 - 1, sqrt(2) - 1]
  constexpr T sqrt2 = 1.41421356237309504880;
  auto large = y_norm > broadcast<N>(sqrt2);
  exp = select(large, exp + one, exp);
  y_norm = select(large, y_norm * half, y_norm);
  auto r = y_norm - one;
  if (p == 1) {
    r = select(exp == zero, x, r);
  }

  // We approximate log(1 + x) ~= x + x^2*P(x)/Q(x). This is similar to how it
  // is done in Eigen, but with higher degree polynomials instead of an x^3
  // term.
  auto log_r_n = eval_polynomial(poly_p, r) * (r * r);
  auto log_r_d = eval_polynomial(poly_q, r);
  auto log_r = r + log_r_n / log_r_d;

  constexpr T ln2 = 6.93147180559945309e-01;
  return multiply_add(exp, broadcast<N>(ln2), log_r);
}

template <int p, size_t N>
static vec<float, N> log_impl(vec<float, N> x) {
  std::array<float, 3> P = {
      -5.2293352783e-03f,
      -3.3127889037e-01f,
      -4.9999964237e-01f,
  };
  std::array<float, 3> Q = {
      3.9658042789e-01f,
      1.3292170763e+00f,
      1.0000000000e+00f,
  };
  return log_impl<p, int32_t>(x, P, Q);
}

template <int p, size_t N>
static vec<double, N> log_impl(vec<double, N> x) {
  std::array<double, 6> P = {
      -2.956046447212538203e-03,
      -6.691314126082094360e-02,
      -4.289980732181625789e-01,
      -1.110965361416291097e+00,
      -1.242991301338429500e+00,
      -4.999999999999990008e-01,
  };
  std::array<double, 7> Q = {
      3.009181610003072455e-03,
      7.942469932969810353e-02,
      6.369060590899747742e-01,
      2.230802780023532605e+00,
      3.823696902394678077e+00,
      3.152649269343555716e+00,
      1.000000000000000000e+00,
  };
  return log_impl<p, int64_t>(x, P, Q);
}

}  // namespace internal

template <size_t N>
static vec<float, N> log(vec<float, N> x) {
  return internal::log_impl<0>(x);
}

template <size_t N>
static vec<float, N> log1p(vec<float, N> x) {
  return internal::log_impl<1>(x);
}

template <size_t N>
static vec<double, N> log(vec<double, N> x) {
  return internal::log_impl<0>(x);
}

template <size_t N>
static vec<double, N> log1p(vec<double, N> x) {
  return internal::log_impl<1>(x);
}

}  // namespace simd

}  // namespace ynn

#endif  // THIRD_PARTY_XNNPACK_YNNPACK_BASE_SIMD_VEC_INC_
