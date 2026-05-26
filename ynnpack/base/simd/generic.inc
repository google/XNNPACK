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
YNN_ALWAYS_INLINE vec<T, N> operator-(vec<T, N> a) {
  return {-lo(a), -hi(a)};
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
YNN_ALWAYS_INLINE vec<T, N> min(vec<T, N> a, T b) {
  return {min(lo(a), b), min(hi(a), b)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> min(T a, vec<T, N> b) {
  return {min(a, lo(b)), min(a, hi(b))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> max(vec<T, N> a, vec<T, N> b) {
  return {max(lo(a), lo(b)), max(hi(a), hi(b))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> max(vec<T, N> a, T b) {
  return {max(lo(a), b), max(hi(a), b)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> max(T a, vec<T, N> b) {
  return {max(a, lo(b)), max(a, hi(b))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> copysign(vec<T, N> mag, vec<T, N> sgn) {
  return {copysign(lo(mag), lo(sgn)), copysign(hi(mag), hi(sgn))};
}
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

// Similar to the above, uses x^2 as well. This has better ILP.
template <typename T, size_t N, size_t Degree>
YNN_ALWAYS_INLINE vec<T, N> eval_polynomial(const std::array<T, Degree>& p,
                                            vec<T, N> x, vec<T, N> x2) {
  auto y0 = broadcast<N>(p[0]);
  auto y1 = broadcast<N>(p[1]);
  for (size_t i = 2; i < p.size(); i += 2) {
    y0 = multiply_add(x2, y0, broadcast<N>(p[i + 0]));
    if (i + 1 < p.size()) {
      y1 = multiply_add(x2, y1, broadcast<N>(p[i + 1]));
    }
  }
  if (Degree % 2 == 1) {
    return multiply_add(x, y1, y0);
  } else {
    return multiply_add(x, y0, y1);
  }
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> divide(vec<T, N> a, vec<T, N> b) {
  return a / b;
}

// A placeholder indicating that a polynomial (usually a denominator polynomial)
// is one.
constexpr std::integral_constant<int, 1> one = {};

// Overloads of `eval_polynomial` and `divide` for the constant one polynomial.
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto eval_polynomial(decltype(one), vec<T, N>) {
  return one;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto eval_polynomial(decltype(one), vec<T, N>, vec<T, N>) {
  return one;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> divide(vec<T, N> a, decltype(one)) {
  return a;
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

// Implements exp(x) (m = 0) or expm1(x) (m = 1), given polynomials
// such that expm1(x) ~= x + x^2*P(x)/Q(x) for x in [-0.5, 0.5], i.e. the
// polynomials P(x)/Q(x) approximate the error after 2 terms of the Taylor
// series.
// See rational_approximation.py for the polynomial generation logic.
template <int m, typename T, size_t N, typename P, typename Q>
YNN_ALWAYS_INLINE vec<T, N> exp_impl(vec<T, N> x, const P& p, const Q& q,
                                     T min_x, T max_x, T log2e,
                                     T ln2_hi, T ln2_lo) {
  auto x_clamped = min(max(x, min_x), max_x);
  auto b = round_small(x_clamped * broadcast<N>(log2e));
  // Compute r = (b - x_clamped) * ln2 with extra precision using both ln2 and
  // the residual T(ln(2)) - ln2_hi
  auto r = multiply_add(b, broadcast<N>(-ln2_hi), x_clamped);
  r = multiply_add(b, broadcast<N>(-ln2_lo), r);

  // Compute 2^b.
  auto exp2_b = exp2_round(b);
  exp2_b = select(isnan(x), x, exp2_b);

  auto r2 = r * r;
  auto p_r = eval_polynomial(p, r, r2);
  auto q_r = eval_polynomial(q, r, r2);
  auto exp2_r = multiply_add(r2, divide(p_r, q_r), r);  // r^2 * p_r + r

  // Compute 2^b * (exp(r) + 1) - m. We avoid catastrophic cancellation
  // by distributing 2^b, and then reassociating the subtraction, so when
  // 2^b = m we keep the low bits of the result.
  auto exp_x =
      multiply_add(exp2_b, exp2_r, exp2_b - broadcast<N>(static_cast<T>(m)));
  return select(isinf(exp2_b), exp2_b, exp_x);
}

template <int m, size_t N>
static vec<float, N> exp_impl(vec<float, N> x) {
  std::array<float, 7> p = {
      -4.9866619520e-05f,
      2.2773783712e-04f,
      1.4132461511e-03f,
      8.3242934197e-03f,
      4.1665628552e-02f,
      1.6666725278e-01f,
      4.9999994040e-01f,
  };
  constexpr float log2e = 1.4426950408889634f;
  constexpr float ln2_hi = 6.9314575195e-01f;
  constexpr float ln2_lo = 1.4286067653e-06f;
  constexpr float min_x = -88.02969360351562f;
  constexpr float max_x = 88.72283935546875f;

  return exp_impl<m>(x, p, /*q=*/one, min_x, max_x, log2e, ln2_hi, ln2_lo);
}

template <int m, size_t N>
static vec<double, N> exp_impl(vec<double, N> x) {
  std::array<double, 11> p = {
      2.119890244143296032e-09,
      2.521079371356939368e-08,
      2.755585399809175769e-07,
      2.755675838881680360e-06,
      2.480158976814372031e-05,
      1.984127071808468106e-04,
      1.388888888680978647e-03,
      8.333333332743943569e-03,
      4.166666666668113195e-02,
      1.666666666666796748e-01,
      4.999999999999995559e-01,
  };
  constexpr double log2e = 1.4426950408889634;
  constexpr double ln2_hi = 6.93147180369123816490e-01;
  constexpr double ln2_lo = 1.90821492927058770002e-10;
  constexpr double min_x = -709.0895657128241;
  constexpr double max_x = 709.782712893384;

  return exp_impl<m>(x, p, /*q=*/one, min_x, max_x, log2e, ln2_hi, ln2_lo);
}

}  // namespace internal

template <size_t N>
static vec<float, N> exp(vec<float, N> x) {
  return internal::exp_impl<0>(x);
}

template <size_t N>
static vec<float, N> expm1(vec<float, N> x) {
  return internal::exp_impl<1>(x);
}

template <size_t N>
static vec<double, N> exp(vec<double, N> x) {
  return internal::exp_impl<0>(x);
}

template <size_t N>
static vec<double, N> expm1(vec<double, N> x) {
  return internal::exp_impl<1>(x);
}

namespace internal {

constexpr int32_t mantissa_mask(float) { return 0x007FFFFF; }
constexpr int64_t mantissa_mask(double) { return 0x000FFFFFFFFFFFFFll; }

// log(x) ~= x + x^2*P(x)/Q(x), i.e. approximating the error after 2 terms of
// the Taylor series.
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
  auto r2 = r * r;
  auto p_r = eval_polynomial(poly_p, r, r2);
  auto q_r = eval_polynomial(poly_q, r, r2);
  auto log_r = multiply_add(r2, divide(p_r, q_r), r);

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

namespace internal {

// Approximate erf(x) ~= x*P(x^2)/Q(x^2)
template <typename T, size_t N, typename P, typename Q>
static vec<T, N> erf_impl(vec<T, N> x, const P& p, const Q& q, T max_abs_x) {
  auto x_clamped = min(max(x, -max_abs_x), max_abs_x);
  auto x2 = x_clamped * x_clamped;
  auto p_x2 = internal::eval_polynomial(p, x2);
  auto q_x2 = internal::eval_polynomial(q, x2);
  return x_clamped * p_x2 / q_x2;
}

// Approximate erf(x) piecewise. When |x| < max_abs_x1, we approximate erf(x)
// directly which is an odd function: erf(x) ~= x*P(x^2)/Q(x^2)
//
// When |x| < max_abs_x2, we approximate erf(x) ~= 1 - exp(-x^2)*R(x)/S(x), i.e.
// R(x)/s(x) ~= exp(x^2)*erfc(x), the "scaled" error function.
template <typename T, size_t N, typename P, typename Q, typename R, typename S>
static vec<T, N> erf_impl(vec<T, N> x, const P& p, const Q& q, const R& r,
                          const S& s, T max_abs_x1, T max_abs_x2) {
  auto abs_x = min(abs(x), max_abs_x2);
  auto x2 = abs_x * abs_x;

  auto p_x2 = internal::eval_polynomial(p, x2);
  auto q_x2 = internal::eval_polynomial(q, x2);
  auto erf_x_small = abs_x * divide(p_x2, q_x2);

  auto r_x = internal::eval_polynomial(r, abs_x);
  auto s_x = internal::eval_polynomial(s, abs_x);
  auto erfc_x = exp(-x2) * divide(r_x, s_x);
  auto erf_x_large = broadcast<N>(static_cast<T>(1.0)) - erfc_x;
  auto erf_x =
      select(abs_x < broadcast<N>(max_abs_x1), erf_x_small, erf_x_large);

  // Restore the sign of the original function.
  return copysign(erf_x, x);
}

// Approximate erf(x) piecewise. When |x| < max_abs_x1, we approximate erf(x)
// directly which is an odd function: erf(x) ~= x*P(x^2)
//
// When |x| < max_abs_x2, we approximate erf(x) ~= 1 - exp(-x^2)*Q(x), i.e.
// Q(x) ~= exp(x^2)*erfc(x), the "scaled" error function.
//
// R(x) is the same as Q(x), but for |x| < max_abs_x3.
template <typename T, size_t N, typename P, typename Q, typename R>
static vec<T, N> erf_impl(vec<T, N> x, const P& p, const Q& q, const R& r,
                          T max_abs_x1, T max_abs_x2, T max_abs_x3) {
  auto abs_x = min(abs(x), max_abs_x3);
  auto x2 = abs_x * abs_x;

  auto p_x2 = internal::eval_polynomial(p, x2);
  auto erf_x_small = abs_x * p_x2;

  auto q_x = internal::eval_polynomial(q, abs_x);
  auto r_x = internal::eval_polynomial(r, abs_x);
  auto erfc_x = exp(-x2) * select(abs_x < broadcast<N>(max_abs_x2), q_x, r_x);
  auto erf_x_large = broadcast<N>(static_cast<T>(1.0)) - erfc_x;
  auto erf_x =
      select(abs_x < broadcast<N>(max_abs_x1), erf_x_small, erf_x_large);

  // Restore the sign of the original function.
  return copysign(erf_x, x);
}

}  // namespace internal

template <size_t N>
static vec<float, N> erf(vec<float, N> x) {
  // For floats, we use a piecewise (non-rational) polynomial approximation.
  std::array<float, 7> P = {
      -5.3975224728e-04f,
      3.3581661410e-04f,
      4.4070379809e-03f,
      -2.6605917141e-02f,
      1.1280037463e-01f,
      -3.7612470984e-01f,
      1.1283792257e+00f,
  };
  std::array<float, 7> Q = {
      4.4372659177e-03f,
      -4.2624399066e-02f,
      1.8425463140e-01f,
      -4.7879990935e-01f,
      8.4492594004e-01f,
      -1.0771834850e+00f,
      9.9257355928e-01f,
  };
  std::array<float, 7> R = {
      1.7328218382e-04f,
      -3.4217946231e-03f,
      2.9065947980e-02f,
      -1.3891060650e-01f,
      4.0989130735e-01f,
      -7.6906991005e-01f,
      8.9860576391e-01f,
  };
  constexpr float max_abs_x1 = 0.8f;
  constexpr float max_abs_x2 = 1.6f;
  constexpr float max_abs_x3 = 3.8f;
  return internal::erf_impl(x, P, Q, R, max_abs_x1, max_abs_x2, max_abs_x3);
}

template <size_t N>
static vec<double, N> erf(vec<double, N> x) {
  std::array<double, 6> P = {
      4.707472733925238136e-06,
      3.014938943048711055e-04,
      4.833419366839060254e-03,
      5.586134426461496910e-02,
      2.245425768504930031e-01,
      1.128379167095512559e+00,
  };
  std::array<double, 6> Q = {
      4.976158428165628768e-05,
      1.342648224661756101e-03,
      1.717640610916350632e-02,
      1.269488309955372762e-01,
      5.323290108487849226e-01,
      1.000000000000000000e+00,
  };
  std::array<double, 9> R = {
      1.042556497169926936e-11,
      -6.304186434426881222e-04,
      -3.920088302332867466e-03,
      -3.103366730622623363e-03,
      6.629233211348599386e-02,
      3.689538153966010481e-01,
      9.764178805043777221e-01,
      1.425066041265239392e+00,
      1.000000055826762768e+00,
  };
  std::array<double, 9> S = {
      -1.117386819582624584e-03,
      -6.948208453039030758e-03,
      -6.058667712054027939e-03,
      1.140182429407544523e-01,
      6.518376428169493364e-01,
      1.792305339163592581e+00,
      2.857670033198485893e+00,
      2.553445786610691215e+00,
      1.000000000000000000e+00,
  };
  constexpr double max_abs_x2 = 5.921591921591922159e+00;
  constexpr double max_abs_x1 = 0.9;
  return internal::erf_impl(x, P, Q, R, S, max_abs_x1, max_abs_x2);
}

// ~3x faster than erf(x), ~2x worse error (~5 ULPs vs. ~2 ULPs), and does not
// preserve NaN.
template <size_t N>
static vec<float, N> fast_erf(vec<float, N> x) {
  // Rational polynomial approximation of erf(x) for [-max_abs_x, max_abs_x]
  constexpr float max_abs_x = 3.8f;
  // erf(x) ~= x*P(x^2)/Q(x^2)
  std::array<float, 6> P = {
      1.6449823761e-06f,
      3.0145223718e-04f,
      5.0891782157e-03f,
      5.6909475476e-02f,
      2.2295241058e-01f,
      1.1283794641e+00f,
  };
  std::array<float, 6> Q = {
      3.4724384022e-05f,
      1.4562678989e-03f,
      1.7675759271e-02f,
      1.2741664052e-01f,
      5.3092014790e-01f,
      1.0000000000e+00f,
  };
  return internal::erf_impl(x, P, Q, max_abs_x);
}

template <size_t N>
static vec<float, N> tanh(vec<float, N> x) {
  static std::array<float, 5> P = {
      2.0879960871e-08f,
      2.5928242394e-05f,
      3.8165273145e-03f,
      1.3651061058e-01f,
      1.0000001192e+00f,
  };
  static std::array<float, 5> Q = {
      1.0818475857e-06f,
      3.8116905489e-04f,
      2.7097105980e-02f,
      4.6984484792e-01f,
      1.0000000000e+00f,
  };
  constexpr float max_abs_x = 7.9974925518035889f;

  auto x_clamped = max(min(x, max_abs_x), -max_abs_x);

  auto x2 = x_clamped * x_clamped;
  auto p_x = internal::eval_polynomial(P, x2) * x_clamped;
  auto q_x = internal::eval_polynomial(Q, x2);
  return p_x / q_x;
}

template <size_t N>
static vec<double, N> tanh(vec<double, N> x) {
  auto two = broadcast<N>(2.0);
  // Clamp large x to avoid Inf/Inf = NaN.
  auto expm1_2x = expm1(min(two*x, 300.0));
  return expm1_2x / (expm1_2x + two);
}

}  // namespace simd

}  // namespace ynn

#endif  // THIRD_PARTY_XNNPACK_YNNPACK_BASE_SIMD_GENERIC_INC_
