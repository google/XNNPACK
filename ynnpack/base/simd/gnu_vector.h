// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_GNU_VECTOR_H_
#define XNNPACK_YNNPACK_BASE_SIMD_GNU_VECTOR_H_

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bit_cast.h"
#include "ynnpack/base/simd/vec.h"

namespace ynn {
namespace simd {

// Trait to get GNU vector type or scalar fallback
template <typename StorageT, size_t M>
struct gnu_vector_type {
  typedef StorageT type __attribute__((vector_size(M * sizeof(StorageT))));
};

template <typename StorageT>
struct gnu_vector_type<StorageT, 1> {
  using type = StorageT;
};

template <typename T>
struct make_signed_int;

template <>
struct make_signed_int<float> {
  using type = int32_t;
};
template <>
struct make_signed_int<double> {
  using type = int64_t;
};
template <>
struct make_signed_int<int8_t> {
  using type = int8_t;
};
template <>
struct make_signed_int<uint8_t> {
  using type = int8_t;
};
template <>
struct make_signed_int<int16_t> {
  using type = int16_t;
};
template <>
struct make_signed_int<uint16_t> {
  using type = int16_t;
};
template <>
struct make_signed_int<int32_t> {
  using type = int32_t;
};
template <>
struct make_signed_int<uint32_t> {
  using type = int32_t;
};
template <>
struct make_signed_int<int64_t> {
  using type = int64_t;
};
template <>
struct make_signed_int<uint64_t> {
  using type = int64_t;
};

template <typename T>
using make_signed_int_t = typename make_signed_int<T>::type;

template <typename T>
struct is_native_vector_type
    : std::integral_constant<bool, std::is_arithmetic_v<T>> {};

// Helper for infinity
template <typename T>
YNN_ALWAYS_INLINE constexpr T get_infinity() {
  if constexpr (std::is_floating_point_v<T>) {
    return std::numeric_limits<T>::infinity();
  } else {
    return T{0};
  }
}

constexpr int32_t sign_mask(float) { return static_cast<int32_t>(0x80000000u); }
constexpr int64_t sign_mask(double) {
  return static_cast<int64_t>(0x8000000000000000ull);
}

constexpr int32_t exponent_mask(float) { return 0x7F800000; }
constexpr int64_t exponent_mask(double) { return 0x7FF0000000000000ll; }

// Helper for elementwise loop operations (N > 1)
template <typename F, typename T, size_t N, typename... Args>
YNN_ALWAYS_INLINE vec<T, N> elementwise_loop(vec<T, N> first, Args... rest) {
  vec<T, N> r;
  for (size_t i = 0; i < N; ++i) {
    r.v_[i] = F{}(first[i], rest[i]...);
  }
  return r;
}

// Specialization of elementwise_loop for N = 1
template <typename F, typename T, typename... Args>
YNN_ALWAYS_INLINE vec<T, 1> elementwise_loop(vec<T, 1> first, Args... rest) {
  return vec<T, 1>{F{}(first.v, rest.v...)};
}

// 3. Macros to define vec<T, N> specializations

#define DEFINE_GNU_VEC_FRIENDS(T)                                            \
  friend YNN_ALWAYS_INLINE auto lo(vec<T, N_> x) {                           \
    constexpr size_t M = (N_ > 1) ? N_ / 2 : 1;                              \
    using Subvec = vec<T, M>;                                                \
    if constexpr (N_ > 1) {                                                  \
      typename gnu_vector_type<T, M>::type rv{};                             \
      std::memcpy(&rv, &x.v_, M * sizeof(T));                                \
      return Subvec{rv};                                                     \
    } else {                                                                 \
      return x;                                                              \
    }                                                                        \
  }                                                                          \
  friend YNN_ALWAYS_INLINE auto hi(vec<T, N_> x) {                           \
    constexpr size_t M = (N_ > 1) ? N_ / 2 : 1;                              \
    using Subvec = vec<T, M>;                                                \
    if constexpr (N_ > 1) {                                                  \
      typename gnu_vector_type<T, M>::type rv{};                             \
      std::memcpy(&rv, reinterpret_cast<const char*>(&x.v_) + M * sizeof(T), \
                  M * sizeof(T));                                            \
      return Subvec{rv};                                                     \
    } else {                                                                 \
      return x;                                                              \
    }                                                                        \
  }                                                                          \
  friend YNN_ALWAYS_INLINE vec<T, N_> fma(vec<T, N_> a, vec<T, N_> b,        \
                                          vec<T, N_> acc) {                  \
    vec<T, N_> r;                                                            \
    using std::fma;                                                          \
    for (size_t i = 0; i < N_; ++i) {                                        \
      r.v_[i] = fma(a.v_[i], b.v_[i], acc.v_[i]);                            \
    }                                                                        \
    return r;                                                                \
  }

#define DEFINE_GNU_VEC(T)                                                   \
  template <>                                                               \
  struct vec<T, 1> {                                                        \
    using value_type = T;                                                   \
    static constexpr std::integral_constant<size_t, 1> N = {};              \
    T v;                                                                    \
    vec() = default;                                                        \
    YNN_ALWAYS_INLINE explicit vec(T val) : v(val) {}                       \
    YNN_ALWAYS_INLINE T operator[](size_t i) const { return v; }            \
  };                                                                        \
  template <size_t N_>                                                      \
  struct vec<T, N_> {                                                       \
    using value_type = T;                                                   \
    static constexpr std::integral_constant<size_t, N_> N = {};             \
    typedef T v __attribute__((vector_size(N_ * sizeof(T))));               \
    typedef T u_v __attribute__((vector_size(N_ * sizeof(T)), aligned(1))); \
    v v_;                                                                   \
    vec() = default;                                                        \
    YNN_ALWAYS_INLINE vec(v val) : v_(val) {}                               \
    YNN_ALWAYS_INLINE explicit vec(T val) {                                 \
      for (size_t i = 0; i < N_; ++i) v_[i] = val;                          \
    }                                                                       \
    YNN_ALWAYS_INLINE T operator[](size_t i) const { return v_[i]; }        \
    DEFINE_GNU_VEC_FRIENDS(T)                                               \
  };

DEFINE_GNU_VEC(int8_t)
DEFINE_GNU_VEC(uint8_t)
DEFINE_GNU_VEC(int16_t)
DEFINE_GNU_VEC(uint16_t)
DEFINE_GNU_VEC(int32_t)
DEFINE_GNU_VEC(uint32_t)
DEFINE_GNU_VEC(int64_t)
DEFINE_GNU_VEC(uint64_t)
DEFINE_GNU_VEC(float)
DEFINE_GNU_VEC(double)

#undef DEFINE_GNU_VEC
#undef DEFINE_GNU_VEC_FRIENDS

// 4. Operations

// Load/Store (without default arguments, as they are defined in vec.h)
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr,
                                 std::integral_constant<size_t, N>, vec<T, N>) {
  typename vec<T, N>::u_v val;
  std::memcpy(&val, ptr, N * sizeof(T));
  return vec<T, N>{val};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load_aligned(const T* ptr,
                                         std::integral_constant<size_t, N>,
                                         vec<T, N>) {
  typename vec<T, N>::u_v val;
  std::memcpy(&val, ptr, N * sizeof(T));
  return vec<T, N>{val};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE void store(T* ptr, vec<T, N> value,
                             std::integral_constant<size_t, N>) {
  std::memcpy(ptr, &value.v_, N * sizeof(T));
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE void store_aligned(T* ptr, vec<T, N> value,
                                     std::integral_constant<size_t, N>) {
  std::memcpy(ptr, &value.v_, N * sizeof(T));
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr, size_t n, vec<T, N> src) {
  assert(n <= N);
  std::memcpy(&src.v_, ptr, n * sizeof(T));
  return src;
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr, size_t n, zeros<N>) {
  return load(ptr, n, vec<T, N>{T{0}});
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr, size_t n, undef<N>) {
  return load(ptr, n, vec<T, N>{});
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE void store(T* ptr, vec<T, N> value, size_t n) {
  assert(n <= N);
  std::memcpy(ptr, &value.v_, n * sizeof(T));
}

template <typename T, size_t N, typename M>
YNN_ALWAYS_INLINE vec<T, N> select(M cond, vec<T, N> a, vec<T, N> b) {
  using IntT = make_signed_int_t<T>;
  using IntVec = vec<IntT, N>;
  IntVec ia = ynn::bit_cast<IntVec>(a);
  IntVec ib = ynn::bit_cast<IntVec>(b);
  IntVec icond = ynn::bit_cast<IntVec>(cond);
  IntVec ir = (ia & icond) | (ib & ~icond);
  return ynn::bit_cast<vec<T, N>>(ir);
}

// Generic templates for native types
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator+(vec<T, N> a, vec<T, N> b) {
  return vec<T, N>{a.v_ + b.v_};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator-(vec<T, N> a, vec<T, N> b) {
  return vec<T, N>{a.v_ - b.v_};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator*(vec<T, N> a, vec<T, N> b) {
  return vec<T, N>{a.v_ * b.v_};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator/(vec<T, N> a, vec<T, N> b) {
  return vec<T, N>{a.v_ / b.v_};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator-(vec<T, N> a) {
  return vec<T, N>{-a.v_};
}

// operator+= etc
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

// Bitwise Operators (not declared in vec.h)
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator&(vec<T, N> a, vec<T, N> b) {
  return vec<T, N>{a.v_ & b.v_};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator|(vec<T, N> a, vec<T, N> b) {
  return vec<T, N>{a.v_ | b.v_};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator^(vec<T, N> a, vec<T, N> b) {
  return vec<T, N>{a.v_ ^ b.v_};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator~(vec<T, N> a) {
  return vec<T, N>{~a.v_};
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

// Shift operators (operator<< is declared in vec.h)
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator<<(vec<T, N> a, int b) {
  return vec<T, N>{a.v_ << b};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator>>(vec<T, N> a, int b) {
  return vec<T, N>{a.v_ >> b};
}

// Comparisons (matching vec.h declarations exactly)
#define DEFINE_COMPARISON_OP(op)                                    \
  template <typename T, size_t N>                                   \
  YNN_ALWAYS_INLINE auto operator op(vec<T, N> a, vec<T, N> b) {    \
    using M = make_signed_int_t<T>;                                 \
    using RetVec = vec<M, N>;                                       \
    return RetVec{ynn::bit_cast<typename RetVec::v>(a.v_ op b.v_)}; \
  }

DEFINE_COMPARISON_OP(==)
DEFINE_COMPARISON_OP(!=)
DEFINE_COMPARISON_OP(<)  // NOLINT
DEFINE_COMPARISON_OP(<=)
DEFINE_COMPARISON_OP(>)  // NOLINT
DEFINE_COMPARISON_OP(>=)
#undef DEFINE_COMPARISON_OP

// Classification: isnan (defined early as it is used by min/max)

// Generic template for native types
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto isnan(vec<T, N> a) {
  using IntT = make_signed_int_t<T>;
  using IntVec = vec<IntT, N>;
  if constexpr (std::is_floating_point_v<T>) {
    IntVec a_int = ynn::bit_cast<IntVec>(a);
    constexpr IntT s_mask = sign_mask(T{});
    constexpr IntT e_mask = exponent_mask(T{});
    return (a_int & IntVec(~s_mask)) > IntVec(e_mask);
  } else {
    return IntVec(IntT{0});
  }
}

// 4. Generic templates for native types
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> abs(vec<T, N> a) {
  if constexpr (std::is_floating_point_v<T>) {
    using IntT = make_signed_int_t<T>;
    using IntVec = vec<IntT, N>;
    IntVec a_int = ynn::bit_cast<IntVec>(a);
    constexpr IntT s_mask = sign_mask(T{});
    IntVec r_int = a_int & IntVec(~s_mask);
    return ynn::bit_cast<vec<T, N>>(r_int);
  } else {
    return vec<T, N>{__builtin_elementwise_abs(a.v_)};
  }
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> min(vec<T, N> a, vec<T, N> b) {
  return vec<T, N>{a.v_ < b.v_ ? a.v_ : b.v_};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> max(vec<T, N> a, vec<T, N> b) {
  return vec<T, N>{a.v_ > b.v_ ? a.v_ : b.v_};
}

// Scalar overloads for min/max (swapped to match x86 asymmetric NaN
// propagation)
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> min(vec<T, N> a, T b) {
  return min(vec<T, N>{b}, a);
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> min(T a, vec<T, N> b) {
  return min(vec<T, N>{a}, b);
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> max(vec<T, N> a, T b) {
  return max(vec<T, N>{b}, a);
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> max(T a, vec<T, N> b) {
  return max(vec<T, N>{a}, b);
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> floor(vec<T, N> a) {
  if constexpr (std::is_floating_point_v<T>) {
    return vec<T, N>{__builtin_elementwise_floor(a.v_)};
  } else {
    return a;
  }
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> ceil(vec<T, N> a) {
  if constexpr (std::is_floating_point_v<T>) {
    return vec<T, N>{__builtin_elementwise_ceil(a.v_)};
  } else {
    return a;
  }
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> round(vec<T, N> a) {
  if constexpr (std::is_floating_point_v<T>) {
    return vec<T, N>{__builtin_elementwise_roundeven(a.v_)};
  } else {
    return a;
  }
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> sqrt(vec<T, N> a) {
  if constexpr (std::is_floating_point_v<T>) {
    return vec<T, N>{__builtin_elementwise_sqrt(a.v_)};
  } else {
    return a;
  }
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> copysign(vec<T, N> mag, vec<T, N> sgn) {
  if constexpr (std::is_floating_point_v<T>) {
    using IntT = make_signed_int_t<T>;
    using IntVec = vec<IntT, N>;
    IntVec mag_int = ynn::bit_cast<IntVec>(mag);
    IntVec sgn_int = ynn::bit_cast<IntVec>(sgn);
    constexpr IntT s_mask = sign_mask(T{});
    IntVec r_int = (mag_int & IntVec(~s_mask)) | (sgn_int & IntVec(s_mask));
    return ynn::bit_cast<vec<T, N>>(r_int);
  } else {
    return mag;
  }
}

// Saturated Arithmetic (matching vec.h declarations exactly)
struct AddSat {
  template <typename T>
  auto operator()(T a, T b) const {
    return ynn::add_sat(a, b);
  }
};
struct SubSat {
  template <typename T>
  auto operator()(T a, T b) const {
    return ynn::sub_sat(a, b);
  }
};

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> add_sat(vec<T, N> a, vec<T, N> b) {
  return elementwise_loop<AddSat>(a, b);
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> sub_sat(vec<T, N> a, vec<T, N> b) {
  return elementwise_loop<SubSat>(a, b);
}

// Classification: isinf, isfinite (isnan is defined earlier)

// Generic templates for native types
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto isinf(vec<T, N> a) {
  using IntT = make_signed_int_t<T>;
  using IntVec = vec<IntT, N>;
  if constexpr (std::is_floating_point_v<T>) {
    IntVec a_int = ynn::bit_cast<IntVec>(a);
    constexpr IntT s_mask = sign_mask(T{});
    constexpr IntT e_mask = exponent_mask(T{});
    return (a_int & IntVec(~s_mask)) == IntVec(e_mask);
  } else {
    return IntVec(IntT{0});
  }
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE auto isfinite(vec<T, N> a) {
  using IntT = make_signed_int_t<T>;
  using IntVec = vec<IntT, N>;
  if constexpr (std::is_floating_point_v<T>) {
    IntVec a_int = ynn::bit_cast<IntVec>(a);
    constexpr IntT mask = exponent_mask(T{});
    return (a_int & IntVec(mask)) < IntVec(mask);
  } else {
    return IntVec(IntT{-1});
  }
}

// Fast exp2_round for float and double using magic constant trick to avoid
// UB/crash on NaN
template <size_t N>
YNN_ALWAYS_INLINE vec<float, N> exp2_round(vec<float, N> a) {
  const auto magic = broadcast<N>(127.0f + static_cast<float>(1 << 23));
  auto res_bits = a + magic;
  using IntVec = vec<int32_t, N>;
  auto res_int = ynn::bit_cast<IntVec>(res_bits);
  auto shifted = res_int << 23;
  return ynn::bit_cast<vec<float, N>>(shifted);
}

template <size_t N>
YNN_ALWAYS_INLINE vec<double, N> exp2_round(vec<double, N> a) {
  const auto magic = broadcast<N>(1023.0 + static_cast<double>(1ll << 52));
  auto res_bits = a + magic;
  using IntVec = vec<int64_t, N>;
  auto res_int = ynn::bit_cast<IntVec>(res_bits);
  auto shifted = res_int << 52;
  return ynn::bit_cast<vec<double, N>>(shifted);
}

template <size_t Size>
YNN_ALWAYS_INLINE vec<float, Size> floor_log2(vec<float, Size> a) {
  using T = float;
  using int_t = int32_t;
  using IntVec = vec<int_t, Size>;
  using FloatVec = vec<T, Size>;

  FloatVec zero(0.0f);
  FloatVec nan(std::numeric_limits<T>::quiet_NaN());
  FloatVec infinity(std::numeric_limits<T>::infinity());

  auto is_nan = isnan(a);
  auto is_neg = a < zero;
  auto is_invalid = is_nan | is_neg;
  auto is_inf = (a == infinity);
  auto is_zero = (a == zero);

  FloatVec sign_mask(-0.0f);
  IntVec a_int = ynn::bit_cast<IntVec>(a);
  IntVec sign_mask_int = ynn::bit_cast<IntVec>(sign_mask);

  a_int = (is_zero & sign_mask_int) | a_int;

  IntVec exp = (a_int & IntVec(static_cast<int_t>(0xFF800000u))) >> 8;
  FloatVec bias_256(256.0f);
  FloatVec bias_383(383.0f);
  IntVec bias_256_int = ynn::bit_cast<IntVec>(bias_256);

  FloatVec res = ynn::bit_cast<FloatVec>(exp | bias_256_int) - bias_383;

  return select(is_invalid, nan, select(is_inf, a, res));
}

template <size_t Size>
YNN_ALWAYS_INLINE vec<double, Size> floor_log2(vec<double, Size> a) {
  using T = double;
  using int_t = int64_t;
  using IntVec = vec<int_t, Size>;
  using FloatVec = vec<T, Size>;

  FloatVec zero(0.0);
  FloatVec nan(std::numeric_limits<T>::quiet_NaN());
  FloatVec infinity(std::numeric_limits<T>::infinity());

  auto is_nan = isnan(a);
  auto is_neg = a < zero;
  auto is_invalid = is_nan | is_neg;
  auto is_inf = (a == infinity);
  auto is_zero = (a == zero);

  FloatVec sign_mask(-0.0);
  IntVec a_int = ynn::bit_cast<IntVec>(a);
  IntVec sign_mask_int = ynn::bit_cast<IntVec>(sign_mask);

  a_int = (is_zero & sign_mask_int) | a_int;

  IntVec exp =
      (a_int & IntVec(static_cast<int_t>(0xFFF0000000000000ull))) >> 11;
  FloatVec bias_2048(2048.0);
  FloatVec bias_3071(3071.0);
  IntVec bias_2048_int = ynn::bit_cast<IntVec>(bias_2048);

  FloatVec res = ynn::bit_cast<FloatVec>(exp | bias_2048_int) - bias_3071;

  return select(is_invalid, nan, select(is_inf, a, res));
}

// Cast (matching vec.h declaration exactly)
// Generic template for native types (using __builtin_convertvector)
template <typename To, typename From, size_t N>
YNN_ALWAYS_INLINE vec<To, N> cast(vec<From, N> from, To) {
  using ToVec = typename vec<To, N>::v;
  return vec<To, N>{__builtin_convertvector(from.v_, ToVec)};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N * 2> concat(vec<T, N> lo, vec<T, N> hi) {
  vec<T, N * 2> r;
  std::memcpy(&r.v_, &lo.v_, N * sizeof(T));
  std::memcpy(reinterpret_cast<char*>(&r.v_) + N * sizeof(T), &hi.v_,
              N * sizeof(T));
  return r;
}

// Extract (recursive, copied from extend_vec.inc)
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

// Horizontal operations (recursive, copied from extend_vec.inc, matching vec.h
// declarations exactly)
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

}  // namespace simd
}  // namespace ynn

// 5. Include target independent math functions (must be outside namespace
// ynn::simd)
#define YNN_HAVE_FMA
#include "ynnpack/base/simd/target_independent.inc"

#endif  // XNNPACK_YNNPACK_BASE_SIMD_GNU_VECTOR_H_
