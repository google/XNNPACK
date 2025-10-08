#ifndef XNNPACK_YNNPACK_BASE_BIT_CAST_H_
#define XNNPACK_YNNPACK_BASE_BIT_CAST_H_

#include <cstring>

namespace ynn {

// Unfortunately, std::bit_cast is C++20, which we can't use. More unfortunately
// it seems impossible to hack together a constexpr bit_cast without compiler
// support.
template <typename To, typename From>
To bit_cast(From x) {
  static_assert(sizeof(To) == sizeof(From), "");
  To result;
  memcpy(&result, &x, sizeof(result));
  return result;
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_BIT_CAST_H_
