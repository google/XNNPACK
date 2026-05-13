#ifndef XNNPACK_YNNPACK_BASE_TEST_TOLERANCE_H_
#define XNNPACK_YNNPACK_BASE_TEST_TOLERANCE_H_

#include <cmath>
#include <type_traits>

#include "ynnpack/base/type.h"

namespace ynn {

// Expresses a tolerance for error as a function of the absolute value of the
// expected result and the epsilon (distance between 1 and the next smallest
// value) of the type.
struct tolerance_spec {
  float relative = 0.0f;
  float absolute = 0.0f;

  template <typename T>
  auto absolute_error(T x) const {
    using Float = std::conditional_t<std::is_same_v<T, double>, double, float>;
    Float fx = static_cast<Float>(x);
    if (!std::isfinite(fx)) {
      // If the reference value is infinity, the computation below will produce
      // NaN. We probably want to compute the tolerance as if the value is the
      // largest value, not infinity.
      fx = std::nextafter(fx, static_cast<Float>(0));
    }

    // Note that `y_ref * rel_tol`, i.e. the expected absolute difference,
    // may round differently than `y_ref * (1 + rel_tol) - y_ref`, i.e. the
    // effective absolute difference computed in `float`s. We therefore use
    // the latter form since it is the true difference between two `float`s
    // within the given relative tolerance.
    Float eps = type_info<T>::epsilon();
    Float rel = std::abs(fx * (1.0f + relative * eps)) - std::abs(fx);
    return std::max<Float>(rel, absolute * eps);
  }
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_TEST_TOLERANCE_H_
