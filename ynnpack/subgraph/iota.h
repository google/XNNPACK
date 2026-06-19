#ifndef XNNPACK_YNNPACK_SUBGRAPH_IOTA_H_
#define XNNPACK_YNNPACK_SUBGRAPH_IOTA_H_

#include <tuple>

namespace ynn {

struct iota_params {
  float scale = 1.0f;
  float offset = 0.0f;
};

inline bool operator==(const iota_params& a, const iota_params& b) {
  return std::tie(a.scale, a.offset) == std::tie(b.scale, b.offset);
}
inline bool operator<(const iota_params& a, const iota_params& b) {
  return std::tie(a.scale, a.offset) < std::tie(b.scale, b.offset);
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_IOTA_H_
