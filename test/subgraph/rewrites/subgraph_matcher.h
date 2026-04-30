#ifndef XNNPACK_TEST_SUBGRAPH_REWRITES_SUBGRAPH_MATCHER_H_
#define XNNPACK_TEST_SUBGRAPH_REWRITES_SUBGRAPH_MATCHER_H_

#include <ostream>
#include <type_traits>

#include "include/xnnpack.h"

// This needs to be in the global namespace for ADL to pick it up for GTest.
void PrintTo(xnn_subgraph_t subgraph, std::ostream* os);

namespace xnnpack {

class IsomorphicGraphMatcher {
 public:
  using is_gtest_matcher = void;

  explicit IsomorphicGraphMatcher(xnn_subgraph_t subgraph)
      : subgraph_(subgraph) {}

  bool MatchAndExplain(xnn_subgraph_t subgraph, std::ostream* listener) const;

  template <
      class GraphHolder,
      class = std::enable_if_t<std::is_same_v<
          decltype(std::declval<GraphHolder>()->subgraph()), xnn_subgraph_t>>>
  bool MatchAndExplain(const GraphHolder& subgraph,
                       std::ostream* listener) const {
    if (!subgraph) {
      if (listener) {
        *listener << "Invalid graph handle: nullptr.";
      }
      return false;
    }
    return MatchAndExplain(subgraph->subgraph(), listener);
  }

  void DescribeTo(std::ostream* os) const;
  void DescribeNegationTo(std::ostream* os) const;

  xnn_subgraph_t subgraph_;
};

IsomorphicGraphMatcher IsIsomorphicTo(xnn_subgraph_t subgraph);

template <
    class GraphHolder,
    class = std::enable_if_t<std::is_same_v<
        decltype(std::declval<GraphHolder>()->subgraph()), xnn_subgraph_t>>>
IsomorphicGraphMatcher IsIsomorphicTo(GraphHolder& subgraph) {
  return IsIsomorphicTo(subgraph ? subgraph->subgraph() : nullptr);
}

}  // namespace xnnpack

#endif  // XNNPACK_TEST_SUBGRAPH_REWRITES_SUBGRAPH_MATCHER_H_
