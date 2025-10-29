#ifndef XNNPACK_YNNPACK_SUBGRAPH_PERFETTO_H_
#define XNNPACK_YNNPACK_SUBGRAPH_PERFETTO_H_

#ifdef YNN_ENABLE_PERFETTO

#include <iostream>
#include <memory>

#include "perfetto/include/perfetto/tracing/tracing.h"

namespace ynn {

// A wrapper for generating perfetto trace files:
class perfetto_session {
  std::unique_ptr<perfetto::TracingSession> tracing_session_;
  std::ostream& os_;

 public:
  explicit perfetto_session(std::ostream& os);
  ~perfetto_session();

  void begin(const char* name);
  void end();

  // Return the global instance of tracing, or nullptr if none. Trace files will
  // be written to the path in the `YNN_TRACE` environment variable.
  static perfetto_session* global();
};

// Call `perfetto_session::begin` upon construction, and `perfetto_session::end`
// upon destruction.
class scoped_perfetto_trace {
  perfetto_session* trace;

 public:
  scoped_perfetto_trace(perfetto_session* trace, const char* name)
      : trace(trace) {
    if (trace) {
      trace->begin(name);
    }
  }

  scoped_perfetto_trace(const char* name)  // NOLINT
      : scoped_perfetto_trace(perfetto_session::global(), name) {}

  ~scoped_perfetto_trace() {
    if (trace) {
      trace->end();
    }
  }
};

}  // namespace ynn

#endif  // YNN_ENABLE_PERFETTO

#endif  // XNNPACK_YNNPACK_SUBGRAPH_PERFETTO_H_
