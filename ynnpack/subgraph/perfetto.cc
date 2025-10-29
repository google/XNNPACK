#include "ynnpack/subgraph/perfetto.h"

#ifdef YNN_ENABLE_PERFETTO

#include <fcntl.h>

#include <cstdlib>
#include <fstream>
#include <memory>
#include <ostream>
#include <vector>

// Avoid ODR violations if other YNNPACK embedders also use Perfetto
#define PERFETTO_TRACK_EVENT_NAMESPACE ynn_perfetto_track_event

#include "perfetto/include/perfetto/tracing/backend_type.h"
#include "perfetto/include/perfetto/tracing/core/forward_decls.h"
#include "perfetto/include/perfetto/tracing/core/trace_config.h"  // IWYU pragma: keep  Indirectly includes generated config protos.
#include "perfetto/include/perfetto/tracing/string_helpers.h"
#include "perfetto/include/perfetto/tracing/tracing.h"
#include "perfetto/include/perfetto/tracing/track_event.h"
#include "perfetto/include/perfetto/tracing/track_event_category_registry.h"

namespace ynn {

// Category names must be C strings, and must be constexpr. absl::string_view
// might be preferred here, but is not guaranteed to be null-terminated.
inline constexpr char kYnnCategory[] = "ynn";

}  // namespace ynn

PERFETTO_DEFINE_CATEGORIES(perfetto::Category(ynn::kYnnCategory));

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace ynn {

namespace {

std::unique_ptr<perfetto::TracingSession> CreateTracingSession() {
  perfetto::TracingInitArgs args;
  args.backends |= perfetto::kInProcessBackend;
  perfetto::Tracing::Initialize(args);
  perfetto::TrackEvent::Register();

  perfetto::protos::gen::TrackEventConfig track_event_config;
  track_event_config.add_enabled_categories("*");

  perfetto::TraceConfig trace_config;
  trace_config.add_buffers()->set_size_kb(1000000);  // Record up to 1 GB.
  auto* source_config = trace_config.add_data_sources()->mutable_config();
  source_config->set_name("track_event");
  source_config->set_track_event_config_raw(
      track_event_config.SerializeAsString());

  std::unique_ptr<perfetto::TracingSession> tracing_session =
      perfetto::Tracing::NewTrace();
  tracing_session->Setup(trace_config);
  return tracing_session;
}

}  // namespace

perfetto_session::perfetto_session(std::ostream& os) : os_(os) {
  tracing_session_ = CreateTracingSession();
  tracing_session_->StartBlocking();
}

perfetto_session::~perfetto_session() {
  if (tracing_session_ != nullptr) {
    tracing_session_->StopBlocking();
    std::vector<char> trace_data = tracing_session_->ReadTraceBlocking();
    // Write the trace into a file.
    os_.write(&trace_data[0], trace_data.size());
    os_.flush();
  }
}

void perfetto_session::begin(const char* name) {
  TRACE_EVENT_BEGIN(kYnnCategory, perfetto::StaticString(name));
}
void perfetto_session::end() { TRACE_EVENT_END(kYnnCategory); }

perfetto_session* perfetto_session::global() {
  static const char* path = getenv("YNN_TRACE");
  if (!path) return nullptr;

  static auto file = std::make_unique<std::ofstream>(path);
  static auto trace = std::make_unique<perfetto_session>(*file);
  return trace.get();
}

}  // namespace ynn

#endif  // YNN_ENABLE_PERFETTO
