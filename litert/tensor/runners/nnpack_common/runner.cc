#include "litert/tensor/runners/nnpack_common/runner.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "litert/tensor/buffer.h"
#include "litert/tensor/datatypes.h"

namespace litert::tensor {

namespace nnpack_common::internal {

absl::Status EnsureBufferSize(std::shared_ptr<Buffer>& buffer,
                              size_t required_bytes,
                              absl::string_view tensor_name,
                              bool preserve_data) {
  if (buffer == nullptr) {
    buffer = OwningCpuBuffer::Allocate<Type::kI8>(required_bytes);
    return absl::OkStatus();
  }

  LockedBufferSpan<const std::byte> lock = buffer->Lock();
  const size_t actual_bytes = lock.size();
  if (actual_bytes >= required_bytes) {
    return absl::OkStatus();
  }

  // Check if buffer is owned and can be safely reallocated.
  if (buffer->IsA(OwningCpuBuffer::TypeId())) {
    auto new_buffer = OwningCpuBuffer::Allocate<Type::kI8>(required_bytes);
    if (preserve_data) {
      std::memcpy(new_buffer->data(), lock.data(), actual_bytes);
    }
    buffer = std::move(new_buffer);
    return absl::OkStatus();
  }

  // Non-owning views (SpanCpuBuffer, MutableSpanCpuBuffer, etc.) cannot be
  // resized.
  return absl::InvalidArgumentError(absl::StrFormat(
      "Buffer for tensor '%s' is a non-owning view of size %zu bytes, which is "
      "smaller than the required %zu bytes and cannot be resized",
      tensor_name, actual_bytes, required_bytes));
}

}  // namespace nnpack_common::internal
}  // namespace litert::tensor
