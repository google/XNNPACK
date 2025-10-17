// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstdint>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/slinky_thread_pool.h"
#include "slinky/base/thread_pool.h"

extern "C" {

ynn_status ynn_create_threadpool(ynn_scheduler_t scheduler,
                                 void* scheduler_context, uint32_t flags,
                                 ynn_threadpool_t* threadpool_out) {
  *threadpool_out = reinterpret_cast<ynn_threadpool_t>(
      new ynn::slinky_thread_pool(scheduler, scheduler_context));
  return ynn_status_success;
}

void ynn_delete_threadpool(ynn_threadpool_t threadpool) {
  delete reinterpret_cast<slinky::thread_pool*>(threadpool);
}

}  // extern "C"
