// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef THIRD_PARTY_XNNPACK_SRC_XNNPACK_ATOMICS_H_
#define THIRD_PARTY_XNNPACK_SRC_XNNPACK_ATOMICS_H_

#include <assert.h>
#include <stddef.h>

typedef void *(*xnn_task_fn)(void *);

// Creates a task with the function `task_fn(data)` in a separate thread.
void xnn_task_launch(xnn_task_fn task_fn, void *data);

// Signals that a task has completed (needs to be called at the end of a task's
// critical section).
void xnn_task_done();

// Waits until all the tasks previously created with `xnn_task_launch` have
// completed.
void xnn_task_wait_on_running();

#endif  // THIRD_PARTY_XNNPACK_SRC_XNNPACK_ATOMICS_H_
