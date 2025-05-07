// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/task.h"

#include <stddef.h>

#if !(XNN_PLATFORM_WINDOWS || XNN_PLATFORM_WEB)
#include <pthread.h>
#include <stdatomic.h>

// The number of threads currently running in the background with some task or
// another.
static atomic_size_t xnn_num_running_tasks = 0;

// A condition variable for waiting on `xnn_num_running_tasks` to be zero.
static pthread_mutex_t xnn_num_running_tasks_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t xnn_num_running_tasks_cond = PTHREAD_COND_INITIALIZER;
#endif  // !(XNN_PLATFORM_WINDOWS || XNN_PLATFORM_WEB)

void xnn_task_launch(xnn_task_fn task_fn, void *data) {
#if XNN_PLATFORM_WINDOWS || XNN_PLATFORM_WEB
  task_fn(data);
#else
  // Increment the running task counter on the number of tasks.
  atomic_fetch_add(&xnn_num_running_tasks, 1);

  // Launch a thread with the given task function.
  pthread_t thread;
  pthread_create(&thread, /*attr=*/NULL, task_fn, data);
#endif  // XNN_PLATFORM_WINDOWS || XNN_PLATFORM_WEB
}

void xnn_task_done() {
#if !(XNN_PLATFORM_WINDOWS || XNN_PLATFORM_WEB)
  // Decrement the running tasks counter and signal to all threads waiting on
  // the counter condition.
  pthread_mutex_lock(&xnn_num_running_tasks_mutex);
  // If we were the last task, signal all waiting threads.
  if (atomic_fetch_sub(&xnn_num_running_tasks, 1) == 1) {
    pthread_cond_broadcast(&xnn_num_running_tasks_cond);
  }
  pthread_mutex_unlock(&xnn_num_running_tasks_mutex);
#endif  // !(XNN_PLATFORM_WINDOWS || XNN_PLATFORM_WEB)
}

void xnn_task_wait_on_running() {
#if !(XNN_PLATFORM_WINDOWS || XNN_PLATFORM_WEB)
  pthread_mutex_lock(&xnn_num_running_tasks_mutex);
  while (atomic_load(&xnn_num_running_tasks) > 0) {
    pthread_cond_wait(&xnn_num_running_tasks_cond,
                      &xnn_num_running_tasks_mutex);
  }
  pthread_mutex_unlock(&xnn_num_running_tasks_mutex);
#endif  // !(XNN_PLATFORM_WINDOWS || XNN_PLATFORM_WEB)
}
