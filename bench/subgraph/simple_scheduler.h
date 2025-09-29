// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_BENCH_SUBGRAPH_SIMPLE_SCHEDULER_H_
#define XNNPACK_BENCH_SUBGRAPH_SIMPLE_SCHEDULER_H_

#include <cassert>
#include <condition_variable>  // NOLINT(build/c++11)
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "include/experimental.h"

namespace xnnpack {

// A simple C++11-based scheduler that implements the `xnn_scheduler_v2`
// interface and uses itself as the scheduler context.
class SimpleScheduler : public xnn_scheduler_v2 {
 public:
  using TaskFunction = void (*)(void*);
  using Task = std::pair<TaskFunction, void*>;

  explicit SimpleScheduler(uint32_t num_threads_) {
    num_threads = num_threads_impl;
    schedule = schedule_impl;
    for (int k = 0; k < num_threads_; k++) {
      threads_.emplace_back(SimpleScheduler::ThreadMain, this);
    }
  }
  ~SimpleScheduler() {
    {
      std::unique_lock<std::mutex> lock(task_mutex_);  // NOLINT(build/c++11)
      done_ = true;
      task_cond_var_.notify_all();
    }
    for (auto& thread : threads_) {
      thread.join();
    }
  }

  void* GetContext() { return this; }

  xnn_scheduler_v2 GetXnnSchedulerV2() {
    struct xnn_scheduler_v2 res;
    res.num_threads = this->num_threads;
    res.schedule = this->schedule;
    return res;
  }

  int NumThreads() { return threads_.size(); }

  void Schedule(void* context, TaskFunction task) {
    std::lock_guard<std::mutex> lock(task_mutex_);  // NOLINT(build/c++11)
    tasks_.push_back({task, context});
    task_cond_var_.notify_one();
  }

 private:
  static int num_threads_impl(void* scheduler_context) {
    return reinterpret_cast<SimpleScheduler*>(scheduler_context)->NumThreads();
  }

  static void schedule_impl(void* scheduler_context, void* context,
                            TaskFunction task) {
    reinterpret_cast<SimpleScheduler*>(scheduler_context)
        ->Schedule(context, task);
  }

  static void ThreadMain(SimpleScheduler* threadpool) {
    Task task;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(  // NOLINT(build/c++11)
            threadpool->task_mutex_);
        while (threadpool->tasks_.empty()) {
          if (threadpool->done_) {
            return;
          }
          threadpool->task_cond_var_.wait(lock);
        }
        task = std::move(threadpool->tasks_.front());
        threadpool->tasks_.pop_front();
      }
      task.first(task.second);
    }
  }

  std::vector<std::thread> threads_;
  std::deque<Task> tasks_;
  std::mutex task_mutex_;  // NOLINT(build/c++11)
  std::condition_variable task_cond_var_;
  bool done_ = false;
};

};  // namespace xnnpack

#endif  // XNNPACK_BENCH_SUBGRAPH_SIMPLE_SCHEDULER_H_
