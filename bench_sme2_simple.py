#!/usr/bin/env python3
import subprocess
import json
import statistics
import os
import sys

# --- Configuration ---
SIZES = [240, 312, 405, 527, 685, 891, 1158, 1505, 1957, 2545, 3308, 4096]
NUM_REPETITIONS = 5

BAZEL_FLAGS = [
    "--action_env=CLANG_COMPILER_PATH=/opt/homebrew/Cellar/llvm@20/20.1.8/bin/clang-20",
    "--repo_env=CC=/opt/homebrew/Cellar/llvm@20/20.1.8/bin/clang-20",
    "--repo_env=BAZEL_COMPILER=/opt/homebrew/Cellar/llvm@20/20.1.8/bin/clang-20",
    "--linkopt=--ld-path=/opt/homebrew/bin/ld64.lld"
]

TARGET = "ynnpack/kernels/dot:bench"
# Path to the binary after bazel build -c opt
BINARY_PATH = "bazel-bin/ynnpack/kernels/dot/bench"

def build():
    print("[-] Building benchmark binary...", end="", flush=True)
    cmd = ["bazel", "build", "-c", "opt", TARGET] + BAZEL_FLAGS
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f" Failed!\n{result.stderr}")
        sys.exit(1)
    print(" Done.")

def run_benchmark(size):
    print(f"[-] Benchmarking size {size}...", end="", flush=True)
    # Execute binary directly to avoid bazel log noise on stdout
    cmd = [
        BINARY_PATH,
        "--benchmark_filter=dot_fp32_sme2",
        f"--shape={size}x{size}x{size}",
        f"--benchmark_repetitions={NUM_REPETITIONS}",
        "--benchmark_format=json",
        "--benchmark_min_time=0.1s",
        "--benchmark_display_aggregates_only=false"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(" Done.")
        return json.loads(result.stdout)
    except Exception as e:
        print(f" Failed! {e}")
        return None

def analyze(data):
    if not data: return None
    t_orig, t_opt = [], []
    for b in data.get("benchmarks", []):
        if "aggregate_name" in b: continue
        if "dot_fp32_sme2_opt" in b["name"]: t_opt.append(b["real_time"])
        elif "dot_fp32_sme2" in b["name"]: t_orig.append(b["real_time"])
    
    if not t_orig or not t_opt: return None
    
    mean_orig = statistics.mean(t_orig)
    mean_opt = statistics.mean(t_opt)
    speedup = mean_orig / mean_opt
    return mean_orig, mean_opt, speedup

def main():
    build()
    
    print(f"\n{'Size':<6} | {'Orig(ms)':<10} | {'Opt(ms)':<10} | {'Speedup':<8} | {'Chart'}")
    print("-" * 65)
    
    results = []
    for s in SIZES:
        data = run_benchmark(s)
        stats = analyze(data)
        if stats:
            m_orig, m_opt, speedup = stats
            bar = "#" * int(max(0, (speedup - 0.95) * 40)) if speedup > 0.95 else ""
            print(f"{s:<6} | {m_orig/1e6:<10.3f} | {m_opt/1e6:<10.3f} | {speedup:<8.3f} | {bar}")
            results.append((s, speedup))
        else:
            print(f"{s:<6} | {'ERR':<10} | {'ERR':<10} | {'N/A':<8} |")

if __name__ == "__main__":
    main()