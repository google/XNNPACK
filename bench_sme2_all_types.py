#!/usr/bin/env python3
import subprocess
import json
import statistics
import sys
import math

# --- Configuration ---
SIZES = [240, 1158, 1957, 4096]
TYPES = ["fp32", "bf16", "fp16", "int8"]
NUM_REPETITIONS = 10 

BAZEL_FLAGS = [
    "--action_env=CLANG_COMPILER_PATH=/opt/homebrew/Cellar/llvm@20/20.1.8/bin/clang-20",
    "--repo_env=CC=/opt/homebrew/Cellar/llvm@20/20.1.8/bin/clang-20",
    "--repo_env=BAZEL_COMPILER=/opt/homebrew/Cellar/llvm@20/20.1.8/bin/clang-20",
    "--linkopt=--ld-path=/opt/homebrew/bin/ld64.lld"
]

TARGET = "ynnpack/kernels/dot:bench"
BINARY_PATH = "bazel-bin/ynnpack/kernels/dot/bench"

def build():
    print("[-] Building benchmark binary...", end="", flush=True)
    cmd = ["bazel", "build", "-c", "opt", TARGET] + BAZEL_FLAGS
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(" Failed!")
        print(result.stderr)
        sys.exit(1)
    print(" Done.")

def run_benchmark(size, type_filter):
    cmd = [
        BINARY_PATH,
        f"--benchmark_filter=dot_.*{type_filter}.*sme2",
        f"--shape={size}x{size}x{size}",
        f"--benchmark_repetitions={NUM_REPETITIONS}",
        "--benchmark_format=json",
        "--benchmark_min_time=0.05s",
        "--benchmark_display_aggregates_only=false"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        stdout = result.stdout
        start_idx = stdout.find("{")
        if start_idx == -1: return None
        return json.loads(stdout[start_idx:])
    except Exception as e:
        return None

def analyze(data, type_str):
    if not data: return None
    t_orig, t_opt = [], []
    for b in data.get("benchmarks", []):
        if "aggregate_name" in b: continue
        if "error_occurred" in b and b["error_occurred"]:
            return "ERROR"
        
        name = b["name"]
        if f"dot_{type_str}" in name and "_opt" in name:
            t_opt.append(b["real_time"])
        elif f"dot_{type_str}" in name:
            t_orig.append(b["real_time"])
    
    if not t_orig or not t_opt: return None
    
    m_orig = statistics.mean(t_orig)
    s_orig = statistics.stdev(t_orig) if len(t_orig) > 1 else 0
    m_opt = statistics.mean(t_opt)
    s_opt = statistics.stdev(t_opt) if len(t_opt) > 1 else 0
    
    return (m_orig, s_orig, m_opt, s_opt)

def main():
    build()
    
    header = f"{'Type':<6} | {'Size':<6} | {'Orig (ms)':<15} | {'Opt (ms)':<15} | {'Speedup'}"
    print(f"\n{header}")
    print("-" * 75)
    
    for t in TYPES:
        for s in SIZES:
            type_filter = t if t in ["fp32", "int8"] else f"{t}_{t}_fp32"
            print(f"{t:<6} | {s:<6} | ", end="", flush=True)
            data = run_benchmark(s, type_filter)
            res = analyze(data, type_filter)
            
            if res == "ERROR":
                print(f"{'VERIFY_ERR':<15} | {'VERIFY_ERR':<15} | {'N/A'}")
            elif res:
                m_orig, s_orig, m_opt, s_opt = res
                speedup = m_orig / m_opt
                
                rel_err = math.sqrt((s_orig/m_orig)**2 + (s_opt/m_opt)**2) if m_orig > 0 and m_opt > 0 else 0
                speedup_err = speedup * rel_err
                
                orig_str = f"{m_orig/1e6:>.3f} ±{s_orig/1e6:.2f}"
                opt_str = f"{m_opt/1e6:>.3f} ±{s_opt/1e6:.2f}"
                
                print(f"{orig_str:<15} | {opt_str:<15} | {speedup:.3f}x ±{speedup_err:.3f}")
            else:
                print(f"{'ERR':<15} | {'ERR':<15} | {'N/A'}")

if __name__ == "__main__":
    main()
