#!/usr/bin/env python3
import subprocess
import json
import statistics
import sys

# --- Configurations from Table III of the arXiv paper ---
MODEL_CONFIGS = [
    (1, 64, 2112, 7168),   (2, 64, 24576, 1536), (3, 64, 32768, 512),
    (4, 64, 7168, 16384),  (5, 64, 4096, 7168),  (6, 64, 7168, 2048),
    (7, 128, 2112, 7168),  (8, 128, 24576, 1536),(9, 128, 32768, 512),
    (10, 128, 7168, 16384),(11, 128, 4096, 7168),(12, 128, 7168, 2048),
    (13, 4096, 2112, 7168),(14, 4096, 24576, 1536),(15, 4096, 32768, 512),
    (16, 4096, 7168, 16384),(17, 4096, 4096, 7168),(18, 4096, 7168, 2048),
    (19, 4096, 256, 4096), (20, 11008, 256, 4096),(21, 4096, 256, 11008),
    (22, 5120, 256, 5120), (23, 13824, 256, 5120),(24, 5120, 256, 13824)
]

NUM_REPETITIONS = 5

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

def run_benchmark(m, n, k):
    cmd = [
        BINARY_PATH,
        "--benchmark_filter=dot_fp32_sme2",
        f"--shape={m}x{n}x{k}",
        f"--benchmark_repetitions={NUM_REPETITIONS}",
        "--benchmark_format=json",
        "--benchmark_min_time=0.1s",
        "--benchmark_display_aggregates_only=false"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout = result.stdout
        start_idx = stdout.find("{")
        if start_idx == -1: return None
        return json.loads(stdout[start_idx:])
    except Exception as e:
        print(f" Error: {e}")
        return None

def analyze(data):
    if not data: return None
    t_orig, t_opt = [], []
    for b in data.get("benchmarks", []):
        if "aggregate_name" in b: continue
        if "error_occurred" in b and b["error_occurred"]:
            print(f"\n[!] VERIFICATION FAILED: {b['error_message']}")
            return "VERIFY_ERROR"
        if "dot_fp32_sme2_opt" in b["name"]: t_opt.append(b["real_time"])
        elif "dot_fp32_sme2" in b["name"]: t_orig.append(b["real_time"])
    
    if not t_orig or not t_opt: return None
    return statistics.mean(t_orig), statistics.mean(t_opt)

def main():
    build()
    
    print(f"\n{'ID':<3} | {'M x N x K':<20} | {'Orig (ms)':<10} | {'Opt (ms)':<10} | {'Speedup':<8}")
    print("-" * 65)
    
    for id, m, n, k in MODEL_CONFIGS:
        print(f"[{id:2}] Processing {m}x{n}x{k}...", end="\r", flush=True)
        data = run_benchmark(m, n, k)
        res = analyze(data)
        
        if res == "VERIFY_ERROR":
            print(f"{id:<3} | {f'{m}x{n}x{k}':<20} | {'INVALID':<10} | {'INVALID':<10} | {'N/A':<8}")
        elif res:
            m_orig, m_opt = res
            speedup = m_orig / m_opt
            print(f"{id:<3} | {f'{m}x{n}x{k}':<20} | {m_orig/1e6:<10.3f} | {m_opt/1e6:<10.3f} | {speedup:.3f}x")
        else:
            print(f"{id:<3} | {f'{m}x{n}x{k}':<20} | {'ERR':<10} | {'ERR':<10} | {'N/A':<8}")

if __name__ == "__main__":
    main()