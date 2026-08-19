"""
scripts/benchmark_bot.py
========================
Automated Latency & Round-Trip Performance Benchmark Suite for RiskLens.
Measures:
1. Full Verification Pipeline Latency (/verify: LangGraph agent + web search + Google fact-check + neural ensemble).
2. Neural Ensemble Latency (/predict: feature extraction + 5-model ensemble inference).
3. Warm vs Cold Start Latency Distributions (Min, Max, Mean, P50, P95 across 10+ samples).
4. Persists real benchmark results to CSV and JSON history files.
"""

import os
import sys
import time
import json
import csv
import argparse
import requests
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_SPACE_URL = "https://preeti-antil-risklens.hf.space"
BENCHMARK_DIR = BASE_DIR / "benchmarks"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
CSV_LOG = BENCHMARK_DIR / "bot_latency.csv"
JSON_LOG = BENCHMARK_DIR / "bot_latency.json"

TEST_CLAIMS = [
    "NASA confirms discovery of liquid water oceans beneath the surface of Jupiter moon Europa.",
    "Breaking: Government announces immediate 50% tax cut on all electric vehicle purchases starting tomorrow.",
    "Scientists develop new malaria vaccine showing 90% efficacy in phase 3 clinical trials.",
    "Leaked documents prove international space station was temporarily abandoned due to asteroid swarm.",
    "World Health Organization declares global health emergency following new influenza strain outbreak.",
    "Central Bank announces mandatory phase-out of physical banknotes within 60 days.",
    "James Webb Space Telescope detects artificial transmission signal from Proxima Centauri b.",
    "New study shows drinking three cups of coffee daily reverses cellular aging markers by 5 years.",
    "Ministry of Education announces complete replacement of standardized tests with AI assessments.",
    "Researchers develop room-temperature superconductor operating at ambient atmospheric pressure."
]


def check_hf_space_runtime_state(space_id: str = "Preeti-Antil/RiskLens") -> dict:
    """Queries Hugging Face Hub API for Space runtime state (RUNNING, SLEEPING, BUILDING)."""
    api_url = f"https://huggingface.co/api/spaces/{space_id}"
    try:
        resp = requests.get(api_url, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            runtime = data.get("runtime", {})
            return {
                "stage": runtime.get("stage", "UNKNOWN"),
                "hardware": runtime.get("hardware", {}).get("current", "cpu-basic"),
                "raw": runtime
            }
        return {"stage": "UNKNOWN", "status_code": resp.status_code}
    except Exception as e:
        return {"stage": "UNKNOWN", "error": str(e)}


def measure_single_request(base_url: str, text: str, endpoint: str = "verify") -> dict:
    """
    Sends request to /verify (full LangGraph agent) or /predict (neural ensemble)
    and measures actual round-trip latency.
    """
    target_endpoint = "/verify" if endpoint == "verify" else "/predict"
    url = f"{base_url.rstrip('/')}{target_endpoint}"
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}

    start_t = time.time()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        elapsed_sec = round(time.time() - start_t, 3)

        if resp.status_code == 200:
            data = resp.json()
            if endpoint == "verify":
                risk_level = data.get("risk_level", "Unknown")
                prob = data.get("risk_score", 0.0)
                sources_cnt = len(data.get("sources", []))
            else:
                risk_level = data.get("ensemble", {}).get("risk_level", "Unknown")
                prob = data.get("ensemble", {}).get("probability_fake", 0.0)
                sources_cnt = 0

            return {
                "success": True,
                "latency_sec": elapsed_sec,
                "risk_level": risk_level,
                "probability": prob,
                "sources_count": sources_cnt,
                "status_code": 200
            }
        return {
            "success": False,
            "latency_sec": elapsed_sec,
            "error": f"HTTP {resp.status_code}: {resp.text[:100]}",
            "status_code": resp.status_code
        }
    except Exception as e:
        elapsed_sec = round(time.time() - start_t, 3)
        return {
            "success": False,
            "latency_sec": elapsed_sec,
            "error": str(e),
            "status_code": 0
        }


def log_benchmark_run(run_data: dict):
    """Appends benchmark metrics to CSV and JSON history logs."""
    # 1. Append to CSV
    csv_exists = CSV_LOG.exists()
    try:
        with open(CSV_LOG, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(["timestamp", "endpoint", "test_type", "sample_count", "min_sec", "mean_sec", "p50_sec", "p95_sec", "max_sec", "space_stage"])
            writer.writerow([
                run_data["timestamp"],
                run_data.get("endpoint", "verify"),
                run_data["test_type"],
                run_data["sample_count"],
                run_data["min_sec"],
                run_data["mean_sec"],
                run_data["p50_sec"],
                run_data["p95_sec"],
                run_data["max_sec"],
                run_data.get("space_stage", "N/A")
            ])
    except Exception as e:
        print(f"Warning: Failed to write CSV log: {e}")

    # 2. Append to JSON
    try:
        history = []
        if JSON_LOG.exists():
            with open(JSON_LOG, "r", encoding="utf-8") as f:
                try:
                    history = json.load(f)
                except Exception:
                    history = []
        history.append(run_data)
        with open(JSON_LOG, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to write JSON log: {e}")


def run_benchmark(base_url: str, samples: int = 10, is_cold: bool = False, endpoint: str = "verify"):
    """Executes latency benchmarking and computes statistical distributions."""
    test_type = "COLD_START" if is_cold else "WARM_INFERENCE"
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    ep_label = "Full LangGraph Agent Pipeline (/verify)" if endpoint == "verify" else "Neural Model Ensemble (/predict)"

    print("=" * 80)
    print(f"⏱️  RiskLens Latency Benchmark Suite [{test_type}]")
    print(f"🎯 Target Endpoint : {base_url}/{endpoint} ({ep_label})")
    print(f"📊 Sample Count    : {samples}")
    print(f"🕒 Timestamp        : {ts}")

    # Check HF runtime state
    state_info = check_hf_space_runtime_state()
    space_stage = state_info.get("stage", "UNKNOWN")
    print(f"🛰️  Space Runtime   : Stage '{space_stage}' (Hardware: {state_info.get('hardware', 'cpu-basic')})")
    print("=" * 80)

    latencies = []
    successes = 0

    print(f"\nExecuting {samples} real benchmark requests against /{endpoint}...\n")
    for i in range(samples):
        claim = TEST_CLAIMS[i % len(TEST_CLAIMS)]
        short_claim = (claim[:45] + "...") if len(claim) > 45 else claim
        print(f"  [{i+1:02d}/{samples:02d}] Testing claim: \"{short_claim}\"", end="", flush=True)

        res = measure_single_request(base_url, claim, endpoint=endpoint)
        lat = res["latency_sec"]
        latencies.append(lat)

        if res["success"]:
            successes += 1
            src_str = f" | Sources: {res.get('sources_count', 0)}" if endpoint == "verify" else ""
            print(f" -> ✅ {lat:.3f}s | Verdict: {res['risk_level']} (Prob: {res['probability']:.2f}){src_str}")
        else:
            print(f" -> ❌ {lat:.3f}s | Error: {res.get('error')}")

        # Short delay between samples on warm runs
        if not is_cold and i < samples - 1:
            time.sleep(0.3)

    if not latencies or successes == 0:
        print("\n❌ Benchmark failed: No successful responses received.")
        return

    # Compute Statistics
    lat_arr = np.array(latencies)
    min_lat = round(float(np.min(lat_arr)), 3)
    max_lat = round(float(np.max(lat_arr)), 3)
    mean_lat = round(float(np.mean(lat_arr)), 3)
    p50_lat = round(float(np.percentile(lat_arr, 50)), 3)
    p95_lat = round(float(np.percentile(lat_arr, 95)), 3)

    summary = {
        "timestamp": ts,
        "endpoint": endpoint,
        "test_type": test_type,
        "sample_count": len(latencies),
        "success_rate_pct": round((successes / len(latencies)) * 100, 1),
        "min_sec": min_lat,
        "mean_sec": mean_lat,
        "p50_sec": p50_lat,
        "p95_sec": p95_lat,
        "max_sec": max_lat,
        "space_stage": space_stage,
        "raw_latencies": latencies
    }

    log_benchmark_run(summary)

    # Print Summary Table
    print("\n" + "=" * 80)
    print(f"📈 LATENCY DISTRIBUTION SUMMARY ({test_type} — /{endpoint})")
    print("=" * 80)
    print(f"  • Endpoint     : /{endpoint} ({ep_label})")
    print(f"  • Success Rate : {summary['success_rate_pct']}% ({successes}/{len(latencies)})")
    print(f"  • Min Latency  : {min_lat:.3f} s")
    print(f"  • Mean Latency : {mean_lat:.3f} s")
    print(f"  • Median (P50) : {p50_lat:.3f} s")
    print(f"  • 95th % (P95) : {p95_lat:.3f} s")
    print(f"  • Max Latency  : {max_lat:.3f} s")
    print(f"  • Log Location : {CSV_LOG}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="RiskLens Latency Benchmark Suite")
    parser.add_argument("--url", default=os.getenv("SPACE_URL", DEFAULT_SPACE_URL), help="Hugging Face Space URL")
    parser.add_argument("--samples", type=int, default=10, help="Number of benchmark samples (default: 10)")
    parser.add_argument("--endpoint", choices=["verify", "predict"], default="verify", help="Endpoint to benchmark: 'verify' (Full LangGraph Agent) or 'predict' (Neural Ensemble)")
    parser.add_argument("--cold", action="store_true", help="Record as a cold-start measurement")
    args = parser.parse_args()

    run_benchmark(base_url=args.url, samples=args.samples, is_cold=args.cold, endpoint=args.endpoint)


if __name__ == "__main__":
    main()
