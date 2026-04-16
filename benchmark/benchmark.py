import asyncio
import aiohttp
import json
import time
import argparse
import sys
from datetime import datetime

# python benchmark.py \
#   --api-url "http://35.208.25.175:80/v1/chat/completions" \
#   --input-file "payloads.jsonl" \
#   --output-file "metrics.jsonl" \
#   --rps 5 \
#   --duration 120

async def send_warmup_request(session, payload, api_url):
    """Sends a request purely to warm up the server. Ignores all metrics."""
    # Ensure streaming is off
    payload["stream"] = False
    try:
        async with session.post(api_url, json=payload) as response:
            await response.read()
    except Exception:
        pass

async def send_request(session, payload, request_id, api_url):
    """Sends a request to the inference gateway and records E2E metrics."""
    # Ensure streaming is off for accurate E2E benchmarking
    payload["stream"] = False

    start_time_raw = time.time()
    start_time_iso = datetime.fromtimestamp(start_time_raw).isoformat()

    try:
        async with session.post(api_url, json=payload) as response:
            status = response.status

            # Wait for the entire response to complete
            await response.read()

            end_time_raw = time.time()
            end_time_iso = datetime.fromtimestamp(end_time_raw).isoformat()
            e2e_latency = end_time_raw - start_time_raw

            return {
                "request_id": request_id,
                "status": status,
                "start_time_timestamp": start_time_raw,
                "start_time_iso": start_time_iso,
                "end_time_timestamp": end_time_raw,
                "end_time_iso": end_time_iso,
                "e2e_latency_seconds": e2e_latency
            }

    except Exception as e:
        end_time_raw = time.time()
        end_time_iso = datetime.fromtimestamp(end_time_raw).isoformat()

        return {
            "request_id": request_id,
            "status": "error",
            "error_message": str(e),
            "start_time_timestamp": start_time_raw,
            "start_time_iso": start_time_iso,
            "end_time_timestamp": end_time_raw,
            "end_time_iso": end_time_iso,
            "e2e_latency_seconds": end_time_raw - start_time_raw
        }

async def main(api_url, input_file, output_file, requests_per_second, warmup_requests, duration):
    print(f"Loading payloads from {input_file}...")

    # Read the JSONL file
    payloads = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    payloads.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Could not find the file at {input_file}")
        return

    total_loaded = len(payloads)
    print(f"Successfully loaded {total_loaded} requests.")

    # Split payloads into warmup and benchmark sets
    warmups_to_send = min(warmup_requests, total_loaded)
    warmup_payloads = payloads[:warmups_to_send]
    benchmark_payloads = payloads[warmups_to_send:]

    if not benchmark_payloads:
        print("No payloads left for the main benchmark after warm-up. Exiting.")
        return

    # 5-minute timeout per request to handle heavy vision tasks
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        # --- WARM-UP PHASE ---
        if warmups_to_send > 0:
            print(f"\nSending {warmups_to_send} warm-up requests... (metrics will NOT be recorded)")
            warmup_tasks = []
            for p in warmup_payloads:
                warmup_tasks.append(asyncio.create_task(send_warmup_request(session, p, api_url)))

            # Wait for all warm-up tasks to complete
            await asyncio.gather(*warmup_tasks)
            print("Warm-up complete. Resting for 10 seconds before main benchmark...\n")
            time.sleep(10) # Reduced to 10s for practical testing, adjust if you need 60s

        # --- MAIN BENCHMARK PHASE ---
        total_requests_to_send = int(requests_per_second * duration)
        num_available_payloads = len(benchmark_payloads)

        print(f"Starting benchmark against {api_url}")
        print(f"Targeting   : {requests_per_second} requests/sec for {duration} seconds")
        print(f"Total tasks : {total_requests_to_send} (cycling through {num_available_payloads} unique payloads)\n")

        tasks = []
        interval = 1.0 / requests_per_second

        overall_start_time = time.time()
        loop_start_time = time.monotonic()

        for i in range(total_requests_to_send):
            # Use modulo to cycle back to the beginning of the list if we run out of payloads
            payload = benchmark_payloads[i % num_available_payloads]

            # Calculate exactly when this request *should* start
            target_time = loop_start_time + (i * interval)
            now = time.monotonic()

            # Sleep the exact difference to prevent drift over time
            delay = target_time - now
            if delay > 0:
                await asyncio.sleep(delay)

            # Fire off the task immediately without waiting for it to finish
            task = asyncio.create_task(send_request(session, payload, i, api_url))
            tasks.append(task)

        # Wait for all dispatched benchmark tasks to complete
        print(f"Finished dispatching all {total_requests_to_send} requests. Waiting for final responses...")
        results = await asyncio.gather(*tasks)

    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time

    # Write metrics to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')

    # Calculate Summary Statistics
    successes = [r for r in results if r.get('status') == 200]
    errors = [r for r in results if r.get('status') != 200]

    print("\n" + "="*40)
    print("         BENCHMARK RESULTS         ")
    print("="*40)
    print(f"Target RPS         : {requests_per_second}")
    print(f"Target Duration    : {duration} seconds")
    print(f"Total Dispatched   : {total_requests_to_send}")
    print(f"Successful         : {len(successes)}")
    print(f"Failed/Errors      : {len(errors)}")
    print(f"Total Run Time     : {total_duration:.2f} seconds")

    if total_duration > 0:
        print(f"Actual Throughput  : {len(results) / total_duration:.2f} requests/sec")

    if successes:
        # Calculate E2E Aggregates and Percentiles
        latencies = sorted([r['e2e_latency_seconds'] for r in successes])

        avg_e2e = sum(latencies) / len(latencies)
        min_e2e = latencies[0]
        max_e2e = latencies[-1]

        # Helper to get percentile from sorted list
        def get_percentile(sorted_data, p):
            index = int((p / 100.0) * len(sorted_data))
            return sorted_data[min(index, len(sorted_data) - 1)]

        p25 = get_percentile(latencies, 25)
        p50 = get_percentile(latencies, 50)
        p75 = get_percentile(latencies, 75)
        p95 = get_percentile(latencies, 95)

        print("-" * 40)
        print("End-to-End Latency (E2E):")
        print(f"  Min E2E          : {min_e2e:.3f} s")
        print(f"  p25 E2E          : {p25:.3f} s")
        print(f"  p50 E2E (Median) : {p50:.3f} s")
        print(f"  p75 E2E          : {p75:.3f} s")
        print(f"  p95 E2E          : {p95:.3f} s")
        print(f"  Max E2E          : {max_e2e:.3f} s")
        print(f"  Avg E2E          : {avg_e2e:.3f} s")
        print("-" * 40)

    print(f"\nDetailed metrics saved to: {output_file}")

if __name__ == "__main__":
    # Setup argparse to accept command-line parameters
    parser = argparse.ArgumentParser(description="Time-bound Async API Benchmarking Tool")

    parser.add_argument("--api-url", type=str, required=True,
                        help="The API endpoint URL to benchmark")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to the input JSONL file containing payloads")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to save the output JSONL metrics")
    parser.add_argument("--rps", type=float, default=10.0,
                        help="Requests per second (default: 10)")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Duration to run the benchmark in seconds (default: 60)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Number of warmup requests to send (default: 20)")

    args = parser.parse_args()

    # Windows specific fix for asyncio
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run the main async loop with the parsed arguments
    asyncio.run(main(
        api_url=args.api_url,
        input_file=args.input_file,
        output_file=args.output_file,
        requests_per_second=args.rps,
        warmup_requests=args.warmup,
        duration=args.duration
    ))