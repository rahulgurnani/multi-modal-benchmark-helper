import asyncio
import aiohttp
import json
import time
import argparse
import sys
import os
from urllib.parse import urlparse
from datetime import datetime

# python benchmark.py \
#   --api-url "http://35.208.25.175:80/v1/chat/completions" \
#   --input-file "payloads.jsonl" \
#   --output-file "metrics.jsonl" \
#   --rps 5 \
#   --duration 120

async def send_warmup_request(session, payload, api_url, model_override=None):
    """Sends a request purely to warm up the server. Ignores all metrics."""
    # Ensure streaming is off
    payload["stream"] = False
    if model_override:
        payload["model"] = model_override
    try:
        async with session.post(api_url, json=payload) as response:
            await response.read()
    except Exception:
        pass

async def send_request(session, payload, request_id, api_url, model_override=None):
    """Sends a request to the inference gateway and records E2E metrics."""
    # Ensure streaming is off for accurate E2E benchmarking
    payload["stream"] = False
    if model_override:
        payload["model"] = model_override

    # Calculate request size
    encoded_payload = json.dumps(payload).encode('utf-8')
    request_size_bytes = len(encoded_payload)

    start_time_raw = time.time()
    start_time_iso = datetime.fromtimestamp(start_time_raw).isoformat()

    try:
        # Use 'data' instead of 'json' to use our pre-encoded payload and ensure correct size measurement
        headers = {"Content-Type": "application/json"}
        async with session.post(api_url, data=encoded_payload, headers=headers) as response:
            status = response.status

            # Wait for the entire response to complete
            body = await response.read()
            response_size_bytes = len(body)

            end_time_raw = time.time()
            end_time_iso = datetime.fromtimestamp(end_time_raw).isoformat()
            e2e_latency = end_time_raw - start_time_raw

            # Extract token usage from the response body
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            cached_tokens = 0
            uncached_prompt_tokens = 0
            response_body = None
            try:
                resp_json = json.loads(body)
                response_body = resp_json
                usage = resp_json.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
                prompt_tokens_details = usage.get("prompt_tokens_details", {})
                cached_tokens = prompt_tokens_details.get("cached_tokens", 0)
                uncached_prompt_tokens = prompt_tokens - cached_tokens
                
                # Fallback: derive total if not provided
                if total_tokens == 0 and (prompt_tokens or completion_tokens):
                    total_tokens = prompt_tokens + completion_tokens
            except (json.JSONDecodeError, AttributeError):
                response_body = body.decode('utf-8', errors='replace')

            return {
                "request_id": request_id,
                "status": status,
                "start_time_timestamp": start_time_raw,
                "start_time_iso": start_time_iso,
                "end_time_timestamp": end_time_raw,
                "end_time_iso": end_time_iso,
                "e2e_latency_seconds": e2e_latency,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "uncached_prompt_tokens": uncached_prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "request_size_bytes": request_size_bytes,
                "response_size_bytes": response_size_bytes,
                "response": response_body
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
            "e2e_latency_seconds": end_time_raw - start_time_raw,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "uncached_prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_size_bytes": request_size_bytes,
            "response_size_bytes": 0,
            "response": None
        }

async def main(api_url, input_file, output_file, requests_per_second, warmup_requests, duration, request_timeout, model_override=None):
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

    # Per-request timeout for heavy vision tasks (configurable via --timeout)
    timeout = aiohttp.ClientTimeout(total=request_timeout)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        # --- WARM-UP PHASE ---
        if warmups_to_send > 0:
            print(f"\nSending {warmups_to_send} warm-up requests... (metrics will NOT be recorded)")
            warmup_tasks = []
            for p in warmup_payloads:
                warmup_tasks.append(asyncio.create_task(send_warmup_request(session, p, api_url, model_override)))

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
            task = asyncio.create_task(send_request(session, payload, i, api_url, model_override))
            tasks.append(task)

        # Wait for all dispatched benchmark tasks to complete
        print(f"Finished dispatching all {total_requests_to_send} requests. Waiting for final responses...")
        results = await asyncio.gather(*tasks)

    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time

    # Write metrics to output file (append mode)
    with open(output_file, 'a', encoding='utf-8') as f:
        # Add a run separator with timestamps
        run_start_iso = datetime.fromtimestamp(overall_start_time).isoformat()
        run_end_iso = datetime.fromtimestamp(overall_end_time).isoformat()
        
        f.write(f"\n# === RUN START: {run_start_iso} (RPS: {requests_per_second}, Duration: {duration}s) ===\n")
        for res in results:
            f.write(json.dumps(res) + '\n')
        f.write(f"# === RUN END: {run_end_iso} ===\n")

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

        # Token Usage Statistics
        total_prompt_tokens = sum(r.get('prompt_tokens', 0) for r in successes)
        total_cached_tokens = sum(r.get('cached_tokens', 0) for r in successes)
        total_uncached_prompt_tokens = sum(r.get('uncached_prompt_tokens', 0) for r in successes)
        total_completion_tokens = sum(r.get('completion_tokens', 0) for r in successes)
        total_all_tokens = sum(r.get('total_tokens', 0) for r in successes)
        num_successes = len(successes)

        avg_prompt_tokens = total_prompt_tokens / num_successes
        avg_cached_tokens = total_cached_tokens / num_successes
        avg_uncached_prompt_tokens = total_uncached_prompt_tokens / num_successes
        avg_completion_tokens = total_completion_tokens / num_successes
        avg_total_tokens = total_all_tokens / num_successes

        print("Token Usage:")
        print(f"  Total Prompt Tokens     : {total_prompt_tokens}")
        print(f"  -> Cached Prompt Tokens : {total_cached_tokens}")
        print(f"  -> Uncached (Raw) Prompts: {total_uncached_prompt_tokens}")
        print(f"  Total Completion Tokens : {total_completion_tokens}")
        print(f"  Total Tokens            : {total_all_tokens}")
        print(f"  Avg Prompt Tokens/Req   : {avg_prompt_tokens:.1f}")
        print(f"  Avg Cached Tokens/Req   : {avg_cached_tokens:.1f}")
        print(f"  Avg Completion Tok/Req  : {avg_completion_tokens:.1f}")
        print(f"  Avg Total Tokens/Req    : {avg_total_tokens:.1f}")

        if total_duration > 0:
            print(f"  Prompt Tok Throughput   : {total_prompt_tokens / total_duration:.2f} tokens/sec")
            print(f"  Completion Tok Thruput  : {total_completion_tokens / total_duration:.2f} tokens/sec")
            print(f"  Total Tok Throughput    : {total_all_tokens / total_duration:.2f} tokens/sec")
        print("-" * 40)

    # Traffic Statistics
    total_request_bytes = sum(r.get('request_size_bytes', 0) for r in results)
    total_response_bytes = sum(r.get('response_size_bytes', 0) for r in results)
    total_data_bytes = total_request_bytes + total_response_bytes

    avg_req_size = total_request_bytes / len(results)
    num_successes = len(successes)
    avg_res_size = total_response_bytes / num_successes if num_successes > 0 else 0

    print("Network Traffic:")
    print(f"  Total Request Data      : {total_request_bytes / 1024 / 1024:.2f} MB")
    print(f"  Total Response Data     : {total_response_bytes / 1024 / 1024:.2f} MB")
    print(f"  Total Data Transferred  : {total_data_bytes / 1024 / 1024:.2f} MB")
    print(f"  Avg Request Size        : {avg_req_size / 1024:.2f} KB")
    print(f"  Avg Response Size       : {avg_res_size / 1024:.2f} KB")
    if total_duration > 0:
        print(f"  Overall Bandwidth       : {total_data_bytes / 1024 / 1024 / total_duration:.2f} MB/sec")
    print("-" * 40)

    print(f"\nDetailed metrics saved to: {output_file}")

if __name__ == "__main__":
    # Setup argparse to accept command-line parameters
    parser = argparse.ArgumentParser(description="Time-bound Async API Benchmarking Tool")

    parser.add_argument("--api-url", type=str, required=True,
                        help="The API endpoint URL to benchmark")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to the input JSONL file containing payloads")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to save the output JSONL metrics (default: res-<input-filename>)")
    parser.add_argument("--rps", type=float, default=10.0,
                        help="Requests per second (default: 10)")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Duration to run the benchmark in seconds (default: 60)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Number of warmup requests to send (default: 20)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-request timeout in seconds (default: 3600 = 1 hour)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override the model name in the request payload")

    args = parser.parse_args()

    # Windows specific fix for asyncio
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # If output file is not specified, default to res-<input-filename>-<host:port>
    if not args.output_file:
        input_base = os.path.basename(args.input_file)
        parsed_url = urlparse(args.api_url)
        netloc = parsed_url.netloc.replace(':', '_') # Use underscore for safer filenames
        args.output_file = f"res-{input_base}-{netloc}"

    # Run the main async loop with the parsed arguments
    asyncio.run(main(
        api_url=args.api_url,
        input_file=args.input_file,
        output_file=args.output_file,
        requests_per_second=args.rps,
        warmup_requests=args.warmup,
        duration=args.duration,
        request_timeout=args.timeout,
        model_override=args.model
    ))