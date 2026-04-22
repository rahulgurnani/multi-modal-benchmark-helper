# Multi-Modal Benchmark Helper

A standardized toolset for benchmarking the performance and throughput of multi-modal Large Language Models (LLMs), specifically optimized for vision-language models like Gemma 4 on GKE/TPU infrastructure.

## Overview

This repository provides two core scripts to orchestrate the benchmarking lifecycle:
1. **Data Generation**: Create representative JSONL payloads with configurable image counts, resolutions, and KV-cache prefix sharing patterns.
2. **Benchmark Runner**: An asynchronous load generator that dispatches requests at a controlled Rate Per Second (RPS) and captures deep metrics including E2E latency, token throughput, and network bandwidth.

---

## 1. Data Generation (`data_generation/data_generation_script.py`)

This script generates a `.jsonl` file where each line is a valid OpenAI-compatible Chat Completion request containing multiple images.

### Key Capabilities
- **Prefix Sharing**: Configure a set of "prefix" images shared across groups of requests to test KV-cache hit performance.
- **Resolution Control**: Supports standard resolutions (180p, 360p, 720p, 1080p).
- **Realistic Warmup**: Automatically prepends a warmup request to the file that matches the shape of the main benchmark requests.

### Example Usage

**Generate 100 requests with 80% prefix share (64 shared images, 16 distinct images):**
```bash
python3 data_generation/data_generation_script.py \
  --num_groups 1 \
  --requests_per_group 100 \
  --prefix_image_choices 64 \
  --var_image_choices 16 \
  --resolution 1080p \
  --output_file "100req_80pct_shared.jsonl"
```

---

## 2. Benchmark Runner (`benchmark/benchmark.py`)

The benchmark runner executes the load test against your inference endpoint and produces a detailed performance report.

### Key Features
- **Append-only Results**: Automatically saves results to `res-<input>-<host_port>` and appends new runs with start/end timestamps.
- **Traffic Metrics**: Measures raw byte sizes for every request/response to report total data transferred and aggregate bandwidth (MB/sec).
- **Token Analysis**: Reports prompt tokens, completion tokens, and specific prompt token details (like cached vs. raw).

### Example Usage

**Run a 10 RPS load test for 60 seconds:**
```bash
python3 benchmark/benchmark.py \
  --model google/gemma-4-26B-A4B-it \
  --api-url "http://localhost:8002/v1/chat/completions" \
  --input-file "100req_80pct_shared.jsonl" \
  --rps 10 \
  --duration 60 \
  --warmup 1
```

---

## Interpreting Results

At the end of a run, the script prints a summary table:

| Metric | Description |
|--------|-------------|
| **Successful / Failed** | Count of 200 OK vs Error responses. |
| **p50/p95 Latency** | End-to-end request latency percentiles. |
| **Token Throughput** | Aggregate tokens per second (Prompt, Completion, Total). |
| **Network Traffic** | Total MB transferred and Overall Bandwidth in MB/sec. |

### Output Files
Results are stored in JSONL format with the following extra fields per request:
- `request_size_bytes`
- `response_size_bytes`
- `e2e_latency_seconds`
- `cached_tokens` (if supported by the backend)

---

## Installation

```bash
pip install aiohttp urllib3
```

Ensure your inference server (vLLM or similar) is reachable from the benchmark environment. If running locally against GKE, use `kubectl port-forward` to expose the service.
