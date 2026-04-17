import argparse
import json
import random
import itertools

# python data_generation__url_script.py \
#   --prefix_start 0 \
#   --prefix_end 1000 \
#   --var_start 1000 \
#   --var_end 2000 \
#   --num_groups 200 \
#   --requests_per_group 6 \
#   --prefix_image_choices 0 1 2 3 \
#   --var_image_choices 0 1 \
#   --resolution 720p \
#   --max_tokens 1000 \
#   --model "google/gemma-4-31B-it" \
#   --output_file "360p_qwen_requests.jsonl"

def generate_url_batch(start_idx, end_idx, width, height):
    """Generates a batch of image URLs at the specified resolution."""
    return [
        f"https://picsum.photos/{width}/{height}?random={i}"
        for i in range(start_idx, end_idx)
    ]

def main():
    # --- COMMAND LINE ARGUMENT SETUP ---
    parser = argparse.ArgumentParser(description="Generate a customized visual LLM benchmark dataset using URLs.")
    parser.add_argument("--prefix_start", type=int, default=0, help="Prefix pool start index")
    parser.add_argument("--prefix_end", type=int, default=1000, help="Prefix pool end index")
    parser.add_argument("--var_start", type=int, default=1000, help="Variable pool start index")
    parser.add_argument("--var_end", type=int, default=2000, help="Variable pool end index")
    parser.add_argument("--num_groups", type=int, default=200, help="Number of benchmark groups")
    parser.add_argument("--requests_per_group", type=int, default=8, help="Number of requests per group")
    parser.add_argument("--output_file", type=str, default="custom_qwen_benchmark.jsonl", help="Path to save the JSONL file")

    parser.add_argument("--prefix_image_choices", nargs='+', type=int, default=[0, 1, 2, 3], help="Space-separated choices for prefix images")
    parser.add_argument("--var_image_choices", nargs='+', type=int, default=[0, 1], help="Space-separated choices for variable images")

    # NEW PARAMETERS FOR RESOLUTION
    parser.add_argument("--resolution", type=str, choices=["180p", "360p", "720p", "1080p"], help="Use standard 16:9 resolutions")
    parser.add_argument("--width", type=int, default=640, help="Custom image width (default: 640)")
    parser.add_argument("--height", type=int, default=360, help="Custom image height (default: 360)")
    parser.add_argument("--max_tokens", type=int, default=1, help="Max output token")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name")

    args = parser.parse_args()

    # Determine final width and height
    width = args.width
    height = args.height

    if args.resolution:
        resolution_map = {
            "180p": (320, 180),
            "360p": (640, 360),
            "720p": (1280, 720),
            "1080p": (1920, 1080)
        }
        width, height = resolution_map[args.resolution]
        print(f"Using standard resolution {args.resolution} -> {width}x{height}")
    else:
        print(f"Using custom resolution -> {width}x{height}")

    print("Generating image URL pools...")

    prefix_pool = generate_url_batch(args.prefix_start, args.prefix_end, width, height)
    variable_pool = generate_url_batch(args.var_start, args.var_end, width, height)

    print(f"Generated {len(prefix_pool)} Prefix URLs and {len(variable_pool)} Variable URLs.")
    print("Generating distinct prompts...")

    actions = [
        "Compare", "Identify", "Analyze", "Describe", "Evaluate",
        "Examine", "Review", "Summarize", "Outline", "Assess"
    ]
    subjects = [
        "the main objects", "the color palettes", "the lighting conditions",
        "the background elements", "the foreground subjects", "the overall composition",
        "the structural patterns", "the textural details", "the spatial relationships",
        "the perspective angles"
    ]
    contexts = [
        "across all images.", "between the prefix and variable images.",
        "in the sequence provided.", "from the first to the last image.",
        "in the provided visual context.", "focusing on any distinct differences.",
        "highlighting the core similarities.", "with close attention to fine details.",
        "and output the findings as a JSON list.", "and write a brief descriptive narrative."
    ]

    prompts = [f"{a} {s} {c}" for a, s, c in itertools.product(actions, subjects, contexts)]

    print(f"Successfully generated {len(prompts)} unique prompts. Generating benchmark file...")

    model_name = args.model

    warmup_requests = []
    main_requests = []
    request_count = 0

    # --- GENERATE WARM-UP REQUESTS ---
    print("Generating 20 warm-up requests...")
    combined_pool = variable_pool
    for _ in range(20):
        content_array = []
        num_warmup_images = random.randint(1, 2)
        warmup_images = random.sample(combined_pool, num_warmup_images)

        for url in warmup_images:
            content_array.append({
                "type": "image_url",
                "image_url": {"url": url}
            })

        content_array.append({
            "type": "text",
            "text": random.choice(prompts)
        })

        warmup_requests.append({
            "model": model_name,
            "messages": [{"role": "user", "content": content_array}],
            "max_tokens": 100,
            "temperature": 1
        })

    # --- GENERATE MAIN REQUESTS ---
    for group_id in range(args.num_groups):
        num_prefix_images = random.choice(args.prefix_image_choices)
        group_prefix_images = random.sample(prefix_pool, min(num_prefix_images, len(prefix_pool)))

        if (group_id + 1) % 50 == 0:
            print(f"Group {group_id + 1}/{args.num_groups}: Prepared {args.requests_per_group} requests.")

        for _ in range(args.requests_per_group):
            num_var_images = random.choice(args.var_image_choices)
            request_var_images = random.sample(variable_pool, min(num_var_images, len(variable_pool)))

            content_array = []

            for url in group_prefix_images:
                content_array.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })

            content_array.append({
                "type": "text",
                "text": random.choice(prompts)
            })

            for url in request_var_images:
                content_array.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })

            request = {
                "model": model_name,
                "messages": [{"role": "user", "content": content_array}],
                "max_tokens": args.max_tokens,
                "temperature": 1
            }

            main_requests.append(request)
            request_count += 1

    random.shuffle(main_requests)
    print("\nMain requests generated and randomized.")

    final_requests = warmup_requests + main_requests
    total_requests = len(final_requests)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for req in final_requests:
            f.write(json.dumps(req) + "\n")

    print(f"Successfully saved {total_requests} total requests (20 warm-up + {request_count} main) to '{args.output_file}'.")

if __name__ == "__main__":
    main()