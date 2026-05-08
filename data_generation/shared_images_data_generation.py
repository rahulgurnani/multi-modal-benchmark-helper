import argparse
import json
import random
import base64
import urllib.request
import itertools
import concurrent.futures
import time
import ssl

# Disable SSL verification to work around corporate MITM certificate interception.
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

def fetch_and_encode(url, label):
    """Downloads an image and returns its base64 string."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as response:
            return base64.b64encode(response.read()).decode('utf-8')
    except Exception as e:
        print(f"Failed to fetch {label}: {e}")
        return None

def fetch_batch(start_idx, end_idx, label_prefix, width, height):
    """Fetches a batch of images concurrently at the specified resolution."""
    urls_and_labels = [
        (f"https://picsum.photos/{width}/{height}?random={i}", f"{label_prefix} {i}")
        for i in range(start_idx, end_idx)
    ]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        future_to_url = {executor.submit(fetch_and_encode, url, label): label for url, label in urls_and_labels}

        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
            if result:
                results.append(result)

    return results

def main():
    parser = argparse.ArgumentParser(description="Generate a visual LLM benchmark dataset with shared (but not necessarily prefix) images.")
    parser.add_argument("--shared_start", type=int, default=0, help="Shared pool start index")
    parser.add_argument("--shared_end", type=int, default=500, help="Shared pool end index")
    parser.add_argument("--unique_start", type=int, default=500, help="Unique pool start index")
    parser.add_argument("--unique_end", type=int, default=1500, help="Unique pool end index")
    parser.add_argument("--num_groups", type=int, default=100, help="Number of image sharing groups")
    parser.add_argument("--requests_per_group", type=int, default=5, help="Number of requests that share the same set of images")
    parser.add_argument("--output_file", type=str, default="shared_images_benchmark.jsonl", help="Path to save the JSONL file")

    parser.add_argument("--shared_image_choices", nargs='+', type=int, default=[1, 2], help="Number of shared images per request")
    parser.add_argument("--unique_image_choices", nargs='+', type=int, default=[1], help="Number of unique images per request")

    parser.add_argument("--resolution", type=str, choices=["180p", "360p", "720p", "1080p"], default="360p")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max output tokens")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup requests to send (default: 1)")

    args = parser.parse_args()

    resolution_map = {
        "180p": (320, 180),
        "360p": (640, 360),
        "720p": (1280, 720),
        "1080p": (1920, 1080)
    }
    width, height = resolution_map[args.resolution]

    print(f"Downloading {args.shared_end - args.shared_start} Shared images...")
    shared_pool = fetch_batch(args.shared_start, args.shared_end, "Shared", width, height)

    print(f"Downloading {args.unique_end - args.unique_start} Unique images...")
    unique_pool = fetch_batch(args.unique_start, args.unique_end, "Unique", width, height)

    if not shared_pool or not unique_pool:
        print("Error: Failed to fetch images.")
        exit()

    actions = ["Compare", "Identify", "Analyze", "Describe", "Evaluate"]
    subjects = ["the main objects", "the color palettes", "the lighting", "the background"]
    contexts = ["across all images.", "focusing on differences.", "in the provided context."]
    prompts = [f"{a} {s} {c}" for a, s, c in itertools.product(actions, subjects, contexts)]

    warmup_requests = []
    
    # --- GENERATE WARM-UP REQUESTS ---
    print(f"Generating {args.warmup} warm-up requests...")
    for _ in range(args.warmup):
        # Pick images from pools for warmup
        num_shared = random.choice(args.shared_image_choices)
        w_shared_images = random.sample(shared_pool, min(num_shared, len(shared_pool)))
        
        num_unique = random.choice(args.unique_image_choices)
        w_unique_images = random.sample(unique_pool, min(num_unique, len(unique_pool)))

        components = []
        for b64 in w_shared_images:
            components.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        for b64 in w_unique_images:
            components.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        components.append({"type": "text", "text": random.choice(prompts)})
        
        random.shuffle(components)

        warmup_requests.append({
            "model": args.model,
            "messages": [{"role": "user", "content": components}],
            "max_tokens": args.max_tokens,
            "temperature": 0.7
        })

    main_requests = []
    
    for group_id in range(args.num_groups):
        # Pick the set of images that will be shared across this group of requests
        num_shared = random.choice(args.shared_image_choices)
        group_shared_images = random.sample(shared_pool, min(num_shared, len(shared_pool)))

        for _ in range(args.requests_per_group):
            num_unique = random.choice(args.unique_image_choices)
            request_unique_images = random.sample(unique_pool, min(num_unique, len(unique_pool)))

            # Create components: images and one text prompt
            components = []
            for b64 in group_shared_images:
                components.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            
            for b64 in request_unique_images:
                components.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            
            components.append({"type": "text", "text": random.choice(prompts)})

            # Shuffling the components ensures the shared images are not always at the beginning
            random.shuffle(components)

            request = {
                "model": args.model,
                "messages": [{"role": "user", "content": components}],
                "max_tokens": args.max_tokens,
                "temperature": 0.7
            }
            main_requests.append(request)

    random.shuffle(main_requests)
    
    final_requests = warmup_requests + main_requests

    with open(args.output_file, "w", encoding="utf-8") as f:
        for req in final_requests:
            f.write(json.dumps(req) + "\n")

    print(f"Successfully saved {len(final_requests)} total requests ({args.warmup} warm-up + {len(main_requests)} main) to '{args.output_file}'.")

if __name__ == "__main__":
    main()
