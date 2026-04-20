import base64
import urllib.request
import json
import time
import argparse
import os
from io import BytesIO
from PIL import Image
import xml.etree.ElementTree as ET

def fetch_image_as_base64(url: str) -> str:
    """Fetches an image from a URL and returns its base64 encoded string."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            image_data = response.read()
        return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to fetch image from URL: {e}")

def get_image_dimensions_from_base64(base64_string: str) -> tuple[int, int]:
    """Decodes a base64 image string and returns its exact (width, height)."""
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    base64_string = base64_string.strip()
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)

    try:
        image_data = base64.b64decode(base64_string)

        if b"<svg" in image_data[:100].lower():
            root = ET.fromstring(image_data)
            width = int(float(root.attrib.get('width', 0).replace('px', '')))
            height = int(float(root.attrib.get('height', 0).replace('px', '')))
            return width, height

        with Image.open(BytesIO(image_data)) as img:
            return img.width, img.height

    except Exception as e:
        raise ValueError(f"Failed to decode image or read dimensions: {e}")

def calculate_image_tokens(width: int, height: int, factor:int) -> dict:
    """Calculates the exact theoretical token count for Qwen2.5-VL."""
    patches = (width * height) / (factor * factor)
    tokens = patches + 2 # Add visual start/end markers

    return {
        "model_width": width,
        "model_height": height,
        "tokens": tokens
    }

def process_image_from_url(url: str, factor: int) -> dict:
    base64_string = fetch_image_as_base64(url)
    width, height = get_image_dimensions_from_base64(base64_string)
    token_data = calculate_image_tokens(width, height, factor)

    return {
        "original_width": width,
        "original_height": height,
        "tokens": token_data["tokens"]
    }

def save_result_to_file(filepath: str, data: dict):
    """Appends a dictionary as a JSON string to a file (JSONL format)."""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data) + '\n')

def run_token_comparison(resolutions: list[tuple[int, int]], endpoint: str, iterations: int, output_file: str, model: str, factor: int):
    """
    Runs a test loop calculating theoretical tokens and comparing them
    against actual prompt tokens from the API response across multiple resolutions.
    """
    print(f"API Endpoint: {endpoint}")
    print(f"Output File:  {os.path.abspath(output_file)}")
    print("-" * 50)

    for width, height in resolutions:
        print(f"\n=== Starting tests for resolution: {width}x{height} ===")

        for index in range(1, iterations + 1):
            image_url = f"https://picsum.photos/{width}/{height}?random={index}"
            print(f"[{width}x{height} - Iteration {index}/{iterations}] Processing: {image_url}")

            record = {
                "target_width": width,
                "target_height": height,
                "iteration": index,
                "url": image_url,
                "status": "success",
                "calculated_image_tokens": None,
                "api_prompt_tokens": None,
                "difference": None,
                "error": None
            }

            # 1. Calculate Theoretical Tokens Locally
            try:
                local_result = process_image_from_url(image_url, factor)
                calculated_tokens = local_result['tokens']
                record["calculated_image_tokens"] = calculated_tokens
            except Exception as e:
                error_msg = f"Local processing error: {e}"
                print(f"  -> Skipping due to {error_msg}")
                record["status"] = "failed"
                record["error"] = error_msg
                save_result_to_file(output_file, record)
                continue

            # 2. Prepare API Request Payload
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            },
                            {
                                "type": "text",
                                "text": "Describe"
                            }
                        ]
                    }
                ]
            }

            headers = {"Content-Type": "application/json"}

            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode('utf-8'),
                headers=headers,
                method='POST'
            )

            # 3. Send API Request and Compare
            try:
                with urllib.request.urlopen(req) as response:
                    response_data = json.loads(response.read().decode('utf-8'))

                    actual_prompt_tokens = response_data.get("usage", {}).get("prompt_tokens", 0)
                    difference = actual_prompt_tokens - calculated_tokens

                    record["api_prompt_tokens"] = actual_prompt_tokens
                    record["difference"] = difference

                    print(f"  -> Calculated: {calculated_tokens:.2f} | API: {actual_prompt_tokens} | Diff: {difference:.2f}")

            except urllib.error.HTTPError as e:
                error_msg = f"API Request failed with HTTP {e.code}: {e.read().decode('utf-8')}"
                print(f"  -> {error_msg}")
                record["status"] = "failed"
                record["error"] = error_msg
            except Exception as e:
                error_msg = f"API Request failed: {e}"
                print(f"  -> {error_msg}")
                record["status"] = "failed"
                record["error"] = error_msg

            # Save the result for this iteration
            save_result_to_file(output_file, record)

            # Sleep briefly to avoid hitting rate limits
            time.sleep(1)

# ==========================================
# Configuration and Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and compare image tokens across multiple resolutions.")

    # Accept a list of resolution strings
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=["270x180","540x360","720x480","1080x720","1620x1080"],
        help="List of target resolutions in WxH format (e.g., 1920x1280 800x600 1024x1024)"
    )
    parser.add_argument("--iterations", type=int, default=10, help="Number of images to process per resolution (default: 10)")
    parser.add_argument("--endpoint", type=str, default="http://34.132.102.66:80/v1/chat/completions", help="API endpoint URL")
    parser.add_argument("--output", type=str, default="token_results.jsonl", help="Output file to save JSONL results")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--factor", type=int, default=28)

    args = parser.parse_args()

    # Parse the resolutions strings into a list of tuples (e.g. [(1920, 1280), (800, 600)])
    parsed_resolutions = []
    for res in args.resolutions:
        try:
            w_str, h_str = res.lower().split('x')
            parsed_resolutions.append((int(w_str), int(h_str)))
        except ValueError:
            print(f"Warning: Invalid resolution format '{res}'. Please use WxH (e.g., 1920x1280). Skipping.")

    if not parsed_resolutions:
        print("Error: No valid resolutions provided. Exiting.")
        exit(1)

    # Clear the file or write a fresh one if you want it to overwrite every time
    # (Remove the next two lines if you prefer to append across multiple script runs)
    if os.path.exists(args.output):
        open(args.output, 'w').close()

    run_token_comparison(parsed_resolutions, args.endpoint, args.iterations, args.output, args.model, args.factor)