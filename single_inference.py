import asyncio
import aiohttp
import json
import time
import argparse
import base64
import urllib.request
import sys

def fetch_and_encode(url):
    """Downloads an image and returns its base64 string."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            return base64.b64encode(response.read()).decode('utf-8')
    except Exception as e:
        print(f"Failed to fetch image: {e}")
        return None

async def send_single_request(api_url, model, image_url, prompt, max_tokens):
    print(f"Fetching image from {image_url}...")
    b64_image = fetch_and_encode(image_url)
    if not b64_image:
        return

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "stream": False
    }

    print(f"Sending request to {api_url}...")
    start_time = time.time()
    
    # 5-minute timeout per request to handle heavy vision tasks
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(api_url, json=payload) as response:
                status = response.status
                body = await response.read()
                end_time = time.time()
                
                latency = end_time - start_time
                
                print("\n" + "="*40)
                print("         INFERENCE OUTPUT         ")
                print("="*40)
                print(f"Status Code: {status}")
                
                try:
                    # Attempt to parse and pretty-print JSON response
                    json_response = json.loads(body)
                    print(json.dumps(json_response, indent=2))
                except json.JSONDecodeError:
                    print(body.decode('utf-8', errors='ignore'))
                
                print("-" * 40)
                print(f"Time Taken: {latency:.3f} seconds")
                print("="*40)
                
        except Exception as e:
            print(f"Request failed with exception: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Request Inference Tool with Image")
    parser.add_argument("--api-url", type=str, required=True, help="The API endpoint URL (e.g., http://host:port/v1/chat/completions)")
    parser.add_argument("--model", type=str, default="qwen/Qwen2.5-VL-7B-Instruct", help="Model name")
    parser.add_argument("--image-url", type=str, default=None, help="Explicit URL of the image to send (overrides width/height)")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.", help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=1, help="Max output tokens")
    
    # Image size arguments
    parser.add_argument("--resolution", type=str, choices=["180p", "360p", "720p", "1080p"], help="Use standard 16:9 resolutions")
    parser.add_argument("--width", type=int, default=1280, help="Custom image width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Custom image height (default: 720)")

    args = parser.parse_args()

    # Determine dimensions
    width, height = args.width, args.height
    if args.resolution:
        resolution_map = {
            "180p": (320, 180),
            "360p": (640, 360),
            "720p": (1280, 720),
            "1080p": (1920, 1080)
        }
        width, height = resolution_map[args.resolution]

    # Use explicit URL if provided, otherwise generate random Picsum URL
    image_url = args.image_url if args.image_url else f"https://picsum.photos/{width}/{height}?random={time.time()}"

    # Windows specific fix for asyncio
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(send_single_request(
            api_url=args.api_url,
            model=args.model,
            image_url=image_url,
            prompt=args.prompt,
            max_tokens=args.max_tokens
        ))
    except KeyboardInterrupt:
        print("\nRequest cancelled by user.")
