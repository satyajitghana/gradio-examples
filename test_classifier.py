from gradio_client import Client, handle_file
import concurrent.futures
import time
from pathlib import Path

def make_prediction(client, image_url):
    """Make a single prediction"""
    try:
        result = client.predict(
            # image_list=handle_file(image_url),
            image=handle_file(image_url),
            api_name="/predict"
        )
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # Single test image URL
    image_url = "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"
    
    # Initialize client
    client = Client("http://127.0.0.1:7860/")
    
    print("\nSending 16 concurrent requests with the same image...")
    start_time = time.time()
    
    # Use ThreadPoolExecutor to send 16 requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(make_prediction, client, image_url) 
            for _ in range(16)
        ]
        
        # Collect results as they complete
        results = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                print(f"Completed prediction {i+1}/16")
            except Exception as e:
                print(f"Error in request {i+1}: {str(e)}")
    
    end_time = time.time()
    
    # Print results
    print(f"\nAll predictions completed in {end_time - start_time:.2f} seconds")
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"\nRequest {i+1}:")
        print(result)

if __name__ == "__main__":
    main() 