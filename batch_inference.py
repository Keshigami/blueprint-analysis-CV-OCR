import os
import glob
import requests

def batch_process(input_dir, api_url="http://localhost:8000/analyze"):
    files = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
    print(f"Found {len(files)} files to process.")
    
    for file_path in files:
        print(f"Processing {file_path}...")
        try:
            with open(file_path, "rb") as f:
                response = requests.post(api_url, files={"file": f})
            
            if response.status_code == 200:
                print(f"Success: {response.json()['message']}")
                # print(response.json())
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Failed to connect to API: {e}")

if __name__ == "__main__":
    # Ensure the API is running before executing this
    # For demo purposes, we just point to the sample directory
    batch_process("sample_blueprints")
