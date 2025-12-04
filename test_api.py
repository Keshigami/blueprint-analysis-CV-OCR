"""
Direct API test to verify backend is working
"""
import requests
import os

def test_analyze():
    """Test the /analyze endpoint"""
    print("Testing /analyze endpoint...")
    
    # Use a sample image
    test_image = "english_data/floorplan_cad/images/floorplan_0000.jpg"
   
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    url = "http://localhost:8000/analyze"
    with open(test_image, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        data = response.json()
        if data.get('status') == 'success':
            print(f"✅ OCR found {data.get('num_text_regions', 0)} text regions")
            return True
    
    print("❌ /analyze failed")
    return False

def test_segment():
    """Test the /segment endpoint"""
    print("\nTesting /segment endpoint...")
    
    test_image = "english_data/floorplan_cad/images/floorplan_0000.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    url = "http://localhost:8000/segment"
    with open(test_image, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    
    print(f"Status: {response.status_code}")
    response_data = response.json()
    print(f"Response keys: {response_data.keys()}")
    
    if response.status_code == 200:
        data = response_data
        if data.get('status') == 'success':
            print(f"✅ SAM2 found {data.get('num_masks', 0)} masks")
            return True
        else:
            print(f"Error: {data.get('message')}")
            if 'detail' in data:
                print(f"Detail: {data['detail']}")
    
    print("❌ /segment failed")
    return False

def test_analyze_complete():
    """Test the /analyze_complete endpoint"""
    print("\nTesting /analyze_complete endpoint...")
    
    test_image = "english_data/floorplan_cad/images/floorplan_0000.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    url = "http://localhost:8000/analyze_complete"
    with open(test_image, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    
    print(f"Status: {response.status_code}")
    response_data = response.json()
    print(f"Response keys: {response_data.keys()}")
    
    if response.status_code == 200:
        data = response_data
        if data.get('status') == 'success':
            print(f"✅ Combined analysis successful")
            print(f"   OCR: {len(data.get('ocr_results', []))} regions")
            print(f"   SAM2: {data.get('segmentation', {}).get('num_masks', 0)} masks")
            return True
        else:
            print(f"Error: {data.get('message')}")
            if 'detail' in data:
                print(f"Detail: {data['detail']}")
    
    print("❌ /analyze_complete failed")
    return False

if __name__ == "__main__":
    print("="*60)
    print("API Endpoint Testing")
    print("="*60)
    
    results = {
        'analyze': test_analyze(),
        'segment': test_segment(), 
        'analyze_complete': test_analyze_complete()
    }
    
    print("\n" + "="*60)
    print("Results:")
    for endpoint, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {endpoint}: {status}")
    print("="*60)
