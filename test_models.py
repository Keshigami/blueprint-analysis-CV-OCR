"""
Comprehensive Model Testing Suite
Compare enhanced models vs baseline
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import json
import time

def load_enhanced_model(checkpoint_path="blueprint_classifier_enhanced.pth"):
    """Load the enhanced EfficientNet-B3 model"""
    print("Loading Enhanced EfficientNet-B3 Model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model architecture
    model = models.efficientnet_b3(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2)
    )
    
    # Load weights
    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        model.to(device)
        print(f"✅ Loaded from {checkpoint_path}")
        return model, device
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None, device

def load_baseline_model(checkpoint_path="blueprint_classifier_english.pth"):
    """Load the baseline ResNet18 model"""
    print("\nLoading Baseline ResNet18 Model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create baseline architecture
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    
    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        model.to(device)
        print(f"✅ Loaded from {checkpoint_path}")
        return model, device
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None, device

def test_model(model, test_images, device, model_name="Model"):
    """Test model on sample images"""
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    results = []
    total_time = 0
    
    print(f"\nTesting {model_name} on {len(test_images)} images...")
    
    with torch.no_grad():
        for img_path in test_images:
            # Load and preprocess
            img = Image.open(img_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)
            
            # Inference
            start_time = time.time()
            outputs = model(input_tensor)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Get prediction
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            results.append({
                'image': img_path.name,
                'prediction': 'floorplan' if predicted.item() == 1 else 'other',
                'confidence': confidence.item(),
                'inference_time': inference_time
            })
    
    avg_time = total_time / len(test_images) if test_images else 0
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
    
    print(f"  Average Confidence: {avg_confidence*100:.2f}%")
    print(f"  Average Inference Time: {avg_time*1000:.2f}ms")
    
    return results, avg_confidence, avg_time

def compare_models():
    """Compare enhanced vs baseline models"""
    print("="*70)
    print("MODEL COMPARISON TEST")
    print("="*70)
    
    # Get test images
    test_dir = Path("english_data/floorplan_cad/images")
    test_images = list(test_dir.glob("*.jpg"))[:20]  # Test on 20 images
    
    if not test_images:
        print("❌ No test images found!")
        return
    
    print(f"\n📊 Testing on {len(test_images)} floor plan images")
    
    # Load models
    enhanced_model, device = load_enhanced_model()
    baseline_model, _ = load_baseline_model()
    
    results_comparison = {
        'test_images': len(test_images),
        'models': {}
    }
    
    # Test enhanced model
    if enhanced_model:
        enhanced_results, enhanced_conf, enhanced_time = test_model(
            enhanced_model, test_images, device, "Enhanced EfficientNet-B3"
        )
        results_comparison['models']['enhanced'] = {
            'architecture': 'EfficientNet-B3',
            'avg_confidence': enhanced_conf,
            'avg_inference_time_ms': enhanced_time * 1000,
            'results': enhanced_results
        }
    
    # Test baseline model  
    if baseline_model:
        baseline_results, baseline_conf, baseline_time = test_model(
            baseline_model, test_images, device, "Baseline ResNet18"
        )
        results_comparison['models']['baseline'] = {
            'architecture': 'ResNet18',
            'avg_confidence': baseline_conf,
            'avg_inference_time_ms': baseline_time * 1000,
            'results': baseline_results
        }
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    if enhanced_model and baseline_model:
        conf_improvement = (enhanced_conf - baseline_conf) * 100
        time_diff = (enhanced_time - baseline_time) * 1000
        
        print(f"\n📈 Confidence Improvement: {conf_improvement:+.2f}%")
        print(f"⏱️  Speed Difference: {time_diff:+.2f}ms")
        
        if enhanced_conf > baseline_conf:
            print(f"✅ Enhanced model is MORE confident")
        if enhanced_time < baseline_time:
            print(f"✅ Enhanced model is FASTER")
    
    # Save results
    with open("model_comparison_results.json", 'w') as f:
        json.dump(results_comparison, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to model_comparison_results.json")
    print("="*70)
    
    return results_comparison

if __name__ == "__main__":
    compare_models()
