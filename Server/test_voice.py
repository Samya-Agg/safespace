#!/usr/bin/env python3
"""
Test script for voice audio processing functionality
"""

import requests
import json
import os
from pathlib import Path

def test_voice_only_prediction(audio_file_path):
    """
    Test the voice-only prediction endpoint
    
    Args:
        audio_file_path: Path to the audio file to test
    """
    url = "http://localhost:8000/predict/voice-only"
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        return
    
    print(f"🎤 Testing voice prediction with: {audio_file_path}")
    
    try:
        with open(audio_file_path, 'rb') as f:
            files = {'voice_audio': (os.path.basename(audio_file_path), f, 'audio/wav')}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Voice prediction successful!")
            print(f"📊 Prediction: {result['predictions']['prediction_label']}")
            print(f"🎯 Confidence: {result['predictions']['confidence']:.3f}")
            print(f"📈 Probabilities: Low={result['predictions']['voice_probs'][0]:.3f}, "
                  f"Medium={result['predictions']['voice_probs'][1]:.3f}, "
                  f"High={result['predictions']['voice_probs'][2]:.3f}")
            print(f"📝 Explanation: {result['explanations']['voice']['summary']}")
        else:
            print(f"❌ Voice prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during voice prediction test: {e}")

def test_full_prediction(physio_file_path, dass21_responses, audio_file_path=None):
    """
    Test the full prediction endpoint with all modalities
    
    Args:
        physio_file_path: Path to the physiological CSV file
        dass21_responses: DASS-21 responses as a list
        audio_file_path: Optional path to audio file
    """
    url = "http://localhost:8000/predict"
    
    if not os.path.exists(physio_file_path):
        print(f"❌ Physiological file not found: {physio_file_path}")
        return
    
    print(f"🚀 Testing full prediction with physiological data: {physio_file_path}")
    if audio_file_path:
        print(f"🎤 Including voice audio: {audio_file_path}")
    
    try:
        # Prepare form data
        with open(physio_file_path, 'rb') as f:
            files = {'physiological_file': (os.path.basename(physio_file_path), f, 'text/csv')}
            
            if audio_file_path and os.path.exists(audio_file_path):
                with open(audio_file_path, 'rb') as audio_f:
                    files['voice_audio'] = (os.path.basename(audio_file_path), audio_f, 'audio/wav')
        
        data = {
            'dass21_responses': json.dumps(dass21_responses)
        }
        
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Full prediction successful!")
            print(f"🎯 Final Prediction: {result['predictions']['prediction_label']}")
            print(f"📊 Confidence: {result['predictions']['confidence']:.3f}")
            print(f"📈 Fusion Probabilities: Low={result['predictions']['fusion_probs'][0]:.3f}, "
                  f"Medium={result['predictions']['fusion_probs'][1]:.3f}, "
                  f"High={result['predictions']['fusion_probs'][2]:.3f}")
            
            if result['predictions']['voice_probs']:
                print(f"🎤 Voice Probabilities: Low={result['predictions']['voice_probs'][0]:.3f}, "
                      f"Medium={result['predictions']['voice_probs'][1]:.3f}, "
                      f"High={result['predictions']['voice_probs'][2]:.3f}")
        else:
            print(f"❌ Full prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during full prediction test: {e}")

def test_server_status():
    """Test if the server is running and models are loaded"""
    url = "http://localhost:8000/debug/explanations"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            print("✅ Server is running!")
            print("📊 Model Status:")
            for model, loaded in result['models_loaded'].items():
                status = "✅ Loaded" if loaded else "❌ Not Loaded"
                print(f"  {model}: {status}")
        else:
            print(f"❌ Server status check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")

if __name__ == "__main__":
    print("🧪 Voice Audio Processing Test Suite")
    print("=" * 50)
    
    # Test server status first
    print("\n1. Testing server status...")
    test_server_status()
    
    # Test voice-only prediction (you'll need to provide an audio file)
    print("\n2. Testing voice-only prediction...")
    # Uncomment and modify the path to test with your audio file
    # test_voice_only_prediction("path/to/your/audio.wav")
    print("   ⚠️  Skipped - provide audio file path to test")
    
    # Test full prediction (you'll need to provide files)
    print("\n3. Testing full prediction...")
    # Uncomment and modify paths to test with your files
    # test_full_prediction(
    #     "path/to/physio.csv",
    #     [1, 2, 0, 3, 1, 2, 0],  # DASS-21 responses
    #     "path/to/audio.wav"  # Optional
    # )
    print("   ⚠️  Skipped - provide file paths to test")
    
    print("\n✅ Test suite completed!")
    print("\nTo run tests with your files:")
    print("1. Ensure your FastAPI server is running on localhost:8000")
    print("2. Uncomment the test calls above and provide correct file paths")
    print("3. Run: python test_voice.py") 