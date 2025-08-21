#!/usr/bin/env python3
"""
Test script to verify voice model integration with audio processing
"""

import numpy as np
import librosa
import tempfile
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

# Custom Attention Layer (same as in main.py)
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

def test_voice_model_loading():
    """Test if the voice model loads correctly"""
    try:
        print("🔍 Testing voice model loading...")
        model = load_model("models/model_finetuned.h5", compile=False, custom_objects={'Attention': Attention})
        print("✅ Voice model loaded successfully!")
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"❌ Failed to load voice model: {e}")
        return False

def test_mfcc_extraction():
    """Test MFCC feature extraction"""
    try:
        print("\n🔍 Testing MFCC extraction...")
        
        # Create a dummy audio signal (1 second of random noise)
        sr = 22050
        duration = 1.0
        y = np.random.randn(int(sr * duration))
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = mfcc.T  # Transpose to (time, features)
        
        print(f"✅ MFCC extraction successful!")
        print(f"📊 MFCC shape: {mfcc.shape}")
        print(f"📊 Expected shape: (time_steps, 40)")
        
        return True
    except Exception as e:
        print(f"❌ MFCC extraction failed: {e}")
        return False

def test_model_prediction():
    """Test model prediction with dummy data"""
    try:
        print("\n🔍 Testing model prediction...")
        
        # Load model
        model = load_model("models/model_finetuned.h5", compile=False, custom_objects={'Attention': Attention})
        
        # Create dummy input (batch_size=1, time_steps=228, features=40, channels=1)
        dummy_input = np.random.randn(1, 228, 40, 1).astype(np.float32)
        
        # Make prediction
        prediction = model.predict(dummy_input, verbose=0)
        
        print(f"✅ Model prediction successful!")
        print(f"📊 Prediction shape: {prediction.shape}")
        print(f"📊 Prediction values: {prediction[0]}")
        print(f"📊 Sum of probabilities: {np.sum(prediction[0]):.6f}")
        
        # Check if probabilities sum to 1
        if abs(np.sum(prediction[0]) - 1.0) < 0.01:
            print("✅ Probabilities sum to 1 (softmax working correctly)")
        else:
            print("⚠️  Probabilities don't sum to 1")
        
        return True
    except Exception as e:
        print(f"❌ Model prediction failed: {e}")
        return False

def test_audio_processing_pipeline():
    """Test the complete audio processing pipeline"""
    try:
        print("\n🔍 Testing complete audio processing pipeline...")
        
        # Create a dummy audio file
        sr = 22050
        duration = 2.0
        y = np.random.randn(int(sr * duration))
        
        # Save as temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            import soundfile as sf
            sf.write(temp_file.name, y, sr)
            temp_file_path = temp_file.name
        
        try:
            # Load audio file
            y_loaded, sr_loaded = librosa.load(temp_file_path, sr=None)
            print(f"✅ Audio loading successful: {y_loaded.shape}, sr={sr_loaded}")
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y_loaded, sr=sr_loaded, n_mfcc=40)
            mfcc = mfcc.T
            
            # Pad or truncate to 228 time steps
            if mfcc.shape[0] < 228:
                padding = np.zeros((228 - mfcc.shape[0], 40))
                mfcc = np.vstack([mfcc, padding])
            else:
                mfcc = mfcc[:228, :]
            
            print(f"✅ MFCC processing successful: {mfcc.shape}")
            
            # Reshape for model input
            mfcc_input = np.expand_dims(mfcc, axis=0)  # Add batch dimension
            mfcc_input = np.expand_dims(mfcc_input, axis=-1)  # Add channel dimension
            
            print(f"✅ Input reshaping successful: {mfcc_input.shape}")
            
            # Load model and predict
            model = load_model("models/model_finetuned.h5", compile=False, custom_objects={'Attention': Attention})
            prediction = model.predict(mfcc_input, verbose=0)
            
            print(f"✅ Complete pipeline successful!")
            print(f"📊 Final prediction: {prediction[0]}")
            
            return True
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        print(f"❌ Audio processing pipeline failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Voice Model Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Voice Model Loading", test_voice_model_loading),
        ("MFCC Extraction", test_mfcc_extraction),
        ("Model Prediction", test_model_prediction),
        ("Complete Pipeline", test_audio_processing_pipeline),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed!")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Voice integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 