#!/usr/bin/env python3
"""
Simple test to verify voice model loading
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import numpy as np

# Custom Attention Layer
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

def test_model_loading():
    """Test if the voice model loads correctly"""
    try:
        print("🔍 Testing voice model loading...")
        
        # Load model with custom attention layer
        model = load_model("models/model_finetuned.h5", compile=False, custom_objects={'Attention': Attention})
        
        print("✅ Voice model loaded successfully!")
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
        
        # Test with dummy input
        dummy_input = np.random.randn(1, 228, 40, 1).astype(np.float32)
        prediction = model.predict(dummy_input, verbose=0)
        
        print("✅ Model prediction successful!")
        print(f"📊 Prediction shape: {prediction.shape}")
        print(f"📊 Prediction values: {prediction[0]}")
        print(f"📊 Sum of probabilities: {np.sum(prediction[0]):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load voice model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Voice Model Loading Test")
    print("=" * 30)
    
    success = test_model_loading()
    
    if success:
        print("\n🎉 Model loading test passed!")
    else:
        print("\n❌ Model loading test failed!") 