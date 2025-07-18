from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow as tf

class Attention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros", trainable=True)
        super().build(input_shape)
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        return tf.keras.backend.sum(x * a, axis=1)
    def get_config(self):
        return super().get_config()

# Load the .h5 model
model = load_model("models/Voice.h5", custom_objects={"Attention": Attention})

# Save in SavedModel format (for TFServing or TFSMLayer)
model.export("models/voice_savedmodel_tf")
print("✅ Saved successfully as SavedModel")
