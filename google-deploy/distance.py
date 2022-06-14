import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as k

# L2 Distance Layer
class L2_siamese_dist(Layer):
  def __init__(self, **kwargs):
      super().__init__()

  # Similarity calculation
  def call(self, input_embedding, validation_embedding):
    sum_squared = k.sum(
        k.square(input_embedding - validation_embedding),
        axis=1,
        keepdims=True
    )
    return k.sqrt(k.maximum(
        sum_squared,
        k.epsilon()
    ))
