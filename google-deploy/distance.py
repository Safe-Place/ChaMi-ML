import tensorflow as tf

# L1 Distance Layer
class L1_siamese_dist(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
    
    # Similarity calculation    
    def call(self, input_embedding, validation_embedding):
        return abs(input_embedding-validation_embedding)

# L2 Distance Layer
class L2_siamese_dist(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
      super().__init__()
  
  # Similarity calculation    
  def call(self, input_embedding, validation_embedding):
    sum_squared = tf.keras.backend.sum(
        tf.keras.backend.square(input_embedding - validation_embedding),
        axis=1,
        keepdims=True
    )
    return tf.keras.backend.sqrt(tf.keras.backend.maximum(
        sum_squared, 
        tf.keras.backend.epsilon()
    ))