# Import library required
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, Input, Flatten, Dropout
from tensorflow import random
from tensorflow.math import abs
from tensorflow.keras import backend as k

#-------------
# For Model v1
#-------------

# CNN Embedding Layer
def create_embedding():
    # set random seed
    #random.set_seed(123)
    cnn = Sequential([
        # Input and feature maps
        Conv2D(64, (10,10), activation='relu', input_shape=(105,105,3), name='Conv1'),
        MaxPooling2D(64, (2,2), padding='same', name='Pool1'),
        Conv2D(128, (7,7), activation='relu', name='Conv2'),
        MaxPooling2D(64, (2,2), padding='same', name='Pool2'),
        Conv2D(128, (4,4), activation='relu', name='Conv3'),
        MaxPooling2D(64, (2,2), padding='same', name='Pool3'),
        Conv2D(256, (4,4), activation='relu', name='Conv4'),
        
        # Feature vector
        Flatten(name='FCN'),
        Dense(4096, activation='sigmoid', name='Dense')
    ])
    cnn._name= 'Embedding'
    return cnn

# L1 Distance Layer
class L1_siamese_dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    
    # Similarity calculation    
    def call(self, input_embedding, validation_embedding):
        return abs(input_embedding-validation_embedding)

#-------------
# For Model v2
#-------------

# CNN Embedding Layer
def create_embedding_v2():
  cnn = Sequential([
        Conv2D(96, (11,11), activation='relu', padding='same', input_shape=(105, 105, 3), name='Conv1'),
        MaxPooling2D(pool_size=(2,2), name='Pool1'),
        Dropout(0.3),

        Conv2D(256, (5,5), activation='relu', padding='same', name='Conv2'),
        MaxPooling2D(pool_size=(2,2), name='Pool2'),
        Dropout(0.3),

        Conv2D(384, (3,3), activation='relu', padding='same', name='Conv3'),
        MaxPooling2D(pool_size=(2,2), name='Pool3'),
        Dropout(0.3),
        
        GlobalAveragePooling2D(),
        Dense(1024),
        Dense(128)
    ])
  cnn._name= 'Embedding'
  return cnn

# L2 Distance Layer
class L2_siamese_dist(Layer):
  def __init__(self, **kwargs):
      super().__init__()
  
  # Similarity calculation    
  def call(self, input_embedding, validation_embedding):
    sum_squared = k.sum(k.square(input_embedding - validation_embedding), axis=1, keepdims=True)
    return k.sqrt(k.maximum(sum_squared, k.epsilon()))

# Siamese neural network model
def create_siamese_model(embedding, distance):
    #random.set_seed(123)
    # Anchor image input
    input_img = Input(shape=(105,105,3), name='Input_img')
    
    # Validation image
    validation_img = Input(shape=(105,105,3), name='Validaiton_img')
    
    # Siamese distance
    siamese_dist = distance()
    siamese_dist._name = 'Distance'
    distances = siamese_dist(embedding(input_img), embedding(validation_img))
    
    # Classification layer
    classifier = Dense(1, activation='sigmoid', name='Output')(distances)
    
    return Model(inputs=[input_img, validation_img], outputs=classifier, name='SiameseNeuralNetwork')
