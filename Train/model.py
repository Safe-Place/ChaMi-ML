# Import library required
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow import random
from tensorflow.math import abs

# CNN Embedding Layer
def create_embedding():
    # set random seed
    random.set_seed(123)
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

def create_siamese_model():
    random.set_seed(123)
    # Anchor image input
    input_img = Input(shape=(105,105,3), name='Input_img')
    
    # Validation image
    validation_img = Input(shape=(105,105,3), name='Validaiton_img')
    
    # Siamese distance
    embedding = create_embedding()
    siamese_dist = L1_siamese_dist()
    siamese_dist._name = 'Distance'
    distances = siamese_dist(embedding(input_img), embedding(validation_img))
    
    # Classification layer
    classifier = Dense(1, activation='sigmoid', name='Output')(distances)
    
    return Model(inputs=[input_img, validation_img], outputs=classifier, name='SiameseNeuralNetwork')
