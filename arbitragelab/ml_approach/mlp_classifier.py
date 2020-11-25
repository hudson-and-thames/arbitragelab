from keras import backend as K

from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers import Input

from arbitragelab.ml_approach.base import BaseNeuralNetwork

class MLPClassifier(BaseNeuralNetwork):
    
    def __init__(self, frame_size, activation_function="relu"):
        
        input_layer = Input( (frame_size,) )

        hidden_layer_0 = Dense(frame_size, activation=activation_function)(input_layer)

        hidden_layer = Dense(frame_size//2, activation=activation_function)(hidden_layer_0)

        output_layer = Dense(2, activation='softmax')(hidden_layer)

        model = Model(inputs=[input_layer], outputs = [output_layer] )

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

        self.model = model
