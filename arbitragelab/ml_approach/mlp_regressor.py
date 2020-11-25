from keras import backend as K

from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers import Input

from arbitragelab.ml_approach.base import BaseNeuralNetwork


class MLPRegressor(BaseNeuralNetwork):
    
    def __init__(self, frame_size, activation_function="relu"):
        
        input_layer = Input( (frame_size,) )

        hidden_layer = Dense(frame_size//2, activation=activation_function)(input_layer)

        output_layer = Dense(1, activation='linear')(hidden_layer)

        model = Model(inputs=[input_layer], outputs = [output_layer] )

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[self.coeff_determination]) 

        self.model = model
        
    def coeff_determination(self, y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred )) 
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )
