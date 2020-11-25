
from keras.models import Model
from keras.layers.core import Dense, Activation, Lambda
from keras.layers import Input
import matplotlib.pyplot as plt
import tensorflow as tf

from keras import backend as K

from arbitragelab.ml_approach.base import BaseNeuralNetwork

class PiSigmaRegressor(BaseNeuralNetwork):
    
    def __init__(self, frame_size, activation_function="sigmoid"):
    
        input_layer = Input( (frame_size,) )

        second_sigma_layer = Dense(2, activation='linear')(input_layer)

        pi_layer = Lambda(self._pi_this)(second_sigma_layer)

        act_layer = Activation(activation_function)(pi_layer)

        model = Model(inputs=[input_layer], outputs = [act_layer] )

        model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy']) 
    
        self.model = model

    def _pi_this(self, x):
        prod = tf.math.reduce_prod(x, keepdims=True, axis=1)
        return prod

    def coeff_determination(self, y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred )) 
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )

