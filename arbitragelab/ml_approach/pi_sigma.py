
from keras.models import Model
from keras.layers.core import Dense, Activation, Lambda
from keras.layers import Input
import tensorflow as tf

from keras import backend as K

from arbitragelab.ml_approach.base import BaseNeuralNetwork

class PiSigmaNeuralNetwork(BaseNeuralNetwork):
    
    def __init__(self, frame_size, num_outputs=1, loss_fn="mean_squared_error",
                 optmz="sgd", metrics=["accuracy"],
                 hidden_layer_activation_function="linear", output_layer_act_func="sigmoid"):
        
        self.frame_size = frame_size
        self.output_size = num_outputs
        self.loss_fn = loss_fn
        self.optimizer = optmz
        self.metrics = metrics
        self.hidden_layer_activation_function = hidden_layer_activation_function
        self.output_layer_activation_function = output_layer_act_func
        
    def build(self):
    
        input_layer = Input( (frame_size,) )

        second_sigma_layer = Dense(2, activation=self.hidden_layer_activation_function)(input_layer)

        pi_layer = Lambda(self._pi_this)(second_sigma_layer)

        act_layer = Activation(self.output_layer_activation_function)(pi_layer)

        model = Model(inputs=[input_layer], outputs = [act_layer] )

        model.compile(loss=self.loss_fn, optimizer=self.optimizer, metrics=self.metrics) 
    
        self.model = model
        
        return model

    def _pi_this(self, x):
        prod = tf.math.reduce_prod(x, keepdims=True, axis=1)
        return prod

    def coeff_determination(self, y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred )) 
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )

