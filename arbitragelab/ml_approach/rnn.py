from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Input, LSTM, Dense

from arbitragelab.ml_approach.base import BaseNeuralNetwork

class RecurrentNeuralNetwork(BaseNeuralNetwork):
    """
    Regression: loss_fn="mean_squared_error", optmz="adam", metrics=["r2_score"]
    num_outputs=1
    
    Classification: loss_fn="categorical_crossentropy", optmz="adam", metrics=["accuracy"]
    num_outputs=num_classes?
    
    """
    
    def __init__(self, input_shape, num_outputs=1, loss_fn="mean_squared_error",
                 optmz="adam", metrics=["accuracy"],
                 hidden_layer_activation_function="relu", output_layer_act_func="linear"):
        
        self.input_shape = input_shape
        self.output_size = num_outputs
        self.loss_fn = loss_fn
        self.optimizer = optmz
        self.metrics = metrics
        self.hidden_layer_activation_function = hidden_layer_activation_function
        self.output_layer_activation_function = output_layer_act_func
    
    def build(self):
        
        input_layer = Input( self.input_shape )

        hidden_layer = LSTM(50, activation=self.hidden_layer_activation_function,
                           input_shape=self.input_shape)(input_layer)
        
        output_layer = Dense(self.output_size, activation=self.output_layer_activation_function)(hidden_layer)
                                 
        model = Model(inputs=[input_layer], outputs = [output_layer] )

        model.compile(loss=self.loss_fn, optimizer=self.optimizer, metrics=self.metrics) 
        
        self.model = model

        return model
