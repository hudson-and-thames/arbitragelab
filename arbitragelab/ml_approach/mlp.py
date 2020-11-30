from keras import backend as K

from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers import Input

from arbitragelab.ml_approach.base import BaseNeuralNetwork


class MultiLayerPerceptron(BaseNeuralNetwork):
    """
    Regression: loss_fn="mean_squared_error", optmz="adam", metrics=["r2_score"]
    num_outputs=1
    
    Classification: loss_fn="categorical_crossentropy", optmz="adam", metrics=["accuracy"]
    num_outputs=num_classes?
    
    """
    
    def __init__(self, frame_size, num_outputs=1, loss_fn="mean_squared_error",
                 optmz="adam", metrics=["accuracy"],
                 hidden_layer_activation_function="relu", output_layer_act_func="linear"):
        
        self.frame_size = frame_size
        self.output_size = num_outputs
        self.loss_fn = loss_fn
        self.optimizer = optmz
        self.metrics = metrics
        self.hidden_layer_activation_function = hidden_layer_activation_function
        self.output_layer_activation_function = output_layer_act_func
    
    def build(self):

        input_layer = Input( (self.frame_size,) )

        hidden_layer = Dense(self.frame_size//2, activation=self.hidden_layer_activation_function)(input_layer)

        output_layer = Dense(self.output_size, activation=self.output_layer_activation_function)(hidden_layer)

        model = Model(inputs=[input_layer], outputs = [output_layer] )

        model.compile(loss=self.loss_fn, optimizer=self.optimizer, metrics=self.metrics) 
        
        self.model = model

        return model
        
#     def coeff_determination(self, y_true, y_pred):
#         SS_res =  K.sum(K.square( y_true-y_pred )) 
#         SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
#         return ( 1 - SS_res/(SS_tot + K.epsilon()) )
