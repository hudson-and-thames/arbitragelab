import matplotlib.pyplot as plt

class BaseNeuralNetwork:
    def summary(self):
        self.model.summary()
        
    def fit(self, *args, **kwargs):
        fitted_model = self.model.fit(*args, **kwargs)
        self.fitted_model = fitted_model
        return fitted_model
        
    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def plot_loss(self):
        plt.plot(self.fitted_model.history['loss'])
