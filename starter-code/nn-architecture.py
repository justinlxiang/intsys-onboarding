import numpy as np

class Network(object):

    def __init__(self, num_datapoints : int):

        # TODO: Set up data points and weights 
        raise NotImplementedError
    
    def feedforward(self, x):

        # TODO: Implement a way to feed forward a single data point x
        # Hint: Consider outputting more than just the final approximation y_tilde

        # Perhaps some values in the middle of the network will be useful to compute 
        # things in backprop().
        raise NotImplementedError

    def backprop(self, x, y):

        # TODO: Implement back propagation. 
        # Hint: Consider taking in more than just the data point x and its 
        # label y. Again, using values computed in the middle of the feedforward() method
        # will save you repeated computations.
        raise NotImplementedError

    def train(self):

        # TODO: Implement the training algorithm, using feedforward() and backprop()
        raise NotImplementedError

    @staticmethod
    def sigmoid(x):

        # TODO: Set sigmoid activation function 
        raise NotImplementedError
    
    @staticmethod
    def d_sigmoid(x):

        # TODO: Set derivative of sigmoid activation function 
        raise NotImplementedError
    
    @staticmethod
    def relu(x):

        # TODO: Set ReLU activation function 
        raise NotImplementedError
    
    @staticmethod
    def d_relu(x):

        # TODO: Set derivative of ReLU activation function 
        raise NotImplementedError
    
    @staticmethod
    def MSELoss(y_tilde, y):

        # TODO: Set mean-squared error cost function 
        raise NotImplementedError
    
    @staticmethod
    def d_MSELoss(y_tilde, y):

        # TODO: Set derivative of mean-squared error cost function wrt y_tilde
        raise NotImplementedError       