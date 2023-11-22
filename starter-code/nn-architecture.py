import numpy as np
from data_generator import DataGenerator
import math

class Network(object): 

    def __init__(self, num_datapoints : int):

        # TODO: Set up data points and weights 
        self.data = DataGenerator(num_datapoints)
        self.weights1 = np.ones((3,3), dtype = float)
        self.weights1 = 0.001 * self.weights1
        self.weights2 = np.ones((1,4), dtype=float)
        self.weights2 = 0.001 * self.weights2

    
    def feedforward(self, x):

        # TODO: Implement a way to feed forward a single data point x
        # Hint: Consider outputting more than just the final approximation y_tilde

        # Perhaps some values in the middle of the network will be useful to compute 
        # things in backprop().
        y = x[2]
        x1 = np.array([x[0], x[1], 1]) 
        # print(x1)
        z1 = np.matmul(self.weights1, x1)
        z1 = np.append(z1, [1], axis=0)
        # print(z1)
        x2 = Network.relu(z1)
        # print(x2)
        z2 = np.matmul(self.weights2, x2)
        # print(z2)
        y_tilde = Network.sigmoid(z2[0])
        # print(y_tilde)
        loss = Network.MSELoss(y_tilde, y)
        # print(loss)

        return loss, y_tilde, x1, z1, x2, z2
    
    def backprop(self, x, y, loss, y_tilde, x1, z1, x2, z2, lRate):

        # TODO: Implement back propagation. 
        # Hint: Consider taking in more than just the data point x and its 
        # label y. Again, using values computed in the middle of the feedforward() method
        # will save you repeated computations.
        dL_yt = 2 * (y_tilde-y)
        dyt_z2 = Network.d_sigmoid(z2[0])
        dz2_W2 = np.transpose([x2])
        # print(dL_yt,dyt_z2,dz2_W2)
        dL_W2 = dL_yt * dyt_z2 * dz2_W2
        # print(dL_W2)
        
        for i in range(4):
            self.weights2[0][i] -= (lRate * dL_W2[i][0])
        # print(self.weights2)

        dz2_x2 = np.copy(self.weights2)
        d_relu_z1 = Network.d_relu(z1)
        dx2_z1 = np.zeros((4,4))
        for i in range(4):
            dx2_z1[i][i] = d_relu_z1[i]

        cutoff = dx2_z1[:,0:3]
        # print(cutoff)

        X = np.array([[x1[0], x1[1], x1[2],0,0,0,0,0,0], [0,0,0,x1[0], x1[1], x1[2],0,0,0], [0,0,0,0,0,0,x1[0], x1[1], x1[2]]])
        
        matrix1 = dL_yt * dyt_z2 * dz2_x2
        # print(matrix1)
        result = np.matmul(matrix1, cutoff)
        result = np.matmul(result, X)
        # print(result)

        for i in range(3):
            for j in range(3):
                self.weights1[i][j] -= (lRate * result[0][i*3+j])
        
        # print(self.weights1)


    def train(self):

        # TODO: Implement the training algorithm, using feedforward() and backprop()

        for epoch in range(100):
            for i in range(len(self.data.dataset)):
                dataPoint = self.data.dataset[i]
                x = [dataPoint[0], dataPoint[1], self.data.labels[i]]
                ff = self.feedforward(x)
                self.backprop(x[0:2], x[2], ff[0], ff[1], ff[2], ff[3], ff[4], ff[5], 0.05)
                # print(network.weights1)
                # print(network.weights2)

    @staticmethod
    def sigmoid(x):

        # TODO: Set sigmoid activation function 
        return 1/(1+math.exp(-x))
    
    @staticmethod
    def d_sigmoid(x):

        # TODO: Set derivative of sigmoid activation function 
        return Network.sigmoid(x)*(1-Network.sigmoid(x))
    
    @staticmethod
    def relu(x):

        # TODO: Set ReLU activation function 
        res = np.zeros((4))
        for i in range(0, len(x)):
            value = x[i] if x[i] > 0 else 0
            res[i] = value
        return res
    
    @staticmethod
    def d_relu(x):

        # TODO: Set derivative of ReLU activation function 
        res = np.zeros((4))
        for i in range(0,len(x)):
            if x[i] > 0:
                value = 1
            elif x[i] <= 0:
                value = 0
            res[i] = value
        return res
    
    @staticmethod
    def MSELoss(y_tilde, y):

        # TODO: Set mean-squared error cost function 
        return (y-y_tilde)**2 
    
    @staticmethod
    def d_MSELoss(y_tilde, y):

        # TODO: Set derivative of mean-squared error cost function wrt y_tilde
        return -2*(y-y_tilde)      
    

if __name__ == "__main__":
    network = Network(1000)
    # x = [network.data.dataset[0][0], network.data.dataset[0][1], network.data.labels[0]]
    # print(x)
    # x = [0.93246305, 1.00330477, 0]
    # print("Datapoint" + str(x))
    # ff = network.feedforward(x)
    # print(ff)
    # network.backprop(x[0:2], x[2], ff[0], ff[1], ff[2], ff[3], ff[4], ff[5], 0.5)

    network.train()

    #testing points
    for i in range(len(network.data.dataset)):
        dataPoint = network.data.dataset[i]
        x = [dataPoint[0], dataPoint[1], network.data.labels[i]]
        feedforward = network.feedforward(x)
        print("Loss " + str(feedforward[0]))
        print(x, "Prediction: " + str(feedforward[1]))

