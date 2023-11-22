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
        """
        Taking in the 1x3 datapoint and returning loss, y_tilde, x1, z1, x2, z2
        corresponding to the variable names in the onboarding Neural Network pdf
        """

        y = x[2]
        x1 = np.array([x[0], x[1], 1]) #1x3 array
        # print(x1)

        z1 = np.matmul(self.weights1, x1)
        z1 = np.append(z1, [1], axis=0) #1x4 array
        # print(z1)

        x2 = Network.relu(z1) #1x4 array
        # print(x2)

        z2 = np.matmul(self.weights2, x2) #1x1 array
        # print(z2)

        y_tilde = Network.sigmoid(z2[0]) #scalar
        # print(y_tilde)

        loss = Network.MSELoss(y_tilde, y) #scalar
        # print(loss)

        return loss, y_tilde, x1, z1, x2, z2
    
    def backprop(self, x, y, loss, y_tilde, x1, z1, x2, z2, lRate):
        # TODO: Implement back propagation. 
        # Hint: Consider taking in more than just the data point x and its 
        # label y. Again, using values computed in the middle of the feedforward() method
        # will save you repeated computations.
        """
        Taking in datapoint, everything computed in feedforward(), and a learning rate
        """

        """
        Computing change in loss with respect to 2nd weight layer
        """
        dL_yt = 2 * (y_tilde-y)
        dyt_z2 = Network.d_sigmoid(z2[0])
        dz2_W2 = np.transpose([x2]) # 4x1 vector
        # print(dL_yt,dyt_z2,dz2_W2)
        dL_W2 = dL_yt * dyt_z2 * dz2_W2 # 4x1 vector
        # print(dL_W2)
        for i in range(4):
            self.weights2[0][i] -= (lRate * dL_W2[i][0])
        # print(self.weights2)

        """
        Computing change in loss with respect to 1st weight layer
        """
        dz2_x2 = np.copy(self.weights2) #1x4 copy of weights2
        d_relu_z1 = Network.d_relu(z1) #1x4 derivative of relu of z1
        dx2_z1 = np.zeros((4,4))
        for i in range(4):
            dx2_z1[i][i] = d_relu_z1[i]

        """cutoff is the first 3 columns of dx2_z1 which matches the last equation on page 22"""
        cutoff = dx2_z1[:,0:3] # 4x3 matrix
        # print(cutoff)

        # X is a 3x9 matrix
        X = np.array([[x1[0], x1[1], x1[2],0,0,0,0,0,0], [0,0,0,x1[0], x1[1], x1[2],0,0,0], [0,0,0,0,0,0,x1[0], x1[1], x1[2]]])
        
        """
        Comoputing final result by multiplying the components then updating weights1
        """
        matrix1 = dL_yt * dyt_z2 * dz2_x2
        # print(matrix1)
        result = np.matmul(matrix1, cutoff)
        result = np.matmul(result, X) # result is a 1x9 matrix
        # print(result)

        #updating weights1
        for i in range(3):
            for j in range(3):
                self.weights1[i][j] -= (lRate * result[0][i*3+j])
        
        # print(self.weights1)


    def train(self):
        # TODO: Implement the training algorithm, using feedforward() and backprop()

        for epoch in range(1):
            for i in range(len(self.data.dataset)):
                dataPoint = self.data.dataset[i]
                x = [dataPoint[0], dataPoint[1], self.data.labels[i]]

                """Feeding forward and then backproping each datapoint one by one for each epoch"""
                ff = self.feedforward(x)
                self.backprop(x[0:2], x[2], ff[0], ff[1], ff[2], ff[3], ff[4], ff[5], 0.05)
                # print(network.weights1)
                # print(network.weights2)

    @staticmethod
    def sigmoid(x):
        # Takes in single scalar and outputs it's sigmoid
        return 1/(1+math.exp(-x))
    
    @staticmethod
    def d_sigmoid(x):
        #Takes in a single scalar and outputs the derivative of it's sigmoid
        return Network.sigmoid(x)*(1-Network.sigmoid(x))
    
    @staticmethod
    def relu(x):
        # Takes in a 1x4 vector and outputs a 1x4 vector
        res = np.zeros((4))
        for i in range(0, len(x)):
            value = x[i] if x[i] > 0 else 0
            res[i] = value
        return res
    
    @staticmethod
    def d_relu(x):
        # Takes in a 1x4 vector and outputs a 1x4 vector        
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
        # Returns scalar value
        return (y-y_tilde)**2 
    
    @staticmethod
    def d_MSELoss(y_tilde, y):
        # Returns scalar value
        return -2*(y-y_tilde)      
    

if __name__ == "__main__":
    network = Network(100000)
    # x = [network.data.dataset[0][0], network.data.dataset[0][1], network.data.labels[0]]
    # print(x)
    # x = [0.93246305, 1.00330477, 0]
    # print("Datapoint" + str(x))
    # ff = network.feedforward(x)
    # print(ff)
    # network.backprop(x[0:2], x[2], ff[0], ff[1], ff[2], ff[3], ff[4], ff[5], 0.5)

    network.train()

    # Testing model on sample points
    for i in range(len(network.data.dataset)):
        dataPoint = network.data.dataset[i]
        x = [dataPoint[0], dataPoint[1], network.data.labels[i]]
        feedforward = network.feedforward(x)
        print("Loss " + str(feedforward[0]))
        print(x, "Prediction: " + str(feedforward[1]))

    # # testing model on actual XOR inputs
    # x = [0,0,0]
    # feedforward = network.feedforward(x)
    # print("Loss " + str(feedforward[0]))
    # print(x, "Prediction: " + str(feedforward[1]))

    # x = [1,1,0]
    # feedforward = network.feedforward(x)
    # print("Loss " + str(feedforward[0]))
    # print(x, "Prediction: " + str(feedforward[1]))

    # x = [1,0,1]
    # feedforward = network.feedforward(x)
    # print("Loss " + str(feedforward[0]))
    # print(x, "Prediction: " + str(feedforward[1]))
    # x = [0,1,1]
    # feedforward = network.feedforward(x)
    # print("Loss " + str(feedforward[0]))
    # print(x, "Prediction: " + str(feedforward[1]))

