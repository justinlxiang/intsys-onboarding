import numpy as np
import random

class DataGenerator(object):

    """
    An object Obj of the DataGenerator class defines fields "Obj.dataset",
    which is a list of np.array([x, y]) values, and "Obj.labels", which is itself
    a numpy array object, that looks something like: np.array([1, 0, 1, 1, ..., 0, 1]).

    In particular, the ith element of Obj.dataset is a length-2 numpy array representing
    a data point, and the ith element of Obj.labels is the "classification" of this
    data point.

    """

    def __init__(self, num_datapoints):

        self.num_datapoints = num_datapoints
        self.dataset, self.labels = DataGenerator.create_dataset(
            self.num_datapoints)

    @staticmethod
    def create_datapoint():
        """ Creates a datapoint with noise and corresponding XOR label """

        dp = []
        unaltered = []
        for _ in range(2):

            binary = random.randint(0, 1)
            noise = np.random.normal(loc=0, scale=0.1) # stdev = 0.1, mean = 0

            dp.append(binary + noise)
            unaltered.append(binary)

        return (np.array(dp), 1) if 0 in unaltered and 1 in unaltered else (np.array(dp), 0)

    @staticmethod
    def create_dataset(num_dps):
        """ Creates a dataset of XOR datapoints and labels with noise """

        dataset = []
        labels = np.array([])

        while num_dps > 0:

            dp, label = DataGenerator.create_datapoint()
            dataset.append(dp)
            labels = np.append(labels, label)

            num_dps -= 1

        return dataset, labels


if __name__ == "__main__":

    # Test that data generation of XOR points works correctly
    obj = DataGenerator(10)
    print(obj.dataset)
    print(obj.labels)
