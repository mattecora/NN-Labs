import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self, weights, actfuncs, derfuncs):
        self.weights = weights
        self.actfuncs = actfuncs
        self.derfuncs = derfuncs
    
    def feedforward(self, x):
        v = []
        y = [x.reshape(len(x), 1)]

        # Compute local fields and outputs
        for i in range(len(self.weights)):
            v.append(self.weights[i] @ np.concatenate(([1], y[i]), axis=None).reshape(len(y[i]) + 1, 1))
            y.append(self.actfuncs[i](v[i]).reshape(len(v[i]), 1))

        return v, y

    def backpropagate(self, x, d):
        # Compute local fields and outputs
        v, y = self.feedforward(x)
        delta = [0] * len(self.weights)
        dEW = [0] * len(self.weights)

        # Compute delta signals
        delta[len(self.weights) - 1] = (d - y[len(self.weights)]) * self.derfuncs[len(self.weights) - 1](v[len(self.weights) - 1])
        for i in reversed(range(len(self.weights) - 1)):
            delta[i] = self.weights[i + 1].transpose()[1:] @ delta[i + 1] * self.derfuncs[i](v[i])

        # Compute the gradient
        for i in range(len(self.weights)):
            dEW[i] = -delta[i] @ np.concatenate(([1], y[i]), axis=None).reshape(1, len(y[i]) + 1)

        return dEW

    def errfunc(self, data, labels):
        # Compute the value of the error function
        return sum([(np.linalg.norm(labels[i] - self.feedforward(data[i])[1][-1])) ** 2 for i in range(len(data))]) / len(data)

    def train(self, train_data, train_labels, test_data, test_labels, eta, eps, epoch_limit, save_suffix):
        # Run the gradient descent method
        epochs = 0

        errors = [self.errfunc(train_data, train_labels)]
        training_accuracy = [self.test(train_data, train_labels)]
        test_accuracy = [self.test(test_data, test_labels)]
        print("Epoch {}: eta = {}, err = {}, train = {}, test = {}".format(epochs, eta, errors[epochs], training_accuracy[epochs], test_accuracy[epochs]))

        while epochs < epoch_limit and errors[epochs] >= eps:
            # Increment epochs
            epochs = epochs + 1

            # Update weights
            for i in range(len(train_data)):
                dEW = self.backpropagate(train_data[i], train_labels[i])
                for j in range(len(dEW)):
                    self.weights[j] = self.weights[j] - eta * dEW[j]

            # Register the error
            errors.append(self.errfunc(train_data, train_labels))
            training_accuracy.append(self.test(train_data, train_labels))
            test_accuracy.append(self.test(test_data, test_labels))
            print("Epoch {}: eta = {}, err = {}, train = {}, test = {}".format(epochs, eta, errors[epochs], training_accuracy[epochs], test_accuracy[epochs]))

            # Decrease eta if necessary
            if errors[epochs] > errors[epochs - 1]:
                eta = 0.9 * eta

        if save_suffix is not None:
            # Save weights and errors to file
            #np.save("weights_{}.npy".format(save_suffix), self.weights)
            np.save("errors_{}.npy".format(save_suffix), errors)
            np.save("trainacc_{}.npy".format(save_suffix), training_accuracy)
            np.save("testacc_{}.npy".format(save_suffix), test_accuracy)

        return epochs, errors, training_accuracy, test_accuracy

    def test(self, test_data, test_labels):
        # Count classification errors on the test set
        return sum([
            1 if np.argmax(test_labels[i]) == np.argmax(self.feedforward(test_data[i])[1][-1])
            else 0
            for i in range(len(test_data))
        ]) / len(test_data)