import numpy as np
import random


class Mlp():

    def __init__(self, size_layers, reg_lambda=0, bias_flag=True, momentium=1, learning_rate=1.0, decay_rate=0.0):
        '''
        Constructor method. Defines the characteristics of the MLP
        Arguments:
            size_layers : List with the number of Units for:
                [Input, Hidden1, Hidden2, ... HiddenN, Output] Layers.
            act_funtc   : Activation function for all the Units in the MLP
                default = 'sigmoid'
            reg_lambda: Value of the regularization parameter Lambda
                default = 0, i.e. no regularization
            bias: Indicates is the bias element is added for each layer, but the output
        '''
        self.size_layers = size_layers
        self.n_layers = len(size_layers)
        self.lambda_r = reg_lambda
        self.bias_flag = bias_flag
        self.momentium = momentium
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

        # Ramdomly initialize theta (MLP weights)
        self.initialize_theta_weights()

    def train(self, X, Y, epoch_number, iterations=400, reset=False):
        '''
        Given X (feature matrix) and y (class vector)
        Updates the Theta Weights by running Backpropagation N tines
        Arguments:
            X          : Feature matrix [n_examples, n_features]
            Y          : Sparse class matrix [n_examples, classes]
            iterations : Number of times Backpropagation is performed
                default = 400
            reset      : If set, initialize Theta Weights before training
                default = False
        '''
        self.learning_rate /= (1 + self.decay_rate * epoch_number)
        if reset:
            self.initialize_theta_weights()
        for iteration in range(iterations):
            # self.shuffle(X,Y)
            self.gradients = self.backpropagation(X, Y)
            self.gradients_vector = self.unroll_weights(self.gradients)
            self.theta_vector = self.unroll_weights(self.theta_weights)
            self.theta_vector = self.theta_vector - self.gradients_vector
            self.theta_weights = self.roll_weights(self.theta_vector)

    def shuffle(self, a, b):
        if len(a) == len(b):
            start_state = random.getstate()
            random.shuffle(a)
            random.setstate(start_state)
            random.shuffle(b)

    def predict(self, X):
        '''
        Given X (feature matrix), y_hay is computed
        Arguments:
            X      : Feature matrix [n_examples, n_features]
        Output:
            y_hat  : Computed Vector Class for X
        '''
        A, Z = self.feedforward(X)
        Y_hat = A[-1]
        return Y_hat

    def initialize_theta_weights(self):
        '''
        Initialize theta_weights, initialization method depends
        on the Activation Function and the Number of Units in the current layer
        and the next layer.
        The weights for each layer as of the size [next_layer, current_layer + 1]
        '''
        self.theta_weights = []
        size_next_layers = self.size_layers.copy()
        size_next_layers.pop(0)
        roz = 2.0
        for size_layer, size_next_layer in zip(self.size_layers, size_next_layers):

                # Method presented "Understanding the difficulty of training deep feedforward neurla networks"
                # Xavier Glorot and Youshua Bengio, 2010
            # epsilon = 4.0 * np.sqrt(6) / np.sqrt(size_layer + size_next_layer)
                # Weigts from a uniform distribution [-epsilon, epsion]
            if self.bias_flag:
                # theta_tmp = epsilon * ((np.random.rand(size_next_layer, size_layer + 1) * 2.0) - 1)
                theta_tmp = (np.random.rand(size_next_layer, size_layer + 1) * roz) - roz/2
            else:
                # theta_tmp = epsilon * ((np.random.rand(size_next_layer, size_layer) * 2.0) - 1)
                theta_tmp = (np.random.rand(size_next_layer, size_layer) * roz) - roz/2
            self.theta_weights.append(theta_tmp)
        return self.theta_weights

    def backpropagation(self, X, Y):
        g_dz = lambda x: self.sigmoid_derivative(x)

        n_examples = X.shape[0]
        # Feedforward
        A, Z = self.feedforward(X)


        # Backpropagation
        deltas = [None] * self.n_layers
        deltas[-1] = A[-1] - Y
        # For the second last layer to the second one
        for ix_layer in np.arange(self.n_layers - 1 - 1, 0, -1):
            theta_tmp = self.theta_weights[ix_layer]
            if self.bias_flag:
                # Removing weights for bias
                theta_tmp = np.delete(theta_tmp, np.s_[0], 1)
            deltas[ix_layer] = (np.matmul(theta_tmp.transpose(), deltas[ix_layer + 1].transpose())).transpose() * g_dz(
                Z[ix_layer])

        # Compute gradients
        gradients = [None] * (self.n_layers - 1)
        # prev_grands
        for ix_layer in range(self.n_layers - 1):
            grads_tmp = np.matmul(deltas[ix_layer + 1].transpose(), A[ix_layer])
            grads_tmp = grads_tmp / n_examples
            # Learning rate jest here
            # grads_tmp = grads_tmp * self.learning_rate

            if self.bias_flag:
                # Regularize weights, except for bias weigths
                # grads_tmp[:, 1:] = grads_tmp[:, 1:] + (self.lambda_r / n_examples) * self.theta_weights[ix_layer][:, 1:]
                grads_tmp[:, 1:] = grads_tmp[:, 1:] + self.momentium * self.theta_weights[ix_layer][:, 1:]
            else:
                # Regularize ALL weights
                # grads_tmp = grads_tmp + (self.lambda_r / n_examples) * self.theta_weights[ix_layer]

                grads_tmp = grads_tmp + self.momentium * self.theta_weights[ix_layer]
            gradients[ix_layer] = grads_tmp;
            prev_grads = grads_tmp
        return gradients

    def feedforward(self, X):
        g = lambda x: self.sigmoid(x)

        A = [None] * self.n_layers
        Z = [None] * self.n_layers
        input_layer = X

        for ix_layer in range(self.n_layers - 1):
            n_examples = input_layer.shape[0]
            if self.bias_flag:
                # Add bias element to every example in input_layer
                input_layer = np.concatenate((np.ones([n_examples, 1]), input_layer), axis=1)
            A[ix_layer] = input_layer
            # Multiplying input_layer by theta_weights for this layer
            Z[ix_layer + 1] = np.matmul(input_layer, self.theta_weights[ix_layer].transpose())
            # Activation Function
            output_layer = g(Z[ix_layer + 1])
            # Current output_layer will be next input_layer
            input_layer = output_layer

        A[self.n_layers - 1] = output_layer
        return A, Z

    def unroll_weights(self, rolled_data):
        '''
        Unroll a list of matrices to a single vector
        Each matrix represents the Weights (or Gradients) from one layer to the next
        '''
        unrolled_array = np.array([])
        for one_layer in rolled_data:
            unrolled_array = np.concatenate((unrolled_array, one_layer.flatten(1)))
        return unrolled_array

    def roll_weights(self, unrolled_data):
        '''
        Unrolls a single vector to a list of matrices
        Each matrix represents the Weights (or Gradients) from one layer to the next
        '''
        size_next_layers = self.size_layers.copy()
        size_next_layers.pop(0)
        rolled_list = []
        if self.bias_flag:
            extra_item = 1
        else:
            extra_item = 0
        for size_layer, size_next_layer in zip(self.size_layers, size_next_layers):
            n_weights = (size_next_layer * (size_layer + extra_item))
            data_tmp = unrolled_data[0: n_weights]
            data_tmp = data_tmp.reshape(size_next_layer, (size_layer + extra_item), order='F')
            rolled_list.append(data_tmp)
            unrolled_data = np.delete(unrolled_data, np.s_[0:n_weights])
        return rolled_list

    def sigmoid(self, z):
        '''
        Sigmoid function
        z can be an numpy array or scalar
        '''
        result = 1.0 / (1.0 + np.exp(-z))
        return result

    def sigmoid_derivative(self, z):
        '''
        Derivative for Sigmoid function
        z can be an numpy array or scalar
        '''
        result = self.sigmoid(z) * (1 - self.sigmoid(z))
        return result
