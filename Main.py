import random

import Layer
import numpy as np
from parsing import *
from mlp import *
import mlp_v2
import pickle
import gzip
# from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time

lolol = [1, 0, 0, 0]
trainingSet = [[[1, 0, 0, 0], [1, 0, 0, 0]],
               [[0, 1, 0, 0], [0, 1, 0, 0]],
               [[0, 0, 1, 0], [0, 0, 1, 0]],
               [[0, 0, 0, 1], [0, 0, 0, 1]]]
numOfNeurons = []
'''numLayers = int(input("Podaj ilosc warstw w sieci : "))
for i in range(numLayers):
    numOfNeurons.append(int(input("Podaj ilosc neuronow w warstwie " + str(i + 1) + " : ")))

# to to na razie sugestia tylko potem zobaczymy jak to zrobic na koniec
print("W jakim trybie ma pracować siec : ")
print("1) Tryb nauki")
print("2) Tryb testowania")

print("Czy uzywamy biasa  : ")
print("1) Tak")
print("2) Nie")

print("Niezmienna czy losową kolejność prezentowania wzorców treningowych? : ")
print("1) Niezmienna")
print("2) Losowa")

# P = Parsing()
# P.init_parser();

siusiak = Layer.Layer(4, lolol, 2)
costam = siusiak.out()
for i in costam:
    print(i)
# for i in siusiak.neurons:
# print(i.value)'''

'''layers = [0,2345,13,667]
size_next_layers = layers.copy()
size_next_layers.pop(0)

print(zip(layers,size_next_layers))
print(set(zip(layers,size_next_layers)))
# print(layers)'''



def load_mnist():
    # Import MNIST data
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    # Training data, only
    X = valid_set[0]
    y = valid_set[1]

    # change y [1D] to Y [2D] sparse array coding class
    n_examples = len(y)
    labels = np.unique(y)
    Y = np.zeros((n_examples, len(labels)))
    for ix_label in range(len(labels)):
        # Find examples with with a Label = lables(ix_label)
        ix_tmp = np.where(y == labels[ix_label])[0]
        Y[ix_tmp, ix_label] = 1

    return X, Y, labels, y

epochs = 100
loss = np.zeros([epochs,1])

X, Y, labels, y  = load_mnist()

# X = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
# Y = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

for i in range(1000):
    print(Y[i])
tic = time.time()

# Creating the MLP object initialize the weights
mlp_classifier = mlp_v2.Mlp(size_layers = [784, 100, 10],
                            reg_lambda  = 0,
                            bias_flag   = False)



for ix in range(epochs):
    # mlp_classifier.shuffle(X, Y)
    mlp_classifier.train(X, Y, ix, 1)
    print("kutas")
    # Po każdym nauczeniu sprawdzamy
    Y_hat = mlp_classifier.predict(X)
    # wyliczamy błąd dla danego przejścia
    loss[ix] = (0.5)*np.square(Y_hat - Y).mean()

print(str(time.time() - tic) + ' s')

plt.figure()
ix = np.arange(epochs)
plt.plot(ix, loss)
plt.show()

Y_hat = mlp_classifier.predict(X)
y_tmp = np.argmax(Y_hat, axis=1)
y_hat = labels[y_tmp]

acc = np.mean(1 * (y_hat == y))
print('Training Accuracy: ' + str(acc*100))

