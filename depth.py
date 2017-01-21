"""
A problem frequently mentioned in context of deep neural networks is the problem of vanishing/expanding gradients.

In backpropagation the partial derivatives depend on the multiple of the other weights along the signal channel. 
Each layer means one more member in this multiplication; this can easily lead to two undesired phenomena:

1) vanishing gradients.

Assume we will have again a network taking a fixed input of 1, and desired output of 1e12. 
Let us have 10 layers in this network, each consisting of 3 neurons, except for the last one with a single neuron for output.
If the weights of these neurons are to be initialized to 1e-3, then the partial derivative for each one of those weights should be 
on the order of (1e-3)^(Nlayers - 1) = 1e-27.

This number is tiny, and it would take insanely long amount of time for this network to learn to output high enough numbers by backpropagation.

2) exploding gradients

In trying to fix this problem, we might start increasing the initial weights in our network. In my experiment, I was changing the weights of 8 hidden layers.
For weights [5e-3, 5e-2] the problem remained to be the one of the vanishing gradients. By setting initial weights to 0.5 (5e-1), I got the network to start training;
however on 4th training iteration the loss(error = abs(1e12 - output)) jumps from roughtly 1e12 to Xe17. This means that after coming closer to right answer, the optimization
overshot the target by 5 orders of magnitude. From there on, optimization loses numerical stability. Ofcourse in this demonstration it is also affected by 
impractically large learning rate lr=1. Yet the problem is quite real: deep neural networks are prone to lose numerical stability if no countermeasures are taken.

Countermeasures:
1) (Countering exploding gradients) Cap the gradient (clipnorm / clipvalue attributes in Keras)
2) smart values during initialization

"""


from keras.optimizers import SGD, Adam, Adadelta
from keras.models import Model, load_model, Sequential # basic class for specifying and training a neural network
import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Reshape



l0_w = np.ndarray(buffer=np.array([1e-3,1e-3,1e-3]),shape=(1,3), dtype=float)
l2_w = np.ndarray(buffer=np.array([1e-3,1e-3,1e-3]),shape=(3,1), dtype=float)
l1_w = np.ndarray(shape=(3,3), dtype=float)
l1_w.fill(5e-1)

X=np.ndarray(shape=(100000,1), dtype=float)
X.fill(1)

Y=np.ndarray(shape=(100000,1), dtype=float)
Y.fill(1e12)

model=Sequential()

model.add(Dense(3, input_dim=1, bias=False))
for i in range(8):
    model.add(Dense(3, input_dim=1, bias=False))

model.add(Dense(1, bias=False))

optimizer = SGD(lr=0.01)
model.compile(optimizer=optimizer, loss='mean_absolute_error')


model.layers[0].set_weights((l0_w,))
for i in range(8):
    model.layers[i+1].set_weights((l1_w,))
model.layers[9].set_weights((l2_w,))


for i in range(10):
    print("Epoch #{}".format(i))
    print("-=-" * 40)
    for n,layer in enumerate(model.layers):
        if n != 0:
            print("- -" * 40)
            print("{}: ".format(n) + layer.get_weights().__str__())
    model.fit(X, Y, nb_epoch=1, verbose=True)




