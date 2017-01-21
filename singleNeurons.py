"""
Backpropagation:

Virtually any optimizer use backpropagation to adjust weights of the layers of neural network. The name Backpropagation comes as opposite to forward pass - calculating the output of the network given its input. 
The adjustment made to each weight is initially defined as partial derivative of the output w.r.t. the weight. Such partial derivatives are found using chain rule for the weights of hidden layers.

e.g. assume the network has a single input i=1; 
a hidden layer of size 3 with initial weights W1 = -1, 1, 1
and an output layer with weights W2(T) = 0.5, -2, 2.

Assume we optimize the layer to achieve output of 1e12; let us follow the evolution of weights under backpropagation with learning rate = 1.
(singleNeurons.py)

Epoch #0

1: [array([[-1.,  1.,  1.]]]

2: [array([	[ 0.5],
       		[-2. ],
       		[ 2. ]]]
Epoch #1
1: [array([[-0.5, -1. ,  3. ]]]

2: [array([	[-0.5],
       		[-1. ],
       		[ 3. ]]]
Epoch #2
1: [array([[-1., -2.,  6.]], )]

2: [array([	[-1.],
       		[-2.],
       		[ 6.]], )]
Epoch #3

1: [array([[ -2.,  -4.,  12.]], )]

2: [array([	[ -2.],
       		[ -4.],
       		[ 12.]], )]

in this case the analytical formula for the output is: O = SUMMj (W1j*W2j) where j=1..3, which means that 
dO / dW1j = W2j 
dO / dW2j = W1j

This is indeed verified by the output of the test script.

"""





from keras.optimizers import SGD, Adam, Adadelta
from keras.models import Model, load_model # basic class for specifying and training a neural network
import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Reshape


l1_w = np.ndarray(buffer=np.array([-1.,1. ,1.]),shape=(1,3), dtype=float)
l2_w = np.ndarray(buffer=np.array([.25,-2.,2.]),shape=(3,1), dtype=float)

X=np.ndarray(shape=(1,1), dtype=float)
X.fill(1)

Y=np.ndarray(shape=(1,1), dtype=float)
Y.fill(1e12)

inp = Input(shape=(1,))

l1=Dense(3, input_dim=1, bias=False) (inp)
l2=Dense(1, bias=False) (l1)

model = Model(input=inp, output=l2) # To define a model, just specify its input and output layers
optimizer = SGD(lr=1)
model.compile(optimizer=optimizer, loss='mean_absolute_error')

model.layers[1].set_weights((l1_w,))
model.layers[2].set_weights((l2_w,))


for i in range(4):
    print("Epoch #{}".format(i))
    print("-=-" * 40)
    for n,layer in enumerate(model.layers):
        if n != 0:
            print("- -" * 40)
            print("{}: ".format(n) + layer.get_weights().__str__())
    model.fit(X, Y, nb_epoch=1, verbose=False)




