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




