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




