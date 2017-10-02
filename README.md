# casestudyNN

experiments in learning Machine Learning / Deep Learning

Experiment1: single neurons:

Assumptions about how a MLP should process forward prop and backprop are tested on small number of neurons and simple numbers.

Backpropagation:
Virtually any optimizer use backpropagation to adjust weights of the layers of neural network. The name Backpropagation comes as opposite to forward pass - calculating the output of the network given its input. 
The adjustment made to each weight is initially defined as partial derivative of the output w.r.t. the weight. Such partial derivatives are found using chain rule for the weights of hidden layers.
e.g. assume the network has a single input i=1; 
a hidden layer of size 3 with initial weights W1 = -1, 1, 1
and an output layer with weights W2(T) = 0.5, -2, 2.
Assume we optimize the layer to achieve output of 1e12; let us follow the evolution of weights under backpropagation with learning rate = 1.
(singleNeurons.py)



Experiment2: depth:

Problems of vanishing and exploding gradients are investigated.
