# casestudyNN

experiments in learning Machine Learning / Deep Learning

# *Experiment1: single neurons:*

Assumptions about how a MLP should process forward prop and backprop are tested on small number of neurons and simple numbers.

Backpropagation:
Virtually any optimizer use backpropagation to adjust weights of the layers of neural network. The name Backpropagation comes as opposite to forward pass - calculating the output of the network given its input. 
The adjustment made to each weight is initially defined as partial derivative of the output w.r.t. the weight. Such partial derivatives are found using chain rule for the weights of hidden layers.

(singleNeurons.py)



# *Experiment2: Deep Learning:*

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

(depth.py)
