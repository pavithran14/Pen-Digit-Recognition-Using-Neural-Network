# Pen-Digit-Recognition-Using-Neural-Network
A Neural Network Project to recognize the Pen Digit Data. BPN algorithm.


Feedforward Neural network with single hidden layer is considered to solve the problem.  In this problem, totally 125 patternscorresponding to the digits 0 – 4 are there. Each pattern consists of 25 sample patterns.  Each pattern is with 16 components. The data set is selected from the UCI repository.
	In the input layer 17 neurons including one bias neuron is considered. 10  hidden neurons and 5 output neuron is considered for hidden and output layer respectively. If the pattern is of class 0 then first output neuron will be enabled and others are disabled in the target patterns. Backpropagation training algorithm is used to train the network, Learning parameter value is 0.0005 is fixed using trial and error. The number of hidden neurons is also fixed by trial and error. Training is terminated when mean square error 0.05 is reached. 5740 numbers of epochs are needed for convergence for the considered parameter values. Trained network is tested with different patterns of the digits 0 – 4. 93.6% accuracy is obtained in the testing process.
