# Pattern Classification-Backpropagation Algorithm

Scratch implmentation of Backpropagation Algorithm for Digit/Pattern Classification on MNIST dataset without any use of existing Machine Learning libraries 

Size of the training data set for pattern classification: 60000 * 784

Each input data size: 28 * 28 = 784

Size of the test data set to test patterns: 10000 * 784

## Steps followed:

### Selection of number of inputs and output layer neurons
Since each input data size is 784, I took number of inputs for the Neural Network as 784 and thus the input x dimension is 784 * 1. Now as we need to classify digits here which ranges from 0 to 9, so I considered number of output neurons as 10. 

### Selection of number of hidden layers and number of neurons in hidden layer
Deciding the number of neurons in the hidden layers is a very important part of deciding your overall neural network architecture. Though these layers do not directly interact with the external environment, they have a tremendous influence on the final output. Both the number of hidden layers and the number of neurons in each of these hidden layers must be carefully considered.
Using too few neurons in the hidden layers will result in something called underfitting. Underfitting occurs when there are too few neurons in the hidden layers to adequately detect the signals in a complicated data set.
Using too many neurons in the hidden layers can also result in several problems. 
First, too many neurons in the hidden layers may result in overfitting. Overfitting occurs when the neural network has so much information processing capacity that the limited amount of information contained in the training set is not enough to train all the neurons in the hidden layers. A second problem can occur even when the training data is sufficient. An inordinately large number of neurons in the hidden layers can increase the time it takes to train the network. The amount of training time can increase to the point that it is impossible to adequately train the neural network. 
As per the Heaton Research given in below mentioned URL, generally 2 hidden layers will enable the network to model any arbitrary function.

### Reference: http://www.heatonresearch.com/2017/06/01/hidden-layers.html

#### Deciding factor that I followed for the selection of number of hidden layer and number of neurons in the hidden layer:
1.	Take neuron numbers sufficient enough to train the network and less enough to reduce the training time of the network. Sometimes taking huge number of neurons can increase the training time too much that it becomes impossible to train the network.
2.	Avoid overfitting or under-fitting by taking number of neurons in the range between number of inputs and number of outputs 
Considering above points, I chose 1 hidden layer initially for the design of this neural network. Also, to make to the program work efficiently and to reduce the training time, I took number of neurons in the hidden layer as variable which can be changed to improve the performance as and when required. Initially I took 50 neurons in the hidden layer.

### Selection of transfer function 
Since our task is pattern recognition using backpropagation of Error, we should use odd activation function such as sigmoid function to design the network. Since for back propagation, we need function which gives smooth curve, the hyperbolic tangent (tanh) can be used. tanh is also a ‘sigmoid function’ which generates values between -1.0 and +1.0.

### Selection of eta or learning rate: 
Since we have huge number of inputs, we should try to take small learning rate. Initially I took learning rate as 0.1. But with this learning rate, I was not getting convergence. So, I tried smaller learning late as small as 0.01 which made the network to converge for the output. Also I am changing the learning rate dynamically to smaller value whenever error rate increases to make sure network should not diverge.

### Selection of weights range for hidden layer and output layer neurons
Weights are one of the important factor for a neural network to converge for the output. Weights should not be too high, this can make the network diverge easily. Most of the time, we have to do some trial and error to select proper initial range for weights. I chose initially weights range from -1 to 1, with which I was getting large induced local fields and with this, most of the time tanh derivative was giving zero. Thus, I changed my weights range to -0.25 to 0.25 and then finally to -0.1 to 0.1. With the weights in this range, my neural network was giving good performance.

### Normalizing the input: 
In the given input data set most of the values are zeros and rest are around 250. With this large number of input value, network was not behaving properly.
Since each input variable should be preprocessed so that its mean value, averaged over the entire training sample, is close to zero, or else it will be small compared to its standard deviation, I normalized the input value by diving each input by 256 to make input value close to zero. This helped my network to behave properly and converge. 

### Deciding factor for the output
Since we are using hyperbolic tangent transfer function for the output layer neurons too. Outputs of the network will not be in binary. It will actually range from -1 to 1. So the highest value from the output layer neurons which would be close to 1 can be chosen to get the correct output which classifies the input pattern. From the network output list, we get the index for the output with value close to 1 and this becomes the correct output ‘y’ which classifies the input pattern.

### Calculation of delta:
Since we are getting desired output in the range of 0 to 9 for each input pattern and also output that classifies the input pattern is in the range of 0 to 9, we create two 10*1 matrix ‘D’ and ‘Y’ for desired output and actual output of the network. These matrix have all zeros values except the index which represents the desired output and actual output in the respective matrices. Since we need to update the weights for the entire network in each iteration, considering only ‘d’ and ‘y’ for the calculation of delta for neurons doesn’t bring convergence for the network. So, I used this ‘D’ and ‘Y’ matrices for the calculation of output and hidden layers neurons’ delta using the backpropagation algorithm.

### Calculation of classification error and Mean Squared Error(MSE)
For each input from the input data set, whenever the desired output d for the input x doesn’t match with the actual output y, we increment the classification error by 1 and after each epoch, we show the number of misclassifications for the given input dataset. We need to make sure that after each epoch, our classification error should come down to get the better performance of the network for pattern classification. 
We calculate the MSE by taking the summation of normed square of difference between desired output and actual output in each iteration of the inputs and then diving it by data set size after each epoch. MSE is very important factor as we uses this to find optimal weights using backpropagation algorithm that minimizes the MSE.

### Failure Scenarios:
1.	When learning rate was high: With learning rate of 0.1, my network was not converging but when I took my learning rate as 0.01, it started converging fast and I was able to get nice curve for epochs and misclassification error rate.
2.	When weights had a wide range for initial value: weights in range from -1 to 1 was giving large induced local fields and with this most of the time tanh derivative was giving zero and thus I chose weights in the range of -0.1 to 0.1
3.	When input data was not normalized: With the given input data which has some data as high as 255, delta value was not coming proper. Reason was the large induced local field values which was because of large input value for some input data. When I tried to normalize the data, my network converged easily.

