# LearningML


I have been following an UDEMY course by lazy programmer. This course had a workshop on the following projects.
- MNIST Handrwitten digits recognition
- Parity Problem solving using ANN and RNN

The first problem involves identifying handwritten digits. A training dataset has been provided by kaagle along with a validation set. I have implemented an ANN using only numpy along with RMSProp and velocity to train on this dataset. Following this I try to make predictions on the validation set.

The second problem tries to predict the parity of a binary number. If the number of 1's in a binary representation of a number is odd then the parity is set to 1, otherwise 0. We try to make an ANN learn this problem. Howeve,r we realize that it is difficult for an ANN to understand this problem as it would require a large number of hidden units. Hence, we try using RNN where each bit of the input number is treated as sequence and we feed back the hidden values at each step. There is a marked improvement in the output.

## Data
Contains the following data file 
- train.csv : contains the mnist handrwitten digit training dataset from Kaggle
- test.csv : contains the mnist handrwitten digit validation dataset from Kaggle

## mnist

- ANNHelper.py is the helper file containing derivatives and non linearity for a single layered neural network in numpy
- MYANN.py takes as input handwritten digits in form of 32x32 image and trains these images on a sinle layered Neural Network with 500 hidden untis. The non linearity used is RELU function. 

## util

- Utils.py contains code for loading handwritten digits from csv training and testing file along with some common code.
- util_parity contains code for laoding all possible 2048 combiations of 12 bit binary numbers. This dataset is used for parity problem.

## parity

mlp_parity.py : takes as input an exhaustive permuation of 12 bit binary number and learns the parity problem. Following this it tries to predict the parity of an input sequence. It uses Artificial Neural Network for this problem. 

rnn_parity : tries to learn the same problem like above except that it uses a recurrent neural network.

## Output

MNIST Handwritten difgits classification with ANN yeilded a classification rate of 0.972
Parity with ANN yielded a classification rate of 0.5
Parity with RNN yielded a classification rate of 1.0

Also later in another project MNIST Handwritten difgits classification yielded a rate of 0.99 using Convolutional Neural Network 
