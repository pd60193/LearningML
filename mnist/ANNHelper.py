import numpy as np

#softmax function in numpy
def softmax(A):
    expA = np.exp(A)
    return expA/(expA.sum(axis = 1,keepdims = True))
#relu function which acts as teh non linearity
def relu(X):
    return X*(X>0)

#derivative with respect to W1
def derivative_with_2(T,Y,Z):
    return Z.T.dot(T-Y)

#derivative with respect to W2
def derivative_with_1(X,T,Y,Z,W2):
    return X.T.dot((T-Y).dot(W2.T)*np.sign(Z))

#derivative with respect to b2
def derivative_with_b2(T,Y):
    return (T-Y).sum(axis = 0)

#derivative with respect to b1
def derivative_with_b1(T,Y,Z,W2):
    return ((T-Y).dot(W2.T)*np.sign(Z)).sum(axis=0)

#Forward method for one layered ANN
def Forward(X,W1,b1,W2,b2):
    Z = relu(X.dot(W1)+b1)
    return softmax(Z.dot(W2)+b2),Z
    
    
