import numpy as np
import pandas as pd
import os

#cross entropy cost function
def cost(T,Y):
    np.place(Y,Y==0,10e-280)
    return (T*np.log(Y)).sum()

#get data from training set and normalizing it
def get_data(path):
    df = pd.read_csv(path)
    data = df.as_matrix().astype(np.float32)
    #print(data)
    X = data[:,1:]
    Y = data[:,0]
    N,D = X.shape
    X = X/X.max()
    #mu = X.mean(axis=0)
    #var = X.var(axis=0)
    #np.place(var,var==0,1)
    #X = (X-mu)/var
    return X,Y

#get data from test set and normalizing it
def get_test_data(path):
    df = pd.read_csv(path)
    data = df.as_matrix().astype(np.float32)
    #print(data)
    X = data
    N,D = X.shape
    X = X/X.max()
    #mu = X.mean(axis=0)
    #var = X.var(axis=0)
    #np.place(var,var==0,1)
    #X = (X-mu)/var
    return X

#initializing weights and bias
def init_weight_and_bias(M1,M2):
    W = np.random.randn(M1,M2)/np.sqrt(M1+M2)
    b = np.zeros(M2)
    return W,b

#calculate error rate between output predictiona nd input labels
def error_rate(T,P):
    test_matrix = np.argmax(T,axis=1)
    predict_matrix = np.argmax(P,axis=1)
    return np.mean(test_matrix!=predict_matrix )


#converting from input labels to one hot encoded vector
def y2indicator(Y):
    Y = Y.astype(np.int32)
    N = len(Y)
    K = len(set(Y))
    T = np.zeros((N,K))
    for n in range(N):
        T[n,Y[n]]=1
    return T

